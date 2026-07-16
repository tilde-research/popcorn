#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 256;
constexpr int kPackBytes = 16;

int blocks_for(int64_t work) {
  return static_cast<int>(std::min<int64_t>((work + kThreads - 1) / kThreads, 65535));
}

template <typename scalar_t, int N>
struct alignas(sizeof(scalar_t) * N) Pack {
  scalar_t v[N];
};

template <typename opmath_t>
__device__ __forceinline__ opmath_t sigmoid(opmath_t x) {
  return opmath_t(1) / (opmath_t(1) + ::exp(-x));
}

template <typename scalar_t, int N>
__global__ void fwd_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, int64_t packs) {
  using opmath_t = at::opmath_type<scalar_t>;
  using P = Pack<scalar_t, N>;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < packs; i += stride) {
    const P pa = reinterpret_cast<const P*>(a)[i];
    const P pb = reinterpret_cast<const P*>(b)[i];
    P po;
#pragma unroll
    for (int j = 0; j < N; ++j) {
      const opmath_t x = static_cast<opmath_t>(pa.v[j]);
      po.v[j] = static_cast<scalar_t>(x * sigmoid(x) * static_cast<opmath_t>(pb.v[j]));
    }
    reinterpret_cast<P*>(out)[i] = po;
  }
}

template <typename scalar_t, int N>
__global__ void bwd_kernel(const scalar_t* g, const scalar_t* a, const scalar_t* b, scalar_t* da, scalar_t* db,
                           int64_t packs) {
  using opmath_t = at::opmath_type<scalar_t>;
  using P = Pack<scalar_t, N>;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < packs; i += stride) {
    const P pg = reinterpret_cast<const P*>(g)[i];
    const P pa = reinterpret_cast<const P*>(a)[i];
    const P pb = reinterpret_cast<const P*>(b)[i];
    P pda, pdb;
#pragma unroll
    for (int j = 0; j < N; ++j) {
      const opmath_t x = static_cast<opmath_t>(pa.v[j]);
      const opmath_t y = static_cast<opmath_t>(pb.v[j]);
      const opmath_t go = static_cast<opmath_t>(pg.v[j]);
      const opmath_t s = sigmoid(x);
      pda.v[j] = static_cast<scalar_t>(go * y * s * (opmath_t(1) + x * (opmath_t(1) - s)));
      pdb.v[j] = static_cast<scalar_t>(go * x * s);
    }
    reinterpret_cast<P*>(da)[i] = pda;
    reinterpret_cast<P*>(db)[i] = pdb;
  }
}

bool packable(std::initializer_list<const void*> pointers) {
  for (const void* pointer : pointers) {
    if (reinterpret_cast<uintptr_t>(pointer) % kPackBytes != 0) return false;
  }
  return true;
}

}  // namespace

torch::Tensor fwd(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type());
  const c10::cuda::CUDAGuard guard(a.device());
  a = a.contiguous();
  b = b.contiguous();
  auto out = torch::empty_like(a);
  const int64_t numel = a.numel();
  if (numel == 0) return out;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "swiglu_fwd", [&] {
    const scalar_t* pa = a.const_data_ptr<scalar_t>();
    const scalar_t* pb = b.const_data_ptr<scalar_t>();
    scalar_t* po = out.mutable_data_ptr<scalar_t>();
    const bool aligned = packable({pa, pb, po});
    constexpr int N = kPackBytes / sizeof(scalar_t);
    const int64_t packs = aligned ? numel / N : 0;
    const int64_t tail_at = packs * N;
    if (packs) fwd_kernel<scalar_t, N><<<blocks_for(packs), kThreads, 0, stream>>>(pa, pb, po, packs);
    if (numel - tail_at)
      fwd_kernel<scalar_t, 1>
          <<<blocks_for(numel - tail_at), kThreads, 0, stream>>>(pa + tail_at, pb + tail_at, po + tail_at, numel - tail_at);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::tuple<torch::Tensor, torch::Tensor> bwd(torch::Tensor g, torch::Tensor a, torch::Tensor b) {
  const c10::cuda::CUDAGuard guard(a.device());
  g = g.contiguous();
  a = a.contiguous();
  b = b.contiguous();
  auto da = torch::empty_like(a);
  auto db = torch::empty_like(b);
  const int64_t numel = a.numel();
  if (numel == 0) return {da, db};
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "swiglu_bwd", [&] {
    const scalar_t* pg = g.const_data_ptr<scalar_t>();
    const scalar_t* pa = a.const_data_ptr<scalar_t>();
    const scalar_t* pb = b.const_data_ptr<scalar_t>();
    scalar_t* pda = da.mutable_data_ptr<scalar_t>();
    scalar_t* pdb = db.mutable_data_ptr<scalar_t>();
    const bool aligned = packable({pg, pa, pb, pda, pdb});
    constexpr int N = kPackBytes / sizeof(scalar_t);
    const int64_t packs = aligned ? numel / N : 0;
    const int64_t tail_at = packs * N;
    if (packs) bwd_kernel<scalar_t, N><<<blocks_for(packs), kThreads, 0, stream>>>(pg, pa, pb, pda, pdb, packs);
    if (numel - tail_at)
      bwd_kernel<scalar_t, 1><<<blocks_for(numel - tail_at), kThreads, 0, stream>>>(
          pg + tail_at, pa + tail_at, pb + tail_at, pda + tail_at, pdb + tail_at, numel - tail_at);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {da, db};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &fwd);
  m.def("bwd", &bwd);
}
