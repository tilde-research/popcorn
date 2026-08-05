# Known kernel issues

Confirmed upstream kernel bugs found by the harness. Adapter bugs do not belong
here; they get fixed. Each entry names the op and backend, the failing case,
the observed error, and how to reproduce it. Backends listed here stay
gated with `predicate` / dtype annotations / `forward_only=True`, or excluded by
learned validity regions from the report table, so dispatch avoids the broken
region and the support matrix reports it honestly.

Reproduce any entry with:

```bash
uv run python -m popcorn.bench run <op> --backend <backend>
```

## bit_linear / fla

- Case: any call with a non-default `eps`
- Error: output ignores the requested epsilon
- Cause: `fla.modules.fused_bitlinear.bit_linear` declares `eps=1e-8` but calls
  `layer_norm_linear_quant_fn` without forwarding it, so the kernel always runs
  with that function's default of 1e-6 (flash-linear-attention 0.4.2).
- Marking: the popcorn reference fixes eps at 1e-6 and exposes no `eps`
  argument, matching what the kernel actually computes.

## layer_norm_swish_linear, rms_norm_swish_linear / fla

- Case: any backward through `linear_weight`, e.g. `hidden=1024,out_features=128,float32`
- Error: `grad linear_weight: err 2.268e+01 > max(...)`
- Cause: `LayerNormSwishGateLinearFn.backward` recomputes the pre-gate norm
  output `y` (`layer_norm_gated_bwd(..., recompute_output=True)`) and computes
  `dlinear_weight = dout.T @ y`, dropping the `silu(g)` gate factor that the
  forward applied before the linear (flash-linear-attention 0.4.2,
  `fla/modules/fused_norm_gate.py`). All other gradients are correct.
- Marking: both backends registered with `forward_only=True`.

## group_norm / liger

- Case: any backward with 2-D `[tokens, channels]` input, e.g.
  `channels=128,tokens=33,num_groups=4,+bias`
- Error: `backward: RuntimeError: shape '[33, 128]' is invalid for input of size 540672`
- Cause: `LigerGroupNormFunction.backward` reshapes gradients assuming a 3-D
  `[batch, channels, length]` layout; the forward happens to accept 2-D inputs
  (liger-kernel 0.6.4, `liger_kernel/ops/group_norm.py`).
- Marking: backend registered with `forward_only=True`.

## embedding / liger (16-bit atomic accumulation)

- Case: 16-bit backward with repeated indices, especially small vocabularies,
  e.g. `batch=8,seq=33,vocab=2,bfloat16`
- Error: `grad weight: err 9.673e-01 > max(2*2.318e-01, 2e-02*34.3)`
- Cause: the backward atomically accumulates directly into 16-bit
  `grad_weight`; collision-heavy rows compound rounding beyond the dtype floor
  (liger-kernel 0.7.0, `liger_kernel/ops/experimental/embedding.py`).
- Marking: backend gated to fp32.

## bias_gelu / fla

- Case: any call with `requires_grad` inputs, e.g. `hidden=8,batch=(2, 3),float32`
- Error: `backward: RuntimeError: expected Variable or None (got tuple)`
- Cause: `fla.modules.activations.GeLUFunction.backward` computes
  `tmp = bias_gelu_bwd(...)`, which returns the `(grad_input, grad_bias)` tuple,
  then returns `tmp, tmp` instead of unpacking it (flash-linear-attention 0.4.2,
  `fla/modules/activations.py`). `bias_gelu_bwd` also reduces the bias gradient
  with `sum(dim=0)`, which is only correct for 2-D inputs.
- Marking: backend registered with `forward_only=True`.

## group_norm / liger (narrow groups)

- Case: groups narrower than 8 channels, in every dtype, e.g.
  `channels=8,tokens=4096,num_groups=4,+bias,float32`
- Error: `out0: err 1.093e-02 > max(2*3.964e-05, 2e-05*4.6)`
- Cause: the kernel's group statistics lose precision on tiny groups where
  torch stays exact; fp32 drifts, fp16/bf16 lose whole digits
  (liger-kernel 0.6.4, `liger_kernel/ops/group_norm.py`).
- Marking: predicate skips groups narrower than 8 channels.

## group_norm, group_norm_linear / fla (narrow groups)

- Case: fp32 backward with groups narrower than 8 channels, e.g.
  `channels=8,tokens=8,num_groups=4,+norm_bias,float32`
- Error: `grad x` off by ~2e-3 relative on a few percent of elements
  (caught by the strict unit test, within harness floors)
- Cause: the kernel's group statistics round differently than torch through
  the tiny per-group variance (flash-linear-attention 0.4.2,
  `fla/modules/layernorm.py`).
- Marking: predicate skips groups narrower than 8 channels.

## layer_norm / liger (tiny rows)

- Case: `normalized_shape < 8`, e.g. `normalized_shape=2,batch=(2, 2048),float16,+bias`
- Error: `grad x: err 3.918e+02 > max(2*4.344e-01, 1e-03*1341.4)`
- Cause: the backward loses catastrophically on rows of 2-3 elements while
  torch stays exact (liger-kernel 0.6.4, `liger_kernel/ops/layer_norm.py`).
- Marking: learned validity region (was `supports` Range starting at 8); re-map with `bench map`.

## layer_norm family / fla (rows of one element)

- Case: `normalized_shape=1` (or `hidden=1` for the fused variants)
- Error: `grad x: err 2.829e-02 > max(2*0.000e+00, 2e-05*1.0)`
- Cause: normalizing a single element is degenerate (the output is exactly
  `bias`, `grad x` is exactly zero); fla's backward emits nonzero junk that
  grows with `out_features` (flash-linear-attention 0.4.2,
  `fla/modules/layernorm.py`). Applies to `layer_norm`, `layer_norm_gated`,
  `layer_norm_linear`, and `layer_norm_linear_quant`.
- Marking: learned validity region (was `supports` Range starting at 2); re-map with `bench map`.

## linear_attn / fla (short sequences)

- Case: any call with `seq < heads`, e.g. `seq=2,heads=4`
- Error: `crash: DeprecationWarning: Input tensor shape suggests potential format mismatch`
- Cause: `chunk_linear_attn` raises (not warns) on `seq_len < num_heads`,
  guessing the layout is head-first; short sequences are legitimate
  (flash-linear-attention 0.4.2, `fla/ops/linear_attn/chunk.py`).
  The fused recurrent path also multiplies by `scale` without filling a
  missing value.
- Marking: predicate skips `seq < heads`; recurrent calls require an explicit
  `softmax_scale`.

## rebased / fla (16-bit gradients)

- Case: fp16 with `use_normalize=True`, e.g. `key_dim=16,seq=8,float16`
- Error: `grad q: err inf > max(...)` (fp16), `grad q: err 1.689e+00` (bf16)
- Cause: gradients through the quadratic-feature normalizer overflow fp16 and
  miss bf16 tolerance (flash-linear-attention 0.4.2, `fla/ops/rebased`).
- Marking: backend gated to float32 inputs.

## gsa / fla (backward is not release-safe)

- Case: any backward with `kv_heads < q_heads`, e.g. `q_heads=4,kv_heads=2,seq=2`
- Error: `backward: RuntimeError: The size of tensor a (4) must match the size
  of tensor b (2) at non-singleton dimension 2`; on some cases the kernel
  instead dies with `CUDA error: an illegal memory access was encountered`,
  poisoning the CUDA context.
- Cause: `chunk_gsa`'s backward does not implement the kv-head grouping its
  forward accepts (flash-linear-attention 0.4.2, `fla/ops/gsa/chunk.py`).
  On Torch 2.13 and Triton 3.7.1, even the smallest ungrouped backward did not
  complete within 180 seconds; its forward passed in 103 seconds.
- Marking: predicate skips grouped kv heads (`kv_heads != heads`) and the
  surviving implementation is forward-only.

## kda / fla (dtype and backward restrictions)

- Case: fp32, e.g. `key_dim=64,seq=64,float32`
- Error: `grad k: err 1.959e-04 > max(2*0.000e+00, 2e-05*8.9)`
- Cause: the chunked kernel's fp32 accumulation order lands 1.1-1.5x past the
  harness floor even with `TRITON_F32_DEFAULT=ieee`; 16-bit inputs pass
  (flash-linear-attention 0.4.2, `fla/ops/kda`).
  Under Torch 2.13 and Triton 3.7.1, backward on the minimal fp16 smoke case
  also triggers an illegal memory access while forward passes.
- Marking: backend gated to 16-bit inputs and forward-only.

## mesa_net / fla (approximate solver)

- Case: every case, all dtypes, e.g. `key_dim=32,seq=2,float32`
- Error: `out0: err 1.284e-02 > max(2*0.000e+00, 2e-05*1.7)`
- Cause: the kernel solves the per-step ridge system with truncated conjugate
  gradients (30 iterations) instead of exactly; the ~1% residual is inherent to
  the algorithm, not a precision artifact (flash-linear-attention 0.4.2,
  `fla/ops/mesa_net`). The backend stays registered so the matrix reports it,
  but it cannot pass an exactness harness.
- Marking: no dispatch gate; the GPU smoke test is a non-strict xfail and the
  full grid shows the backend as unsupported (✘).

## rwkv4 / fla (16-bit state trajectory)

- Case: any 16-bit backward, e.g. `channels=64,seq=33,bfloat16`
- Error: `grad w: err 1.374e+01 > max(2*7.656e-02, 2e-02*25.0)`
- Cause: the kernel saves the per-step `(alpha, beta, eps)` trajectory in the
  input dtype and the backward reconsumes it; 16-bit quantization of the `eps`
  log offset compounds through the recurrence (flash-linear-attention 0.4.2,
  `fla/ops/rwkv4/fused_recurrent.py`).
- Marking: backend gated to float32 inputs.

## rwkv7_channel_mixing / fla (unmasked 4096-element tile)

- Case: `batch * seq * intermediate` is not divisible by 4096, e.g.
  `batch=2,seq=2,intermediate=128,float32`
- Error: `grad x: err is non-finite`; other input and weight gradients are
  likewise non-finite, with corrupted forward values on some allocations.
- Cause: `rwkv_channel_mixing_pow_and_relu` launches 4096 offsets per program
  but loads and stores without a boundary mask, so a partial final tile reads
  and writes out of bounds (flash-linear-attention 0.4.2,
  `fla/ops/rwkv7/channel_mixing.py`).
- Marking: predicate requires the fused intermediate tensor size to be
  divisible by 4096.

## log_linear_attn / fla (multiple heads)

- Case: any call with more than one query head, e.g. `heads=4,seq=64`
- Error: `out0: err 8.857e+01` against a scale of 66, i.e. a completely
  different result
- Cause: the kernel accepts `heads > 1` with its required single-kv-head `k`
  but disagrees with fla's own `naive_log_linear_attn` on those inputs
  (`heads=1` matches to 1e-4); its backward also returns grad `q` with `k`'s
  single-head shape (flash-linear-attention 0.4.2, `fla/ops/log_linear_attn`).
- Marking: runtime guard enforces the upstream launch contract: one head,
  `key_dim` divisible by 64, and power-of-two `value_dim`.

## ttt / fla (backward accuracy)

- Case: any backward, all dtypes, e.g. `head_dim=32,seq=64,float32`
- Error: `grad q: err 3.626e-02 > max(2*0.000e+00, 2e-05*630.0)`
- Cause: the fused backward misses the torch-autograd gradient by roughly 3x
  the tolerance in every dtype, including ieee fp32, so the deviation is in the
  kernel's gradient math rather than matmul precision
  (flash-linear-attention 0.4.2, `fla/ops/ttt`).
- Marking: backend registered with `forward_only=True`.

## bit_linear family / fla (quantization boundaries)

- Case: sporadic, large fp32 cases, e.g. `hidden=8,out_features=128,batch=(2, 2048),float32`
- Error: `out0: err 2.372e-02 > max(2*4.914e-06, 2e-05*15.5)` where the error
  equals exactly one int8 quantization bucket (`max|h| / 127`)
- Cause: the fused kernel computes the pre-quant norm with different rounding
  than `F.layer_norm`, so activations landing on an int8 rounding edge flip one
  bucket. Pointwise comparison of quantized outputs is ill-conditioned at the
  edges; more tokens mean more edge hits. Applies to `bit_linear`,
  `layer_norm_linear_quant`, and `rms_norm_linear_quant`
  (flash-linear-attention 0.4.2, `fla/modules/fused_bitlinear.py`).
- Marking: no dispatch gate; the GPU smoke test is a non-strict xfail and large
  fp32 grids can show occasional fails.

## tvd / liger (tie-breaking)

- Case: sporadic, large fp32 grids with `reduction=sum` or `batchmean`, e.g.
  `tokens=1024,vocab=4096,float32,reduction=sum`
- Error: `grad p: err 3.489e-01 > max(2*5.837e-08, 2e-05*1.0)`, consistent with
  single elements where `p == q` exactly
- Cause: the gradient of `|p - q|` is undefined at ties; torch's `sign(0)` is 0
  while the kernel emits ±0.5, so exact fp32 ties disagree by one subgradient
  choice (liger-kernel 0.6.4, `liger_kernel/ops/tvd.py`).
- Marking: no dispatch gate; the GPU smoke test is a non-strict xfail and large
  fp32 grids can show occasional fails.

## comba / fla (naive reference disagrees with the kernel)

- Case: any input, e.g. `seq=33,float32`: `naive_recurrent_comba` vs
  `chunk_comba` differ by 6.6e-2 where the chunk kernel is self-consistent
- Cause: the chunk kernel computes the delta correction `v - S^T p` from the
  pre-decay state, `naive_recurrent_comba` decays first and then corrects
  (flash-linear-attention 0.4.2, `fla/ops/comba/naive.py`).
- Marking: none; our torch reference follows the chunk kernel (pre-decay read),
  which matches it to 5e-7 under ieee fp32.

## poly_norm / liger (fp16 overflow)

- Case: fp16 with wide rows, e.g. `hidden=4096,float16`
- Error: `out0: err 1.550e+01`, plus grad errors in the hundreds
- Cause: the kernel forms `x^3 * x^3` in the input dtype for the cubic term's
  rms, so fp16 draws past ~6 sigma overflow toward the 65504 ceiling and
  poison the row statistics (liger-kernel 0.6.4, `liger_kernel/ops/poly_norm.py`).
- Marking: backend gated to `Float32 | BFloat16`.

## poly_norm / liger (tiny rows)

- Case: bf16 with rows of 1-2 elements, e.g. `hidden=2,batch=(2, 3),bfloat16`
- Error: `grad x: err 2.660e-01 > max(2*3.451e-02, 2e-02*11.7)`; `hidden >= 3`
  passes across the grid
- Cause: with 1-2 elements the rms terms are nearly degenerate (`norm(x)` is
  close to `sign(x)`), and the kernel's bf16 backward loses the cancellation
  torch preserves (liger-kernel 0.6.4, `liger_kernel/ops/poly_norm.py`).
- Marking: learned validity region (was `supports` Range starting at 3); re-map with `bench map`.

## comba / fla (backward unsafe)

- Case: chunked backward, including
  `batch=8,heads=2,key_dim=32,seq=65,value_dim=16,bfloat16`.
- Error: `CUDA error: an illegal memory access was encountered`. Earlier fp16
  cases also exceeded the gradient tolerance without crashing.
- Cause: flash-linear-attention 0.4.2's chunked backward writes out of bounds
  on valid layouts under Torch 2.13 and Triton 3.7.1.
- Marking: both FLA implementations are forward-only. The recurrent path was
  already forward-only; the chunked path is now marked the same.

## gated_delta_product / fla (fp16 outputs)

- Case: fp16, e.g. `seq=32,num_householder=2,float16`
- Error: `out0: err 1.117e-03 > max(2*3.467e-04, 1e-03*1.0)`, marginal but
  recurring; the kernel also asserts against float32 inputs outright
- Cause: fp16 accumulation across the `num_householder` inner writes lands just
  past the dtype floor (flash-linear-attention 0.4.2,
  `fla/ops/gated_delta_product/chunk.py`).
- Marking: backend gated to `BFloat16` only.

## softmax / quack (strictly 2-D, broken backward)

- Case: any backward, e.g. a contiguous `(8, 32)` fp32 input
- Error: `ValueError: Mismatched mdY.strides[1]` from the cute-DSL launcher
- Cause: the backward kernel's stride contract rejects plain contiguous
  cotangents, and the forward asserts `x.ndim == 2` outright
  (quack-kernels 0.5.0, `quack/softmax.py`).
- Marking: backend not registered; liger serves the op.

## topk / quack (alignment constraints, fp32 mismatch)

- Case: `topk(x, k)` on `(16, 32)` fp32, any k
- Error: `k=1/4` crash with stride-alignment `ValueError`s, `N=33` asserts
  `N must be a power of 2`, and at `k=8` fp32 values disagree with
  `torch.topk` while fp16/bf16 agree
- Cause: the kernel requires power-of-two rows, value strides divisible by 8,
  returns int32 indices, and its fp32 path selects different elements
  (quack-kernels 0.5.0, `quack/topk.py`).
- Marking: op dropped; the surviving envelope (16-bit, k=8, pow2 rows) is too
  narrow to justify it.

## nsa / fla, popcorn (fp16 gradients)

- Case: fp16 backward, e.g. `batch=2,head_dim=64,kv_heads=1,q_heads=16,seq=2,float16,softmax_scale=0.25`
- Error: `grad k: err 7.429e-03 > max(2*2.514e-03, 1e-03*6.1)` (fla); the
  popcorn composition fails the same cases by similar margins (~2-3x budget)
- Cause: fp16 accumulation noise through the three-branch composition (softmax
  in the compression/selection kernels plus the gated blend) lands just past
  the harness budget on both stacks; bf16 passes everywhere, so it is dtype
  noise rather than gradient math.
- Marking: both backends are gated to bfloat16 and reject launch groups where
  `q_heads / kv_heads` is not a multiple of 16.

## nsa upstream (tilde-research/nsa-release), fixed in the vendored copy

`popcorn.impls.nsa_tl` vendors the tilde one-pass selection kernel and
composition with these upstream bugs fixed (details in the file header):

- The compression causal mask was inverted (flex mask functions return True
  where attention is allowed), leaking future compressed blocks; confirmed by
  upstream issue reports of loss collapsing to ~0 after ~200 steps.
- flex lse is returned as `[B, HQ, T]` but consumed as `[B, T, HQ]` by
  `parallel_nsa_topk`, so every topk probability used the wrong normalizer.
- The selection backward kernel dropped the `col >= 0` sentinel guard its
  forward has, turning fla's -1 "unused slot" indices into negative-offset
  atomic writes.
- The selection backward read the incoming gradient with the output's strides;
  a permuted (e.g. einsum-produced) cotangent was read element-shuffled,
  silently corrupting dq/dk/dv.
- The compression flex call ignored custom softmax scales, and the
  sliding-window branch ran flash-attn 2 with `window_size=(-1, 0)` (full
  causal) instead of the requested window.

## wall_attn / popcorn (16-bit gate gradients)

- Case: 16-bit backward on short sequences, e.g.
  `kv_heads=2,q_heads=4,seq=8,+g_scalar,bfloat16`
- Error: `grad g: err 3.49e-02 > max(2*..., 2e-2*...)` (bf16), `grad k` on some
  fp16 short-seq cases; the forward and `q`/`k`/`v` gradients stay within budget
- Cause: the kernel returns analytic gate gradients derived structurally
  (`dg = LN2 * q * dq`, then a reverse cumsum); in 16-bit that chain lands
  ~1.1-1.7x past the dtype floor while the value gradients stay clean
  (github.com/tilde-research/wall-attention-release). fp32 is exact because
  popcorn pins `TRITON_F32_DEFAULT=ieee`.
- Marking: backend gated to float32.
- Note (adapter, not a kernel bug): the harness seeds `g`/`g_scalar` in the
  log-sigmoid (<= 0) domain. The kernel's per-block reference frame assumes a
  monotone non-increasing prefix; a mixed-sign gate overflows its intra-block
  `exp2` and diverges from the reference by ~10-40% in every dtype (upstream
  only tests the decay domain).

## path_attn / fla (backward miscompiles)

- Case: any backward, e.g. bf16 `(1, 8, 4, 64)` with fp32 `w`/`beta`
- Error: `CompilationError: T marked as constexpr and listed in
  do_not_specialize/do_not_specialize_on_alignment` in
  `chunk_cumprod_householder_bwd_kernel`
- Cause: the backward kernel declares `T: tl.constexpr` while also listing `T`
  in `do_not_specialize`, which current triton rejects at compile time
  (flash-linear-attention 0.4.x, `fla/ops/path_attn/cumprod_householder_bwd.py`).
- Marking: `forward_only=True`.

## kda / fla:recurrent (memory-state-dependent non-finite outputs)

- Case: 5 of 12 sampled grid cases, both fp32 and bf16, assorted shapes
  (e.g. `batch=8, heads=1, key_dim=192, seq=65, value_dim=256`)
- Error: `out0: err is non-finite` — one `value_dim`-sized row of the output
  (256 of 133120 elements) comes back NaN.
- Cause: `fused_recurrent_kda` only misbehaves under specific CUDA caching
  allocator states: the same tensors run clean in a fresh process, and the
  fp64/fp32 references and all inputs are finite. That signature points at an
  uninitialized or out-of-bounds read in the upstream triton kernel
  (flash-linear-attention 0.4.2), triggered when preceding allocations leave
  the right garbage behind. Reproduce by instrumenting `_error` inside
  `run_case` on case `069be4f4b502`; direct calls on the same inputs pass.
- Marking: none — the fail rows stay in the report, so tuned dispatch never
  routes those cases to `fla:recurrent`, and dispatch-time validation blocks
  it elsewhere. Worth an upstream report.

## gated_delta_rule / fla (chunk path does not compile in budget)

- Case: the smallest valid forward,
  `batch=1,heads=1,key_dim=16,seq=2,value_dim=16,float32`.
- Error: the chunk kernel does not complete compilation within 150 seconds;
  the isolated 300-second smoke watchdog also expired.
- Cause: flash-linear-attention 0.4.2's chunk path does not compile to a usable
  kernel under Torch 2.13 and Triton 3.7.1.
- Marking: the chunk adapter is removed. The independently verified recurrent
  forward path remains registered.

## gated_oja_rule / fla (chunk backward does not compile in budget)

- Case: the smallest valid case,
  `batch=1,heads=1,key_dim=16,seq=2,value_dim=16,float32`.
- Error: forward passes in 99 seconds, while backward exceeds both the
  180-second focused probe and the isolated 300-second smoke watchdog.
- Cause: the upstream chunk backward does not produce a runnable kernel on
  Torch 2.13 and Triton 3.7.1.
- Marking: chunk and recurrent implementations are forward-only.

## rwkv7 / fla (float32 unsupported on the upgraded stack)

- Case: the smallest fp32 smoke case.
- Error: upstream warns that `ChunkDeltaRuleFunction` does not support float32
  and does not finish within the smoke watchdog.
- Cause: RWKV-7 lowers through flash-linear-attention's generalized delta-rule
  chunk path, whose current fp32 specialization is unsupported.
- Marking: the adapter is gated to bfloat16; its bfloat16 forward/backward
  smoke passes.

## deltaformer / fla (undeclared FlashAttention-2 dependency)

- Case: every call in the isolated FLA environment.
- Error: the adapter is skipped because `flash_attn` is absent; forcing the
  upstream function fails when its second stage imports FlashAttention-2.
- Cause: `fla.ops.deltaformer.deltaformer_attn` depends on FlashAttention-2,
  but flash-linear-attention does not declare it and Popcorn's `fla` extra
  cannot promise a compatible source build for every supported torch/CUDA
  pair.
- Marking: the adapter is removed rather than making isolated FLA correctness
  depend on an undeclared source build. The torch reference remains available.

## rms_norm / liger (DTensor check assumes an import someone else did)

- Case: every liger `rms_norm` call in a minimal environment.
- Error: `AttributeError: module 'torch.distributed' has no attribute 'tensor'`
- Cause: liger 0.7.0's kernel guard runs
  `isinstance(X, torch.distributed.tensor.DTensor)` without importing
  `torch.distributed.tensor`; the attribute only exists after something else
  (typically transformers) imports that submodule, so environments with bare
  torch expose the bug.
- Marking: none. `popcorn.kernels.rms_norm` imports the submodule at module
  load, which keeps liger's check working everywhere.

## grpo_offpolicy / liger (grpo_loss requires transformers)

- Case: every liger `grpo_offpolicy` call without transformers installed.
- Error: `ImportError: The attribute 'grpo_loss' requires the 'transformers'
  library` at call time; liger-kernel declares no runtime dependency on it.
- Marking: the adapter's predicate skips liger when transformers is absent, so
  dispatch falls back cleanly. The sweep's typed liger phase installs
  transformers to measure the path.

## linear_cross_entropy / quack (no dtype round-trips in 0.5.0)

- Case: every scheduled quack `linear_cross_entropy` case.
- Error: `TypeError: a_dtype should be float16 or float8` for fp32 inputs; fp16
  inputs run but return an fp32 loss, which breaks the op's same-dtype return.
- Cause: quack 0.5.0's `chunked_linear_cross_entropy` gemm accepts only
  fp16/fp8 activations while the loss is always fp32. Earlier releases took
  fp32 activations, which is what the adapter was written against.
- Marking: the adapter is removed rather than narrowed, since no annotation
  satisfies both constraints and adapters do not cast outputs. Restore it if a
  quack release accepts fp32 again or returns the input dtype.

## linear_jsd / liger (non-finite valid inputs in 0.7.0)

- Case: ordinary finite inputs at model widths, including
  `hidden=4096,teacher_hidden=3072,tokens=1448,vocab=129,bfloat16`.
- Error: non-finite loss, student gradient, and student-weight gradient.
- Cause: the fused path forms logits before its fp32 cast, then exponentiates
  log-probabilities before mixing them. Large but finite logit ranges can
  therefore overflow the projection or underflow both probabilities to zero.
- Marking: the adapter is removed because no dtype or shape annotation
  guarantees its value-dependent numerical contract. Restore it when upstream
  computes the projection and mixture stably.

## log_linear_attn, linear_attn / fla (hard crashes that poison the worker)

- Case: grid rows crash the worker process outright (segfault, no Python
  traceback), concentrated in `log_linear_attn` and `linear_attn`.
- Error: the crashing case records `crash`; before the sweep recycled poisoned
  workers, the cases that followed recorded
  `inputs: AcceleratorError: CUDA error: an illegal memory access was encountered`.
- Cause: fla Triton kernels write out of bounds on some shapes. CUDA surfaces
  the fault asynchronously at the next sync point and never recovers the
  context, so every later kernel launch in that process fails too.
- Marking: none. `crash` rows are honest per-case evidence and dispatch never
  routes to an implementation without a passing row. The sweep worker exits
  after recording any row whose reason carries an unrecoverable CUDA mark
  (`scripts/bench_sweep.py POISON_MARKS`), so the supervisor restarts it with a
  clean context and one bad case can no longer contaminate its neighbors.

## wall_attn, nsa / popcorn (first-party kernels that build on fla)

- Case: the isolated `popcorn` sweep phase recorded only `skip` rows for these
  impls; every other eligible first-party impl measured normally.
- Cause: `popcorn.impls.wall_attn_tl` and `popcorn.impls.nsa_tl` are adapted
  from fla and import its utility ops, so their predicates decline whenever
  fla is absent. The phase installed bare `.` without the fla extra.
- Marking: the typed popcorn phase installs `.[fla]` now
  (`popcorn.bench.sweep.PhaseSpec`). `nsa:popcorn` keeps its valid rows; `wall_attn:popcorn` has
  none since the reference's stable-form rewrite and needs one re-measurement
  before dispatch will route to it.

## attn, attn_varlen / fa3 (rows predate the current references)

- Case: `attn:fa3` and `attn_varlen:fa3` have no passing row at the current
  reference fingerprints; their passes predate the reference cleanups.
- Cause: flash-attn-3 compiles against torch's CUDA at install time and the
  benchmark cluster's nvcc is older than torch 2.11's CUDA runtime, so the
  sweep's fa3 phase preflight-skips and the rows cannot be refreshed there.
- Marking: none. The pairs stay registered and dispatch falls back to
  implementations with current evidence; re-measure on a machine whose nvcc
  matches torch's CUDA to restore them.

## attn / cudnn (hosts exposing two CUDA runtime majors)

- Case: Torch 2.13's CUDA 13 wheel on a host that also exposes
  `/usr/local/cuda/.../libcudart.so.12`.
- Error: `Multiple libcudart libraries found: libcudart.so.12 and libcudart.so.13`.
- Cause: cuDNN Frontend 1.22 probes both runtime majors with `dlopen` and aborts
  when both resolve, even though Torch and cuDNN are consistently CUDA 13.
- Marking: no fallback or repository loader hack. The sweep runs a direct
  forward/backward preflight and skips `cudnn` on a mixed-runtime host. Use a
  CUDA 13-only image to benchmark and publish this backend.

# Deferred ops

Ops we looked at and did not port, and why.

- `swiglu_mlp`: torch reference only. The fused `[hidden, 2 * intermediate]`
  weight layout cannot be expressed by the shape grammar, and its only kernel
  was a vendored tilelang implementation.
- `attn_decoding_one_step`: beyond `cu_seqlens` (which the harness now
  generates, see `attn_varlen`) it needs kv caches the shape grammar cannot
  express. (`nsa` used to sit here for its block indices; the ported op
  derives them internally from the gate tensors instead.)
- `selection_attn`, `flash_lla`, `attn_res_pool`, `moba`: vendored triton
  kernels in the old tree (no external backend); porting means adopting
  first-party sources (as now done for `nsa` and `wall_attn`).
- `fft_conv`, `causal_conv1d`: the old tree vendored fla's triton kernels;
  today's fla exposes no public op for them.
- `tiled_mlp` (liger): takes an `nn.Module` and a callable and shards the
  forward; orchestration around autograd, not a kernel.
- quack `gemm`/`linear`/`mlp` family: training-graph fusion APIs that store
  and reconsume pre-activations (`fuse_grad_accum`, `store_preact`); no
  stateless functional contract to adapt, and bare gemm is cuBLAS territory.
- fla `fused_chunk_*`/`parallel_*` variants of ops we already serve via
  `chunk_*`: not blocked, just not registered yet; they would slot in as
  additional `fla:*` backends and multiply the benchmark grid, so they wait
  until tuned dispatch earns it. (The `fused_recurrent_*` variants are
  registered as `fla:recurrent`, forward-only: step-wise kernels for the
  decode regime, so autograd calls route to `chunk_*`/torch instead.)
