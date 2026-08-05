# Examples

| Notebook | What it shows |
| --- | --- |
| [01_optimize_modeling_code.ipynb](./01_optimize_modeling_code.ipynb) | Swap plain PyTorch ops for `popcorn.kernels`. |
| [02_torch_compile_experimental.ipynb](./02_torch_compile_experimental.ipynb) | Add experimental compile rewrites without touching the model. |
| [03_write_a_kernel.ipynb](./03_write_a_kernel.ipynb) | Add a new fused op: reference, Triton implementation, correctness, and benchmarks. |

[`modeling.py`](./modeling.py) is the tiny plain PyTorch model the first two notebooks use. Pass its ops one by one or as a provider.

```bash
uv pip install -e ".[fla,liger]" jupyter
```
