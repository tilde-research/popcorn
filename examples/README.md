# Examples

| Notebook | What it shows |
| --- | --- |
| [01_optimize_modeling_code.ipynb](./01_optimize_modeling_code.ipynb) | Swap plain-torch ops for `popcorn.kernels` in a hybrid Wall-Attention / attention LM. |
| [02_torch_compile_experimental.ipynb](./02_torch_compile_experimental.ipynb) | Experimental: inject kernels via `popcorn.compile` without editing the model. |

[`modeling_butter.py`](./modeling_butter.py) — HF-style model, no popcorn dependency. Pass any `ops` namespace with the same names at construction.

```bash
uv pip install -e ".[fla,liger]" jupyter
```
