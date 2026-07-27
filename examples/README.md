# Examples

| Notebook | What it shows |
| --- | --- |
| [01_optimize_modeling_code.ipynb](./01_optimize_modeling_code.ipynb) | Butter, a hybrid Gated-DeltaNet / attention LM written on popcorn kernels, profiled against its plain-torch references with popcorn's own harness. Forcing backends per call, per region, or via config. |
| [02_torch_compile_experimental.ipynb](./02_torch_compile_experimental.ipynb) | The experimental `popcorn.compile` pass injecting kernels into `torch.compile`d modeling code that never imports popcorn, and how to read the time/memory trade it makes. |

[`modeling_butter.py`](./modeling_butter.py) is the shared HF-style modeling file: `ButterConfig(dispatch=False)` binds every call site to the kernels' plain-torch references instead of the dispatcher — same weights, same code, no monkey patching.

Run them with a CUDA GPU and the optional backends installed:

```bash
uv pip install -e ".[fla,liger]" jupyter
```
