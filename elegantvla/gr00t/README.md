# GR00T integration

| Directory | Contents |
|---|---|
| `core/` | Model adapter, action-head compiler, cache controller, scheduler policies and RL environment |
| `entrypoints/` | `evaluate.py`, `benchmark.py`, `train_rl.py`, `eval_rl.py` and `train_bc.py` |
| `analysis/` | Trajectory replay, result summaries and `open_loop_error/` |
| `scripts/` | `run.sh`, RL launchers and batch evaluation helpers |
| `integration/` | GR00T model hooks, fused kernels and CUDA Graph wrappers |
| `tests/` | Cache, schedule, error-metric and operator regression tests |

Run from the ElegantVLA root after preparing the GR00T environment:

```bash
python -m pip install --no-deps -e .
bash scripts/install_gr00t_integration.sh
bash elegantvla/gr00t/scripts/run.sh evaluate --help
bash elegantvla/gr00t/scripts/run.sh train_rl --help
```

Method imports use `elegantvla.gr00t.core`. Evaluation and training imports use
`elegantvla.gr00t.entrypoints`; offline tools use `elegantvla.gr00t.analysis`.

`core/adapter.py` handles model inputs, backbone recomputation and action-head
calls. `core/action_head.py` contains the optional compiled action-head refresh
and residual-reuse implementation. `core/wrapper.py` controls execution and cache
lifetime, while `rule_policy.py` and `rl_policy.py` select scheduling actions.
Schedule types, rule/RL policy implementations and `SkipController` live in
`elegantvla.scheduling` and are shared with OpenPI. The GR00T module paths remain
available as compatibility exports; its adapter and GPU runtime stay model-specific.

`PYTHON_BIN` selects the interpreter. Install SimplerEnv in that environment, or
set `SIMPLERENV_ROOT` for an external source checkout and
`SIMPLERENV_SITE_PACKAGES` for an optional dependency directory. Launchers add the
ElegantVLA package and pinned GR00T checkout to `PYTHONPATH`.

Pass model paths through CLI arguments or `MODEL_PATH` for the specialized shell
launchers. GR00T run directories default to `outputs/gr00t/`.

## Optional GPU Runtime

The integration includes model-specific kernels and graph wrappers. They are
separate from the shared scheduler and are not enabled by every launcher.

| Environment variable | Effect | Default |
|---|---|---|
| `GROOT_REALTIME_EAGLE` | Select realtime defaults, including static Qwen attention, boundary graphs and the Action DiT graph | `0` |
| `GROOT_TRITON_EAGLE_KERNELS` | Install Triton kernels; with realtime defaults, also enable gate/up, QKV and SigLIP FC2/residual fusion | Explicit opt-in; `GROOT_TRITON_EAGLE` is a fallback alias |
| `GROOT_CUDA_GRAPH_QWEN_BOUNDARIES` | Capture first/last Qwen layers and the refresh suffix | Enabled by realtime defaults unless the full-decoder graph is selected |
| `GROOT_CUDA_GRAPH_EAGLE` | Capture the whole Qwen decoder, an alternative to boundary graphs | `0` |
| `GROOT_CUDA_GRAPH_VISION` | Capture the fixed-shape SigLIP encoder | `0` |
| `GROOT_CUDA_GRAPH_ACTION_DIT` | Capture the Action DiT forward | Enabled by realtime defaults |

For temporal first/last recomputation, use boundary graphs rather than the
whole-decoder graph. Triton also requires a compatible compiler environment;
enabling `GROOT_REALTIME_EAGLE` alone does not opt into Triton.

Projected DiT K/V reuse is conditional on the model configuration and unchanged
conditioning. Action-head residual-reuse modes use a separate compiled path and
do not stack that K/V cache protocol on top. This is not a fully preallocated,
end-to-end CUDA Graph runtime.

These are implementation entrypoints, not a validated performance preset.
Clean-clone full-checkpoint inference, GPU timing and closed-loop validation are
still required for the packaged backend. See the
[four-layer runtime](../../docs/soft_hardware_architecture_zh.md) for cache boundaries.
