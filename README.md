# ElegantVLA

ElegantVLA is a phase-adaptive inference framework for accelerating
vision-language-action models. It schedules the VLM backbone and action head
using temporal representation similarity and robot motion.

The repository uses one Python package, `elegantvla`, with matching
GR00T and OpenPI layouts. Base models are pinned Git submodules.

## Repository layout

```text
ElegantVLA/
├── elegantvla/
│   ├── scheduling/            # Shared actions, teacher/RL policy and cache controller
│   ├── gr00t/
│   │   ├── core/              # Scheduler, adapter and cache runtime
│   │   ├── entrypoints/       # Evaluation, benchmarks and scheduler training
│   │   ├── analysis/          # Trajectory replay and representation error
│   │   ├── scripts/           # Shell launchers
│   │   ├── integration/       # Upstream model hooks and GPU kernels
│   │   └── tests/
│   └── openpi/
│       ├── core/              # Same wrapper/adapter/action-head roles as GR00T
│       ├── legacy/            # Previous OpenPI experiments, opt-in compatibility
│       ├── entrypoints/       # Serving, evaluation and benchmarks
│       ├── analysis/          # Trajectory summaries and similarity plots
│       ├── scripts/           # Shell launchers
│       ├── integration/       # Upstream policy and server hooks
│       └── tests/
├── third_party/
│   ├── Isaac-GR00T/
│   └── openpi/
├── scripts/                   # Integration installation
├── tests/                     # Source layout and launcher checks
├── docs/
└── pyproject.toml
```

Both backends use the same roles: `core` implements the method,
`entrypoints` runs it, `analysis` inspects results, `scripts` launches commands,
and `integration` connects the method to the upstream model.
Both backends share schedule types, the threshold policy, the RL inference
adapter and the episode-local cache controller. GR00T also includes its training
environment and RL training commands; an OpenPI training environment is not yet
included.

The model interfaces differ. GR00T sends VLM output tokens to a separate DiT,
which projects its own cross-attention K/V. OpenPI supplies layer-wise VLM prefix
K/V to its action expert. Both backends default to one fresh first layer,
a cached middle residual, and one fresh last layer. OpenPI additionally retains
the middle layers' prefix K/V, as required by its model interface. The older
OpenPI first-two/last-two setup remains available through `legacy/`.
See the [backend architecture](docs/backend_architecture_zh.md),
[GR00T guide](elegantvla/gr00t/README.md) and
[OpenPI guide](elegantvla/openpi/README.md).

The algorithm interface is shared, but the GPU optimizations are backend-specific.
GR00T includes optional fused kernels and CUDA Graph wrappers. OpenPI implements
the scheduling and cache paths in PyTorch; those GR00T-specific optimizations are
not ported. See the [four-layer runtime and implementation status](docs/soft_hardware_architecture_zh.md).

## Pinned upstream versions

| Dependency | Revision |
|---|---|
| NVIDIA Isaac-GR00T | `ead52833afbbf4243f8cd5e7664f48a94de03b19` (`n1.6-release`) |
| Physical Intelligence OpenPI | `c23745b5ad24e98f66967ea795a07b2588ed6c79` |

## Prepare the source tree

From the repository root, initialize the pinned dependencies:

```bash
git submodule update --init --recursive
```

Set up the selected upstream model's Python environment using its instructions.
In that environment, install this package and apply the corresponding integration:

```bash
python -m pip install --no-deps -e .

# Select the backend to prepare.
bash scripts/install_gr00t_integration.sh
bash scripts/install_openpi_integration.sh
```

GR00T and OpenPI may use separate environments. Install this package in each
environment you use. The installers check the pinned revision and destination
files before copying integration hooks. Add `--check` to inspect without copying.

## Run

The launchers resolve paths from their own location and accept arguments for the
underlying Python entrypoint:

```bash
bash elegantvla/gr00t/scripts/run.sh evaluate --help
bash elegantvla/gr00t/scripts/run.sh benchmark --help
bash elegantvla/openpi/scripts/run.sh evaluate --help
bash elegantvla/openpi/scripts/run.sh benchmark --help
bash elegantvla/openpi/scripts/run.sh serve --help
```

The matching Python module names are `elegantvla.gr00t.entrypoints.evaluate`
and `elegantvla.openpi.entrypoints.evaluate`. See the backend guides above for
available commands and environment options.

Model checkpoints, datasets, generated features, experiment logs and Python
environments are excluded. Some legacy configuration and trace field names are
retained so existing experiment records remain readable.

## Tests And Validation

Run the source layout, launcher and dependency-free configuration checks with:

```bash
python -m unittest discover -s tests
python -m unittest discover -s elegantvla/openpi/tests -p test_config.py
```

The OpenPI backend also has CPU forward tests with tiny real Gemma layers; these
require the OpenPI PyTorch environment. These tests do not validate full-size
checkpoint performance or closed-loop success rates. Clean-clone inference and
GPU latency validation remain outstanding for both packaged backends.
See the [release verification checklist](docs/release_checklist_zh.md).

## Third-party software

The contents of `third_party/` and upstream-derived integration code remain
subject to their original licenses. A top-level `LICENSE` and `CITATION.cff`
are not yet included; they are required release-checklist items.
