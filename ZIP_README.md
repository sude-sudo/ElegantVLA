# ZIP Source Snapshot

This archive contains the ElegantVLA source code and current documentation.
It excludes Git metadata, upstream repository contents, Python environments,
model weights, datasets, generated features, experiment outputs and internal notes.

The upstream URLs are recorded in `.gitmodules`; the exact revisions are in
`README.md`. Unlike a Git clone, this ZIP has no Git index to record submodules.
To restore them, run the following from the extracted `ElegantVLA` directory:

```bash
git init
git add .gitmodules
git update-index --add --cacheinfo 160000,ead52833afbbf4243f8cd5e7664f48a94de03b19,third_party/Isaac-GR00T
git update-index --add --cacheinfo 160000,c23745b5ad24e98f66967ea795a07b2588ed6c79,third_party/openpi
git submodule update --init --recursive
```

Then follow `README.md` and the selected backend's guide to prepare its environment
and install the integration. Initializing dependencies downloads source code;
it does not download model weights or establish runtime validation.

The archive includes the latest working-copy documentation, not only staged
Git contents. See `docs/soft_hardware_architecture_zh.md` for the distinction
between GR00T's optional GPU optimizations and OpenPI's algorithm-only port.
