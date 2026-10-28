# configure/

One `<Model>.txt` file per model in the `key = value` format (any key can be overridden on the command line, e.g.
`--lr 0.0005`). `utility.utility_function.tools.read_configuration` parses values to their types: `none`,
`true`/`false`, numbers, lists such as `[20, 10]`; anything else stays a string. Lines starting with `#` are
comments. A run writes its resolved configuration (`config_resolved.json`) and its SHA-256 next to its log.
`profiles/*.json` describe the execution profiles. Terms are defined in `docs/GLOSSARY.md`.

| Key | Meaning |
|---|---|
| `dataset`, `split_seed` | data contract `dataset_verify/<dataset>/<split_seed>/` |
| `gate` | `none` (LightGCN) · `global` (GlobalGate) · `degree` (DegreeGate) · `degree_content` (DegreeContentGate) |
| `source` | prototype source: `taxonomy` · `shuffled` · `global` · `kmeans` |
| `shuffle_seed` | `shuffled` source only: `100 + k` for the k-th training seed (paired by position); `none` otherwise |
| `fuse_mask` | `taxonomy`: every model fuses exactly the items with a defined taxonomy prototype and train degree ≥ 1 |
| `leave_one_out` | leave-one-out prototypes (`false` = self-inclusive ablation) |
| `K`, `dim`, `init_std` | LightGCN layers, embedding size, N(0, init_std²) initialisation |
| `adj_layout` | `csr` or `coo` (identical values; CSR is faster on GPU), or `segment`: same values, product computed by gather + segment sum, bitwise reproducible on CUDA but about 4x slower (cuSPARSE products differ by ~1e-7 between repeated runs) |
| `lr`, `batch_size`, `reg` | Adam learning rate, BPR batch, L2 on layer-0 parameters of the batch |
| `gate_a_init`, `gate_b_init` | initial a (after softplus) and b of the gate |
| `epochs`, `patience` | epoch cap and early-stopping patience (epochs without improvement of either checkpoint) |
| `delta_sel` | margin of the selection rule (0.5%) |
| `ks`, `selection_k` | evaluated cut-offs; cut-off used for selection |
