# HiTaxProto-CF (working name)

Taxonomy prototypes fused into the scoring representation of LightGCN with a degree-monotone gate, evaluated against
matched prototype sources (global mean, shuffled labels, K-means) under a protocol fixed before the test split is
opened. Every name used in the code, configuration files, logs and result files is defined in
[docs/GLOSSARY.md](docs/GLOSSARY.md); the experimental protocol is [protocol/protocol_plan.md](protocol/protocol_plan.md).

```
config_path/        path contract (data, metadata, results; optional TaxPro-CL repository copy of the raw files)
configure/          default configurations (<Model>.txt, key = value) + profiles/
models/             LightGCN, HiTaxProtoCF (prototype memory, gates, fusion, ancestor prototypes), TFLightGCN, NCL, SimGCL
utility/
  utility_data/     split, data_graph (COO/CSR/segment adjacency), data_loader (sealed test), taxonomy, controls, sampler
  utility_function/ statistics, tools (configuration reader, hash)
  utility_train/    trainer (EarlyStopper, train, select_config), group_evaluator, model_factory
tools/
  data/             sources, download_data, build_interactions, prepare_metadata, parse_yelp_categories, build_taxonomy
  protocol/         audit_contract, audit_models, audit_protocol, leak_check, record_locks, freeze_protocol, eval_test,
                    ncl_parity (official NCL in RecBole vs this NCL on the same contract, validation only)
  experiments/      run_queue, select_config, make_pilot_phase2, pilot_decision
  analysis/         collect_results, export_gates, analyze_test
  toy/              independent NumPy reference of the worked example
plans/              run plans executed by tools/experiments/run_queue.py (one run at a time)
protocol/           protocol plan, sealed test hashes, frozen protocol (once frozen)
tests/              python -m pytest
```

## Workflow

0. Raw sources: `python -m tools.data.download_data` (URLs and SHA-256 in `tools/data/sources.py`; TLS is always
   verified: if the machine's trust store lacks a host's root, add a standard bundle with `--ca_bundle`, e.g.
   `"C:/Program Files/Git/mingw64/etc/ssl/certs/ca-bundle.crt"`). The same files in a local copy of the TaxPro-CL
   repository (Amazon and Yelp metadata, ~38 GB) are read in place instead if `HITAX_TAXPROCL_ROOT` points to it;
   they must have the recorded SHA-256. With that variable set, `python -m pytest` also cross-checks the evaluator
   against the TaxPro-CL one.
1. Data contracts (CPU): `tools/data/build_interactions.py`, `prepare_metadata.py`, `parse_yelp_categories.py`,
   `build_taxonomy.py`; then `tools/protocol/audit_contract.py`, `leak_check.py`, `record_locks.py`.
2. Model and protocol audits: `tools/protocol/audit_models.py`, `audit_protocol.py`; `python -m pytest`.
3. Pilot (validation only): `run_queue.py plans/pilot_<dataset>_phase1.json` → `select_config.py` (reference
   LightGCN) → `make_pilot_phase2.py` → `run_queue.py plans/pilot_<dataset>_phase2.json` → `select_config.py` per
   arm → `pilot_decision.py`; gate distribution by degree band: `tools/analysis/export_gates.py` per gated arm.
   yelp2018 first, then amazon-book.
4. Main experiments (validation only) → selection → `tools/protocol/freeze_protocol.py`.
5. Sealed test once: `tools/protocol/eval_test.py`, then `tools/analysis/analyze_test.py`.

Data are not distributed with the code: `dataset/`, `dataset_verify/` and `metadata/` are downloaded and
rebuilt from the original sources by steps 0–1 (LightGCN releases of amazon-book and yelp2018, Amazon 2014/2018/2023 metadata, Amazon 2018
Grocery 5-core, MovieLens-1M). The Yelp category hierarchy is parsed from the official page
https://docs.developer.yelp.com/docs/resources-categories. The page is not versioned and its markup changes
with every site build, so its identity is the content of the parsed hierarchy (1 564 aliases, 16 with several
parents): `hierarchy_sha256` = `e23796e433ecd85a0fb4b2713703977a2befef86f3b88d3da8d0a287b4275f96` (page retrieved
on 2026-10-26); a page whose hierarchy differs is rejected by the download tool.

Training runs read train and validation only; the test split is refused (`SealedTestError`) without the frozen
protocol, and the confirmation dataset cannot be trained on before freezing.

Environment: Python 3.10, torch 2.1.0 (CUDA 12.1), NumPy 1.26.4, SciPy 1.11.4 (`requirements.txt`).

## Reproducibility

* Data: every contract is rebuilt byte for byte from the raw sources by the `tools/data` steps (seeded split,
  deterministic label resolution and shuffles); `tools/protocol/audit_contract.py` and `leak_check.py` verify it.
* Runs: each run folder stores `config_resolved.json` (with its hash), `environment.json` (library versions, GPU,
  `code_sha256` of the code and default configurations, deterministic-algorithm flag) and `run_manifest.json`
  (selection threshold, selected epoch, status). `main.py` enables PyTorch deterministic algorithms; with
  `adj_layout = segment` a run is bitwise reproducible on the same GPU and software stack, with `csr` repeated runs
  differ by about 1e-7 in the parameters (cuSPARSE).
* Interruptions: `last_model.pt` is written atomically after every epoch (model, optimizer, sampler and RNG states,
  early-stopping state); `--resume` (used by `run_queue.py`) continues an interrupted run exactly, and a queue killed
  without clean-up can be restarted (its stale lock is detected).
