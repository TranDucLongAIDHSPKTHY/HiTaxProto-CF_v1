# Glossary

Names used in the code, configuration files, logs and result files.

## Method

| Term | Meaning |
|---|---|
| **HiTaxProto-CF** | LightGCN whose item representation used for scoring is fused with a taxonomy prototype: `e*_i = g_i·e_i + (1 − g_i)·p_i`. Training uses BPR + L2 only (no taxonomy-defined loss). |
| **prototype** `p_i` | Mean final LightGCN embedding of the *other* training items that share a valid taxonomy node with item *i* (leave-one-out), recomputed at every step without gradient; equal-weight mean over the item's nodes. |
| **leave-one-out** / **self-inclusive** | Prototype computed without / with the item itself (`leave_one_out = true / false`; the self-inclusive variant is an ablation). |
| **gate** `g_i` | Share of the item's own collaborative embedding kept in `e*_i`. |
| `LightGCN` (`gate = none`) | No fusion; the reference model. |
| `GlobalGate` (`gate = global`) | `g_i = σ(b)`: one learned weight shared by every item. |
| `DegreeGate` (`gate = degree`) | `g_i = σ(a·log(1 + d_i) + b)`, `a = softplus(·) ≥ 0`, so items with more training interactions borrow less. `GlobalGate` is `DegreeGate` with `a = 0`. |
| `DegreeContentGate` (`gate = degree_content`) | `DegreeGate` plus `w_eᵀe_i + w_pᵀp_i`. |
| **ancestor prototype** (`AncestorPrototypeMemory`) | Prototype over the multi-level ancestor set `A(i)`; variants `mean` and `attention`. |
| **fuse mask** | Items that are fused in every compared model: items with a defined taxonomy prototype and at least one training interaction. |

## Prototype sources (`source`)

| Value | Groups used to build prototypes |
|---|---|
| `taxonomy` | Valid taxonomy labels of the item. |
| `shuffled` | Label sets permuted among training items of the same stratum (degree band × number of labels), so node sizes and each item's number of labels are kept; seeds 101–105. |
| `global` | One group containing every training item (shrinkage to the global mean). |
| `kmeans` | K-means clusters of the final item embeddings (K = number of valid taxonomy nodes). |

## Comparators

| Name | Meaning |
|---|---|
| `TF-LightGCN` | Taxonomy offsets added to layer-0 item vectors (adaptation of Kanagal et al., PVLDB 2012); direct-label (mean) and ancestor (sum) variants. |
| `NCL`, `NCL+Taxonomy` | NCL (Lin et al., WWW 2022); `NCL+Taxonomy` replaces the item-side K-means groups of its prototype loss by valid taxonomy nodes. |

## Data

| Term | Meaning |
|---|---|
| Λ(i) | Direct labels of item *i*, reduced to the most specific ones. |
| valid label | Taxonomy node with ≥ 2 training items; an invalid label is lifted to its parent(s). |
| A(i) | Ancestor set of *i*, deduplicated over paths, super-root and nodes covering > τ = 0.5 of training items removed. |
| item groups (train degree *d*) | `strict_cold` d = 0 · `near_cold` (NC) 1–5 · `mid_tail` (MT) 6–10 · `long_tail` (LT) 1–10 · `warm` > 10 · `overall` all items. Groups restrict the relevant items and eligible users; the candidates are the whole catalogue, except for the within-group rankings `<group>_within` (e.g. `long_tail_within`, the primary endpoint), whose candidates are the unseen items of the group. |
| data contract | `dataset_verify/<dataset>/<split_seed>/`: interactions, ID maps, taxonomy, labels, ancestor sets, shuffled labels, manifests. Every model input is recomputable from training rows only (`tools/protocol/leak_check.py`). |

## Protocol

| Term | Meaning |
|---|---|
| **selection rule** | Among configurations whose validation Overall Recall@20 ≥ (1 − 0.5%) × reference LightGCN, choose the highest validation endpoint (`long_tail_within` Recall@20); otherwise fall back to the best Overall and report "infeasible". The reference LightGCN is selected on validation Overall. |
| **non-inferiority** | Overall Recall@20 of a model is non-inferior to the reference if the one-sided 95% lower bound of mean(m_A − 0.99·m_ref) over users is > 0 (margin 1%). |
| **sealed test** | The test rows are stored apart (`sealed_test/test_interactions.csv`, SHA-256 in `split_manifest.json`) and the data loader refuses to return them (`SealedTestError`) unless given a frozen protocol file whose hash matches. Training, early stopping, configuration choice and the pilot use validation only; the test split is evaluated once, after the protocol is frozen. |
| `hit_cap` | The run reached the epoch cap before early stopping. |

## Studies (`--study`, first folder of `log/`)

| Study | Purpose |
|---|---|
| `convergence` | LightGCN runs measuring the number of epochs to convergence. |
| `pilot` | Validation-only pilot (selection half / assessment half of validation users). |
| `controls` | Gates × prototype sources, and the self-inclusive ablation. |
| `comparison` | HiTaxProto-CF against baselines and same-information comparators. |
| `hierarchy` | Ancestor prototypes (multi-level taxonomy). |
| `cold_start` | Inductive new items (prototype-only representation). |
| `transfer` | Fusion inside SimGCL. |
| `ablation` | Further single-question ablations. |
| `dev` | Development and smoke runs (never reported). |

## Log layout

`log/<study>/<dataset>/<model>/<setting>/seed<seed>/` — e.g. `log/convergence/yelp2018/LightGCN/lr0.001_reg0.0001/seed42/`.
`<setting>` is built from the key hyperparameters of the model (`lr`, `reg` for LightGCN; `source`, `binit` for the
gated models; `self-inclusive` when leave-one-out is off). Files: `training.log`, `config_resolved.json`,
`environment.json`, `validation_metrics.json`, `best_validation_model.pt`, `last_model.pt`, `run_manifest.json`.

## Worked example

`tools/toy/toy_example.py`: a bookshop with 5 users and 7 books in two categories (detective A, science fiction B);
N is a new science-fiction book with a single purchase. Every number of the example is recomputed by the tests.
