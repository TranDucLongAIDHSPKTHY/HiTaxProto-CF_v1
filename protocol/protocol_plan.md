# Experimental protocol (plan, not yet frozen)

This document states what is run, what is measured and how claims are decided, before any test result is seen.
It is frozen (`tools/protocol/freeze_protocol.py`) after the pilot and before the sealed test split is opened; any
change before freezing is recorded with its date. Terms are defined in `docs/GLOSSARY.md`.

## 1. Questions

| Question | Comparison (same gate, same fused items) | Status |
|---|---|---|
| Q1 — information: does the benefit come from the taxonomy itself? | DegreeGate-taxonomy vs DegreeGate-shuffled; DegreeGate-taxonomy vs DegreeGate-global (main); vs DegreeGate-kmeans (exploratory) | main |
| Q2 — mechanism: does the degree term of the gate add value over one learned weight? | DegreeGate-taxonomy vs GlobalGate-taxonomy (main); DegreeContentGate vs DegreeGate (secondary) | main |
| Q3 — conditions and trade-off: when does fusion help, and at what cost on warm items / Overall? | strata by degree band, node size, label coverage, taxonomy type | exploratory |
| Q4 — structure: do multi-level ancestors add value over direct labels? | ancestor prototypes vs DegreeGate (only if the pilot passes) | conditional |

Negative results (taxonomy not better than its controls, gate not better than one weight) are valid outcomes and are
reported with the same detail. No target improvement is set in advance.

## 2. Data

| Dataset | Role | Notes |
|---|---|---|
| yelp2018 (LightGCN release, re-split) | development; main family | official Yelp category hierarchy (DAG), 835 valid nodes |
| amazon-book (LightGCN release, re-split) | development; main family | Books metadata 2014/2018/2023 (earliest informative release), 1 058 valid nodes |
| amazon2018-grocery (5-core) | **confirmation**: locked until the protocol is frozen; one run per configuration | 1 005 valid nodes, never used in development |
| ml-1m (3 splits) | secondary: flat multi-label genres, split variability, Overall / warm | fewer than 1 000 users with a long-tail test positive: long-tail only described |

Split: per user 70/10/20, `numpy.random.default_rng([seed, user])`, seed 2026 (ml-1m: 1, 2, 3). The 10-core filter of
the LightGCN releases makes their sparse items "sparse in the training part of this partition", not rare overall; this
is stated with every long-tail result. Every contract is audited (`tools/protocol/audit_contract.py`).

## 3. Models, grids and budget

All models use LightGCN-family conventions (K = 3, d = 64, Adam, batch 2048, N(0, 0.1²) initialisation, BPR, one
uniform negative per interaction, early stopping with patience 20 and an epoch cap of 1 000, validation every epoch).
(lr, reg) are tuned once for the reference LightGCN and inherited by every model built on it.

| Model | Grid | Configurations |
|---|---|---|
| LightGCN (reference) | lr ∈ {1e-3, 5e-4} × reg ∈ {1e-5, 1e-4, 1e-3}; chosen on validation Overall Recall@20 | 6 |
| BPR-MF | as LightGCN (own tuning) | 6 |
| TF-LightGCN | as LightGCN (own tuning) | 6 |
| DegreeGate, GlobalGate, DegreeContentGate × source | b_init ∈ {−1, 0, 1} | 3 per arm |
| HiTaxProto-CF in the comparison | {DegreeGate, DegreeContentGate} × b_init | 6 |
| NCL, NCL+Taxonomy | proto_lambda ∈ {1e-8, 1e-7, 1e-6} × temperature ∈ {0.05, 0.1} | 6 |
| SimGCL | eps ∈ {0.05, 0.1, 0.2} × tau ∈ {0.1, 0.2} (cl_lambda = 0.5) | 6 |
| LAGCL | official implementation, 6 configurations declared before freezing | 6 |

Seeds: tuning 42, 0; final 42, 0, 1, 2, 3 (pilot: 42, 0, 1). The k-th shuffled-label seed (101–105) is paired with the
k-th training seed. Selection rule (every model except the reference): highest validation endpoint (section 5) among
configurations whose validation Overall Recall@20 ≥ 99.5% of the reference; otherwise the best Overall, reported as
infeasible. Within a run, the checkpoint follows the same rule with the threshold taken from the reference LightGCN
run of the SAME seed (same initialisation and sampling stream; two checkpoints kept, patience on both); between
configurations, means over seeds are compared.

Reproducibility: training uses cuSPARSE products (`adj_layout = csr`), which are not bitwise deterministic on the GPU;
every final evaluation of a selected checkpoint uses a deterministic propagation, so all reported numbers are
reproduced exactly from the saved checkpoints. The run-to-run variation of training is measured once, by repeating
one reference LightGCN run of the pilot (same configuration and seed), and reported next to the variation between
seeds.

Declared reduction order if the measured GPU time exceeds the budget: (1) LAGCL and BPR-MF; (2) tuning with one seed;
(3) DegreeContentGate and the self-inclusive ablation on amazon-book; (4) K-means source only with DegreeGate.
TF-LightGCN, NCL+Taxonomy and NCL are never removed. Runs are executed one at a time on one GPU.

## 4. Pilot (validation only)

Datasets: yelp2018 first; amazon-book only after the Yelp decision. Arms: reference LightGCN (6 configurations);
DegreeGate × {taxonomy, shuffled, global} and GlobalGate-taxonomy, b_init ∈ {−1, 0, 1}; seeds 42, 0, 1. The validation
users are split once into a selection half (early stopping, configuration choice, reference value) and an assessment
half evaluated once per run on its selected checkpoint.

Pass on a dataset (assessment half): relative difference of the endpoint (section 5), DegreeGate-taxonomy vs
DegreeGate-shuffled, ≥ +1%; two-level (seed × user) 95% CI excluding 0; positive for all three seeds; Overall Recall@20
of DegreeGate-taxonomy ≥ 99% of the reference (screening only). If no dataset passes: stop, analyse (label coverage,
node sizes, gate distribution by degree, strata, taxonomy vs K-means), do not add modules, and re-frame as a controlled
study of when fusion helps.

## 5. Endpoint and statistics (test split, after freezing)

Primary endpoint: long-tail Recall@20 ranked among the long-tail items (`long_tail_within`): relevant items and
candidates are the items with 1–10 training interactions (items already seen by the user excluded). Secondary:
long-tail Recall@20 and Recall@100 with the whole catalogue as candidates, near-cold and mid-tail within-group
Recall@20.
Unit: the user; per-user metrics averaged over the 5 seeds.

* Main family (Holm, α = 0.05): {taxonomy − shuffled, taxonomy − global, DegreeGate − GlobalGate} × {yelp2018,
  amazon-book, amazon2018-grocery} = 9 cells.
* Secondary families (separate Holm): HiTaxProto-CF − TF-LightGCN and HiTaxProto-CF − NCL+Taxonomy × 3 datasets;
  near-cold, NDCG@20, Recall@10, warm. Q3 and ml-1m: point estimates with CIs, exploratory.
* Test: paired sign-flip on user differences (10⁵ sign vectors); interval: two-level bootstrap (5 000 resamples).
  A positive claim needs Holm-adjusted p < 0.05 AND an interval excluding 0. With five seeds the two-level interval can
  fall slightly below its nominal level (simulated coverage 0.93 for a nominal 0.95 over 300 trials;
  `tools/protocol/audit_protocol.py`), so it is never used alone. "Five of five seeds positive" is descriptive only.
* Conjunction "taxonomy better than both shuffled and global": intersection-union test (maximum of the two Holm
  p-values, both differences positive).
* Non-inferiority on Overall Recall@20 vs the reference LightGCN, margin 1%: one-sided 95% lower bound of
  mean(m_A − 0.99 · m_ref) > 0.
* A result is "confirmed" only if the corresponding confirmation-dataset cell passes Holm.

## 6. Sealed test

The test rows of every contract are stored apart; their SHA-256 values are recorded in `protocol/sealed_test_hashes.json`.
`tools/protocol/freeze_protocol.py` records this plan, the selected runs (configuration and checkpoint hashes) and the
code hash; `tools/protocol/eval_test.py` then evaluates each selected run once; `tools/analysis/analyze_test.py`
applies section 5 exactly. Nothing in sections 1–5 changes after the test is opened.

## 7. Changes before freezing

Any change to sections 1–6 made before the protocol is frozen is listed here with its date and reason.

| Date | Change | Reason |
|---|---|---|
