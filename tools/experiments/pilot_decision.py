"""Pilot decision (validation only, assessment half evaluated once per run).

Inputs: the selection reports of the reference LightGCN and of DegreeGate with the taxonomy and shuffled sources
(tools/experiments/select_config.py on the pilot runs). On the ASSESSMENT half of the validation users:
  * relative difference of the endpoint (long-tail Recall@20 ranked among the long-tail items, long_tail_within),
    taxonomy vs shuffled (ratio of means, paired by seed position),
  * two-level (seed x user) bootstrap 95% CI, sign of the difference for every seed,
  * Overall Recall@20 of DegreeGate-taxonomy >= 0.99 x the reference LightGCN (screening only).
The pilot passes on a dataset when the relative difference >= +1%, the CI excludes 0, every seed is positive and the
Overall screen holds. Writes results/pilot/<dataset>.json.

    python -m tools.experiments.pilot_decision --dataset yelp2018
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import MODEL_OUTPUT_DIR, RESULT_DIR  # noqa: E402
from utility.utility_function.statistics import rel_delta, two_level_bootstrap  # noqa: E402
from utility.utility_train.group_evaluator import ENDPOINT_GROUP  # noqa: E402

SEEDS = (42, 0, 1)


def per_user(run_dir, group, metric="Recall@20"):
    z = np.load(run_dir / "assessment_per_user.npz")
    return dict(zip(z[group + "/users"].tolist(), z["{}/{}".format(group, metric)].tolist()))


def matrix(dirs, group):
    """[seeds x users] on the users eligible in every run (identical by construction: same split, same half)."""
    tables = [per_user(d, group) for d in dirs]
    users = sorted(set.intersection(*[set(t) for t in tables]))
    if any(len(t) != len(users) for t in tables):
        raise ValueError("eligible users differ between runs of the pilot")
    return np.array([[t[u] for u in users] for t in tables]), users


def selected_dir(dataset, model, report_name, reference=None):
    """Run folders of the chosen setting; the report must cover the pilot seeds on the selection half and, for a gated
    model, use the reference value of the selected LightGCN."""
    rep = json.loads((RESULT_DIR / "selection" / report_name).read_text(encoding="utf-8"))
    if rep["seeds"] != list(SEEDS) or rep["chosen_metrics"].get("validation_users") != "selection":
        raise SystemExit("{}: not a pilot selection (seeds {}, selection half)".format(report_name, list(SEEDS)))
    if reference is not None and abs(rep["reference_overall"] - reference) > 1e-12:
        raise SystemExit("{}: reference_overall {} differs from the reference LightGCN ({})".format(
            report_name, rep["reference_overall"], reference))
    return [MODEL_OUTPUT_DIR / "pilot" / dataset / model / rep["chosen"] / "seed{}".format(s) for s in SEEDS], rep


def _finite(x):
    """Strict JSON: NaN / inf become null."""
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--n_boot", type=int, default=5000)
    a = p.parse_args(argv)
    ref_dirs, ref_rep = selected_dir(a.dataset, "LightGCN", "pilot__{}__LightGCN.json".format(a.dataset))
    ref_value = ref_rep["chosen_metrics"]["val_Overall"]
    tax_dirs, tax_rep = selected_dir(a.dataset, "DegreeGate", "pilot__{}__DegreeGate__taxonomy.json".format(a.dataset),
                                     ref_value)
    shu_dirs, shu_rep = selected_dir(a.dataset, "DegreeGate", "pilot__{}__DegreeGate__shuffled.json".format(a.dataset),
                                     ref_value)
    T, users = matrix(tax_dirs, ENDPOINT_GROUP)
    S, users_s = matrix(shu_dirs, ENDPOINT_GROUP)
    if users != users_s:
        raise ValueError("taxonomy and shuffled runs are not evaluated on the same users")
    boot = two_level_bootstrap(T, S, n_boot=a.n_boot, seed=0)
    per_seed = [rel_delta(T[k], S[k]) for k in range(len(SEEDS))]
    To, _ = matrix(tax_dirs, "overall"); Ro, _ = matrix(ref_dirs, "overall")
    overall_ratio = float(To.mean() / Ro.mean())
    checks = {"relative_difference_>=_1%": boot["rel"] >= 0.01, "ci_excludes_0": boot["ci_abs"][0] > 0,
              "all_seeds_positive": all(x > 0 for x in per_seed), "overall_>=_0.99_reference": overall_ratio >= 0.99}
    checks = {k: bool(v) for k, v in checks.items()}
    report = {"dataset": a.dataset, "half": "assessment", "seeds": SEEDS, "endpoint": ENDPOINT_GROUP + " Recall@20",
              "users_endpoint": len(users),
              "chosen": {"reference": ref_rep["chosen"], "taxonomy": tax_rep["chosen"], "shuffled": shu_rep["chosen"]},
              "endpoint_mean": {"taxonomy": float(T.mean()), "shuffled": float(S.mean())},
              "relative_difference": boot["rel"], "absolute_difference": boot["abs"],
              "ci95_absolute": list(boot["ci_abs"]), "ci95_relative": list(boot["ci_rel"]),
              "relative_difference_per_seed": per_seed, "overall_ratio_to_reference": overall_ratio,
              "checks": checks, "pass": all(checks.values())}
    if S.mean() == 0:                                     # relative difference undefined (0 / 0 or x / 0)
        report["note"] = "endpoint of the shuffled source is 0: relative difference undefined, not a pass"
        report["pass"] = False
    report = _finite(report)
    out = RESULT_DIR / "pilot" / "{}.json".format(a.dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1, allow_nan=False); f.write("\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
