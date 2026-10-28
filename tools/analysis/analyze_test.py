"""Pre-declared analysis of the sealed test results (run only after tools/protocol/eval_test.py).

The analysis is read from the FROZEN protocol ("analysis" entry), so it cannot change after the test is opened:
  "analysis": {
    "endpoint": {"group": "long_tail_within", "metric": "Recall@20"},
    "seeds": [42, 0, 1, 2, 3],
    "families": {"main": [{"label", "dataset", "a": <run>, "b": <run>[, "group", "metric"]}, ...], "secondary": [...]},
    "conjunctions": [{"label", "cells": [<label>, <label>]}],
    "noninferiority": [{"label", "arm": <run>, "reference": <run>, "margin": 0.01}]}
A cell may name its own "group" / "metric" (secondary outcomes); otherwise the endpoint is used.
<run> = log/<study>/<dataset>/<model>/<setting>; seeds are paired by position (the k-th shuffled-label seed goes with
the k-th training seed). For every cell: absolute and relative difference (ratio of means), paired sign-flip test on
user differences averaged over seeds (10^5 sign vectors), two-level (seed x user) bootstrap 95% CI, Holm within the
family, per-seed signs (descriptive). A positive claim needs Holm p < 0.05 AND a CI excluding 0. A conjunction uses the
intersection-union test (max of the Holm p-values, every difference positive). Writes results/analysis/test_analysis.json
and test_analysis.md.

    python -m tools.analysis.analyze_test --protocol protocol/frozen_protocol.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import PROJECT_ROOT, RESULT_DIR  # noqa: E402
from utility.utility_function.statistics import (claim_positive, holm, iut, noninferiority, rel_delta,  # noqa: E402
                                                 signflip_p, two_level_bootstrap)


def load_matrix(run, seeds, group, metric):
    """[seeds x users] of one run on the test split; users must be identical across seeds."""
    tables = []
    for s in seeds:
        d = PROJECT_ROOT / run / "seed{}".format(s)
        if not (d / "test_evaluated.json").is_file():
            raise SystemExit("{} has no sealed test evaluation".format(d))
        z = np.load(d / "test_per_user.npz")
        tables.append(dict(zip(z[group + "/users"].tolist(), z["{}/{}".format(group, metric)].tolist())))
    users = sorted(tables[0])
    if any(sorted(t) != users for t in tables):
        raise SystemExit("{}: eligible users differ between seeds".format(run))
    return np.array([[t[u] for u in users] for t in tables]), users


def paired(a, b, seeds, group, metric):
    A, ua = load_matrix(a, seeds, group, metric)
    B, ub = load_matrix(b, seeds, group, metric)
    if ua != ub:
        raise SystemExit("{} and {} are not evaluated on the same users".format(a, b))
    return A, B


def cell(spec, seeds, group, metric, n_perm, n_boot):
    group, metric = spec.get("group", group), spec.get("metric", metric)
    A, B = paired(spec["a"], spec["b"], seeds, group, metric)
    D = A.mean(0) - B.mean(0)
    boot = two_level_bootstrap(A, B, n_boot=n_boot, seed=0)
    per_seed = [float(A[k].mean() - B[k].mean()) for k in range(len(seeds))]
    return {"label": spec["label"], "dataset": spec["dataset"], "a": spec["a"], "b": spec["b"], "group": group,
            "metric": metric, "users": int(A.shape[1]),
            "mean_a": float(A.mean()), "mean_b": float(B.mean()), "abs_diff": boot["abs"],
            "rel_diff": rel_delta(A, B) if B.mean() > 0 else None, "ci95_abs": list(boot["ci_abs"]),
            "p_signflip": signflip_p(D, B=n_perm, seed=0), "per_seed_abs_diff": per_seed,
            "seeds_positive": int(sum(x > 0 for x in per_seed))}


def finite(x):
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [finite(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def analyse(proto, n_perm=100_000, n_boot=5000):
    an = proto["analysis"]
    seeds = an.get("seeds", [42, 0, 1, 2, 3])
    group, metric = an["endpoint"]["group"], an["endpoint"]["metric"]
    out = {"endpoint": an["endpoint"], "seeds": seeds, "families": {}, "conjunctions": [], "noninferiority": []}
    by_label = {}
    for fam, cells in an["families"].items():
        rows = [cell(c, seeds, group, metric, n_perm, n_boot) for c in cells]
        adj = holm([r["p_signflip"] for r in rows]) if rows else []
        for r, p in zip(rows, adj):
            r["p_holm"] = float(p)
            r["claim_positive"] = claim_positive(r["p_holm"], r["ci95_abs"])
            by_label[r["label"]] = r
        out["families"][fam] = rows
    for c in an.get("conjunctions", []):
        parts = [by_label[x] for x in c["cells"]]
        p = iut(*[r["p_holm"] for r in parts])
        out["conjunctions"].append({"label": c["label"], "cells": c["cells"], "p_iut": float(p),
                                    "claim_positive": bool(p < 0.05 and all(r["abs_diff"] > 0 and r["ci95_abs"][0] > 0
                                                                            for r in parts))})
    for ni in an.get("noninferiority", []):
        A, R = paired(ni["arm"], ni["reference"], seeds, "overall", metric)
        r = noninferiority(A, R, delta=float(ni.get("margin", 0.01)), n_boot=n_boot, seed=0)
        out["noninferiority"].append({"label": ni["label"], "arm": ni["arm"], "reference": ni["reference"],
                                      "margin": float(ni.get("margin", 0.01)), **r})
    return finite(out)


def markdown(res):
    lines = ["# Test analysis ({} {})".format(res["endpoint"]["group"], res["endpoint"]["metric"]), ""]
    for fam, rows in res["families"].items():
        lines += ["## Family: {}".format(fam), "",
                  "| comparison | dataset | A | B | Δ abs | Δ rel | CI95 abs | p | p Holm | seeds + | claim |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            rel = "n/a" if r["rel_diff"] is None else "{:+.2%}".format(r["rel_diff"])
            lines.append("| {label} | {dataset} | {ma:.6f} | {mb:.6f} | {d:+.6f} | {rel} | [{c0:+.6f}, {c1:+.6f}] | {p:.4f} "
                         "| {ph:.4f} | {sp}/{n} | {cl} |".format(
                             label=r["label"], dataset=r["dataset"], ma=r["mean_a"], mb=r["mean_b"], d=r["abs_diff"],
                             rel=rel, c0=r["ci95_abs"][0], c1=r["ci95_abs"][1], p=r["p_signflip"], ph=r["p_holm"],
                             sp=r["seeds_positive"], n=len(r["per_seed_abs_diff"]), cl="yes" if r["claim_positive"] else "no"))
        lines.append("")
    if res["conjunctions"]:
        lines += ["## Conjunctions (intersection-union)", "", "| claim | p IUT | positive |", "|---|---|---|"]
        lines += ["| {} | {:.4f} | {} |".format(c["label"], c["p_iut"], "yes" if c["claim_positive"] else "no")
                  for c in res["conjunctions"]] + [""]
    if res["noninferiority"]:
        lines += ["## Non-inferiority (Overall)", "", "| arm vs reference | margin | lower bound of D' | rel vs ref | pass |",
                  "|---|---|---|---|---|"]
        lines += ["| {} | {:.0%} | {:+.6f} | {:+.2%} | {} |".format(n["label"], n["margin"], n["lower_bound_Dprime"],
                                                                     n["rel_vs_ref"], "yes" if n["pass"] else "no")
                  for n in res["noninferiority"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    p.add_argument("--n_perm", type=int, default=100_000)
    p.add_argument("--n_boot", type=int, default=5000)
    a = p.parse_args(argv)
    proto = json.loads(Path(a.protocol).read_text(encoding="utf-8"))
    if proto.get("frozen") is not True:
        raise SystemExit("the analysis runs only under a frozen protocol")
    res = analyse(proto, a.n_perm, a.n_boot)
    out = RESULT_DIR / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "test_analysis.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, allow_nan=False); f.write("\n")
    with open(out / "test_analysis.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(markdown(res))
    print(markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
