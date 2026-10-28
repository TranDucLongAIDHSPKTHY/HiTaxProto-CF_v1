"""Apply the selection rule to the runs of one model (validation only).

For every setting folder of log/<study>/<dataset>/<model>/, the validation metrics at the selected checkpoint of each
tuning seed are averaged; the reference LightGCN is chosen on validation Overall Recall@20, every other model by the
selection rule (highest validation long_tail_within Recall@20 among settings whose validation Overall Recall@20 is at least
(1 - 0.5%) times the reference). The reference value is read from results/selection/<study>__<dataset>__LightGCN.json
(an explicit --reference_overall must agree with it), and every run must have been trained with its threshold: the
reference run of the same seed ('per_seed', recorded per seed in the report) or the mean ('single').
Seeds: 42, 0, 1 for the pilot, 42, 0 (tuning seeds) otherwise. Writes results/selection/<study>__<dataset>__<model>.json.

    python -m tools.experiments.select_config --study pilot --dataset yelp2018 --model LightGCN
    python -m tools.experiments.select_config --study pilot --dataset yelp2018 --model DegreeGate --setting_prefix taxonomy_
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
from utility.utility_train.group_evaluator import ENDPOINT_GROUP  # noqa: E402
from utility.utility_train.trainer import select_config  # noqa: E402


PILOT_SEEDS, TUNING_SEEDS = (42, 0, 1), (42, 0)


def reference_value(study, dataset, explicit=None):
    """(mean, per-seed values) of the validation Overall Recall@20 of the selected reference LightGCN (its selection
    report); per-seed values are None when only an explicit value is available."""
    path = RESULT_DIR / "selection" / "{}__{}__LightGCN.json".format(study, dataset)
    if not path.is_file():
        if explicit is None:
            raise SystemExit("no reference selection report {}; select LightGCN first".format(path))
        return explicit, None
    chosen = json.loads(path.read_text(encoding="utf-8"))["chosen_metrics"]
    value = chosen["val_Overall"]
    if explicit is not None and abs(explicit - value) > 1e-12:
        raise SystemExit("--reference_overall {} differs from the reference report ({})".format(explicit, value))
    return value, {s: v["val_Overall"] for s, v in chosen.get("per_seed", {}).items()} or None


def collect(study, dataset, model, seeds, prefix="", k=20, reference=None, reference_per_seed=None):
    base = MODEL_OUTPUT_DIR / study / dataset / model
    out = []
    for setting in sorted(p.name for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)):
        rows = []
        for s in seeds:
            m_path = base / setting / "seed{}".format(s) / "run_manifest.json"
            if not m_path.is_file():
                continue
            m = json.loads(m_path.read_text(encoding="utf-8"))
            if m.get("status") != "completed":
                continue
            if model != "LightGCN":                     # every run used the threshold of the reference report
                used, mode = m.get("reference_overall"), m.get("reference_mode", "single")
                expected = reference_per_seed.get(str(s)) if mode == "per_seed" and reference_per_seed else reference
                if used is None or expected is None or abs(used - expected) > 1e-12:
                    raise SystemExit("{}: trained with reference_overall={} ({}), expected {}".format(
                        m_path, used, mode, expected))
            v = m["selected_validation"]
            if ENDPOINT_GROUP not in v:
                raise SystemExit("{}: no {} in the selected validation metrics".format(m_path, ENDPOINT_GROUP))
            rows.append((v[ENDPOINT_GROUP]["Recall@%d" % k], v["overall"]["Recall@%d" % k], bool(m.get("feasible")),
                         m.get("validation_users", "all"), s))
        if len(rows) != len(seeds):
            out.append({"name": setting, "complete": False, "seeds_found": len(rows)})
            continue
        if len({r[3] for r in rows}) != 1:
            raise ValueError("{}: seeds mix validation-user modes".format(setting))
        out.append({"name": setting, "complete": True, "val_LT": float(np.mean([r[0] for r in rows])),
                    "val_Overall": float(np.mean([r[1] for r in rows])),
                    "infeasible_seeds": int(sum(not r[2] for r in rows)), "validation_users": rows[0][3],
                    "per_seed": {str(r[4]): {"val_LT": r[0], "val_Overall": r[1], "feasible": r[2]} for r in rows}})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="default: 42 0 1 (pilot), 42 0 (otherwise)")
    p.add_argument("--setting_prefix", default="")
    p.add_argument("--reference_overall", type=float, default=None)
    p.add_argument("--delta_sel", type=float, default=0.005)
    a = p.parse_args(argv)
    if a.seeds is None:
        a.seeds = list(PILOT_SEEDS if a.study == "pilot" else TUNING_SEEDS)
    per_seed = None
    if a.model != "LightGCN":
        a.reference_overall, per_seed = reference_value(a.study, a.dataset, a.reference_overall)
    cands = collect(a.study, a.dataset, a.model, a.seeds, a.setting_prefix, reference=a.reference_overall,
                    reference_per_seed=per_seed)
    incomplete = [c["name"] for c in cands if not c["complete"]]
    if incomplete:
        raise SystemExit("incomplete settings (all tuning seeds required): {}".format(incomplete))
    if a.model == "LightGCN":
        chosen, feasible = max(cands, key=lambda c: c["val_Overall"]), True
        rule = "reference: highest validation Overall Recall@20"
    else:
        chosen, feasible = select_config(cands, a.reference_overall, a.delta_sel)
        rule = "selection rule, delta_sel = {}".format(a.delta_sel)
    report = {"study": a.study, "dataset": a.dataset, "model": a.model, "seeds": a.seeds, "rule": rule,
              "reference_overall": a.reference_overall, "candidates": cands, "chosen": chosen["name"],
              "chosen_metrics": chosen, "feasible": feasible}
    out = RESULT_DIR / "selection" / "{}__{}__{}{}.json".format(a.study, a.dataset, a.model,
                                                               "__" + a.setting_prefix.strip("_") if a.setting_prefix else "")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1, allow_nan=False); f.write("\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
