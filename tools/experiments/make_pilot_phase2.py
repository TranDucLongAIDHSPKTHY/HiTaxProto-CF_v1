"""Write the phase-2 plan of a pilot from the phase-1 selection of the reference LightGCN.

Reads results/selection/pilot__<dataset>__LightGCN.json (validation Overall Recall@20 on the SELECTION half, mean over
seeds 42, 0, 1) and writes plans/pilot_<dataset>_phase2.json: DegreeGate x {taxonomy, shuffled, global} and
GlobalGate x taxonomy, b_init in {-1, 0, 1}, seeds 42, 0, 1. Every run takes --reference_selection (each seed is held
to the validation Overall of the reference run of the same seed) and inherits the reference's split seed, (lr, reg)
and adjacency layout.

    python -m tools.experiments.make_pilot_phase2 --dataset yelp2018
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import MODEL_OUTPUT_DIR, PROJECT_ROOT, RESULT_DIR  # noqa: E402

ARMS = (("degree", "taxonomy"), ("degree", "shuffled"), ("degree", "global"), ("global", "taxonomy"))
B_INIT = ("-1", "0", "1")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    a = p.parse_args(argv)
    rep = json.loads((RESULT_DIR / "selection" / "pilot__{}__LightGCN.json".format(a.dataset)).read_text(encoding="utf-8"))
    if rep["seeds"] != [42, 0, 1] or rep["chosen_metrics"].get("validation_users") != "selection":
        raise SystemExit("the reference selection must use seeds 42 0 1 on the selection half of the validation users")
    if sorted(rep["chosen_metrics"].get("per_seed", {})) != ["0", "1", "42"]:
        raise SystemExit("the reference selection report has no per-seed values; rerun select_config for LightGCN")
    report_name = "pilot__{}__LightGCN.json".format(a.dataset)
    ref_dir = MODEL_OUTPUT_DIR / "pilot" / a.dataset / "LightGCN" / rep["chosen"] / "seed42"
    cfg = json.loads((ref_dir / "config_resolved.json").read_text(encoding="utf-8"))
    reference = rep["chosen_metrics"]["val_Overall"]
    runs = []
    for gate, source in ARMS:
        for b in B_INIT:
            runs.append(["--model", "HiTaxProtoCF", "--dataset", a.dataset, "--split_seed", str(cfg["split_seed"]),
                         "--study", "pilot",
                         "--validation_users", "selection", "--seeds", "42", "0", "1", "--gate", gate,
                         "--source", source, "--gate_b_init", b, "--lr", str(cfg["lr"]), "--reg", str(cfg["reg"]),
                         "--adj_layout", str(cfg["adj_layout"]), "--reference_selection", report_name, "--device", "cuda"])
    plan = {"name": "pilot_{}_phase2".format(a.dataset),
            "purpose": "gated models of the pilot; reference LightGCN {} (validation Overall Recall@20, selection half "
                       "= {:.8f})".format(rep["chosen"], reference),
            "reference": {"setting": rep["chosen"], "val_Overall": reference, "lr": cfg["lr"], "reg": cfg["reg"],
                          "per_seed_val_Overall": {s: v["val_Overall"] for s, v in rep["chosen_metrics"]["per_seed"].items()}},
            "runs": runs}
    out = PROJECT_ROOT / "plans" / "pilot_{}_phase2.json".format(a.dataset)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(plan, f, indent=1); f.write("\n")
    print(out, len(runs), "entries x 3 seeds; reference", rep["chosen"], reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
