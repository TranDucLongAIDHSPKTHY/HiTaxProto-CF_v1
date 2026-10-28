"""Collect every run of log/ into results/per_seed_metrics.csv (validation / assessment metrics only; test metrics
are added from test_evaluated.json once the sealed test evaluation has run).

    python -m tools.analysis.collect_results
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import MODEL_OUTPUT_DIR, RESULT_DIR  # noqa: E402

GROUPS = ("overall", "near_cold", "mid_tail", "long_tail", "warm", "near_cold_within", "mid_tail_within",
          "long_tail_within")
FIELDS = ["study", "dataset", "model", "setting", "seed", "status", "validation_users", "selected_epoch",
          "completed_epochs", "hit_cap", "feasible", "duration_seconds", "config_hash"]
for split in ("val", "assessment", "test"):
    FIELDS += ["{}_{}_{}".format(split, g, m) for g in GROUPS for m in ("Recall@20", "NDCG@20")]


def rows(root=None):
    root = Path(root or MODEL_OUTPUT_DIR)
    for m_path in sorted(root.glob("*/*/*/*/seed*/run_manifest.json")):
        d = m_path.parent
        study, dataset, model, setting = d.parts[-5:-1]
        m = json.loads(m_path.read_text(encoding="utf-8"))
        r = {"study": study, "dataset": dataset, "model": model, "setting": setting, "seed": d.name[4:]}
        for k in FIELDS[5:13]:
            r[k] = m.get(k, "")
        sources = {"val": m.get("selected_validation"), "assessment": m.get("assessment")}
        t = d / "test_evaluated.json"
        if t.is_file():
            sources["test"] = json.loads(t.read_text(encoding="utf-8"))["metrics"]
        for split, metrics in sources.items():
            if metrics:
                for g in GROUPS:
                    for k in ("Recall@20", "NDCG@20"):
                        r["{}_{}_{}".format(split, g, k)] = metrics.get(g, {}).get(k, "")
        yield r


def main(argv=None):
    out = RESULT_DIR / "per_seed_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = list(rows())
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n", extrasaction="ignore")
        w.writeheader(); w.writerows(data)
    print(out, len(data), "runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
