"""Leakage check: every model-input file must be recomputable from TRAIN rows only.

Copies the interaction contract of one dataset with every val/test row removed into a temporary directory,
re-runs degree groups, label resolution, ancestor sets and shuffled labels there, and compares the model-input
files byte for byte with the frozen contract.

    python -m tools.protocol.leak_check --dataset yelp2018
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.data.build_taxonomy as bt  # noqa: E402
from config_path.config_path import contract_dir  # noqa: E402
from tools.data.build_interactions import group_of, write_csv  # noqa: E402

MODEL_INPUTS = ("degree_groups.csv", "item_labels.csv", "ancestor_sets.csv",
                "shuffled_labels_s101.csv", "shuffled_labels_s102.csv", "shuffled_labels_s103.csv",
                "shuffled_labels_s104.csv", "shuffled_labels_s105.csv")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(dataset, seed=2026):
    src = Path(contract_dir(dataset, seed))
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / dataset / str(seed)
        dst.mkdir(parents=True)
        for name in ("item_map.csv", "user_map.csv", "split_manifest.json"):
            shutil.copyfile(src / name, dst / name)
        n_items = len((src / "item_map.csv").read_text(encoding="utf-8").splitlines()) - 1
        deg = [0] * n_items
        with open(src / "interactions.csv", encoding="utf-8", newline="") as f, \
                open(dst / "interactions.csv", "w", encoding="utf-8", newline="") as g:
            r = csv.reader(f); g.write(",".join(next(r)) + "\n")
            for u, i, s in r:
                if s == "train":
                    g.write("{},{},{}\n".format(u, i, s)); deg[int(i)] += 1
        write_csv(dst / "degree_groups.csv", ("item_id", "d_train", "group"),
                  [(i, deg[i], group_of(deg[i])) for i in range(n_items)])
        orig = bt.contract_dir
        bt.contract_dir = lambda ds, s: Path(tmp) / ds / str(s)
        try:
            bt.build(dataset, seed, bt.DEFAULT["policy"], bt.DEFAULT["kindle"])
        finally:
            bt.contract_dir = orig
        result = {name: digest(src / name) == digest(dst / name) for name in MODEL_INPUTS}
    return {"dataset": dataset, "all_equal": all(result.values()), "files": result}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args(argv)
    res = check(a.dataset, a.seed)
    print(json.dumps(res, indent=1))
    return 0 if res["all_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
