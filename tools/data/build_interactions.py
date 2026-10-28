"""Interaction part of the data contract.

Sources (tools/data/sources.py): LightGCN release files (amazon-book, yelp2018: train + test merged), the Amazon 2018 5-core
file of one category (amazon2018-grocery, 5-core re-checked) or MovieLens-1M ratings (ml-1m, every rating implicit).
Remaps users and items in
ascending order of their RAW id strings, splits every user 70/10/20 (utility/utility_data/split.py, seed 2026), and writes
dataset_verify/<dataset>/<seed>/: interactions.csv (train + val), sealed_test/test_interactions.csv, user_map.csv, item_map.csv, degree_groups.csv (train-only),
split_check.csv and split_report.json (val/test based, never a model input), split_manifest.json.

    python -m tools.data.build_interactions --dataset yelp2018
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import DATASET_DIR, contract_dir, taxprocl_file  # noqa: E402
from tools.data.sources import expected_sha256  # noqa: E402
from utility.utility_data.split import RATIOS, split_all  # noqa: E402

SOURCE_FILES = ("train.txt", "test.txt", "user_list.txt", "item_list.txt")
GROUP_BOUNDS = (("near_cold", 1, 5), ("mid_tail", 6, 10), ("warm", 11, 10**12))
SEALED_DIR, SEALED_FILE = "sealed_test", "test_interactions.csv"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_sources(dataset):
    """Raw LightGCN files in dataset/<dataset>/ (downloaded by tools/data/download_data.py, or copied once from the
    TaxPro-CL repository copy if HITAX_TAXPROCL_ROOT is set); every file must have the SHA-256 recorded in
    tools/data/sources.py."""
    target = DATASET_DIR / dataset
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in SOURCE_FILES:
        src, dst = taxprocl_file("dataset/{}/{}".format(dataset, name)), target / name
        if not dst.is_file():
            if src is None or not src.is_file():
                raise FileNotFoundError("{} missing: run python -m tools.data.download_data --datasets {}".format(
                    dst, dataset))
            shutil.copyfile(src, dst)
        hashes[name] = sha256_file(dst)
        if hashes[name] != expected_sha256("dataset/{}/{}".format(dataset, name)):
            raise ValueError("source file differs from the recorded SHA-256 (tools/data/sources.py): {}".format(dst))
    return target, hashes


def read_pairs(path):
    pairs = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            tokens = line.split()
            if len(tokens) > 1:
                u = int(tokens[0])
                pairs.extend((u, int(i)) for i in tokens[1:])
    return pairs


def read_id_list(path):
    """LightGCN list file 'org_id remap_id' -> {remap_id: org_id}."""
    out = {}
    with open(path, encoding="utf-8") as stream:
        header = stream.readline().split()
        if header[:2] != ["org_id", "remap_id"]:
            raise ValueError("unexpected header in {}: {}".format(path, header))
        for line in stream:
            tokens = line.split()
            if len(tokens) >= 2:
                out[int(tokens[-1])] = tokens[0]
    return out


def group_of(d):
    if d == 0:
        return "strict_cold"
    return next(g for g, lo, hi in GROUP_BOUNDS if lo <= d <= hi)


def write_csv(path, header, rows):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp, path)


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(tmp, path)


DATASETS = {"amazon-book": "lightgcn", "yelp2018": "lightgcn", "amazon2018-grocery": "amazon2018", "ml-1m": "movielens"}
ROLES = {"amazon-book": "development", "yelp2018": "development", "amazon2018-grocery": "confirmation",
         "ml-1m": "secondary"}
RAW_FILES = {"amazon2018-grocery": "Grocery_and_Gourmet_Food_5.json.gz", "ml-1m": "ml-1m.zip"}


def iterative_k_core(pairs, k):
    """Repeatedly drop users and items with fewer than k interactions (ported from the TaxPro-CL repository)."""
    current, rounds = set(pairs), 0
    while True:
        du = collections.Counter(u for u, _ in current); di = collections.Counter(i for _, i in current)
        kept = {(u, i) for u, i in current if du[u] >= k and di[i] >= k}
        rounds += 1
        if len(kept) == len(current):
            return kept, rounds
        current = kept


def load_raw(dataset):
    """-> (set of (raw user id, raw item id) strings, source hashes, info)."""
    kind = DATASETS[dataset]
    if kind == "lightgcn":
        src_dir, hashes = copy_sources(dataset)
        raw = read_pairs(src_dir / "train.txt") + read_pairs(src_dir / "test.txt")
        users_raw, items_raw = read_id_list(src_dir / "user_list.txt"), read_id_list(src_dir / "item_list.txt")
        pairs = {(users_raw[u], items_raw[i]) for u, i in raw}
        return pairs, hashes, {"raw_rows": len(raw), "unique_pairs": len(pairs)}
    path = DATASET_DIR / "raw" / dataset / RAW_FILES[dataset]
    if not path.is_file():
        raise FileNotFoundError("{} missing: run python -m tools.data.download_data --datasets {}".format(path, dataset))
    if sha256_file(path) != expected_sha256("dataset/raw/{}/{}".format(dataset, RAW_FILES[dataset])):
        raise ValueError("raw file differs from the recorded SHA-256 (tools/data/sources.py): {}".format(path))
    hashes = {path.name: sha256_file(path)}
    if kind == "amazon2018":
        raw = []
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rec = json.loads(line)
                    raw.append((rec["reviewerID"], rec["asin"]))       # interaction keys only
        pairs = set(raw)
        core, rounds = iterative_k_core(pairs, 5)                    # re-check 5-core on the whole data
        return core, hashes, {"raw_rows": len(raw), "unique_pairs": len(pairs), "five_core_pairs": len(core),
                              "five_core_rounds": rounds}
    with zipfile.ZipFile(path) as z:                                 # every rating is an implicit interaction
        lines = z.read("ml-1m/ratings.dat").decode("latin-1").splitlines()
    raw = [tuple(line.split("::")[:2]) for line in lines if line.strip()]
    pairs = set(raw)
    return pairs, hashes, {"raw_rows": len(raw), "unique_pairs": len(pairs)}


def build(dataset, seed=2026, force=False):
    out_dir = contract_dir(dataset, seed)
    if (out_dir / "split_manifest.json").is_file() and not force:
        raise FileExistsError("contract exists (use --force): {}".format(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # merge + deduplicate
    pairs, source_hashes, info = load_raw(dataset)

    # deterministic remap by ascending RAW id string
    u_order = sorted({u for u, _ in pairs})
    i_order = sorted({i for _, i in pairs})
    u_new = {r: k for k, r in enumerate(u_order)}
    i_new = {r: k for k, r in enumerate(i_order)}
    by_user = collections.defaultdict(list)
    for u, i in pairs:
        by_user[u_new[u]].append(i_new[i])

    # per-user split
    rows, dropped = split_all(by_user, seed)
    rows.sort()
    # the test split is sealed: written apart from the training contract and read only with a frozen protocol
    write_csv(out_dir / "interactions.csv", ("user_id", "item_id", "split"), [r for r in rows if r[2] != "test"])
    (out_dir / SEALED_DIR).mkdir(exist_ok=True)
    write_csv(out_dir / SEALED_DIR / SEALED_FILE, ("user_id", "item_id", "split"), [r for r in rows if r[2] == "test"])
    write_csv(out_dir / "user_map.csv", ("id", "raw_id"), list(enumerate(u_order)))
    write_csv(out_dir / "item_map.csv", ("id", "raw_id"), list(enumerate(i_order)))

    # train degrees and groups (train-only) + split check (val/test, never a model input)
    n_items = len(i_order)
    d_train = np.zeros(n_items, dtype=np.int64); d_all = np.zeros(n_items, dtype=np.int64)
    d_trainval = np.zeros(n_items, dtype=np.int64)
    counts = collections.Counter()
    for _, i, s in rows:
        counts[s] += 1; d_all[i] += 1
        if s == "train":
            d_train[i] += 1
        if s in ("train", "val"):
            d_trainval[i] += 1
    write_csv(out_dir / "degree_groups.csv", ("item_id", "d_train", "group"),
              [(i, int(d_train[i]), group_of(int(d_train[i]))) for i in range(n_items)])
    share = np.where(d_all > 0, (d_all - d_train) / np.maximum(d_all, 1), 0.0)
    write_csv(out_dir / "split_check.csv", ("item_id", "d_trainval", "heldout_share"),
              [(i, int(d_trainval[i]), "%.6f" % share[i]) for i in range(n_items)])

    groups = np.array([group_of(int(d)) for d in d_train])
    test_users = collections.defaultdict(set)
    for u, i, s in rows:
        if s == "test":
            g = groups[i]
            test_users["overall"].add(u); test_users[g].add(u)
            if g in ("near_cold", "mid_tail"):
                test_users["long_tail"].add(u)
    report = {
        "dataset": dataset, "split_seed": seed,
        "users_with_test_positive": {g: len(v) for g, v in sorted(test_users.items())},
        "items_per_group_train_degree": {g: int((groups == g).sum()) for g in ("strict_cold", "near_cold", "mid_tail", "warm")},
        "heldout_share_item_mean": {g: float(share[groups == g].mean()) for g in ("near_cold", "mid_tail", "warm")
                                    if (groups == g).any()},
        "heldout_share_pooled": {g: float((d_all - d_train)[groups == g].sum() / d_all[groups == g].sum())
                                 for g in ("near_cold", "mid_tail", "warm") if (groups == g).any()},
        "near_cold_items_with_trainval_degree_above_5": float((d_trainval[groups == "near_cold"] > 5).mean())
        if (groups == "near_cold").any() else None,
        "check_long_tail_test_users": {"value": len(test_users["long_tail"]), "threshold": 1000,
                                       "pass": len(test_users["long_tail"]) >= 1000},
        "note": "data description only (counts of held-out rows); no model is evaluated on the test split here",
    }
    write_json(out_dir / "split_report.json", report)
    total = sum(counts.values())
    manifest = {
        "dataset": dataset, "role": ROLES[dataset], "protocol": "per-user 70/10/20, default_rng([seed, user_id]), floor, min 1",
        "ratios_nominal": list(RATIOS), "ratios_actual": [counts[s] / total for s in ("train", "val", "test")],
        "seed": seed, "numpy_version": np.__version__, "n_dropped_users": len(dropped),
        "counts": {"users": len(u_order), "items": n_items, **info,
                   "train": counts["train"], "val": counts["val"], "test": counts["test"]},
        "source_files": {name: {"sha256": h} for name, h in source_hashes.items()},
        "sha256": {name: sha256_file(out_dir / name) for name in
                   ("interactions.csv", "user_map.csv", "item_map.csv", "degree_groups.csv")},
        "sealed_test": {"file": "{}/{}".format(SEALED_DIR, SEALED_FILE),
                        "sha256": sha256_file(out_dir / SEALED_DIR / SEALED_FILE),
                        "rule": "read only through a frozen protocol (protocol/frozen_protocol.json)"},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out_dir / "split_manifest.json", manifest)
    return manifest, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    manifest, report = build(args.dataset, args.seed, args.force)
    print(json.dumps({"counts": manifest["counts"], "report": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
