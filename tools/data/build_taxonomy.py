"""Taxonomy part of the data contract and its train-only statistics.

1. Labels per item: one metadata release per item — 'first_informative' (default: the earliest release with a path
   of depth >= 2, all its paths kept), 'deepest' (deepest path over releases, as in the TaxPro-CL repository) or
   'earliest_release' (earliest release with any path). Amazon "Kindle Store" paths duplicating "Books" paths are
   dropped. Node identity: the FULL path for Amazon, the alias of the official hierarchy for Yelp (flat names if the
   hierarchy file is absent); every parentless node hangs under a super-root ROOT. Λ is reduced to the most specific
   labels; cycles are rejected.
2. Valid labels on TRAIN items (>= 2 members); an invalid label is lifted to its parents.
3. Ancestor sets A(i) with hop distance and coverage on train items (τ = 0.5).
4. Shuffled label sets, seeds 101..105, strata = degree bin x |Λ(i)|, strata below 20 items pooled; mask check.

Only the default policy writes contract files; any other policy writes its statistics to
preprocessed/data_checks/<dataset>__<policy>__<variant>.json for comparison.

    python -m tools.data.build_taxonomy --dataset amazon-book
"""
import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_path.config_path import METADATA_DIR, PREPROCESSED_DIR, contract_dir  # noqa: E402
from models.HiTaxProtoCF import PrototypeMemory  # noqa: E402
from utility.utility_data.controls import shuffle_labels  # noqa: E402
from utility.utility_data.taxonomy import ancestor_sets, reduce_to_most_specific, valid_labels  # noqa: E402

ROOT = "ROOT"
RELEASES = {"amazon-book": ("2014", "2018", "2023"), "yelp2018": ("2018", "2021", "2022"),
            "amazon2018-grocery": ("2018",), "ml-1m": ("genres",)}
DEFAULT = {"policy": "first_informative", "kindle": "drop_if_books"}
SHUFFLE_SEEDS = (101, 102, 103, 104, 105)
TAU = 0.5
SEP = " > "


def choose_paths(by_release, releases, policy):
    """-> (release, paths) for one item."""
    avail = [(r, by_release.get(r) or []) for r in releases]
    avail = [(r, p) for r, p in avail if p]
    if not avail:
        return None, []
    if policy == "earliest_release":
        return avail[0]
    if policy == "first_informative":
        for r, p in avail:
            if any(len(x) >= 2 for x in p):
                return r, p
        return avail[0]
    if policy == "deepest":                          # deepest, then more paths, then newest
        return max(avail, key=lambda rp: (max(map(len, rp[1])), len(rp[1]), int(rp[0])))
    raise ValueError(policy)


def apply_kindle(paths, mode):
    if mode == "keep":
        return paths, False
    books = [p for p in paths if p[0].casefold() == "books"]
    if books and len(books) < len(paths) and any(p[0].casefold() == "kindle store" for p in paths):
        return [p for p in paths if p[0].casefold() != "kindle store"], True
    return paths, False


def read_contract(dataset, seed):
    d = contract_dir(dataset, seed)
    with open(d / "item_map.csv", encoding="utf-8", newline="") as f:
        r = csv.reader(f); next(r)
        raw_of = {int(i): raw for i, raw in r}
    deg = {}
    with open(d / "degree_groups.csv", encoding="utf-8", newline="") as f:
        r = csv.reader(f); next(r)
        for i, dt, _ in r:
            deg[int(i)] = int(dt)
    return d, raw_of, deg


def write_csv(path, header, rows):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(header); w.writerows(rows)
    os.replace(tmp, path)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write("\n")
    os.replace(tmp, path)


def check_acyclic(parents):
    state = {}

    def visit(n):
        if state.get(n) == 1:
            raise ValueError("cycle through node {}".format(n))
        if state.get(n) == 2:
            return
        state[n] = 1
        for p in parents.get(n, []):
            visit(p)
        state[n] = 2
    for n in list(parents):
        visit(n)


def prototype_mask(labels, deg, n_items):
    names = sorted({n for ns in labels.values() for n in ns})
    if not names:
        return torch.zeros(n_items, dtype=torch.bool)
    idx = {n: k for k, n in enumerate(names)}
    rows = [(i, idx[n]) for i, ns in labels.items() for n in ns]
    is_tr = torch.tensor([deg[i] >= 1 for i in range(n_items)])
    mem = PrototypeMemory(n_items, len(names), torch.tensor(rows, dtype=torch.long), is_tr)
    return mem(torch.zeros(n_items, 1))[1] & is_tr


YELP_HIERARCHY = METADATA_DIR / "yelp2018" / "yelp_categories.json"


def load_yelp_hierarchy():
    """-> (title casefold -> alias, alias -> parents). Ambiguous titles prefer the alias available 'All'/US."""
    h = json.loads(YELP_HIERARCHY.read_text(encoding="utf-8"))
    cands = collections.defaultdict(list)
    for alias, title in h["titles"].items():
        cands[title.casefold()].append(alias)
    rank = lambda a: (h["locales"][a].strip() != "All", "US" not in h["locales"][a], a)
    t2a = {t: sorted(al, key=rank)[0] for t, al in cands.items()}
    parents = {a: (h["parents"].get(a) or [ROOT]) for a in h["titles"]}
    return t2a, parents


def rejoin_titles(fragments, known):
    """The Yelp CATEGORIES column is comma separated, which splits titles containing a comma
    ('Beer, Wine & Spirits'); greedily re-join consecutive fragments whose join is a known title."""
    out, i = [], 0
    while i < len(fragments):
        for j in range(len(fragments), i, -1):
            cand = ", ".join(fragments[i:j])
            if j == i + 1 or cand.casefold() in known:
                out.append(cand); i = j
                break
    return out


def node_depths(nodes, parents):
    memo = {ROOT: 0}

    def dep(n):
        if n not in memo:
            memo[n] = 1 + max(dep(p) for p in parents.get(n, [ROOT]))
        return memo[n]
    return {n: dep(n) for n in nodes}


def build(dataset, seed, policy, kindle, hierarchy=True):
    d, raw_of, deg = read_contract(dataset, seed)
    n_items = len(raw_of)
    by_raw = json.loads((METADATA_DIR / dataset / "paths_by_release.json").read_text(encoding="utf-8"))
    yelp = dataset == "yelp2018"
    yelp_hier = yelp and hierarchy and YELP_HIERARCHY.is_file()
    flat = (yelp and not yelp_hier) or dataset == "ml-1m"
    kindle_rule = dataset == "amazon-book"
    release_count, kindle_dropped = collections.Counter(), 0
    direct, parents, depth = {}, {}, {}
    unmatched = collections.Counter()
    if yelp_hier:
        t2a, yelp_parents = load_yelp_hierarchy()
    for i in range(n_items):
        rel, paths = choose_paths(by_raw.get(raw_of[i], {}), RELEASES[dataset], policy)
        release_count[rel or "none"] += 1
        if kindle_rule:
            paths, dropped = apply_kindle(paths, kindle)
            kindle_dropped += dropped
        labels = []
        if yelp_hier:
            for title in rejoin_titles([p[0] for p in paths], t2a):
                alias = t2a.get(title.casefold())
                if alias is None:
                    unmatched[title] += 1
                    alias = "unmatched:" + title
                    yelp_parents.setdefault(alias, [ROOT])
                labels.append(alias)
            direct[i] = sorted(set(labels))
            continue
        for p in paths:
            if flat:
                labels.append(p[0]); depth[p[0]] = 1
                continue
            for k in range(1, len(p) + 1):                       # node identity = full path prefix
                node = SEP.join(p[:k])
                depth[node] = k
                parent = SEP.join(p[:k - 1]) if k > 1 else ROOT
                parents.setdefault(node, [])
                if parent not in parents[node]:
                    parents[node].append(parent)
            labels.append(SEP.join(p))
        direct[i] = sorted(set(labels))
    if yelp_hier:                                                # keep the labels used and all their ancestors
        used, stack = set(), [n for ns in direct.values() for n in ns]
        while stack:
            n = stack.pop()
            if n != ROOT and n not in used:
                used.add(n); stack.extend(yelp_parents.get(n, [ROOT]))
        parents = {n: list(yelp_parents.get(n, [ROOT])) for n in used}
        depth = node_depths(used, parents)
    if flat:                                                     # flat taxonomy: every label hangs under ROOT
        parents = {n: [ROOT] for n in depth}
    check_acyclic(parents)
    struct, n_reduced = reduce_to_most_specific(direct, parents)
    info = {}
    labels, valid = valid_labels(struct, deg, parents=parents, root=ROOT, info=info)

    train_items = [i for i in range(n_items) if deg[i] >= 1]
    lt_items = [i for i in train_items if deg[i] <= 10]
    member = collections.Counter(n for i in train_items for n in labels[i])
    cover = {n: c / len(train_items) for n, c in member.items()}
    raw_sets, anc_cover, kept, hop = ancestor_sets(labels, parents, train_items, TAU, ROOT)
    sizes = np.array(sorted(member.values())) if member else np.array([0])
    n_lab = np.array([len(labels[i]) for i in train_items])
    stats = {
        "dataset": dataset, "split_seed": seed, "policy": policy, "kindle": kindle if kindle_rule else None,
        "release_of_item": dict(release_count), "kindle_paths_dropped_items": kindle_dropped if kindle_rule else None,
        "yelp_unmatched_titles": dict(unmatched.most_common()) if yelp_hier else None,
        "share_train_items_with_valid_label": float((n_lab > 0).mean()),
        "share_long_tail_items_with_valid_label": float(np.mean([len(labels[i]) > 0 for i in lt_items])),
        # informativeness: a long-tail item has at least one valid label covering <= 50% of train items
        "share_long_tail_items_with_informative_label": float(np.mean([any(cover[n] <= TAU for n in labels[i])
                                                                for i in lt_items])),
        "labels_removed_by_reduction": n_reduced, "labels_lifted": info["n_lifted"],
        "labels_dropped": info["n_dropped"], "lift_rounds": info["rounds"],
        "valid_nodes_K": len(valid), "taxonomy_nodes": len(depth),
        "nodes_with_more_than_one_parent": sum(len(v) > 1 for v in parents.values()),
        "max_depth": max(depth.values()) if depth else 0,
        "mean_valid_labels_per_train_item": float(n_lab.mean()),
        "share_multilabel_train_items": float((n_lab > 1).mean()),
        "node_size_quantiles_25_50_75": [float(np.percentile(sizes, q)) for q in (25, 50, 75)],
        "largest_valid_node": max(member.items(), key=lambda kv: kv[1]) if member else None,
        "measured_type": ("flat multi-label" if flat else
                          ("DAG (official Yelp hierarchy)" if yelp_hier else "forest of trees (full-path identity)")
                          + (", multi-label" if (n_lab > 1).any() else "")),
        "checks": {"train_items_labelled_>=90%": bool((n_lab > 0).mean() >= 0.9),
               "long_tail_items_labelled_>=90%": bool(np.mean([len(labels[i]) > 0 for i in lt_items]) >= 0.9)},
    }
    variant = "{}__{}__{}".format(dataset, policy, (("hier" if yelp_hier else "flat") if yelp else
                                                  (kindle if kindle_rule else "default")))
    if (policy, kindle) != (DEFAULT["policy"], DEFAULT["kindle"]) or (yelp and not yelp_hier):
        write_json(PREPROCESSED_DIR / "data_checks" / (variant + ".json"), stats)
        return stats

    # ---- contract files (default policy only) ----
    names = [ROOT] + sorted(depth)
    nid = {n: k for k, n in enumerate(names)}
    write_csv(d / "taxonomy_nodes.csv", ("node_id", "name", "source", "is_super_root"),
              [(nid[n], n, dataset, n == ROOT) for n in names])
    edges = [(nid[c], nid[p]) for c, ps in sorted(parents.items()) for p in ps]
    if flat:
        edges = [(nid[n], 0) for n in sorted(depth)]
    write_csv(d / "taxonomy_edges.csv", ("child_id", "parent_id"), edges)
    write_csv(d / "item_labels_struct.csv", ("item_id", "node_id"),
              [(i, nid[n]) for i in range(n_items) for n in struct[i]])
    write_csv(d / "item_labels.csv", ("item_id", "node_id"), [(i, nid[n]) for i in range(n_items) for n in labels[i]])
    write_csv(d / "ancestor_sets.csv", ("item_id", "node_id", "hop", "coverage_train", "kept"),
              [(i, nid[n], hop[(i, n)], "%.6f" % anc_cover.get(n, 0.0), n in kept[i] and n != ROOT)
               for i in range(n_items) for n in sorted(raw_sets[i])])
    tax_mask = prototype_mask(labels, deg, n_items)
    shuf = {}
    for s in SHUFFLE_SEEDS:
        sinfo = {}
        out = shuffle_labels(labels, deg, s, info=sinfo)
        mask_equal = bool(torch.equal(prototype_mask(out, deg, n_items), tax_mask))
        if not mask_equal:
            raise AssertionError("shuffled-label mask differs from the taxonomy mask (seed {})".format(s))
        sizes_equal = collections.Counter(n for i in train_items for n in out[i]) == member
        counts_equal = all(len(out[i]) == len(labels[i]) for i in range(n_items))
        write_csv(d / "shuffled_labels_s{}.csv".format(s), ("item_id", "node_id"),
                  [(i, nid[n]) for i in range(n_items) for n in out[i]])
        shuf[str(s)] = {"pool_sizes": sinfo["pool_sizes"], "pool_label_counts": sinfo["pool_counts"],
                        "share_unchanged": sinfo["share_unchanged"], "mask_equals_taxonomy": mask_equal,
                        "node_sizes_equal": sizes_equal, "label_counts_equal": counts_equal}
    write_json(d / "shuffle_manifest.json", {"strata": "degree bins x |Λ(i)|, pooled to >= 20 items",
                                              "fuse_mask_size": int(tax_mask.sum()), "seeds": shuf})
    stats["fuse_mask_size"] = int(tax_mask.sum())
    write_json(d / "stats_train.json", stats)
    write_json(PREPROCESSED_DIR / "data_checks" / (variant + ".json"), stats)
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=tuple(RELEASES))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--policy", default=DEFAULT["policy"], choices=("first_informative", "deepest", "earliest_release"))
    p.add_argument("--kindle", default=DEFAULT["kindle"], choices=("drop_if_books", "keep"))
    p.add_argument("--no_hierarchy", action="store_true", help="Yelp: flat labels even if the hierarchy exists")
    a = p.parse_args(argv)
    print(json.dumps(build(a.dataset, a.seed, a.policy, a.kindle, hierarchy=not a.no_hierarchy), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
