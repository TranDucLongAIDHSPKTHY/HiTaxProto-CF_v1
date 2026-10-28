"""Independent audit of one data contract. Written WITHOUT importing the data pipeline: every invariant is recomputed
from the raw sources and the contract files, and every failure is listed.

Checks
  raw      the contract (train + val + sealed test) is exactly the deduplicated raw interactions (after the 5-core
           re-check where it applies), mapped through user_map / item_map
  maps     ids contiguous, raw ids unique and assigned in ascending order of the raw id string
  split    per user: disjoint parts, sizes test = max(1, floor(0.2 n)), val = max(1, floor(0.1 n)), and the exact
           assignment reproduced from numpy.random.default_rng([seed, user]) over the user's sorted items
  degree   degree_groups.csv = train degree recount, group thresholds
  labels   item_labels.csv: unique pairs, existing non-root nodes, every used node has >= 2 train members, labels are
           most specific, each label is an ancestor-or-self of a structural label of the item
  ancestor ancestor_sets.csv = closure over taxonomy_edges.csv, shortest hop, coverage on train items, kept rule
  shuffled each seed: label counts per item and node sizes over train items preserved, untouched items unchanged,
           every train item labelled in the taxonomy has a defined leave-one-out prototype in the shuffled labels
  sealed   interactions.csv has no test row; the sealed file hash matches the manifest
  spot     random items: labels re-derived from the raw metadata record of the release reported for the item

    python -m tools.protocol.audit_contract --dataset amazon-book --seed 2026
"""
import argparse
import ast
import collections
import csv
import gzip
import hashlib
import json
import random
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import DATASET_DIR, DATASET_VERIFY_DIR, metadata_source  # noqa: E402


class Audit:
    def __init__(self):
        self.results = collections.OrderedDict()

    def check(self, name, ok, detail=""):
        self.results.setdefault(name, []).append((bool(ok), detail))

    def report(self):
        out = {}
        for name, items in self.results.items():
            bad = [d for ok, d in items if not ok]
            out[name] = {"checks": len(items), "failures": len(bad), "examples": bad[:5]}
        return out


def rows_of(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        return header, [tuple(x) for x in r]


def raw_pairs(dataset):
    """Deduplicated raw interactions as (raw user, raw item) strings, recomputed here."""
    if dataset in ("amazon-book", "yelp2018"):
        d = DATASET_DIR / dataset
        def ids(name):
            with open(d / name, encoding="utf-8") as f:
                f.readline()
                return {int(t[-1]): t[0] for t in (line.split() for line in f) if len(t) >= 2}
        users, items = ids("user_list.txt"), ids("item_list.txt")
        pairs = set()
        for name in ("train.txt", "test.txt"):
            with open(d / name, encoding="utf-8") as f:
                for line in f:
                    t = line.split()
                    for i in t[1:]:
                        pairs.add((users[int(t[0])], items[int(i)]))
        return pairs
    if dataset == "amazon2018-grocery":
        pairs = set()
        with gzip.open(DATASET_DIR / "raw" / dataset / "Grocery_and_Gourmet_Food_5.json.gz", "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    pairs.add((r["reviewerID"], r["asin"]))
        while True:                                   # 5-core, recomputed independently
            du = collections.Counter(u for u, _ in pairs); di = collections.Counter(i for _, i in pairs)
            kept = {p for p in pairs if du[p[0]] >= 5 and di[p[1]] >= 5}
            if kept == pairs:
                return pairs
            pairs = kept
    with zipfile.ZipFile(DATASET_DIR / "raw" / dataset / "ml-1m.zip") as z:
        return {tuple(line.split("::")[:2]) for line in z.read("ml-1m/ratings.dat").decode("latin-1").splitlines()
                if line.strip()}


def audit(dataset, seed, n_spot=40):
    A = Audit()
    d = DATASET_VERIFY_DIR / dataset / str(seed)
    manifest = json.loads((d / "split_manifest.json").read_text(encoding="utf-8"))
    _, umap = rows_of(d / "user_map.csv"); _, imap = rows_of(d / "item_map.csv")
    users = [r for _, r in umap]; items = [r for _, r in imap]
    # maps
    A.check("maps", [int(k) for k, _ in umap] == list(range(len(umap))), "user ids contiguous")
    A.check("maps", [int(k) for k, _ in imap] == list(range(len(imap))), "item ids contiguous")
    A.check("maps", users == sorted(set(users)), "user raw ids unique and ascending")
    A.check("maps", items == sorted(set(items)), "item raw ids unique and ascending")
    # interactions
    h, inter = rows_of(d / "interactions.csv")
    _, sealed = rows_of(d / manifest["sealed_test"]["file"])
    A.check("sealed", all(s != "test" for _, _, s in inter), "interactions.csv contains test rows")
    A.check("sealed", all(s == "test" for _, _, s in sealed), "sealed file contains non-test rows")
    sha = hashlib.sha256((d / manifest["sealed_test"]["file"]).read_bytes()).hexdigest()
    A.check("sealed", sha == manifest["sealed_test"]["sha256"], "sealed hash differs from manifest")
    allrows = [(int(u), int(i), s) for u, i, s in inter + sealed]
    contract = {(users[u], items[i]) for u, i, _ in allrows}
    A.check("raw", len(contract) == len(allrows), "duplicate (user, item) rows in the contract")
    raw = raw_pairs(dataset)
    A.check("raw", contract == raw, "contract != raw: missing {} extra {}".format(len(raw - contract), len(contract - raw)))
    # split
    by_user = collections.defaultdict(lambda: collections.defaultdict(list))
    for u, i, s in allrows:
        by_user[u][s].append(i)
    for u in sorted(by_user):
        parts = by_user[u]
        its = sorted(i for s in parts for i in parts[s])
        n = len(its)
        n_test, n_val = max(1, int(0.2 * n)), max(1, int(0.1 * n))
        ok_sizes = len(parts["test"]) == n_test and len(parts["val"]) == n_val and len(parts["train"]) == n - n_test - n_val
        A.check("split", ok_sizes, "user {} sizes".format(u))
        perm = np.random.default_rng([seed, u]).permutation(n)
        exp_test = {its[k] for k in perm[:n_test]}; exp_val = {its[k] for k in perm[n_test:n_test + n_val]}
        A.check("split", set(parts["test"]) == exp_test and set(parts["val"]) == exp_val, "user {} assignment".format(u))
    # degree
    deg = collections.Counter(i for _, i, s in allrows if s == "train")
    _, dg = rows_of(d / "degree_groups.csv")
    def grp(x):
        return "strict_cold" if x == 0 else "near_cold" if x <= 5 else "mid_tail" if x <= 10 else "warm"
    for i, dt, g in dg:
        A.check("degree", int(dt) == deg.get(int(i), 0) and g == grp(int(dt)), "item {}".format(i))
    A.check("degree", len(dg) == len(items), "one row per item")
    train_items = {i for i in range(len(items)) if deg.get(i, 0) >= 1}
    # taxonomy
    _, nodes = rows_of(d / "taxonomy_nodes.csv")
    name = {int(k): n for k, n, _, _ in nodes}
    root = [int(k) for k, _, _, r in nodes if r == "True"]
    A.check("labels", len(root) == 1, "exactly one super-root")
    _, edges = rows_of(d / "taxonomy_edges.csv")
    parents = collections.defaultdict(set)
    for c, p in edges:
        parents[int(c)].add(int(p))

    def anc(n):
        out, stack = set(), [n]
        while stack:
            for p in parents.get(stack.pop(), ()):
                if p not in out:
                    out.add(p); stack.append(p)
        return out
    _, lab = rows_of(d / "item_labels.csv")
    _, struct = rows_of(d / "item_labels_struct.csv")
    A.check("labels", len(set(lab)) == len(lab), "duplicate (item, node) rows")
    L = collections.defaultdict(set); S = collections.defaultdict(set)
    for i, n in lab:
        L[int(i)].add(int(n))
    for i, n in struct:
        S[int(i)].add(int(n))
    members = collections.Counter(n for i in train_items for n in L.get(i, ()))
    for i, ns in L.items():
        for n in ns:
            A.check("labels", n in name and n != root[0], "item {} node {} unknown or root".format(i, n))
            A.check("labels", members[n] >= 2, "node {} has {} train members".format(n, members[n]))
            A.check("labels", not any(n in anc(m) for m in ns if m != n), "item {} label {} not most specific".format(i, n))
            A.check("labels", any(n == s or n in anc(s) for s in S[i]), "item {} label {} not from its structure".format(i, n))
    # ancestor sets
    _, anc_rows = rows_of(d / "ancestor_sets.csv")
    got = collections.defaultdict(dict)
    for i, n, hop, cov, kept in anc_rows:
        got[int(i)][int(n)] = (int(hop), float(cov), kept == "True")
    closure = {}
    for i in range(len(items)):
        dist = {n: 0 for n in L.get(i, ())}
        frontier = list(dist)
        while frontier:
            nxt = []
            for n in frontier:
                for p in parents.get(n, ()):
                    if p not in dist:
                        dist[p] = dist[n] + 1; nxt.append(p)
            frontier = nxt
        closure[i] = dist
    cover = collections.Counter(n for i in train_items for n in closure[i])
    for i, dist in closure.items():
        A.check("ancestor", set(dist) == set(got.get(i, {})), "item {} ancestor set".format(i))
        for n, h in dist.items():
            if n in got.get(i, {}):
                gh, gc, gk = got[i][n]
                cv = cover[n] / len(train_items)
                A.check("ancestor", gh == h and abs(gc - cv) < 1e-6 and gk == (n != root[0] and cv <= 0.5),
                        "item {} node {}".format(i, n))
    # shuffled labels
    for s in (101, 102, 103, 104, 105):
        _, sh = rows_of(d / "shuffled_labels_s{}.csv".format(s))
        SH = collections.defaultdict(set)
        for i, n in sh:
            SH[int(i)].add(int(n))
        A.check("shuffled", all(len(SH.get(i, ())) == len(L.get(i, ())) for i in range(len(items))),
                "seed {} label counts".format(s))
        A.check("shuffled", collections.Counter(n for i in train_items for n in SH.get(i, ())) == members,
                "seed {} node sizes".format(s))
        A.check("shuffled", all(SH.get(i, set()) == L.get(i, set()) for i in range(len(items))
                                if i not in train_items or not L.get(i)), "seed {} untouched items".format(s))
        msize = collections.Counter(n for i in train_items for n in SH.get(i, ()))
        A.check("shuffled", all(msize[n] >= 2 for i in train_items for n in SH.get(i, ())),
                "seed {} leave-one-out prototype defined".format(s))
        moved = sum(SH.get(i, set()) != L.get(i, set()) for i in train_items if L.get(i))
        A.check("shuffled", moved > 0, "seed {} moved nothing".format(s))
        for pool in strata_pools(L, deg, train_items):     # label sets only move inside their (pooled) stratum
            A.check("shuffled", collections.Counter(frozenset(L[i]) for i in pool)
                    == collections.Counter(frozenset(SH.get(i, ())) for i in pool), "seed {} stratum".format(s))
    # spot check against raw metadata
    spot = spot_check(dataset, d, items, S, name, n_spot)
    for ok, detail in spot:
        A.check("spot", ok, detail)
    return A.report()


BINS = ((1, 5), (6, 10), (11, 20), (21, 100), (101, 10**12))


def strata_pools(L, deg, train_items, min_size=20):
    """Strata of the shuffled source, recomputed: (degree band, number of labels); for each label count, adjacent
    bands are pooled upwards until a pool holds >= min_size items, a too-small last pool joins the previous one."""
    strata = collections.defaultdict(list)
    for i in sorted(train_items):
        if L.get(i):
            b = next(k for k, (lo, hi) in enumerate(BINS) if lo <= deg[i] <= hi)
            strata[(len(L[i]), b)].append(i)
    pools = []
    for c in sorted({k[0] for k in strata}):
        cp, cur = [], []
        for b in range(len(BINS)):
            cur += strata.get((c, b), [])
            if len(cur) >= min_size:
                cp.append(cur); cur = []
        if cur:
            if cp:
                cp[-1] = cp[-1] + cur
            else:
                cp.append(cur)
        pools += cp
    return pools


def _amazon_records(path, asins, key, gz=False):
    """One pass over a metadata dump: {asin: first record} for the requested ids."""
    pat = re.compile(rb"['\"]" + key.encode() + rb"['\"]\s*:\s*['\"]([^'\"]+)['\"]")
    out = {}
    opener = gzip.open if gz else open
    with opener(path, "rb") as f:
        for line in f:
            m = pat.search(line)
            if m is not None:
                a = m.group(1).decode("utf-8", errors="replace")
                if a in asins and a not in out:
                    text = line.decode("utf-8", errors="replace")
                    out[a] = ast.literal_eval(text) if text.lstrip().startswith("{'") else json.loads(text)
    return out


def _amazon_expected(rec, rel, kindle):
    import html as _h
    if rel == "2014":
        paths = [list(p) for p in rec.get("categories", []) if p]
    else:
        p = rec.get("category") if rel == "2018" else rec.get("categories")
        main = rec.get("main_cat") if rel == "2018" else rec.get("main_category")
        paths = [list(p)] if p else ([[main]] if main else [])
    paths = [[_h.unescape(str(x)).strip() for x in p if _h.unescape(str(x)).strip()] for p in paths]
    paths = [p for p in paths if p]
    if kindle and any(p[0] == "Books" for p in paths):
        paths = [p for p in paths if p[0] != "Kindle Store"]
    full = {" > ".join(p) for p in paths}
    return sorted(f for f in full if not any(g != f and g.startswith(f + " > ") for g in full))


def spot_check(dataset, d, items, S, name, n):
    """Re-derive the STRUCTURAL labels of random items from their raw metadata record."""
    rnd = random.Random(7)
    picks = rnd.sample(range(len(items)), min(n, len(items)))
    out = []
    if dataset in ("amazon-book", "amazon2018-grocery"):
        if dataset == "amazon-book":
            files = {"2014": (metadata_source("amazon/2014/meta_Books.json"), "asin", False),
                     "2018": (metadata_source("amazon/2018/meta_Books.json"), "asin", False),
                     "2023": (metadata_source("amazon/2023/meta_Books.jsonl"), "parent_asin", False)}
        else:
            files = {"2018": (DATASET_DIR / "raw" / dataset / "meta_Grocery_and_Gourmet_Food.json.gz", "asin", True)}
        want = {items[i] for i in picks}
        recs = {rel: _amazon_records(path, want, key, gz) for rel, (path, key, gz) in files.items()}
        for i in picks:
            a = items[i]
            labels = sorted(name[x] for x in S.get(i, ()))
            cand = [(rel, _amazon_expected(recs[rel][a], rel, dataset == "amazon-book")) for rel in files if a in recs[rel]]
            chosen = next((c for c in cand if any(e.count(" > ") >= 1 for e in c[1])), cand[0] if cand else None)
            expect = chosen[1] if chosen else []
            out.append((labels == expect, "item {} release {}: labels {} expected {}".format(
                a, chosen[0] if chosen else None, labels, expect)))
    elif dataset == "ml-1m":
        with zipfile.ZipFile(DATASET_DIR / "raw" / dataset / "ml-1m.zip") as z:
            genres = {line.split("::")[0]: line.split("::")[2] for line in
                      z.read("ml-1m/movies.dat").decode("latin-1").splitlines() if line.strip()}
        for i in picks:
            labels = sorted(name[x] for x in S.get(i, ()))
            out.append((labels == sorted(genres[items[i]].split("|")), "movie {}: {}".format(items[i], labels)))
    else:                                              # yelp2018: raw CATEGORIES -> official aliases -> most specific
        h = json.loads((ROOT / "metadata" / "yelp2018" / "yelp_categories.json").read_text(encoding="utf-8"))
        by_title = collections.defaultdict(list)
        for al, t in h["titles"].items():
            by_title[t.casefold()].append(al)
        pick_alias = {t: sorted(v, key=lambda a: (h["locales"][a].strip() != "All", "US" not in h["locales"][a], a))[0]
                      for t, v in by_title.items()}

        def ancestors(a):
            out_, stack = set(), [a]
            while stack:
                for q in h["parents"].get(stack.pop(), []):
                    if q not in out_:
                        out_.add(q); stack.append(q)
            return out_
        want = {items[i] for i in picks}
        raw = {}
        with open(metadata_source("yelp/2018/Yelp2018.item"), encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["ITEM"] in want:
                    raw[r["ITEM"]] = r["CATEGORIES"] or ""
        for i in picks:
            b = items[i]
            frags = [x.strip() for x in raw.get(b, "").replace("\/", "/").split(",") if x.strip()]
            titles, k = [], 0
            while k < len(frags):                      # longest run of fragments forming an official title
                for j in range(len(frags), k, -1):
                    t = ", ".join(frags[k:j])
                    if j == k + 1 or t.casefold() in pick_alias:
                        titles.append(t); k = j
                        break
            al = {pick_alias.get(t.casefold(), "unmatched:" + t) for t in titles}
            expect = sorted(x for x in al if not any(x in ancestors(y) for y in al if y != x))
            labels = sorted(name[x] for x in S.get(i, ()))
            out.append((labels == expect, "business {}: labels {} expected {}".format(b, labels, expect)))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--spot", type=int, default=40)
    a = p.parse_args(argv)
    rep = audit(a.dataset, a.seed, a.spot)
    out = ROOT / "preprocessed" / "data_checks" / "audit__{}__{}.json".format(a.dataset, a.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"dataset": a.dataset, "seed": a.seed, "spot_items": a.spot, "results": rep}, f, indent=1,
                  ensure_ascii=False)
        f.write("\n")
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0 if all(v["failures"] == 0 for v in rep.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
