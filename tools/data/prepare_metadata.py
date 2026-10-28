"""Category paths of every item, per metadata release (category fields only).

Amazon-Book: the 2014 (`categories`, list of paths), 2018 (`category` + `main_cat`, one path) and 2023 (`categories`
+ `main_category`, one path, keyed by `parent_asin`, joined directly to the ASIN) Books dumps (tools/data/sources.py)
are scanned once; ONLY those category fields are read. Yelp2018: the `CATEGORIES` column of the RecBole
Yelp 2018 / 2021 / 2022 item files (flat category titles; the hierarchy comes from tools/data/parse_yelp_categories.py). Parsing helpers are
ported from the TaxPro-CL repository (tools/data/prepare_metadata.py: html unescape, main-category root insertion, path de-duplication).

Output metadata/<dataset>/paths_by_release.json  {raw item id: {release: [path, ...]}} + scan_summary.json.

    python -m tools.data.prepare_metadata --dataset amazon-book
"""
import argparse
import ast
import csv
import gzip
import html
import json
import re
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import DATASET_DIR, DATASET_VERIFY_DIR, METADATA_DIR, metadata_source  # noqa: E402

AMAZON_FILES = {"2014": "amazon/2014/meta_Books.json", "2018": "amazon/2018/meta_Books.json",
                "2023": "amazon/2023/meta_Books.jsonl"}
AMAZON_ID = {"2014": re.compile(rb"['\"]asin['\"]\s*:\s*['\"]([^'\"]+)"),
             "2018": re.compile(rb'"asin"\s*:\s*"([^"]+)"'),
             "2023": re.compile(rb'"parent_asin"\s*:\s*"([^"]+)"')}
AMAZON2018_META = {"amazon2018-grocery": "meta_Grocery_and_Gourmet_Food.json.gz"}
YELP_FILES = {"2018": "yelp/2018/Yelp2018.item", "2021": "yelp/2021/Yelp2021.item", "2022": "yelp/2022/Yelp2022.item"}


def clean_text(value):
    if value is None:
        return None
    text = html.unescape(str(value)).strip()
    return text or None


def dedup_paths(paths):
    out, seen = [], set()
    for p in paths:
        key = tuple(x.casefold() for x in p)
        if key and key not in seen:
            seen.add(key); out.append(p)
    return out


def paths_2014(value):
    if not isinstance(value, (list, tuple)):
        return []
    if all(not isinstance(x, (list, tuple)) for x in value):
        value = [value]
    out = []
    for raw in value:
        if isinstance(raw, (list, tuple)):
            p = [c for c in (clean_text(x) for x in raw) if c]
            if p:
                out.append(p)
    return dedup_paths(out)


def single_path(value, root_category):
    if isinstance(value, (list, tuple)):
        p = [c for c in (clean_text(x) for x in value) if c]
    else:
        c = clean_text(value)
        p = [c] if c else []
    if not p:                                    # the store's main category is used only when no path exists:
        root = clean_text(root_category)         # prepending it would create false roots such as
        p = [root] if root else []               # "Buy a Kindle > Books > ..."
    return [p] if p else []


def amazon_paths(release, line):
    """Category fields only: no field derived from behaviour or reviews is read."""
    text = line.decode("utf-8", errors="replace")
    rec = ast.literal_eval(text) if release == "2014" else json.loads(text)
    if release == "2014":
        return paths_2014(rec.get("categories"))
    if release == "2018":
        return single_path(rec.get("category"), rec.get("main_cat"))
    return single_path(rec.get("categories"), rec.get("main_category"))


def item_ids(dataset):
    """Raw item ids of the dataset, read from the item map of its data contract (any split seed)."""
    path = sorted((DATASET_VERIFY_DIR / dataset).glob("*/item_map.csv"))[0]
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream); next(reader)
        return {raw for _, raw in reader}


def scan_amazon2018(dataset):
    """One Amazon 2018 category: the 2018 metadata of the same release as the interactions (category field)."""
    targets = item_ids(dataset)
    out = {a: {} for a in targets}
    path = DATASET_DIR / "raw" / dataset / AMAZON2018_META[dataset]
    found = 0
    with gzip.open(path, "rb") as stream:
        for line in stream:
            m = AMAZON_ID["2018"].search(line)
            if m is None:
                continue
            a = m.group(1).decode("utf-8", errors="replace")
            if a in targets and "2018" not in out[a]:
                out[a]["2018"] = amazon_paths("2018", line); found += 1
    summary = {"2018": {"items_found": found, "file": path.name}}
    print(summary, flush=True)
    return out, summary


def scan_movielens(dataset):
    """MovieLens-1M genres (flat, multi-label; movies.dat is Latin-1)."""
    targets = item_ids(dataset)
    with zipfile.ZipFile(DATASET_DIR / "raw" / dataset / "ml-1m.zip") as z:
        lines = z.read("ml-1m/movies.dat").decode("latin-1").splitlines()
    out = {}
    for line in lines:
        if line.strip():
            mid, _, genres = line.split("::")
            if mid in targets:
                out[mid] = {"genres": [[g] for g in genres.split("|") if g]}
    summary = {"genres": {"items_found": len(out)}}
    print(summary, flush=True)
    return out, summary


def scan_amazon(dataset):
    targets = item_ids(dataset)
    out = {a: {} for a in targets}
    summary = {}
    for release, rel in AMAZON_FILES.items():
        t0 = time.time(); matched = dup = bad = 0
        with open(metadata_source(rel), "rb") as stream:
            for line in stream:
                m = AMAZON_ID[release].search(line)
                if m is None:
                    continue
                a = m.group(1).decode("utf-8", errors="replace")
                if a not in targets:
                    continue
                try:
                    paths = amazon_paths(release, line)
                except (SyntaxError, ValueError, TypeError):
                    bad += 1; continue
                if release in out[a]:
                    dup += 1
                    if sum(map(len, paths)) <= sum(map(len, out[a][release])):
                        continue
                out[a][release] = paths; matched += 1
        summary[release] = {"items_found": sum(1 for v in out.values() if release in v), "duplicate_rows": dup,
                            "unparsable_rows": bad, "seconds": round(time.time() - t0, 1)}
        print(release, summary[release], flush=True)
    return out, summary


def scan_yelp(dataset):
    targets = item_ids(dataset)
    out = {b: {} for b in targets}
    summary = {}
    for release, rel in YELP_FILES.items():
        n = 0
        with open(metadata_source(rel), encoding="utf-8", errors="replace", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            item_col = next(c for c in reader.fieldnames if c.split(":")[0] == "business_id" or c == "ITEM")
            cat_col = next(c for c in reader.fieldnames if c.split(":")[0].lower() == "categories")
            for row in reader:
                b = (row.get(item_col) or "").strip()
                if b in targets:
                    cats = [c for c in (clean_text(x) for x in (row.get(cat_col) or "").replace("\\/", "/").split(","))
                            if c]
                    out[b][release] = dedup_paths([[c] for c in cats]); n += 1
        summary[release] = {"items_found": n, "columns": [item_col, cat_col]}
        print(release, summary[release], flush=True)
    return out, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    scanners = {"amazon-book": scan_amazon, "yelp2018": scan_yelp, "amazon2018-grocery": scan_amazon2018,
                "ml-1m": scan_movielens}
    parser.add_argument("--dataset", required=True, choices=tuple(scanners))
    args = parser.parse_args(argv)
    out, summary = scanners[args.dataset](args.dataset)
    target = METADATA_DIR / args.dataset
    target.mkdir(parents=True, exist_ok=True)
    with open(target / "paths_by_release.json", "w", encoding="utf-8", newline="\n") as stream:
        json.dump(out, stream, ensure_ascii=False, sort_keys=True)
    with open(target / "scan_summary.json", "w", encoding="utf-8", newline="\n") as stream:
        json.dump({"dataset": args.dataset, "fields_read": "category fields only", "releases": summary},
                  stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
