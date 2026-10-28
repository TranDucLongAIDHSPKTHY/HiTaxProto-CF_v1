"""Yelp category hierarchy from the official documentation page
https://docs.developer.yelp.com/docs/resources-categories (nested HTML list "Title (alias, locales)").

The page embeds the list several times; only the first block is parsed. Nesting gives the parent of each entry;
an alias listed under several parents becomes a DAG node with several parents.

    python -m tools.data.parse_yelp_categories --html metadata/yelp2018/resources-categories.html
"""
import argparse
import collections
import hashlib
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import METADATA_DIR  # noqa: E402

URL = "https://docs.developer.yelp.com/docs/resources-categories"
ENTRY = re.compile(r"\s*(.+?)\s*\(([a-z0-9_\-]+),\s*(.*)\)\s*$")


def parse(page_text):
    t = html.unescape(html.unescape(page_text)).replace("\\n", "\n").replace('\\"', '"').replace("\\/", "/")
    a = t.find('<ul class="bullet-list-round category-list">')
    b = t.find("</body>", a)
    if a < 0 or b < 0:
        raise ValueError("category list not found")
    stack, last, parents, titles, locales = [], None, collections.defaultdict(set), {}, {}
    for m in re.finditer(r"<(/?)ul[^>]*>|<li>([^<]*)</li>", t[a:b]):
        if m.group(2) is None:
            if m.group(1):
                stack.pop()
            else:
                stack.append(last)
            continue
        mm = ENTRY.match(m.group(2))
        if not mm:
            raise ValueError("unparsed entry: {!r}".format(m.group(2)))
        title, alias, loc = mm.groups()
        titles.setdefault(alias, title); locales.setdefault(alias, loc)
        if stack[-1] is not None:
            parents[alias].add(stack[-1])
        last = alias
    if stack:
        raise ValueError("unbalanced list")
    return titles, locales, {k: sorted(v) for k, v in parents.items()}


def hierarchy_sha256(titles, locales, parents):
    """SHA-256 of the canonical JSON of the parsed hierarchy: identifies its CONTENT, independently of the page markup
    (the page itself is not versioned and its bytes change with every site build)."""
    canon = json.loads(json.dumps({"titles": titles, "locales": locales, "parents": parents},
                                  default=lambda o: sorted(o)))
    text = json.dumps(canon, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--html", type=Path, default=METADATA_DIR / "yelp2018" / "resources-categories.html")
    p.add_argument("--retrieved", default="2026-10-26")
    a = p.parse_args(argv)
    raw = a.html.read_bytes()
    titles, locales, parents = parse(raw.decode("utf-8"))
    out = {"source": URL, "retrieved": a.retrieved, "html_sha256": hashlib.sha256(raw).hexdigest(),
           "hierarchy_sha256": hierarchy_sha256(titles, locales, parents),
           "titles": titles, "locales": locales, "parents": parents}
    with open(METADATA_DIR / "yelp2018" / "yelp_categories.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("aliases", len(titles), "multi-parent", sum(len(v) > 1 for v in parents.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
