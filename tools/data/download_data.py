"""Download the raw sources of the data contracts (tools/data/sources.py) and verify their SHA-256.

Each file is downloaded to <path>.part, decompressed or extracted when needed, hashed, and moved into place only if
its SHA-256 equals the recorded one; a file already present with the right hash is skipped. Downloads are large
(Amazon Books metadata: about 7 GB compressed, 21 GB stored).

    python -m tools.data.download_data --list
    python -m tools.data.download_data --datasets yelp2018 ml-1m
    python -m tools.data.download_data --verify_only

TLS certificates are always verified. If the trust store of the machine lacks the root of a host (e.g. Windows without
the emSign roots used by files.grouplens.org), add a standard root bundle with --ca_bundle, e.g. the Mozilla bundle
shipped with Git for Windows: --ca_bundle "C:/Program Files/Git/mingw64/etc/ssl/certs/ca-bundle.crt". If a host
still cannot be reached, the tool prints the URL and the expected path: download the file there by any means and run
--verify_only, which accepts it only if its SHA-256 equals the recorded one.
"""
import argparse
import gzip
import hashlib
import os
import shutil
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import PROJECT_ROOT, taxprocl_file  # noqa: E402
from tools.data.sources import SOURCES  # noqa: E402

CHUNK = 1 << 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, target, context):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (data download for research)"})
    with urllib.request.urlopen(req, timeout=120, context=context) as r, open(target, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        for block in iter(lambda: r.read(CHUNK), b""):
            f.write(block); done += len(block)
            if total and done % (256 * CHUNK) < CHUNK:
                print("    {:.0f}% of {:.0f} MB".format(100 * done / total, total / 2**20), flush=True)


def unpack(raw, kind, part):
    if kind == "file":
        os.replace(raw, part)
    elif kind == "gunzip":
        with gzip.open(raw, "rb") as src, open(part, "wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK)
        raw.unlink()
    elif kind.startswith("zip:"):
        suffix = kind[4:]
        with zipfile.ZipFile(raw) as z:
            member = next(n for n in z.namelist() if n.endswith(suffix))
            with z.open(member) as src, open(part, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
        raw.unlink()
    else:
        raise ValueError(kind)


def accepted(path, src):
    """The file matches the recorded SHA-256 or, for the Yelp category page, the recorded parsed hierarchy."""
    if sha256_file(path) == src["sha256"]:
        return True
    if src.get("hierarchy_sha256"):
        from tools.data.parse_yelp_categories import hierarchy_sha256, parse
        return hierarchy_sha256(*parse(Path(path).read_bytes().decode("utf-8"))) == src["hierarchy_sha256"]
    return False


def status(src):
    """'ok' (local file with the right hash), 'ok (TaxPro-CL)' (same file in the TaxPro-CL repository), or missing."""
    local = PROJECT_ROOT / src["path"]
    if local.is_file() and accepted(local, src):
        return "ok"
    if local.is_file():
        return "WRONG HASH"
    alt = taxprocl_copy(src["path"])
    if alt is not None and alt.is_file() and sha256_file(alt) == src["sha256"]:
        return "ok (TaxPro-CL)"
    return "missing"


def taxprocl_copy(rel):
    """Location of the same source in the TaxPro-CL repository copy (HITAX_TAXPROCL_ROOT), if one is set and has it."""
    if rel.startswith("dataset/raw/metadata/"):
        return taxprocl_file("metadata/" + rel[len("dataset/raw/metadata/"):])
    if rel.startswith(("dataset/amazon-book/", "dataset/yelp2018/")):
        return taxprocl_file(rel)
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    p.add_argument("--list", action="store_true", help="print the sources and exit")
    p.add_argument("--verify_only", action="store_true", help="check the hashes of the files present, download nothing")
    p.add_argument("--ca_bundle", default=None, help="extra trusted root certificates (PEM bundle); still verified")
    a = p.parse_args(argv)
    context = ssl.create_default_context()
    if a.ca_bundle:
        context.load_verify_locations(cafile=a.ca_bundle)
    bad = 0
    for ds in a.datasets:
        for src in SOURCES[ds]:
            if a.list:
                print("{:20s} {:55s} {}".format(ds, src["path"], src["url"]))
                continue
            st = status(src)
            if st.startswith("ok") or a.verify_only:
                print("{:20s} {:55s} {}".format(ds, src["path"], st))
                bad += not st.startswith("ok")
                continue
            target = PROJECT_ROOT / src["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            raw, part = Path(str(target) + ".download"), Path(str(target) + ".part")
            print("{:20s} downloading {}".format(ds, src["url"]), flush=True)
            try:
                fetch(src["url"], raw, context)
            except (OSError, ssl.SSLError) as e:              # URLError is an OSError
                print("    download failed ({}): save {} as {} (stored as downloaded{}) and run --verify_only".format(
                    e, src["url"], target, "" if src["kind"] == "file" else ", then " + src["kind"]))
                raw.unlink(missing_ok=True)
                bad += 1
                continue
            unpack(raw, src["kind"], part)
            if not accepted(part, src):
                print("    SHA-256 MISMATCH ({}): kept as {} for inspection, not used".format(sha256_file(part), part.name))
                bad += 1
                continue
            if sha256_file(part) != src["sha256"]:
                print("    page bytes differ from the {} copy, parsed hierarchy identical: accepted".format(
                    src.get("retrieved")))
            os.replace(part, target)
            print("{:20s} {:55s} ok".format(ds, src["path"]))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
