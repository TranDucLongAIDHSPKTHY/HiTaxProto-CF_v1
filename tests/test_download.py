"""Download tool, offline: hash check, gunzip and zip-member unpacking, files with a wrong hash are never used."""
import gzip
import hashlib
import zipfile

import tools.data.download_data as dl
from tools.data import sources


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def test_unpack_kinds(tmp_path):
    body = b"line 1\nline 2\n"
    (tmp_path / "a.gz").write_bytes(gzip.compress(body))
    dl.unpack(tmp_path / "a.gz", "gunzip", tmp_path / "a.part")
    assert (tmp_path / "a.part").read_bytes() == body and not (tmp_path / "a.gz").exists()
    with zipfile.ZipFile(tmp_path / "b.zip", "w") as z:
        z.writestr("Yelp2018/Yelp2018.item", body); z.writestr("Yelp2018/Yelp2018.inter", b"x")
    dl.unpack(tmp_path / "b.zip", "zip:Yelp2018.item", tmp_path / "b.part")
    assert (tmp_path / "b.part").read_bytes() == body


def test_download_verifies_the_hash(tmp_path, monkeypatch):
    good, bad = b"right bytes", b"other bytes"
    monkeypatch.setattr(dl, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dl, "taxprocl_file", lambda rel: None)
    monkeypatch.setattr(dl, "SOURCES", {"toy": [dict(url="https://example.invalid/f", path="dataset/raw/toy/f.bin",
                                                     kind="file", sha256=_sha(good))]})
    payload = {"data": bad}
    monkeypatch.setattr(dl, "fetch", lambda url, target, context: target.write_bytes(payload["data"]))
    assert dl.main([]) == 1                                         # wrong bytes: refused, kept as .part only
    assert not (tmp_path / "dataset/raw/toy/f.bin").exists() and (tmp_path / "dataset/raw/toy/f.bin.part").exists()
    payload["data"] = good
    assert dl.main([]) == 0 and (tmp_path / "dataset/raw/toy/f.bin").read_bytes() == good
    calls = []
    monkeypatch.setattr(dl, "fetch", lambda *a: calls.append(a))
    assert dl.main([]) == 0 and not calls                           # present with the right hash: skipped
    assert dl.main(["--verify_only"]) == 0


def test_every_source_has_a_url_a_path_and_a_sha256():
    paths = [s["path"] for items in sources.SOURCES.values() for s in items]
    assert len(paths) == len(set(paths))
    for items in sources.SOURCES.values():
        for s in items:
            assert s["url"].startswith("https://") and len(s["sha256"]) == 64 and s["kind"].split(":")[0] in (
                "file", "gunzip", "zip")
