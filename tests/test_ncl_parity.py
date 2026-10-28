"""The RecBole export of the NCL parity check never carries test rows: its "test" file is a copy of validation."""
import json

from config_path import config_path
from tools.protocol import ncl_parity


def test_export_has_no_test_rows(tmp_path, monkeypatch):
    d = tmp_path / "tiny" / "2026"
    (d / "sealed_test").mkdir(parents=True)
    (d / "interactions.csv").write_text("user_id,item_id,split\n0,0,train\n0,1,val\n1,1,train\n1,0,val\n",
                                        encoding="utf-8")
    (d / "sealed_test" / "test_interactions.csv").write_text("user_id,item_id,split\n0,2,test\n1,2,test\n",
                                                             encoding="utf-8")
    (d / "split_manifest.json").write_text(json.dumps({"counts": {"users": 2, "items": 3}}), encoding="utf-8")
    monkeypatch.setattr(config_path, "contract_dir", lambda ds, s: tmp_path / ds / str(s))
    monkeypatch.setattr(ncl_parity, "OUT", tmp_path / "out")
    ncl_parity.export("tiny", 2026)
    out = tmp_path / "out" / "recbole_data" / "tiny_p1"
    read = lambda part: (out / "tiny_p1.{}.inter".format(part)).read_text(encoding="utf-8").splitlines()
    assert read("train") == ["user_id:token\titem_id:token", "0\t0", "1\t1"]
    assert read("valid") == ["user_id:token\titem_id:token", "0\t1", "1\t0"]
    assert read("test") == read("valid")
    assert (out / "tiny_p1.item").read_text(encoding="utf-8").splitlines() == ["item_id:token", "0", "1", "2"]
