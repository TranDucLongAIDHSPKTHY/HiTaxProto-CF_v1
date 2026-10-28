"""The test split is sealed: no read without a frozen protocol whose hash matches manifest and file."""
import hashlib
import json

import pytest

from utility.utility_data import data_loader
from utility.utility_data.data_loader import ContractData, SealedTestError


@pytest.fixture()
def contract(tmp_path, monkeypatch):
    d = tmp_path / "tiny" / "2026"
    (d / "sealed_test").mkdir(parents=True)
    (d / "interactions.csv").write_text("user_id,item_id,split\n0,0,train\n0,1,val\n1,1,train\n1,0,val\n",
                                        encoding="utf-8")
    test = d / "sealed_test" / "test_interactions.csv"
    test.write_text("user_id,item_id,split\n0,2,test\n1,2,test\n", encoding="utf-8")
    sha = hashlib.sha256(test.read_bytes()).hexdigest()
    manifest = {"counts": {"users": 2, "items": 3},
                "sealed_test": {"file": "sealed_test/test_interactions.csv", "sha256": sha}}
    (d / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(data_loader, "contract_dir", lambda ds, s: tmp_path / ds / str(s))
    return d, sha


def _protocol(path, sha, frozen=True, key="tiny/2026"):
    path.write_text(json.dumps({"frozen": frozen, "datasets": {key: {"sealed_test_sha256": sha}}}), encoding="utf-8")
    return path


def test_training_load_has_no_test_rows(contract):
    data = ContractData("tiny", 2026)
    assert set(data.edges) == {"train", "val"}


def test_test_split_refused_without_protocol(contract):
    with pytest.raises(SealedTestError):
        ContractData("tiny", 2026, splits=("train", "val", "test"))


@pytest.mark.parametrize("frozen, key, good_hash", [(False, "tiny/2026", True), (True, "other/2026", True),
                                                    (True, "tiny/2026", False)])
def test_test_split_refused_with_invalid_protocol(contract, tmp_path, frozen, key, good_hash):
    _, sha = contract
    proto = _protocol(tmp_path / "p.json", sha if good_hash else "0" * 64, frozen, key)
    with pytest.raises(SealedTestError):
        ContractData("tiny", 2026, splits=("train", "val", "test"), protocol=proto)


def test_test_split_read_with_frozen_protocol(contract, tmp_path):
    _, sha = contract
    data = ContractData("tiny", 2026, splits=("train", "val", "test"), protocol=_protocol(tmp_path / "p.json", sha))
    assert data.by_user("test") == {0: [2], 1: [2]}


def test_tampered_sealed_file_is_refused(contract, tmp_path):
    d, sha = contract
    (d / "sealed_test" / "test_interactions.csv").write_text("user_id,item_id,split\n0,1,test\n", encoding="utf-8")
    with pytest.raises(SealedTestError):
        ContractData("tiny", 2026, splits=("test",), protocol=_protocol(tmp_path / "p.json", sha))


def test_contract_with_test_rows_in_interactions_is_refused(contract):
    d, _ = contract
    with open(d / "interactions.csv", "a", encoding="utf-8") as f:
        f.write("0,2,test\n")
    with pytest.raises(SealedTestError):
        ContractData("tiny", 2026)
