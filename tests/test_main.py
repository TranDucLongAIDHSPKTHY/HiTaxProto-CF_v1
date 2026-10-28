"""End-to-end runner on a tiny synthetic contract: output layout, log structure, no test access, exact resume."""
import hashlib
import json

import numpy as np
import pytest
import torch

import main
from utility.utility_data import data_loader
from utility.utility_data.split import split_all


@pytest.fixture()
def tiny(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    users = {u: sorted(set(rng.choice(40, size=rng.integers(4, 12), replace=False).tolist())) for u in range(30)}
    rows, _ = split_all(users, 2026)
    d = tmp_path / "dv" / "tiny" / "2026"
    d.mkdir(parents=True)
    with open(d / "interactions.csv", "w", encoding="utf-8", newline="") as f:
        f.write("user_id,item_id,split\n")
        f.writelines("%d,%d,%s\n" % r for r in sorted(rows) if r[2] != "test")
    (d / "sealed_test").mkdir()
    with open(d / "sealed_test" / "test_interactions.csv", "w", encoding="utf-8", newline="") as f:
        f.write("user_id,item_id,split\n")
        f.writelines("%d,%d,%s\n" % r for r in sorted(rows) if r[2] == "test")
    manifest = {"counts": {"users": 30, "items": 40},
                "sha256": {"interactions.csv": hashlib.sha256((d / "interactions.csv").read_bytes()).hexdigest()},
                "sealed_test": {"file": "sealed_test/test_interactions.csv",
                                "sha256": hashlib.sha256((d / "sealed_test" / "test_interactions.csv").read_bytes()).hexdigest()}}
    (d / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(data_loader, "contract_dir", lambda ds, s: tmp_path / "dv" / ds / str(s))
    monkeypatch.setattr(main, "run_dir",
                        lambda st, ds, m, se, s: tmp_path / "log" / st / ds / m / se / ("seed%d" % s))
    yield tmp_path
    torch.set_default_dtype(torch.float64)            # main.py switches to float32 for real runs


def _argv(*extra):
    return ["--model", "LightGCN", "--dataset", "tiny", "--device", "cpu", "--K", "2", "--dim", "8",
            "--batch_size", "16", "--patience", "50", *extra]


def _out(root, study, setting="lr0.001_reg0.0001"):
    return root / "log" / study / "tiny" / "LightGCN" / setting / "seed42"


def test_runner_layout_log_structure_and_no_test_access(tiny, monkeypatch):
    seen_splits = []
    orig = data_loader.ContractData.__init__

    def spy(self, dataset, split_seed, splits=("train", "val")):
        seen_splits.append(tuple(splits)); orig(self, dataset, split_seed, splits)
    monkeypatch.setattr(data_loader.ContractData, "__init__", spy)
    assert main.main(_argv("--epochs", "3", "--study", "dev")) == 0
    out = _out(tiny, "dev")
    m = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["status"] == "completed" and m["hit_cap"] and m["completed_epochs"] == 3 and m["model"] == "LightGCN"
    assert len(json.loads((out / "validation_metrics.json").read_text(encoding="utf-8"))) == 3
    assert (out / "best_validation_model.pt").is_file() and (out / "last_model.pt").is_file()
    assert all("test" not in s for s in seen_splits)
    log = (out / "training.log").read_text(encoding="utf-8")
    for line in ("Start Training: model=LightGCN dataset=tiny seed=42", "CLI configuration override: K=2 (default=3)",
                 "Random seed: 42", "Dataset statistics:", "Model Initialized: LightGCN", "Configuration lr=0.001",
                 "Model parameters:", "Optimizer Initialized: Adam", "Start Epoch: 1", "Train Loss: epoch=1",
                 "End Epoch: 1 training_time=", "Best Validation Updated: epoch=1", "Saving Checkpoint: last_model=",
                 "Validation Metrics: epoch=1", "Validation Groups: epoch=1", "Epoch Progress: 1/3",
                 "Training loop completed:", "Test evaluation deferred", "Training Finished", "Seed 42 Training Time:"):
        assert line in log, line
    assert " | INFO | trainer | Start Epoch: 1" in log


def test_setting_folder_refuses_a_different_configuration(tiny):
    assert main.main(_argv("--epochs", "1", "--study", "dev")) == 0
    with pytest.raises(FileExistsError):
        main.main(_argv("--epochs", "2", "--study", "dev"))


def test_resume_refuses_a_changed_configuration(tiny):
    assert main.main(_argv("--epochs", "2", "--study", "dev", "--setting", "s")) == 0
    with pytest.raises(FileExistsError):
        main.main(_argv("--epochs", "4", "--study", "dev", "--setting", "s", "--resume"))


def test_resume_continues_exactly(tiny):
    assert main.main(_argv("--epochs", "4", "--study", "dev", "--setting", "full")) == 0
    full, part, mid = (_out(tiny, "dev", s) for s in ("full", "part", "mid"))
    assert main.main(_argv("--epochs", "2", "--study", "dev", "--setting", "mid")) == 0   # interrupted after 2 epochs
    part.mkdir(parents=True)
    ref_hash = json.loads((full / "run_manifest.json").read_text(encoding="utf-8"))["config_hash"]
    ck = torch.load(mid / "last_model.pt"); ck["config_hash"] = ref_hash      # the cap is the only difference
    torch.save(ck, part / "last_model.pt")
    hist = json.loads((mid / "validation_metrics.json").read_text(encoding="utf-8"))
    (part / "validation_metrics.json").write_text(json.dumps(hist), encoding="utf-8")
    assert main.main(_argv("--epochs", "4", "--study", "dev", "--setting", "part", "--resume")) == 0
    a = torch.load(full / "best_validation_model.pt")["model"]; b = torch.load(part / "best_validation_model.pt")["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    h_full = json.loads((full / "validation_metrics.json").read_text(encoding="utf-8"))
    h_part = json.loads((part / "validation_metrics.json").read_text(encoding="utf-8"))
    assert [r["metrics"] for r in h_full] == [r["metrics"] for r in h_part]


def test_confirmation_dataset_runs_only_under_a_frozen_protocol(tiny):
    mpath = tiny / "dv" / "tiny" / "2026" / "split_manifest.json"
    m = json.loads(mpath.read_text(encoding="utf-8")); m["role"] = "confirmation"
    mpath.write_text(json.dumps(m), encoding="utf-8")
    assert main.main(_argv("--epochs", "1", "--study", "dev", "--setting", "a")) == 1
    draft = tiny / "draft.json"
    draft.write_text(json.dumps({"frozen": False, "datasets": {"tiny/2026": {}}}), encoding="utf-8")
    assert main.main(_argv("--epochs", "1", "--study", "dev", "--setting", "b", "--protocol", str(draft))) == 1
    frozen = tiny / "frozen.json"
    frozen.write_text(json.dumps({"frozen": True, "datasets": {"tiny/2026": {}}}), encoding="utf-8")
    assert main.main(_argv("--epochs", "1", "--study", "dev", "--setting", "c", "--protocol", str(frozen))) == 0


def test_pilot_halves_are_disjoint_and_assessment_is_written_once(tiny):
    assert main.main(_argv("--epochs", "3", "--study", "pilot", "--validation_users", "selection")) == 0
    out = _out(tiny, "pilot")
    m = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["validation_users"] == "selection" and "assessment" in m and "selected_validation" in m
    sel = np.load(out / "validation_per_user.npz")["overall/users"]
    ass = np.load(out / "assessment_per_user.npz")["overall/users"]
    val_users = {int(line.split(",")[0]) for line in (tiny / "dv" / "tiny" / "2026" / "interactions.csv")
                 .read_text(encoding="utf-8").splitlines()[1:] if line.endswith(",val")}
    assert not (set(sel) & set(ass)) and set(sel) | set(ass) == val_users
    hist = json.loads((out / "validation_metrics.json").read_text(encoding="utf-8"))
    assert all(r["metrics"]["overall"]["n_users"] == len(sel) for r in hist)     # selection half only
    log = (out / "training.log").read_text(encoding="utf-8")
    assert log.count("Assessment half (evaluated once)") == 1
    z = np.load(out / "validation_per_user.npz")             # manifest = exact evaluation of the selected checkpoint
    for g in ("overall", "long_tail_within"):
        if len(z[g + "/users"]):
            assert m["selected_validation"][g]["Recall@20"] == float(z[g + "/Recall@20"].mean())
    assert "selected_validation_training" in m
