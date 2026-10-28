"""End-to-end protocol on a tiny contract: pilot -> selection -> pilot decision -> freeze -> sealed test once."""
import json

import pytest

import main
import tools.experiments.pilot_decision as pilot
import tools.experiments.select_config as select
import tools.protocol.eval_test as eval_test
import tools.protocol.freeze_protocol as freeze
from tests.test_model_factory import tiny_tax  # noqa: F401  (fixture)
from tests.test_main import tiny  # noqa: F401  (fixture)


@pytest.fixture()
def wired(tiny_tax, monkeypatch):
    root = tiny_tax
    for mod in (select, pilot):
        monkeypatch.setattr(mod, "MODEL_OUTPUT_DIR", root / "log")
        monkeypatch.setattr(mod, "RESULT_DIR", root / "results")
    monkeypatch.setattr(freeze, "PROJECT_ROOT", root)
    monkeypatch.setattr(freeze, "DATASET_VERIFY_DIR", root / "dv")
    monkeypatch.setattr(freeze, "DRAFT", root / "protocol" / "protocol_draft.json")
    monkeypatch.setattr(freeze, "FROZEN", root / "protocol" / "frozen_protocol.json")
    monkeypatch.setattr(eval_test, "PROJECT_ROOT", root)
    return root


def _train(model, *extra):
    argv = ["--model", model, "--dataset", "tiny", "--device", "cpu", "--K", "2", "--dim", "8", "--batch_size", "16",
            "--epochs", "3", "--study", "pilot", "--validation_users", "selection", "--seeds", "42", "0", "1", *extra]
    assert main.main(argv) == 0


def test_full_protocol(wired, capsys):
    root = wired
    _train("LightGCN")
    assert select.main(["--study", "pilot", "--dataset", "tiny", "--model", "LightGCN", "--seeds", "42", "0", "1"]) == 0
    ref = json.loads((root / "results" / "selection" / "pilot__tiny__LightGCN.json").read_text(encoding="utf-8"))
    ref_overall = str(ref["chosen_metrics"]["val_Overall"])
    for source in ("taxonomy", "shuffled"):
        _train("HiTaxProtoCF", "--source", source, "--reference_overall", ref_overall)
        assert select.main(["--study", "pilot", "--dataset", "tiny", "--model", "DegreeGate", "--seeds", "42", "0", "1",
                            "--setting_prefix", source + "_", "--reference_overall", ref_overall]) == 0
    assert pilot.main(["--dataset", "tiny", "--n_boot", "200"]) == 0
    decision = json.loads((root / "results" / "pilot" / "tiny.json").read_text(encoding="utf-8"))
    assert set(decision["checks"]) == {"relative_difference_>=_1%", "ci_excludes_0", "all_seeds_positive",
                                       "overall_>=_0.99_reference"} and isinstance(decision["pass"], bool)
    # freeze: the protocol lists the reference run; then the test split is opened exactly once
    (root / "protocol").mkdir(exist_ok=True)
    run = "log/pilot/tiny/LightGCN/" + ref["chosen"]
    (root / "protocol" / "protocol_draft.json").write_text(json.dumps(
        {"endpoint": "test long-tail Recall@20", "datasets": ["tiny/2026"], "seeds": [42, 0, 1], "runs": [run]}),
        encoding="utf-8")
    assert freeze.main() == 0
    with pytest.raises(SystemExit):
        freeze.main()                                            # never overwritten
    proto = str(root / "protocol" / "frozen_protocol.json")
    assert eval_test.main(["--protocol", proto, "--device", "cpu"]) == 0
    for s in (42, 0, 1):
        d = root / run / "seed{}".format(s)
        assert (d / "test_per_user.npz").is_file() and (d / "test_evaluated.json").is_file()
    with pytest.raises(SystemExit):
        eval_test.main(["--protocol", proto, "--device", "cpu"])   # opened once only


def test_eval_test_refuses_unfrozen_or_tampered(wired):
    root = wired
    _train("LightGCN")
    run = "log/pilot/tiny/LightGCN/lr0.001_reg0.0001"
    (root / "protocol").mkdir(exist_ok=True)
    (root / "protocol" / "protocol_draft.json").write_text(json.dumps(
        {"datasets": ["tiny/2026"], "seeds": [42, 0, 1], "runs": [run]}), encoding="utf-8")
    draft = str(root / "protocol" / "protocol_draft.json")
    with pytest.raises(SystemExit):
        eval_test.main(["--protocol", draft, "--device", "cpu"])   # not frozen
    assert freeze.main() == 0
    ck = root / run / "seed0" / "best_validation_model.pt"
    ck.write_bytes(ck.read_bytes() + b"x")                         # tampered checkpoint
    with pytest.raises(SystemExit):
        eval_test.main(["--protocol", str(root / "protocol" / "frozen_protocol.json"), "--device", "cpu"])


def test_eval_test_rebuilds_the_kmeans_memory_of_the_selected_epoch(wired):
    root = wired
    _train("HiTaxProtoCF", "--source", "kmeans", "--reference_overall", "0.01")
    run = "log/pilot/tiny/DegreeGate/kmeans_binit0"
    (root / "protocol").mkdir(exist_ok=True)
    (root / "protocol" / "protocol_draft.json").write_text(json.dumps(
        {"datasets": ["tiny/2026"], "seeds": [42, 0, 1], "runs": [run]}), encoding="utf-8")
    assert freeze.main() == 0
    assert eval_test.main(["--protocol", str(root / "protocol" / "frozen_protocol.json"), "--device", "cpu"]) == 0
    assert (root / run / "seed42" / "test_evaluated.json").is_file()


def test_reporting_tools_on_pilot_runs(wired, monkeypatch):
    import tools.analysis.collect_results as cr
    import tools.analysis.export_gates as eg
    root = wired
    _train("HiTaxProtoCF", "--source", "taxonomy", "--reference_overall", "0.01")
    rows = list(cr.rows(root / "log"))
    assert len(rows) == 3 and all(r["status"] == "completed" and r["assessment_overall_Recall@20"] != "" for r in rows)
    monkeypatch.setattr(eg, "PROJECT_ROOT", root)
    assert eg.main(["log/pilot/tiny/DegreeGate/taxonomy_binit0", "--seeds", "42"]) == 0
    z = __import__("numpy").load(root / "log/pilot/tiny/DegreeGate/taxonomy_binit0/seed42/gates.npz")
    assert len(z["gate"]) == len(z["items"]) > 0 and ((z["gate"] > 0) & (z["gate"] < 1)).all()


def test_reference_threshold_is_recorded_and_enforced(wired):
    root = wired
    _train("HiTaxProtoCF", "--source", "taxonomy", "--reference_overall", "0.01")
    m = json.loads((root / "log/pilot/tiny/DegreeGate/taxonomy_binit0/seed42/run_manifest.json").read_text(encoding="utf-8"))
    assert m["reference_overall"] == 0.01 and m["validation_users"] == "selection"
    with pytest.raises(FileExistsError):                          # a completed run is never reused with another threshold
        _train("HiTaxProtoCF", "--source", "taxonomy", "--reference_overall", "0.02")
    _train("LightGCN")
    assert select.main(["--study", "pilot", "--dataset", "tiny", "--model", "LightGCN"]) == 0      # pilot seeds by default
    with pytest.raises(SystemExit):                               # runs trained with 0.01, reference report differs
        select.main(["--study", "pilot", "--dataset", "tiny", "--model", "DegreeGate", "--setting_prefix", "taxonomy_"])
    with pytest.raises(SystemExit):                               # explicit value must agree with the reference report
        select.main(["--study", "pilot", "--dataset", "tiny", "--model", "DegreeGate", "--setting_prefix", "taxonomy_",
                     "--reference_overall", "0.5"])


def test_pilot_phase2_plan_and_input_checks(wired, monkeypatch):
    import tools.experiments.make_pilot_phase2 as mp
    root = wired
    monkeypatch.setattr(mp, "MODEL_OUTPUT_DIR", root / "log")
    monkeypatch.setattr(mp, "RESULT_DIR", root / "results")
    monkeypatch.setattr(mp, "PROJECT_ROOT", root)
    (root / "plans").mkdir()
    _train("LightGCN")
    assert select.main(["--study", "pilot", "--dataset", "tiny", "--model", "LightGCN"]) == 0
    ref = json.loads((root / "results/selection/pilot__tiny__LightGCN.json").read_text(encoding="utf-8"))
    assert mp.main(["--dataset", "tiny"]) == 0
    plan = json.loads((root / "plans/pilot_tiny_phase2.json").read_text(encoding="utf-8"))
    assert len(plan["runs"]) == 12
    for args in plan["runs"]:
        assert args[args.index("--reference_selection") + 1] == "pilot__tiny__LightGCN.json"
        assert args[args.index("--seeds") + 1: args.index("--seeds") + 4] == ["42", "0", "1"]
    assert plan["reference"]["per_seed_val_Overall"] == {s: v["val_Overall"] for s, v in
                                                         ref["chosen_metrics"]["per_seed"].items()}
    assert select.main(["--study", "pilot", "--dataset", "tiny", "--model", "LightGCN", "--seeds", "42", "0"]) == 0
    with pytest.raises(SystemExit):                               # not the pilot seeds
        mp.main(["--dataset", "tiny"])
    with pytest.raises(SystemExit):
        pilot.main(["--dataset", "tiny", "--n_boot", "50"])


def test_per_seed_reference_threshold(wired, monkeypatch):
    root = wired
    monkeypatch.setattr(main, "RESULT_DIR", root / "results")
    _train("LightGCN")
    assert select.main(["--study", "pilot", "--dataset", "tiny", "--model", "LightGCN"]) == 0
    ref = json.loads((root / "results/selection/pilot__tiny__LightGCN.json").read_text(encoding="utf-8"))
    per_seed = ref["chosen_metrics"]["per_seed"]
    assert sorted(per_seed) == ["0", "1", "42"]
    assert abs(sum(v["val_Overall"] for v in per_seed.values()) / 3 - ref["chosen_metrics"]["val_Overall"]) < 1e-12
    _train("HiTaxProtoCF", "--source", "taxonomy", "--reference_selection", "pilot__tiny__LightGCN.json")
    for s in ("42", "0", "1"):                                    # each seed held to the reference run of its seed
        m = json.loads((root / "log/pilot/tiny/DegreeGate/taxonomy_binit0/seed{}/run_manifest.json".format(s))
                       .read_text(encoding="utf-8"))
        assert m["reference_mode"] == "per_seed" and m["reference_overall"] == per_seed[s]["val_Overall"]
        log = (root / "log/pilot/tiny/DegreeGate/taxonomy_binit0/seed{}/training.log".format(s)).read_text(encoding="utf-8")
        assert "reference run of the same seed" in log
    assert select.main(["--study", "pilot", "--dataset", "tiny", "--model", "DegreeGate", "--setting_prefix",
                        "taxonomy_"]) == 0
    with pytest.raises(ValueError):                               # one source of the threshold only
        _train("HiTaxProtoCF", "--source", "global", "--reference_selection", "pilot__tiny__LightGCN.json",
               "--reference_overall", "0.01")
