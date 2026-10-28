"""Pre-declared test analysis on synthetic per-user results with a known effect."""
import json

import numpy as np
import pytest

import tools.analysis.analyze_test as at

SEEDS = [42, 0, 1, 2, 3]


def _write(root, run, overall, long_tail):
    for k, s in enumerate(SEEDS):
        d = root / run / "seed{}".format(s)
        d.mkdir(parents=True)
        users = np.arange(overall.shape[1])
        np.savez(d / "test_per_user.npz", **{"overall/users": users, "overall/Recall@20": overall[k],
                                             "long_tail/users": users, "long_tail/Recall@20": long_tail[k]})
        (d / "test_evaluated.json").write_text("{}", encoding="utf-8")


@pytest.fixture()
def synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(at, "PROJECT_ROOT", tmp_path)
    rng = np.random.default_rng(0)
    base_o = rng.uniform(0, 0.3, (1, 3000)); base_l = rng.uniform(0, 0.05, (1, 3000))
    noise = lambda: rng.normal(0, 0.01, (5, 3000))
    _write(tmp_path, "log/x/ds/DegreeGate/taxonomy", base_o + noise(), base_l + 0.01 + noise())    # real effect
    _write(tmp_path, "log/x/ds/DegreeGate/shuffled", base_o + noise(), base_l + noise())
    _write(tmp_path, "log/x/ds/DegreeGate/global", base_o + noise(), base_l + 0.0 + noise())
    _write(tmp_path, "log/x/ds/LightGCN/ref", base_o + noise(), base_l - 0.005 + noise())
    proto = {"frozen": True, "analysis": {
        "endpoint": {"group": "long_tail", "metric": "Recall@20"}, "seeds": SEEDS,
        "families": {"main": [
            {"label": "taxonomy-shuffled", "dataset": "ds", "a": "log/x/ds/DegreeGate/taxonomy",
             "b": "log/x/ds/DegreeGate/shuffled"},
            {"label": "shuffled-global", "dataset": "ds", "a": "log/x/ds/DegreeGate/shuffled",
             "b": "log/x/ds/DegreeGate/global"}]},
        "conjunctions": [{"label": "both", "cells": ["taxonomy-shuffled", "shuffled-global"]}],
        "noninferiority": [{"label": "taxonomy vs ref", "arm": "log/x/ds/DegreeGate/taxonomy",
                            "reference": "log/x/ds/LightGCN/ref", "margin": 0.01}]}}
    return proto


def test_known_effect_detected_and_null_not_claimed(synthetic):
    res = at.analyse(synthetic, n_perm=20000, n_boot=500)
    cells = {r["label"]: r for r in res["families"]["main"]}
    assert cells["taxonomy-shuffled"]["claim_positive"] and abs(cells["taxonomy-shuffled"]["abs_diff"] - 0.01) < 0.002
    assert not cells["shuffled-global"]["claim_positive"]
    assert cells["taxonomy-shuffled"]["p_holm"] >= cells["taxonomy-shuffled"]["p_signflip"]
    assert not res["conjunctions"][0]["claim_positive"]            # one component is null
    assert res["noninferiority"][0]["pass"]                         # same Overall distribution
    assert "| taxonomy-shuffled | ds |" in at.markdown(res)


def test_missing_sealed_evaluation_is_refused(synthetic, tmp_path):
    (tmp_path / "log/x/ds/DegreeGate/taxonomy/seed3/test_evaluated.json").unlink()
    with pytest.raises(SystemExit):
        at.analyse(synthetic, n_perm=1000, n_boot=50)


def test_refuses_unfrozen_protocol(tmp_path):
    p = tmp_path / "draft.json"
    p.write_text(json.dumps({"frozen": False, "analysis": {}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        at.main(["--protocol", str(p)])
