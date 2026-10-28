"""Every model of the study builds from its configuration and trains end to end on a tiny contract."""
import json

import numpy as np
import pytest

import main
from tests.test_main import tiny  # noqa: F401  (fixture)
from utility.utility_data.controls import shuffle_labels
from utility.utility_train import model_factory


@pytest.fixture()
def tiny_tax(tiny):
    d = tiny / "dv" / "tiny" / "2026"
    rng = np.random.default_rng(1)
    labels = {i: sorted(set(rng.choice(["n1", "n2", "n3", "n4"], size=rng.integers(1, 3)).tolist())) for i in range(40)}
    node = {"ROOT": 0, "n1": 1, "n2": 2, "n3": 3, "n4": 4}
    with open(d / "taxonomy_nodes.csv", "w", encoding="utf-8", newline="") as f:
        f.write("node_id,name,source,is_super_root\n")
        f.writelines("%d,%s,tiny,%s\n" % (k, n, n == "ROOT") for n, k in node.items())

    def write(name, lab):
        with open(d / name, "w", encoding="utf-8", newline="") as f:
            f.write("item_id,node_id\n")
            f.writelines("%d,%d\n" % (i, node[n]) for i in sorted(lab) for n in lab[i])
    write("item_labels.csv", labels)
    deg = {}
    for line in (d / "interactions.csv").read_text(encoding="utf-8").splitlines()[1:]:
        u, i, s = line.split(",")
        if s == "train":
            deg[int(i)] = deg.get(int(i), 0) + 1
    deg = {i: deg.get(i, 0) for i in range(40)}
    for k, seed in enumerate((101, 102, 103, 104, 105)):
        write("shuffled_labels_s%d.csv" % seed, shuffle_labels(labels, deg, seed))
    return tiny


def _run(root, model, *extra, epochs="2"):
    argv = ["--model", model, "--dataset", "tiny", "--device", "cpu", "--dim", "8", "--batch_size", "16",
            "--epochs", epochs, "--study", "dev", *extra]
    if model != "BPRMF":
        argv += ["--K", "2"]
    return main.main(argv)


def _manifest(root, name, setting):
    return json.loads((root / "log" / "dev" / "tiny" / name / setting / "seed42" / "run_manifest.json")
                      .read_text(encoding="utf-8"))


@pytest.mark.parametrize("gate, source, name", [
    ("degree", "taxonomy", "DegreeGate"), ("degree", "shuffled", "DegreeGate"), ("degree", "global", "DegreeGate"),
    ("degree", "kmeans", "DegreeGate"), ("global", "taxonomy", "GlobalGate"),
    ("degree_content", "taxonomy", "DegreeContentGate")])
def test_gated_models_train_and_share_the_fuse_mask(tiny_tax, gate, source, name):
    assert _run(tiny_tax, "HiTaxProtoCF", "--gate", gate, "--source", source, "--reference_overall", "0.01") == 0
    m = _manifest(tiny_tax, name, "{}_binit0".format(source))
    assert m["status"] == "completed" and m["fuse_mask_size"] > 0


def test_fuse_mask_identical_across_sources(tiny_tax):
    sizes = set()
    for source in ("taxonomy", "shuffled", "global", "kmeans"):
        assert _run(tiny_tax, "HiTaxProtoCF", "--source", source, "--reference_overall", "0.01") == 0
        sizes.add(_manifest(tiny_tax, "DegreeGate", "{}_binit0".format(source))["fuse_mask_size"])
    assert len(sizes) == 1


def test_self_inclusive_setting_and_shuffle_seed_pairing(tiny_tax):
    assert _run(tiny_tax, "HiTaxProtoCF", "--leave_one_out", "false", "--reference_overall", "0.01") == 0
    assert _manifest(tiny_tax, "DegreeGate", "taxonomy_binit0_self-inclusive")["status"] == "completed"
    assert _run(tiny_tax, "HiTaxProtoCF", "--source", "shuffled", "--reference_overall", "0.01") == 0
    assert _manifest(tiny_tax, "DegreeGate", "shuffled_binit0")["shuffle_seed"] == 101    # seed 42 -> 101
    assert model_factory.shuffle_seed_for({"shuffle_seed": None}, 3) == 105
    with pytest.raises(ValueError):
        model_factory.shuffle_seed_for({"shuffle_seed": None}, 7)


@pytest.mark.parametrize("model, extra, name, setting", [
    ("BPRMF", (), "BPR-MF", "lr0.001_reg0.0001"),
    ("TFLightGCN", (), "TF-LightGCN", "lr0.001_reg0.0001"),
    ("NCL", ("--warm_up", "0"), "NCL", "lamp1e-07_tau0.05"),
    ("NCL", ("--warm_up", "0", "--item_groups", "taxonomy"), "NCL+Taxonomy", "lamp1e-07_tau0.05"),
    ("SimGCL", (), "SimGCL", "eps0.1_tau0.2"),
    ("SimGCL", ("--gate", "degree"), "SimGCL+DegreeGate", "taxonomy_binit0_eps0.1_tau0.2")])
def test_comparators_train(tiny_tax, model, extra, name, setting):
    assert _run(tiny_tax, model, *extra, "--reference_overall", "0.01") == 0
    assert _manifest(tiny_tax, name, setting)["status"] == "completed"


def test_non_reference_model_requires_reference_overall(tiny_tax):
    with pytest.raises(ValueError, match="reference_overall"):    # refused before any run folder is created
        _run(tiny_tax, "HiTaxProtoCF")
    assert not (tiny_tax / "log" / "dev" / "tiny" / "DegreeGate").exists()
