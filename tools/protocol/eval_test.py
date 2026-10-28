"""Sealed test evaluation: evaluate every run listed in the frozen protocol ONCE on the test split.

Checks before opening the test split: the protocol is frozen, the configuration hash and the checkpoint hash of each
run match the protocol, and the run has not been evaluated before (test_evaluated.json is the once-only marker).
Ranking excludes train + validation items; per-user metrics are written to test_per_user.npz for the statistics.

    python -m tools.protocol.eval_test --protocol protocol/frozen_protocol.json
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as runner  # noqa: E402
from config_path.config_path import PROJECT_ROOT  # noqa: E402
from models.HiTaxProtoCF import PrototypeMemory  # noqa: E402
from utility.utility_data.data_graph import exact_propagation  # noqa: E402
from utility.utility_data.data_loader import ContractData  # noqa: E402
from utility.utility_function.tools import config_hash  # noqa: E402
from utility.utility_train import model_factory  # noqa: E402
from utility.utility_train.group_evaluator import WITHIN_GROUPS, evaluate  # noqa: E402

GROUPS = ("overall", "near_cold", "mid_tail", "long_tail", "warm")
KEYS = GROUPS + tuple(g + "_within" for g in WITHIN_GROUPS)        # + within-group rankings (endpoint)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_selected(run_dir, data, device):
    config = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8"))
    base = {"LightGCN": "LightGCN", "BPR-MF": "BPRMF", "TF-LightGCN": "TFLightGCN", "NCL": "NCL",
            "NCL+Taxonomy": "NCL"}.get(run_dir.parent.parent.name)
    if base is None:
        base = "SimGCL" if run_dir.parent.parent.name.startswith("SimGCL") else "HiTaxProtoCF"
    seed = int(run_dir.name[4:])
    model, _, _ = model_factory.build(base, config, data, device, seed)
    state = torch.load(run_dir / "best_validation_model.pt", map_location=device)["model"]
    if model_factory.gate_type(config) != "none" and config.get("source") == "kmeans":  # memory of the selected epoch
        k = int(state["memory.node_count"].numel())
        pairs = torch.stack([state["memory.pair_item"], state["memory.pair_node"]], 1).cpu()
        model.memory = PrototypeMemory(data.n_items, k, pairs, (data.train_degree >= 1)).to(device)
    model.load_state_dict(state)
    return model.eval(), config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    proto = json.loads(Path(a.protocol).read_text(encoding="utf-8"))
    if proto.get("frozen") is not True:
        raise SystemExit("protocol is not frozen")
    runner.enable_determinism()
    device = torch.device(a.device)
    torch.set_default_dtype(torch.float32)
    for run, seeds in proto["runs"].items():
        for s, rec in seeds.items():
            d = PROJECT_ROOT / run / "seed{}".format(s)
            if (d / "test_evaluated.json").exists():
                raise SystemExit("{} was already evaluated on test: the test split is opened once".format(d))
            config = json.loads((d / "config_resolved.json").read_text(encoding="utf-8"))
            if config_hash(config) != rec["config_hash"] or sha(d / "best_validation_model.pt") != rec["checkpoint_sha256"]:
                raise SystemExit("{} differs from the frozen protocol".format(d))
            data = ContractData(config["dataset"], config["split_seed"], splits=("train", "val", "test"),
                                protocol=a.protocol)
            model, _ = load_selected(d, data, device)
            with torch.no_grad(), exact_propagation(model):   # bitwise reproducible from the checkpoint
                eu, ei, _ = model.representations()
            res = evaluate(eu, ei, data.seen_for("test"), data.by_user("test"), data.train_degree,
                           ks=tuple(config["ks"]), groups=GROUPS, within=WITHIN_GROUPS)
            runner.save_per_user(d / "test_per_user.npz", res)
            summary = {g: {**res[g]["mean"], "n_users": res[g]["n_users"]} for g in KEYS}
            with open(d / "test_evaluated.json", "w", encoding="utf-8", newline="\n") as f:
                json.dump({"evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "protocol": a.protocol,
                           "metrics": summary}, f, indent=1); f.write("\n")
            print(d, {g: round(summary[g]["Recall@20"], 6) for g in KEYS}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
