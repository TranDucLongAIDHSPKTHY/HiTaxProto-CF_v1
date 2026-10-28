"""Gate values of a trained gated model at its selected checkpoint, by item degree band (does the gate separate sparse
from warm items, or is it saturated?). Recomputed from best_validation_model.pt; no test data is read.

Writes <run>/seed<seed>/gates.npz (item ids, train degree, gate) and prints mean / quartiles per band.

    python -m tools.analysis.export_gates log/pilot/yelp2018/DegreeGate/taxonomy_binit0 --seeds 42 0 1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import PROJECT_ROOT  # noqa: E402
from tools.protocol.eval_test import load_selected  # noqa: E402
from utility.utility_data.data_loader import ContractData  # noqa: E402

BANDS = (("near_cold", 1, 5), ("mid_tail", 6, 10), ("11-20", 11, 20), ("21-100", 21, 100), ("> 100", 101, 10**12))


def gates_of(run_dir, device):
    config = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8"))
    data = ContractData(config["dataset"], config["split_seed"])
    model, _ = load_selected(run_dir, data, device)
    with torch.no_grad():
        _, _, extra = model.representations()
    if extra is None:
        raise SystemExit("{} has no gate".format(run_dir))
    g, act = extra
    items = act.nonzero().flatten().cpu().numpy()
    deg = data.train_degree.numpy()[items]
    return items, deg, g.detach().cpu().numpy()


def summary(deg, gate):
    out = {}
    for name, lo, hi in BANDS:
        m = (deg >= lo) & (deg <= hi)
        if m.any():
            q = np.percentile(gate[m], [25, 50, 75])
            out[name] = {"items": int(m.sum()), "mean": float(gate[m].mean()), "q25": float(q[0]), "median": float(q[1]),
                         "q75": float(q[2])}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    torch.set_default_dtype(torch.float32)
    report = {}
    for s in a.seeds:
        d = PROJECT_ROOT / a.run / "seed{}".format(s)
        items, deg, gate = gates_of(d, torch.device(a.device))
        np.savez(d / "gates.npz", items=items, train_degree=deg, gate=gate)
        report[str(s)] = summary(deg, gate)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
