"""NCL parity check: the official NCL (RecBole 1.2.0, third_party/recbole_env) and the NCL of this codebase trained on
the SAME data contract (train + validation; the sealed test is never read) with the same hyper-parameters, compared on
validation Recall@20 / NDCG@20.

To isolate the implementation, "our" side runs in a matched mode: RecBole's Xavier initialisation (separate user and
item tables), RecBole's EmbLoss (sum of L2 norms / batch size, not squared), and the structure/prototype losses summed
over the actual batch (batch_ref = batch size). The only remaining difference is the K-means routine (faiss vs torch
Lloyd on train items). RecBole's "test" file is a COPY of the validation file, only to satisfy its loader (its test
numbers are meaningless and ignored); an .item file lists the whole catalogue so both sides rank the same items.

    python -m tools.protocol.ncl_parity export --dataset yelp2018            # writes the RecBole benchmark files
    third_party/recbole_env/Scripts/python.exe -m tools.protocol.ncl_parity recbole --dataset yelp2018
    python -m tools.protocol.ncl_parity ours --dataset yelp2018
    python -m tools.protocol.ncl_parity compare --dataset yelp2018
"""
import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "ncl_parity"
YELP = dict(ssl_temp=0.05, ssl_reg=1e-6, alpha=1.5, proto_reg=1e-7, num_clusters=2000)    # RUCAIBox/NCL yelp.yaml
COMMON = dict(embedding_size=64, n_layers=3, hyper_layers=1, reg_weight=1e-4, warm_up_step=20, learning_rate=1e-3,
              train_batch_size=2048, epochs=1000, stopping_step=20, seed=42)


def params(dataset, epochs=None):
    p = dict(COMMON, **YELP)
    if dataset == "ml-1m":                                        # small smoke settings
        p.update(num_clusters=100)
    if epochs:
        p.update(epochs=int(epochs))
    return p


def export(dataset, seed):
    from config_path.config_path import contract_dir
    src = Path(contract_dir(dataset, seed)) / "interactions.csv"
    d = OUT / "recbole_data" / "{}_p1".format(dataset)
    d.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "val": []}
    with open(src, encoding="utf-8") as f:
        f.readline()
        for line in f:
            u, i, s = line.rstrip("\n").split(",")
            rows[s].append((u, i))
    for part, key in (("train", "train"), ("valid", "val"), ("test", "val")):  # test = COPY of validation
        with open(d / "{}_p1.{}.inter".format(dataset, part), "w", encoding="utf-8", newline="\n") as f:
            f.write("user_id:token\titem_id:token\n")
            f.writelines("{}\t{}\n".format(u, i) for u, i in rows[key])
    n_items = json.loads((src.parent / "split_manifest.json").read_text(encoding="utf-8"))["counts"]["items"]
    with open(d / "{}_p1.item".format(dataset), "w", encoding="utf-8", newline="\n") as f:
        f.write("item_id:token\n")
        f.writelines("{}\n".format(i) for i in range(n_items))
    print("exported", d, {k: len(v) for k, v in rows.items()}, "items", n_items)


def run_recbole(dataset, device, epochs=None):
    import faiss
    from recbole.config import Config                             # not recbole.quick_start (needs ray)
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer, init_logger, init_seed
    original = faiss.Kmeans

    class CpuKmeans(original):                                    # faiss-cpu build: force gpu=False
        def __init__(self, *a, **k):
            k["gpu"] = False
            super().__init__(*a, **k)
    faiss.Kmeans = CpuKmeans
    p = params(dataset, epochs)
    cfg = dict(p, data_path=str(OUT / "recbole_data"), benchmark_filename=["train", "valid", "test"],
               load_col={"inter": ["user_id", "item_id"], "item": ["item_id"]}, user_inter_num_interval="[0,inf)",
               item_inter_num_interval="[0,inf)", eval_args={"split": None, "group_by": "user", "order": "RO",
                                                             "mode": "full"},
               train_neg_sample_args={"distribution": "uniform", "sample_num": 1}, metrics=["Recall", "NDCG"],
               topk=[10, 20], valid_metric="Recall@20", metric_decimal_place=6, eval_batch_size=4096, reproducibility=True,
               use_gpu=device == "cuda", gpu_id="0", show_progress=False, checkpoint_dir=str(OUT / "recbole_ckpt"))
    run_dir = OUT / "recbole_run"                                 # RecBole writes ./log and ./log_tensorboard
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    t0 = time.time()
    config = Config(model="NCL", dataset="{}_p1".format(dataset), config_dict=cfg)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    ds = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, ds)      # the third loader (copy of validation) is unused
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    _, best = trainer.fit(train_data, valid_data, saved=True, show_progress=False)
    out = {"side": "recbole 1.2.0", "dataset": dataset, "params": p, "best_valid": dict(best),
           "n_users": int(ds.user_num) - 1, "n_items": int(ds.item_num) - 1,
           "n_train": int(len(train_data._dataset)), "seconds": round(time.time() - t0, 1)}
    write(dataset, "recbole", out)


def run_ours(dataset, device, epochs=None):
    import torch
    import main as runner
    from utility.utility_data.data_loader import ContractData
    from utility.utility_data.sampler import make_sampler
    from utility.utility_function.tools import read_configuration
    from utility.utility_train import model_factory
    from utility.utility_train.group_evaluator import evaluate
    from utility.utility_train.trainer import EarlyStopper, train
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(module)s | %(message)s")
    torch.set_default_dtype(torch.float32)
    p = params(dataset, epochs)
    runner.enable_determinism()
    runner.set_seed(p["seed"])
    seed_split = 2026 if dataset != "ml-1m" else 1
    data = ContractData(dataset, seed_split)
    cfg = read_configuration(ROOT / "configure" / "NCL.txt")
    cfg.update(dataset=dataset, split_seed=seed_split, K=p["n_layers"], dim=p["embedding_size"], lr=p["learning_rate"],
               batch_size=p["train_batch_size"], reg=p["reg_weight"], ssl_lambda=p["ssl_reg"], proto_lambda=p["proto_reg"],
               alpha=p["alpha"], temperature=p["ssl_temp"], hyper_layers=p["hyper_layers"], num_clusters=p["num_clusters"],
               warm_up=p["warm_up_step"], item_groups="kmeans")
    dev = torch.device(device)
    model, callback, _ = model_factory.build("NCL", cfg, data, dev, p["seed"])
    bb, U, I, d = model.bb, data.n_users, data.n_items, p["embedding_size"]
    with torch.no_grad():                                         # matched: RecBole xavier on separate tables
        bu, bi = math.sqrt(6.0 / (U + d)), math.sqrt(6.0 / (I + d))
        bb.emb.weight[:U].uniform_(-bu, bu); bb.emb.weight[U:].uniform_(-bi, bi)
    model.batch_ref = p["train_batch_size"]                       # matched: losses summed over the batch

    def emb_loss(users, pos, neg):                                # matched: RecBole EmbLoss (norms, not squared)
        w = bb.emb.weight
        return (w[users].norm(2) + w[U + pos].norm(2) + w[U + neg].norm(2)) / len(users)
    bb.reg_terms = emb_loss
    seen, truth = data.seen_for("val"), data.by_user("val")

    def val_fn(m):
        with torch.no_grad():
            eu, ei, _ = m.representations()
        r = evaluate(eu, ei, seen, truth, data.train_degree, ks=(10, 20), groups=("overall",))["overall"]["mean"]
        val_fn.last = r
        return r["Recall@20"], r["Recall@20"]
    best = {"value": -1}

    def on_end(ep, opt, rec):
        if val_fn.last["Recall@20"] > best["value"]:
            best.update(value=val_fn.last["Recall@20"], epoch=ep + 1, metrics=dict(val_fn.last))
    t0 = time.time()
    train(model, make_sampler(data.edges["train"], data.n_items, seed=p["seed"]), len(data.edges["train"]), p["epochs"],
          p["train_batch_size"], reg=p["reg_weight"], val_fn=val_fn, stopper=EarlyStopper(None, patience=p["stopping_step"]),
          epoch_callback=callback, on_epoch_end=on_end)
    write(dataset, "ours", {"side": "this codebase (matched mode)", "dataset": dataset, "params": p,
                            "best_valid": {"recall@10": best["metrics"]["Recall@10"], "recall@20": best["metrics"]["Recall@20"],
                                           "ndcg@10": best["metrics"]["NDCG@10"], "ndcg@20": best["metrics"]["NDCG@20"]},
                            "best_epoch": best["epoch"], "seconds": round(time.time() - t0, 1)})


def write(dataset, side, value):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "{}__{}.json".format(dataset, side), "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=1, default=float); f.write("\n")
    print(json.dumps(value, indent=1, default=float))


def compare(dataset):
    r = json.loads((OUT / "{}__recbole.json".format(dataset)).read_text(encoding="utf-8"))["best_valid"]
    o = json.loads((OUT / "{}__ours.json".format(dataset)).read_text(encoding="utf-8"))["best_valid"]
    rows = {k: {"recbole": float(r[k]), "ours": float(o[k]), "rel": float(o[k]) / float(r[k]) - 1}
            for k in ("recall@10", "recall@20", "ndcg@10", "ndcg@20")}
    out = {"dataset": dataset, "metrics": rows, "tolerance": 0.05,
           "pass": all(abs(v["rel"]) <= 0.05 for v in rows.values())}
    write(dataset, "comparison", out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("export", "recbole", "ours", "compare"))
    p.add_argument("--dataset", default="yelp2018")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=None, help="smoke tests only")
    a = p.parse_args(argv)
    if a.action == "export":
        export(a.dataset, a.seed)
    elif a.action == "recbole":
        run_recbole(a.dataset, a.device, a.epochs)
    elif a.action == "ours":
        run_ours(a.dataset, a.device, a.epochs)
    else:
        compare(a.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
