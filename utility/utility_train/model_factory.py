"""Build every model of the study from its configuration and the data contract.

configure/<base>.txt   folder name in log/                 reference for the selection rule
LightGCN.txt           LightGCN                             none (it IS the reference)
BPRMF.txt              BPR-MF (LightGCN with K = 0)         reference LightGCN
HiTaxProtoCF.txt       GlobalGate | DegreeGate | DegreeContentGate (gate key)
TFLightGCN.txt         TF-LightGCN                          reference LightGCN
NCL.txt                NCL | NCL+Taxonomy (item_groups key)  reference LightGCN
SimGCL.txt             SimGCL | SimGCL+DegreeGate ... (gate key; fusion on top of SimGCL = transfer study)
"""
import logging

import torch

from models.HiTaxProtoCF import DISPLAY_NAMES, Gate, HiTaxProtoCF, PrototypeMemory, make_fuse_mask
from models.LightGCN import LightGCN
from models.NCL import NCL
from models.SimGCL import SimGCL
from models.TFLightGCN import TFLightGCN
from utility.utility_data.controls import global_pairs, kmeans_labels
from utility.utility_data.data_graph import build_norm_adj

logger = logging.getLogger("hitax")

TRAINING_SEEDS = (42, 0, 1, 2, 3)       # shuffled labels: the k-th training seed is paired with shuffle seed 101 + k
SETTING_KEYS = {"LightGCN": ("lr", "reg"), "BPRMF": ("lr", "reg"), "TFLightGCN": ("lr", "reg"),
                "HiTaxProtoCF": ("source", "gate_b_init"), "NCL": ("proto_lambda", "temperature"),
                "SimGCL": ("eps", "tau")}
LABELS = {"gate_b_init": "binit", "lr": "lr", "reg": "reg", "proto_lambda": "lamp", "temperature": "tau",
          "eps": "eps", "tau": "tau"}


def gate_type(config):
    """`gate = none` in a configuration file reads as None: no gate."""
    return config.get("gate") or "none"


def display_name(base, config):
    if base == "HiTaxProtoCF":
        return DISPLAY_NAMES[gate_type(config)]
    if base == "NCL":
        return "NCL+Taxonomy" if config.get("item_groups") == "taxonomy" else "NCL"
    if base == "SimGCL":
        return "SimGCL" if gate_type(config) == "none" else "SimGCL+" + DISPLAY_NAMES[gate_type(config)]
    return {"LightGCN": "LightGCN", "BPRMF": "BPR-MF", "TFLightGCN": "TF-LightGCN"}[base]


def is_reference(base):
    return base == "LightGCN"


def setting_name(base, config):
    """Readable setting folder from the key hyperparameters, e.g. 'lr0.001_reg0.0001' or 'taxonomy_binit-1'."""
    parts = []
    keys = SETTING_KEYS.get(base, ())
    if base == "SimGCL" and gate_type(config) != "none":
        keys = ("source", "gate_b_init") + keys
    for key in keys:
        value = config[key]
        parts.append("{}{}".format(LABELS.get(key, ""), "{:g}".format(value) if isinstance(value, (int, float))
                                   else value))
    if config.get("leave_one_out") is False:
        parts.append("self-inclusive")
    return "_".join(parts) or "default"


def shuffle_seed_for(config, seed):
    if config.get("shuffle_seed") is not None:
        return int(config["shuffle_seed"])
    if seed not in TRAINING_SEEDS:
        raise ValueError("shuffled source: seed {} has no paired shuffle seed; set shuffle_seed".format(seed))
    return 101 + TRAINING_SEEDS.index(seed)


class KMeansSource:
    """epoch_callback of the K-means source: re-cluster the FINAL embeddings of train items at the start of every
    epoch (epoch 0: k-means++, 10 initialisations, <= 100 Lloyd iterations; later: warm start from the previous
    centres, <= 20 iterations) and swap the prototype memory. K = number of valid taxonomy nodes."""

    def __init__(self, k, is_train_item, leave_one_out, seed):
        self.k, self.is_train, self.loo, self.seed = k, is_train_item, leave_one_out, seed
        self.train_items = is_train_item.nonzero().flatten().tolist()
        self.centres = None

    def __call__(self, model, epoch):
        with torch.no_grad():
            _, ei = model.backbone()
        if self.centres is None:
            labels, sse, sses, k_eff, C = kmeans_labels(ei, self.train_items, self.k, self.seed, n_init=10, iters=100,
                                                        return_centers=True)
        else:
            labels, sse, sses, k_eff, C = kmeans_labels(ei, self.train_items, self.k, self.seed, init=self.centres,
                                                        iters=20, return_centers=True)
        self.centres = C
        pairs = torch.tensor([(i, int(c[0][1:])) for i, c in labels.items()], dtype=torch.long)
        mem = PrototypeMemory(len(self.is_train), self.k, pairs, self.is_train.cpu(), leave_one_out=self.loo)
        model.memory = mem.to(ei.device)
        logger.info("K-means source: epoch=%d k=%d effective_k=%d sse=%.4f runs=%d", epoch + 1, self.k, k_eff, sse,
                    len(sses))


def build(base, config, data, device, seed):
    """-> (model, epoch_callback or None, resolved extras for the run manifest)."""
    edges = torch.as_tensor(data.edges["train"])
    adj = build_norm_adj(data.n_users, data.n_items, edges, dtype=torch.float32, layout=config["adj_layout"])
    dim, K, std = int(config["dim"]), int(config["K"]), float(config["init_std"])
    deg = data.train_degree
    is_train = deg >= 1
    extras = {}
    if base in ("LightGCN", "BPRMF"):
        if base == "BPRMF" and K != 0:
            raise ValueError("BPR-MF is LightGCN with K = 0")
        return HiTaxProtoCF(LightGCN(data.n_users, data.n_items, dim, K, adj, std), "none", deg).to(device), None, extras
    tax_pairs = data.label_pairs("item_labels.csv")
    n_nodes = data.n_nodes()
    if base == "TFLightGCN":
        bb = TFLightGCN(data.n_users, data.n_items, dim, K, adj, tax_pairs, n_nodes, std, agg="mean")
        return HiTaxProtoCF(bb, "none", deg).to(device), None, extras
    if base == "NCL":
        bb = LightGCN(data.n_users, data.n_items, dim, K, adj, std).to(device)
        groups = config["item_groups"]
        model = NCL(bb, groups, tax_pairs if groups == "taxonomy" else None, n_nodes if groups == "taxonomy" else None,
                    is_train, k=int(config["num_clusters"]), lam_s=float(config["ssl_lambda"]),
                    lam_p=float(config["proto_lambda"]), alpha=float(config["alpha"]),
                    tau=float(config["temperature"]), hyper=int(config["hyper_layers"]),
                    warm_up=int(config["warm_up"])).to(device)
        return model, (lambda m, ep: m.e_step(ep, seed=seed * 1000 + ep)), extras
    if base == "SimGCL":
        bb = SimGCL(data.n_users, data.n_items, dim, K, adj, std, eps=float(config["eps"]), tau=float(config["tau"]),
                    cl_lambda=float(config["cl_lambda"]))
        if gate_type(config) == "none":
            return HiTaxProtoCF(bb, "none", deg).to(device), None, extras
    elif base == "HiTaxProtoCF":
        bb = LightGCN(data.n_users, data.n_items, dim, K, adj, std)
    else:
        raise ValueError("unknown model {}".format(base))
    gate = gate_type(config)
    loo = bool(config["leave_one_out"])
    tax_mem = PrototypeMemory(data.n_items, n_nodes, tax_pairs, is_train)
    _, tax_has = tax_mem(torch.zeros(data.n_items, 1))
    fuse = make_fuse_mask(tax_has, deg)
    source, callback = config["source"], None
    if source == "taxonomy":
        mem = PrototypeMemory(data.n_items, n_nodes, tax_pairs, is_train, leave_one_out=loo)
    elif source == "shuffled":
        s = shuffle_seed_for(config, seed)
        extras["shuffle_seed"] = s
        mem = PrototypeMemory(data.n_items, n_nodes, data.label_pairs("shuffled_labels_s{}.csv".format(s)), is_train,
                              leave_one_out=loo)
    elif source == "global":
        mem = PrototypeMemory(data.n_items, 1, global_pairs(data.n_items)[is_train], is_train, leave_one_out=loo)
    elif source == "kmeans":
        k = int(torch.unique(tax_pairs[:, 1]).numel())
        extras["kmeans_k"] = k
        callback = KMeansSource(k, is_train, loo, seed)
        mem = None                                   # created by the callback at the start of epoch 0
    else:
        raise ValueError("unknown prototype source {}".format(source))
    g = Gate(gate, dim, a_init=float(config["gate_a_init"]), b_init=float(config["gate_b_init"]))
    extras["fuse_mask_size"] = int(fuse.sum())
    return HiTaxProtoCF(bb, gate, deg, memory=mem, gate=g, fuse_mask=fuse).to(device), callback, extras
