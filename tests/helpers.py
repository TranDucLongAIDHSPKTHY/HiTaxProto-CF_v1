"""Builders for the bookshop worked example on top of the library classes (reference: tools/toy/toy_example.py)."""
import math

import numpy as np
import torch

from models.HiTaxProtoCF import Gate, HiTaxProtoCF, PrototypeMemory, make_fuse_mask
from models.LightGCN import LightGCN
from tools.toy import toy_example as toy
from utility.utility_data.data_graph import build_norm_adj

N_USERS, N_ITEMS = 5, 7
IDX = {name: k for k, name in enumerate(toy.ITEMS)}


def toy_edges():
    return torch.tensor(toy.edges(), dtype=torch.long)


def toy_item_degree():
    return torch.bincount(toy_edges()[:, 1], minlength=N_ITEMS)


def toy_backbone(layout="coo"):
    adj = build_norm_adj(N_USERS, N_ITEMS, toy_edges(), layout=layout)
    bb = LightGCN(N_USERS, N_ITEMS, 2, 1, adj)
    with torch.no_grad():
        bb.emb.weight.copy_(torch.tensor(np.vstack([toy.E0_USERS, toy.E0_ITEMS])))
    return bb


def label_pairs(labels):
    """labels: list (per item) of node-name lists -> (pairs [P,2], node name -> id)."""
    names = sorted({n for ns in labels for n in ns})
    node = {n: k for k, n in enumerate(names)}
    rows = [(i, node[n]) for i, ns in enumerate(labels) for n in ns]
    return torch.tensor(rows, dtype=torch.long).reshape(-1, 2), node


def memory_for(labels, leave_one_out=True):
    pairs, node = label_pairs(labels)
    return PrototypeMemory(N_ITEMS, len(node), pairs, toy_item_degree() >= 1, leave_one_out=leave_one_out)


def gate_for(gate_type, alpha=0.5):
    if gate_type == "global":
        return Gate("global", 2, b_init=math.log(alpha / (1 - alpha)))
    g = Gate(gate_type, 2, a_init=2.0, b_init=-1.0)
    if gate_type == "degree_content":
        with torch.no_grad():
            g.w_e.copy_(torch.tensor(toy.GATE["degree_content"]["w_e"]))
            g.w_p.copy_(torch.tensor(toy.GATE["degree_content"]["w_p"]))
    return g


def toy_model(gate_type, source="taxonomy", leave_one_out=True, alpha=0.5, layout="coo", labels=None):
    bb = toy_backbone(layout)
    deg = toy_item_degree()
    if gate_type == "none":
        return HiTaxProtoCF(bb, "none", deg)
    _, tax_has = memory_for(toy.LABELS["taxonomy"])(bb()[1])
    mem = memory_for(labels if labels is not None else toy.LABELS[source], leave_one_out=leave_one_out)
    return HiTaxProtoCF(bb, gate_type, deg, memory=mem, gate=gate_for(gate_type, alpha),
                        fuse_mask=make_fuse_mask(tax_has, deg))


def u4_order(model):
    eu, ei, _ = model.representations()
    s = (eu[3] @ ei.T).detach()
    cand = [i for i in range(N_ITEMS) if i not in toy.TRAIN[3]]
    return [toy.ITEMS[i] for i in sorted(cand, key=lambda i: (-float(s[i]), i))], s
