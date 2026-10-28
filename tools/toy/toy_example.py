"""Independent NumPy reference of the bookshop worked example (docs/GLOSSARY.md, "worked example"), written from the
method definition only (it does not import the library). Items A1, A2, A3, A4, B1, B2, N -> 0..6; users U1..U5 -> 0..4.
Two categories: A (detective) and B (science fiction); N is a new science-fiction book with one purchase.

    python -m tools.toy.toy_example          # prints every reference number
"""
import json
import math

import numpy as np

USERS = ["U1", "U2", "U3", "U4", "U5"]
ITEMS = ["A1", "A2", "A3", "A4", "B1", "B2", "N"]
TRAIN = {0: [0, 1], 1: [0, 1, 3], 2: [4, 5, 2, 6], 3: [4, 5], 4: [4, 2]}
TEST = {3: [2, 6]}                                   # U4 later buys A3 and N
E0_USERS = np.array([[1, .1], [1, 0], [0, .9], [.3, .9], [0, 1]], dtype=float)
E0_ITEMS = np.array([[.7, 0], [1, .4], [.3, .7], [.6, .2], [.1, 1], [.4, 1], [.5, .1]], dtype=float)
LABELS = {"taxonomy": [["A"], ["A"], ["A"], ["A"], ["B"], ["B"], ["B"]],
          "global": [["G"]] * 7,
          "shuffled": [["A"], ["A"], ["A"], ["B"], ["B"], ["B"], ["A"]],     # A4 <-> N (same degree band d = 1)
          "kmeans": [["K0"], ["K0"], ["K1"], ["K0"], ["K1"], ["K1"], ["K0"]],  # exact 2-means optimum, SSE 0.3673
          "multilabel": [["A"], ["A"], ["A", "B"], ["A"], ["B"], ["B"], ["B"]]}  # A3 carries both labels
GATE = {"degree": dict(a=2.0, b=-1.0), "degree_content": dict(a=2.0, b=-1.0, w_e=(0.0, 1.0), w_p=(-1.0, 0.0))}


def edges():
    return [(u, i) for u, its in TRAIN.items() for i in its]


def degrees():
    du = np.zeros(5); di = np.zeros(7)
    for u, i in edges():
        du[u] += 1; di[i] += 1
    return du, di


def lightgcn_one_layer():
    du, di = degrees()
    e1u = np.zeros_like(E0_USERS); e1i = np.zeros_like(E0_ITEMS)
    for u, i in edges():
        c = 1.0 / math.sqrt(du[u] * di[i])
        e1u[u] += c * E0_ITEMS[i]; e1i[i] += c * E0_USERS[u]
    return (E0_USERS + e1u) / 2, (E0_ITEMS + e1i) / 2


def prototypes(ei, labels, leave_one_out=True):
    P = np.zeros_like(ei); has = np.zeros(len(ei), bool)
    for i in range(len(ei)):
        ps = []
        for n in labels[i]:
            mem = [j for j in range(len(ei)) if n in labels[j] and (j != i or not leave_one_out)]
            if mem:
                ps.append(ei[mem].mean(0))
        if ps:
            P[i] = np.mean(ps, 0); has[i] = True
    return P, has


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def gates(gate_type, ei, P, alpha=0.5):
    _, di = degrees()
    if gate_type == "global":
        return np.full(len(ei), alpha)
    p = GATE[gate_type]
    z = p["a"] * np.log1p(di) + p["b"]
    if gate_type == "degree_content":
        z = z + ei @ np.array(p["w_e"]) + P @ np.array(p["w_p"])
    return sigmoid(z)


def fused(gate_type, source="taxonomy", leave_one_out=True, alpha=0.5):
    eu, ei = lightgcn_one_layer()
    if gate_type == "none":
        return eu, ei, None
    P, has = prototypes(ei, LABELS[source], leave_one_out)
    g = gates(gate_type, ei, P, alpha)
    return eu, g[:, None] * ei + (1 - g[:, None]) * P, g


def ranking_u4(ei_star, eu):
    cand = [i for i in range(7) if i not in TRAIN[3]]
    s = {i: float(eu[3] @ ei_star[i]) for i in cand}
    return sorted(cand, key=lambda i: (-s[i], i)), s


def metrics_at_2(order):
    _, di = degrees()
    out = {}
    for g, keep in (("overall", lambda i: True), ("sparse", lambda i: di[i] == 1), ("warm", lambda i: di[i] >= 2)):
        T = [i for i in TEST[3] if keep(i)]
        hits = [1.0 if i in T else 0.0 for i in order[:2]]
        dcg = sum(h / math.log2(r + 2) for r, h in enumerate(hits))
        idcg = sum(1 / math.log2(r + 2) for r in range(min(2, len(T))))
        out[g] = (sum(hits) / len(T), dcg / idcg)
    return out


# name: (gate type, prototype source, leave-one-out, alpha of the global gate)
CONFIGS = {"lightgcn": ("none", "taxonomy", True, 0.5),
           "global_gate": ("global", "taxonomy", True, 0.5),
           "global_gate_alpha075": ("global", "taxonomy", True, 0.75),
           "degree_gate": ("degree", "taxonomy", True, 0.5),
           "degree_content_gate": ("degree_content", "taxonomy", True, 0.5),
           "degree_gate_self_inclusive": ("degree", "taxonomy", False, 0.5),
           "degree_gate_shuffled": ("degree", "shuffled", True, 0.5),
           "degree_gate_global": ("degree", "global", True, 0.5),
           "degree_gate_kmeans": ("degree", "kmeans", True, 0.5)}


def compute_all():
    res = {}
    for name, (gate_type, src, loo, alpha) in CONFIGS.items():
        eu, es, g = fused(gate_type, src, loo, alpha)
        order, s = ranking_u4(es, eu)
        res[name] = {"ei_star": es.tolist(), "eu": eu.tolist(), "gate": None if g is None else g.tolist(),
                     "scores_u4": {ITEMS[i]: v for i, v in s.items()}, "order_u4": [ITEMS[i] for i in order],
                     "metrics_at_2": metrics_at_2(order)}
    eu, es, _ = fused("degree")
    x = float(eu[2] @ es[6] - eu[2] @ es[0])
    res["bpr_U3_N_A1_degree_gate"] = {"x": x, "loss": float(-math.log(sigmoid(x)))}
    _, ei = lightgcn_one_layer()
    res["multilabel_prototypes"] = prototypes(ei, LABELS["multilabel"])[0].tolist()
    res["new_item_prototype_only_B"] = ei[[4, 5, 6]].mean(0).tolist()
    return res


if __name__ == "__main__":
    print(json.dumps(compute_all(), indent=1, ensure_ascii=False))
