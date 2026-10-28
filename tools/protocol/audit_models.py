"""Independent audit of the model code: every component is recomputed with dense NumPy (float64) from its written
definition, on random graphs and labels, and compared with the library. Nothing of the model code is reused here.

    python -m tools.protocol.audit_models
"""
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.HiTaxProtoCF import AncestorPrototypeMemory, Gate, HiTaxProtoCF, PrototypeMemory, make_fuse_mask  # noqa
from models.LightGCN import LightGCN  # noqa: E402
from models.NCL import NCL  # noqa: E402
from models.SimGCL import SimGCL, info_nce as simgcl_info_nce  # noqa: E402
from models.TFLightGCN import TFLightGCN  # noqa: E402
from utility.utility_data.data_graph import build_norm_adj  # noqa: E402
from utility.utility_train.group_evaluator import evaluate  # noqa: E402

torch.set_default_dtype(torch.float64)
np.seterr(divide="ignore")              # isolated nodes: d ** -0.5 is masked by np.where
RESULTS = collections.OrderedDict()


def check(name, ok, detail=""):
    RESULTS.setdefault(name, []).append((bool(ok), detail))


def random_graph(rng, U=30, I=45, E=260):
    e = set()
    while len(e) < E:
        e.add((int(rng.integers(U)), int(rng.integers(I))))
    return U, I, np.array(sorted(e), dtype=np.int64)


def dense_lightgcn(U, I, edges, E0, K):
    """out = mean_{k=0..K} Â^k E0, Â = D^-1/2 A D^-1/2 of the bipartite graph built from train edges only."""
    n = U + I
    A = np.zeros((n, n))
    for u, i in edges:
        A[u, U + i] = A[U + i, u] = 1.0
    d = A.sum(1)
    inv = np.where(d > 0, d ** -0.5, 0.0)
    Ah = inv[:, None] * A * inv[None, :]
    h, acc = E0.copy(), E0.copy()
    for _ in range(K):
        h = Ah @ h; acc = acc + h
    out = acc / (K + 1)
    return out[:U], out[U:]


def dense_prototypes(e, labels, is_train, loo=True):
    """p_i = equal-weight mean over the item's nodes of the mean final embedding of the OTHER train items of the node."""
    I, d = e.shape
    P, has = np.zeros((I, d)), np.zeros(I, bool)
    for i in range(I):
        per = []
        for n in labels.get(i, ()):
            mem = [j for j in range(I) if is_train[j] and n in labels.get(j, ()) and (j != i or not loo)]
            if mem:
                per.append(e[mem].mean(0))
        if per:
            P[i], has[i] = np.mean(per, 0), True
    return P, has


def pairs_of(labels):
    return torch.tensor([(i, n) for i, ns in labels.items() for n in ns], dtype=torch.long).reshape(-1, 2)


def main():
    rng = np.random.default_rng(0)
    for trial in range(5):
        U, I, edges = random_graph(rng)
        K, d = int(rng.integers(1, 4)), 6
        E0 = rng.normal(0, 0.1, (U + I, d))
        for layout in ("coo", "csr", "segment"):
            bb = LightGCN(U, I, d, K, build_norm_adj(U, I, torch.as_tensor(edges), layout=layout))
            with torch.no_grad():
                bb.emb.weight.copy_(torch.as_tensor(E0))
            eu, ei = bb()
            ru, ri = dense_lightgcn(U, I, edges, E0, K)
            check("lightgcn_propagation", np.abs(eu.detach().numpy() - ru).max() < 1e-12 and
                  np.abs(ei.detach().numpy() - ri).max() < 1e-12, "trial {} {}".format(trial, layout))
        users, pos, neg = torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]), torch.tensor([6, 7, 8])
        reg = 0.5 * (np.sum(E0[[0, 1, 2]] ** 2, 1) + np.sum(E0[U + np.array([3, 4, 5])] ** 2, 1)
                     + np.sum(E0[U + np.array([6, 7, 8])] ** 2, 1)).mean()
        check("lightgcn_l2", abs(float(bb.reg_terms(users, pos, neg)) - reg) < 1e-12, "trial {}".format(trial))
        deg = np.bincount(edges[:, 1], minlength=I)
        is_train = deg >= 1
        labels = {i: sorted(set(rng.choice(6, size=int(rng.integers(0, 3)), replace=False).tolist())) for i in range(I)}
        for loo in (True, False):
            mem = PrototypeMemory(I, 6, pairs_of(labels), torch.as_tensor(is_train), leave_one_out=loo)
            P, has = mem(torch.as_tensor(ri))
            rP, rhas = dense_prototypes(ri, labels, is_train, loo)
            check("prototype", np.array_equal(has.numpy(), rhas) and np.abs(P.numpy()[rhas] - rP[rhas]).max() < 1e-12,
                  "trial {} leave_one_out={}".format(trial, loo))
        # gates + fusion (full model), user side untouched, prototypes without gradient
        tax_has = PrototypeMemory(I, 6, pairs_of(labels), torch.as_tensor(is_train))(torch.as_tensor(ri))[1]
        fuse = make_fuse_mask(tax_has, torch.as_tensor(deg))
        for gate, (a, b) in (("global", (0.0, 0.3)), ("degree", (1.7, -0.4)), ("degree_content", (0.9, 0.2))):
            g = Gate(gate, d, a_init=max(a, 1e-3), b_init=b)
            we, wp = rng.normal(0, 0.5, d), rng.normal(0, 0.5, d)
            if gate == "degree_content":
                with torch.no_grad():
                    g.w_e.copy_(torch.as_tensor(we)); g.w_p.copy_(torch.as_tensor(wp))
            m = HiTaxProtoCF(bb, gate, torch.as_tensor(deg), memory=PrototypeMemory(I, 6, pairs_of(labels),
                                                                                  torch.as_tensor(is_train)),
                             gate=g, fuse_mask=fuse)
            meu, mei, _ = m.representations()
            rP, _ = dense_prototypes(ri, labels, is_train)
            sig = lambda z: 1 / (1 + np.exp(-z))
            if gate == "global":
                gg = np.full(I, sig(b))
            else:
                z = math.log1p(math.expm1(max(a, 1e-3))) * 0 + max(a, 1e-3) * np.log1p(deg) + b
                if gate == "degree_content":
                    z = z + ri @ we + rP @ wp
                gg = sig(z)
            f = fuse.numpy()
            expect = ri.copy()
            expect[f] = gg[f, None] * ri[f] + (1 - gg[f, None]) * rP[f]
            check("fusion", np.abs(mei.detach().numpy() - expect).max() < 1e-10 and
                  np.abs(meu.detach().numpy() - ru).max() < 1e-12, "trial {} {}".format(trial, gate))
            if gate == "degree":
                gd = gg[np.argsort(deg)]
                check("degree_gate_monotone", np.all(np.diff(gd[deg[np.argsort(deg)] > 0]) >= -1e-15), str(trial))
            # BPR = -mean log sigma(score(u, i+) - score(u, i-)) + reg * L2
            x = (ru[[0, 1, 2]] * expect[[3, 4, 5]]).sum(1) - (ru[[0, 1, 2]] * expect[[6, 7, 8]]).sum(1)
            bpr = float(np.mean(np.log1p(np.exp(-x))))
            check("bpr", abs(float(m.bpr_loss(users, pos, neg, reg=1e-3)) - (bpr + 1e-3 * reg)) < 1e-10, gate)
        # no gradient through the prototypes: d loss / d E0 equals the gradient with P treated as a constant
        m.zero_grad(); m.bpr_loss(users, pos, neg, reg=0.0).backward()
        grad = bb.emb.weight.grad.clone()
        Pc, _ = m.memory(torch.as_tensor(ri))

        class Const(torch.nn.Module):
            def forward(self, _):
                return Pc, torch.as_tensor(dense_prototypes(ri, labels, is_train)[1])
        m.memory = Const(); bb.emb.weight.grad = None
        m.bpr_loss(users, pos, neg, reg=0.0).backward()
        check("stop_gradient", torch.allclose(grad, bb.emb.weight.grad, atol=1e-14), str(trial))
        # ancestor prototypes (mean variant == prototype memory on the same pairs)
        pairs3 = torch.tensor([(i, n, int(rng.integers(0, 3))) for i, ns in labels.items() for n in ns]).reshape(-1, 3)
        am = AncestorPrototypeMemory(I, 6, pairs3, torch.as_tensor(is_train), dim=d, variant="mean")
        Pa, ha = am(torch.as_tensor(ri))
        rP, rh = dense_prototypes(ri, labels, is_train)
        check("ancestor_mean", np.abs(Pa.detach().numpy()[rh] - rP[rh]).max() < 1e-12 and np.array_equal(ha.numpy(), rh),
              str(trial))
        # TF-LightGCN: h_i = x_i + mean_n z_n, separate L2
        Z = rng.normal(0, 0.2, (6, d))
        tf = TFLightGCN(U, I, d, K, build_norm_adj(U, I, torch.as_tensor(edges)), pairs_of(labels), 6)
        with torch.no_grad():
            tf.emb.weight.copy_(torch.as_tensor(E0)); tf.z.weight.copy_(torch.as_tensor(Z))
        H = E0.copy()
        for i, ns in labels.items():
            if ns:
                H[U + i] += Z[ns].mean(0)
        tu, ti = tf()
        ru2, ri2 = dense_lightgcn(U, I, edges, H, K)
        check("tf_lightgcn", np.abs(ti.detach().numpy() - ri2).max() < 1e-12 and np.abs(tu.detach().numpy() - ru2).max()
              < 1e-12, str(trial))
        expect_reg = reg + 0.5 * np.mean([sum(np.sum(Z[n] ** 2) for n in labels.get(p, ())) +
                                          sum(np.sum(Z[n] ** 2) for n in labels.get(q, ()))
                                          for p, q in ((3, 6), (4, 7), (5, 8))])
        check("tf_lightgcn_l2", abs(float(tf.reg_terms(users, pos, neg)) - expect_reg) < 1e-12, str(trial))
        # SimGCL: unperturbed propagation WITHOUT layer 0; noise = sign(e) * eps * unit vector; official InfoNCE
        sg = SimGCL(U, I, d, K, build_norm_adj(U, I, torch.as_tensor(edges)), eps=0.1, tau=0.2, cl_lambda=0.5)
        with torch.no_grad():
            sg.emb.weight.copy_(torch.as_tensor(E0))
        n = U + I
        A = np.zeros((n, n))
        for u, i in edges:
            A[u, U + i] = A[U + i, u] = 1.0
        dd = A.sum(1); inv = np.where(dd > 0, dd ** -0.5, 0.0); Ah = inv[:, None] * A * inv[None, :]
        h, layers = E0.copy(), []
        for _ in range(K):
            h = Ah @ h; layers.append(h)
        su, si = sg()
        ref = np.mean(layers, 0)
        check("simgcl_propagation", np.abs(su.detach().numpy() - ref[:U]).max() < 1e-12 and
              np.abs(si.detach().numpy() - ref[U:]).max() < 1e-12, str(trial))
        h0 = torch.as_tensor(Ah @ E0)
        torch.manual_seed(trial)
        noisy = h0 + torch.sign(h0) * F.normalize(torch.rand_like(h0), dim=-1) * 0.1
        delta = (noisy - h0).numpy()
        check("simgcl_noise", np.allclose(np.linalg.norm(delta, axis=1)[np.abs(h0.numpy()).sum(1) > 0], 0.1) and
              np.all(np.sign(delta) * np.sign(h0.numpy()) >= 0), str(trial))
        v1, v2 = rng.normal(size=(7, d)), rng.normal(size=(7, d))
        c = (v1 / np.linalg.norm(v1, axis=1, keepdims=True)) @ (v2 / np.linalg.norm(v2, axis=1, keepdims=True)).T / 0.2
        ref_nce = -np.mean(np.diag(c) - np.log(np.exp(c).sum(1)))
        check("simgcl_infonce", abs(float(simgcl_info_nce(torch.as_tensor(v1), torch.as_tensor(v2), 0.2)) - ref_nce)
              < 1e-12, str(trial))
        # evaluator vs brute force (ties broken by lower item index)
        S_u, S_i = rng.normal(size=(U, d)), np.round(rng.normal(size=(I, d)), 1)
        seen = {u: [int(i) for i in edges[edges[:, 0] == u][:, 1]] for u in range(U)}
        truth = {u: [int(i) for i in rng.choice(I, 3, replace=False) if i not in seen[u]] for u in range(U)}
        truth = {u: v for u, v in truth.items() if v}
        out = evaluate(torch.as_tensor(S_u), torch.as_tensor(S_i), seen, truth, torch.as_tensor(deg), ks=(5, 10),
                       groups=("overall", "long_tail"), within=("long_tail",))
        lt = lambda i: 1 <= deg[i] <= 10
        for g, keep, cand_keep in (("overall", lambda i: True, lambda i: True), ("long_tail", lt, lambda i: True),
                                   ("long_tail_within", lt, lt)):
            rec, nd = [], []
            for u in sorted(truth):
                T = [i for i in truth[u] if keep(i)]
                if not T:
                    continue
                cand = [i for i in range(I) if i not in seen[u] and cand_keep(i)]
                order = sorted(cand, key=lambda i: (-float(S_u[u] @ S_i[i]), i))[:10]
                hits = [1.0 if i in T else 0.0 for i in order]
                rec.append(sum(hits) / len(T))
                nd.append(sum(h / math.log2(r + 2) for r, h in enumerate(hits)) /
                          sum(1 / math.log2(r + 2) for r in range(min(10, len(T)))))
            check("evaluator", abs(out[g]["mean"]["Recall@10"] - np.mean(rec)) < 1e-12 and
                  abs(out[g]["mean"]["NDCG@10"] - np.mean(nd)) < 1e-12 and out[g]["n_users"] == len(rec),
                  "trial {} {}".format(trial, g))
        # NCL losses vs the written definition (batch_ref = batch size -> batch SUM)
        nb = LightGCN(U, I, d, 3, build_norm_adj(U, I, torch.as_tensor(edges)))
        with torch.no_grad():
            nb.emb.weight.copy_(torch.as_tensor(E0))
        for groups in ("kmeans", "taxonomy"):
            ncl = NCL(nb, groups, pairs_of({i: ns for i, ns in labels.items() if ns and is_train[i]})
                      if groups == "taxonomy" else None, 6 if groups == "taxonomy" else None,
                      torch.as_tensor(is_train), k=4, lam_s=1.0, lam_p=1.0, alpha=1.7, tau=0.3, warm_up=0, batch_ref=3)
            ncl.e_step(0, seed=1)
            Ls = [E0]
            Ah = None
            n = U + I
            A = np.zeros((n, n))
            for u, i in edges:
                A[u, U + i] = A[U + i, u] = 1.0
            dd = A.sum(1); inv = np.where(dd > 0, dd ** -0.5, 0.0); Ah = inv[:, None] * A * inv[None, :]
            for _ in range(3):
                Ls.append(Ah @ Ls[-1])
            nrm = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)

            def info_nce(q, cands, pos_idx, tau):
                logits = nrm(q) @ nrm(cands).T / tau
                return float(np.sum(-logits[np.arange(len(q)), pos_idx] + np.log(np.exp(logits).sum(1))))
            ls_u = info_nce(Ls[2][[0, 1, 2]], Ls[0][:U], [0, 1, 2], 0.3)
            ls_i = info_nce(Ls[2][U + np.array([3, 4, 5])], Ls[0][U:], [3, 4, 5], 0.3)
            uc, ua = ncl.user_c.numpy(), ncl.user_a.numpy()
            lp_u = info_nce(E0[[0, 1, 2]], uc, ua[[0, 1, 2]], 0.3)
            ic, ok = ncl.item_c.numpy(), ncl.item_ok.numpy()
            logits = nrm(E0[U + np.array([3, 4, 5])]) @ ic.T / 0.3
            logits[:, ~ok] = -np.inf
            logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
            if groups == "kmeans":
                ia = ncl.item_a.numpy()
                lp_i = float(sum(-logp[r, ia[i]] for r, i in enumerate((3, 4, 5)) if ia[i] >= 0))
            else:
                tr_labels = {i: ns for i, ns in labels.items() if ns and is_train[i]}
                lp_i = float(sum(np.mean([-logp[r, n] for n in tr_labels[i]]) for r, i in enumerate((3, 4, 5))
                                 if i in tr_labels))
            mean_layers = np.mean(Ls[:4], 0)
            x = (mean_layers[[0, 1, 2]] * mean_layers[U + np.array([3, 4, 5])]).sum(1) - \
                (mean_layers[[0, 1, 2]] * mean_layers[U + np.array([6, 7, 8])]).sum(1)
            bpr = float(np.mean(np.log1p(np.exp(-x))))
            expect = bpr + (ls_u + 1.7 * ls_i) + (lp_u + lp_i)
            got = float(ncl.bpr_loss(users, pos, neg, reg=0.0))
            check("ncl_objective", abs(got - expect) < 1e-8, "trial {} {}: {} vs {}".format(trial, groups, got, expect))
    report = {k: {"checks": len(v), "failures": sum(not ok for ok, _ in v),
                  "examples": [dtl for ok, dtl in v if not ok][:3]} for k, v in RESULTS.items()}
    out = ROOT / "preprocessed" / "model_checks" / "audit_models.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1); f.write("\n")
    for k, v in report.items():
        print("{:22s} {:3d}/{:3d} {}".format(k, v["checks"] - v["failures"], v["checks"], v["examples"]))
    return 0 if all(v["failures"] == 0 for v in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
