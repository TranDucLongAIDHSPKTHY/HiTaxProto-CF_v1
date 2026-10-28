"""Forward pass of every worked-example configuration vs the independent NumPy reference, and hand vs machine."""
import math

import numpy as np
import pytest
import torch
from torch.func import functional_call

from tests import helpers as H
from tools.toy import toy_example as toy

REF = toy.compute_all()


@pytest.mark.parametrize("name", list(toy.CONFIGS))
def test_forward_matches_reference(name):
    gate, source, loo, alpha = toy.CONFIGS[name]
    model = H.toy_model(gate, source, loo, alpha)
    eu, ei, _ = model.representations()
    assert np.abs(ei.detach().numpy() - np.array(REF[name]["ei_star"])).max() <= 1e-6
    assert np.abs(eu.detach().numpy() - np.array(REF[name]["eu"])).max() <= 1e-6


@pytest.mark.parametrize("name, top2", [("lightgcn", ["A3", "A2"]), ("global_gate", ["N", "A3"]),
                                        ("global_gate_alpha075", ["A3", "N"]), ("degree_gate", ["A3", "N"]),
                                        ("degree_content_gate", ["A3", "N"]), ("degree_gate_global", ["A3", "A2"]),
                                        ("degree_gate_shuffled", ["A3", "A4"]), ("degree_gate_kmeans", ["A3", "A2"]),
                                        ("degree_gate_self_inclusive", ["A3", "N"])])
def test_top2_for_u4(name, top2):
    gate, source, loo, alpha = toy.CONFIGS[name]
    order, _ = H.u4_order(H.toy_model(gate, source, loo, alpha))
    assert order[:2] == top2 == REF[name]["order_u4"][:2]


def test_hand_vs_machine_worked_example():
    """4-decimal hand values of the worked example; tolerance = rounding (5e-5)."""
    dg = H.toy_model("degree")
    eu, ei_star, (g, act) = dg.representations()
    _, e_final = dg.backbone()
    P, _ = dg.memory(e_final)
    n, a3, a1 = H.IDX["N"], H.IDX["A3"], H.IDX["A1"]
    hand = {
        "e_N": (e_final[n], (0.2500, 0.2750)), "e_U4": (eu[3], (0.2704, 0.9041)),
        "p_N": (P[n], (0.1931, 0.9509)), "e*_N": (ei_star[n], (0.2270, 0.5485)),
    }
    for key, (machine, value) in hand.items():
        assert np.abs(machine.detach().numpy() - np.array(value)).max() <= 5e-5, key
    gate = torch.zeros(H.N_ITEMS); gate[act] = g.detach()
    assert abs(float(gate[n]) - 0.5954) <= 5e-5 and abs(float(gate[a3]) - 0.7680) <= 5e-5
    assert abs(float(eu[3] @ ei_star[n]) - 0.5573) <= 5e-5
    assert abs(float(eu[3] @ ei_star[a3]) - 0.6318) <= 5e-5
    x = eu[2] @ ei_star[n] - eu[2] @ ei_star[a1]
    assert abs(float(x) - 0.2725) <= 5e-5 and abs(float(-torch.nn.functional.logsigmoid(x)) - 0.5661) <= 5e-5
    gg = H.toy_model("global")
    eu_g, ei_g, _ = gg.representations()
    assert abs(float(eu_g[3] @ ei_g[a3]) - 0.5220) <= 5e-5


def test_global_gate_equals_fixed_alpha_and_learns_b():
    for alpha in (0.5, 0.75):
        _, ei, (g, _) = H.toy_model("global", alpha=alpha).representations()
        assert torch.allclose(g, torch.full_like(g, alpha))
    gg = H.toy_model("global")
    gg.bpr_loss(torch.tensor([2]), torch.tensor([6]), torch.tensor([0])).backward()
    assert gg.gate.b.grad is not None and float(gg.gate.b.grad.abs()) > 0


def test_prototypes_carry_no_gradient():
    dg = H.toy_model("degree")
    P, _ = dg.memory(dg.backbone()[1])
    assert not P.requires_grad


def test_gradcheck_module_bpr_with_fixed_prototypes():
    """Gradient of the MODULE's BPR + L_reg (DegreeContentGate) wrt layer-0 embeddings and gate params, prototypes held fixed."""
    dcg = H.toy_model("degree_content")
    P_fixed, has_fixed = dcg.memory(dcg.backbone()[1])

    class Fixed(torch.nn.Module):
        def forward(self, _):
            return P_fixed, has_fixed
    dcg.memory = Fixed()
    names = ["backbone.emb.weight", "gate.b", "gate.a_raw", "gate.w_e", "gate.w_p"]
    params = dict(dcg.named_parameters())
    users, pos, neg = torch.tensor([2, 3, 4]), torch.tensor([6, 4, 2]), torch.tensor([0, 3, 1])

    def _bpr(replaced):
        return functional_call(_BPR(dcg), {"m." + k: v for k, v in replaced.items()}, (users, pos, neg))

    inputs = tuple(params[k].detach().clone().requires_grad_(True) for k in names)
    assert torch.autograd.gradcheck(lambda *t: _bpr(dict(zip(names, t))), inputs, eps=1e-6, atol=1e-6)


class _BPR(torch.nn.Module):
    """functional_call needs forward(); route it to the module's own bpr_loss (no formula re-implemented)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, users, pos, neg):
        return self.m.bpr_loss(users, pos, neg, reg=1e-2)


def test_singleton_node_under_loo_has_no_prototype():
    labels = [["A"], ["A"], ["A"], ["A"], ["B"], ["B"], ["S"]]       # N alone in node S
    m = H.toy_model("degree", labels=labels)
    m.fuse_mask = None
    _, e = m.backbone()
    _, has = m.memory(e)
    assert not bool(has[H.IDX["N"]])
    _, ei_star, _ = m.representations()
    assert torch.allclose(ei_star[H.IDX["N"]], e[H.IDX["N"]])


def test_duplicate_rows_removed_no_self_leakage():
    pairs, node = H.label_pairs(toy.LABELS["taxonomy"])
    dup = torch.cat([pairs, pairs[pairs[:, 0] == H.IDX["N"]]])
    from models.HiTaxProtoCF import PrototypeMemory
    m_dup = PrototypeMemory(7, len(node), dup, H.toy_item_degree() >= 1)
    m_ref = PrototypeMemory(7, len(node), pairs, H.toy_item_degree() >= 1)
    e = H.toy_backbone()()[1]
    assert torch.allclose(m_dup(e)[0], m_ref(e)[0])


def test_multilabel_equal_weight_mean():
    e = H.toy_backbone()()[1]
    P, _ = H.memory_for(toy.LABELS["multilabel"])(e)
    assert np.abs(P[H.IDX["A3"]].numpy() - np.array([0.4972, 0.4211])).max() <= 5e-5
    assert np.abs(P[H.IDX["N"]].numpy() - np.array([0.1787, 0.8870])).max() <= 5e-5


def test_m2_gate_monotone_in_degree():
    torch.manual_seed(0)
    d = torch.log1p(torch.arange(0, 200, dtype=torch.float64))
    for _ in range(20):
        g = H.Gate("degree", 2, a_init=float(torch.rand(1) * 5 + 1e-3), b_init=float(torch.randn(1) * 3))
        out = g(d, None, None)
        assert bool((out[1:] >= out[:-1]).all())


def test_shared_fuse_mask_and_strict_guarantee():
    # unlabelled A4 -> no taxonomy prototype -> left as e for every source (global fuses exactly the same items)
    labels = [["A"], ["A"], ["A"], [], ["B"], ["B"], ["B"]]
    tax = H.memory_for(labels)
    e = H.toy_backbone()()[1]
    _, tax_has = tax(e)
    mask = H.make_fuse_mask(tax_has, H.toy_item_degree())
    glob = H.HiTaxProtoCF(H.toy_backbone(), "degree", H.toy_item_degree(), memory=H.memory_for([["G"]] * 7),
                          gate=H.gate_for("degree"), fuse_mask=mask)
    _, ei_star, (_, act) = glob.representations()
    assert torch.equal(act, mask) and not bool(act[H.IDX["A4"]])
    assert torch.allclose(ei_star[H.IDX["A4"]], e[H.IDX["A4"]])
    # a source lacking a prototype for a masked item must raise
    bad = H.HiTaxProtoCF(H.toy_backbone(), "degree", H.toy_item_degree(),
                         memory=H.memory_for([["A"], ["A"], ["A"], ["A"], ["B"], ["B"], ["S"]]),
                         gate=H.gate_for("degree"), fuse_mask=torch.ones(7, dtype=torch.bool))
    with pytest.raises(RuntimeError):
        bad.representations()


def test_fuse_mask_excludes_degree_zero():
    deg = torch.tensor([2, 2, 2, 1, 3, 2, 0])
    assert not bool(H.make_fuse_mask(torch.ones(7, dtype=torch.bool), deg)[6])


def test_adjacency_train_only_and_coefficients():
    adj = H.build_norm_adj(5, 7, H.toy_edges()).to_dense()
    assert torch.allclose(adj, adj.T)
    u4, b1, n = 3, 5 + H.IDX["B1"], 5 + H.IDX["N"]
    assert float(adj[u4, n]) == 0.0                                       # held-out pair absent
    assert abs(float(adj[u4, b1]) - 1 / math.sqrt(6)) < 1e-12              # c(U4, B1) = 0.4082
    assert abs(float(adj[2, n]) - 0.5) < 1e-12                             # c(U3, N) = 0.5


def test_csr_equals_coo_forward_and_gradient():
    torch.manual_seed(1)
    U, I = 40, 60
    e = torch.unique(torch.stack([torch.randint(U, (400,)), torch.randint(I - 5, (400,))], 1), dim=0)  # 5 isolated
    assert len(torch.unique(e[:, 1])) < I                                   # items: empty CSR rows
    grads = []
    for layout in ("coo", "csr", "segment"):
        bb = H.LightGCN(U, I, 8, 3, H.build_norm_adj(U, I, e, layout=layout))
        with torch.no_grad():
            bb.emb.weight.copy_(torch.linspace(-1, 1, (U + I) * 8).reshape(U + I, 8))
        eu, ei = bb()
        (eu.pow(2).sum() + (ei * torch.arange(8.)).sum()).backward()
        grads.append((torch.cat([eu, ei]).detach(), bb.emb.weight.grad.clone()))
    for other in grads[1:]:
        assert torch.allclose(grads[0][0], other[0], atol=1e-12)
        assert torch.allclose(grads[0][1], other[1], atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_segment_layout_is_bitwise_reproducible():
    torch.manual_seed(2)
    U, I = 300, 500
    e = torch.unique(torch.stack([torch.randint(U, (6000,)), torch.randint(I, (6000,))], 1), dim=0)
    adj = H.build_norm_adj(U, I, e, dtype=torch.float32, layout="segment").to("cuda")
    h0 = torch.randn(U + I, 64, device="cuda")
    runs = []
    torch.use_deterministic_algorithms(True)            # as in main.py (backward of index_select = index_add)
    try:
        for _ in range(3):
            h = h0.clone().requires_grad_(True); x, acc = h, h
            for _ in range(3):
                x = adj.mm(x); acc = acc + x
            (acc.sin() * acc).sum().backward()
            runs.append((acc.detach(), h.grad))
    finally:
        torch.use_deterministic_algorithms(False)
    assert all(torch.equal(runs[0][0], r[0]) and torch.equal(runs[0][1], r[1]) for r in runs[1:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_csr_model_matches_cpu_and_backpropagates():
    cpu = H.toy_model("degree_content", layout="csr")
    gpu = H.toy_model("degree_content", layout="csr").to("cuda")
    ref = cpu.representations()[1]
    out = gpu.representations()[1]
    assert torch.allclose(out.cpu(), ref, atol=1e-12)
    loss = gpu.bpr_loss(torch.tensor([2, 3], device="cuda"), torch.tensor([6, 4], device="cuda"),
                        torch.tensor([0, 3], device="cuda"))
    loss.backward()
    assert gpu.backbone.emb.weight.grad is not None and gpu.gate.a_raw.grad is not None


@pytest.mark.parametrize("layout", ["coo", "csr"])
def test_exact_propagation_same_values_and_restores_the_adjacency(layout):
    from utility.utility_data.data_graph import SegmentAdjacency, exact_propagation
    m = H.toy_model("degree", layout=layout)
    keys = set(m.state_dict())
    ref = m.representations()[1]
    with torch.no_grad(), exact_propagation(m):
        assert isinstance(m.backbone.adj, SegmentAdjacency)
        out = m.representations()[1]
    assert torch.allclose(out, ref, atol=1e-12)
    assert isinstance(m.backbone.adj, torch.Tensor) and "adj" in m.backbone._buffers and set(m.state_dict()) == keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_exact_propagation_is_bitwise_reproducible_on_cuda():
    from utility.utility_data.data_graph import exact_propagation
    torch.manual_seed(3)
    U, I = 300, 500
    e = torch.unique(torch.stack([torch.randint(U, (6000,)), torch.randint(I, (6000,))], 1), dim=0)
    bb = H.LightGCN(U, I, 64, 3, H.build_norm_adj(U, I, e, dtype=torch.float32, layout="csr")).to("cuda")
    bb.emb.weight.data = bb.emb.weight.data.float()
    torch.use_deterministic_algorithms(True)
    try:
        with torch.no_grad(), exact_propagation(bb):
            a = torch.cat(bb())
            b = torch.cat(bb())
    finally:
        torch.use_deterministic_algorithms(False)
    assert torch.equal(a, b)
