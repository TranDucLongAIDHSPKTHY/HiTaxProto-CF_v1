"""Ancestor prototype memory, TF-LightGCN and NCL / NCL+Taxonomy."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from models.HiTaxProtoCF import AncestorPrototypeMemory
from models.NCL import NCL
from models.TFLightGCN import TFLightGCN
from tests import helpers as H
from tools.toy import toy_example as toy

DAG_PAIRS = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [3, 0, 0], [4, 1, 0], [5, 1, 0], [6, 1, 0]])


def test_ancestor_mean_reproduces_worked_example_prototype():
    e = H.toy_backbone()()[1]
    P, has = AncestorPrototypeMemory(7, 2, DAG_PAIRS, H.toy_item_degree() >= 1, dim=2, variant="mean")(e)
    assert np.abs(P[2].detach().numpy() - np.array([0.4972, 0.4211])).max() <= 5e-5 and bool(has.all())


def test_ancestor_attention_moves_towards_higher_scored_node_and_learns_only_q_gamma():
    bb = H.toy_backbone()
    e = bb()[1]
    mem = AncestorPrototypeMemory(7, 2, DAG_PAIRS, H.toy_item_degree() >= 1, dim=2, variant="attention")
    with torch.no_grad():
        mem.q.copy_(torch.tensor([0.0, 3.0]))           # node B prototype has the larger 2nd coordinate
    P, _ = mem(e)
    assert float(P[2, 1]) > 0.4211
    gq, ge = torch.autograd.grad(P.sum(), [mem.q, bb.emb.weight], allow_unused=True)
    assert gq is not None and float(gq.abs().sum()) > 0 and ge is None


def test_ancestor_dedup_keeps_shortest_hop():
    pairs = torch.tensor([[0, 0, 2], [0, 0, 1], [1, 0, 0]])
    mem = AncestorPrototypeMemory(2, 1, pairs, torch.tensor([True, True]), dim=2, H=3)
    assert len(mem.hop) == 2 and int(mem.hop[0]) == 1


def _tf(agg, pairs, n_nodes):
    bb = H.toy_backbone()
    tf = TFLightGCN(5, 7, 2, 1, bb.adj, pairs, n_nodes, agg=agg)
    with torch.no_grad():
        tf.emb.weight.copy_(bb.emb.weight)
    return bb, tf


def test_tf_lightgcn_with_zero_nodes_equals_lightgcn():
    pairs, node = H.label_pairs(toy.LABELS["taxonomy"])
    bb, tf = _tf("mean", pairs, len(node))
    for a, b in zip(bb(), tf()):
        assert torch.allclose(a, b)


def test_tf_lightgcn_mean_and_sum_aggregation_and_separate_l2():
    pairs, node = H.label_pairs(toy.LABELS["multilabel"])                 # A3 has two labels
    z = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    for agg, expect in (("mean", torch.tensor([0.5, 1.0])), ("sum", torch.tensor([1.0, 2.0]))):
        _, tf = _tf(agg, pairs, len(node))
        with torch.no_grad():
            tf.z.weight.copy_(z)
        a3 = tf.layer0()[5 + 2]
        assert torch.allclose(a3, tf.emb.weight[5 + 2] + expect)
    _, tf = _tf("mean", pairs, len(node))
    with torch.no_grad():
        tf.z.weight.copy_(z)
    users, pos, neg = torch.tensor([2]), torch.tensor([2]), torch.tensor([0])
    w = tf.emb.weight
    base = 0.5 * (w[2].pow(2).sum() + w[5 + 2].pow(2).sum() + w[5 + 0].pow(2).sum())
    z_pen = 0.5 * ((1.0 + 4.0) + 1.0)                             # A3: ||z_A||^2 + ||z_B||^2; A1: ||z_A||^2
    assert abs(float(tf.reg_terms(users, pos, neg)) - float(base + z_pen)) < 1e-12


def _ncl(item_groups="kmeans", labels=None, **kw):
    bb = H.toy_backbone()
    if item_groups == "taxonomy":
        pairs, node = H.label_pairs(labels)
        return NCL(bb, "taxonomy", pairs, len(node), H.toy_item_degree() >= 1, k=2, **kw)
    return NCL(bb, "kmeans", is_train_item=H.toy_item_degree() >= 1, k=2, **kw)


def test_ncl_no_prototype_loss_before_first_e_step_or_during_warm_up():
    users, pos, neg = torch.tensor([2, 3]), torch.tensor([6, 4]), torch.tensor([0, 3])
    m = _ncl(warm_up=0)
    l_before = float(m.bpr_loss(users, pos, neg))                 # epoch = -1: no E-step yet, no L_P, no error
    m.e_step(0)
    assert float(m.bpr_loss(users, pos, neg)) != l_before
    w = _ncl(warm_up=20)
    w.e_step(5)
    assert abs(float(w.bpr_loss(users, pos, neg)) - l_before) < 1e-12


def test_ncl_structure_reduction_is_batch_sum_at_batch_ref_and_alpha_only_on_item_side():
    users, pos, neg = torch.tensor([0, 2, 3]), torch.tensor([1, 6, 4]), torch.tensor([3, 0, 2])
    m = _ncl(batch_ref=3, lam_s=1.0, lam_p=0.0, tau=0.5)
    L = m.layers()
    z0 = F.normalize(L[0][:5], dim=1); z2 = F.normalize(L[2][users], dim=1)
    manual = -F.log_softmax(z2 @ z0.T / 0.5, dim=1)[torch.arange(3), users].sum()
    assert abs(float(m._structure(L, users, 0, 5)) - float(manual)) < 1e-10
    l1 = float(_ncl(alpha=1.0, lam_s=1.0, lam_p=0.0, tau=0.5).bpr_loss(users, pos, neg))
    l2 = float(_ncl(alpha=2.0, lam_s=1.0, lam_p=0.0, tau=0.5).bpr_loss(users, pos, neg))
    item_term = float(m._structure(L, pos, 5, 7)) * 4096 / 3
    assert abs((l2 - l1) - item_term) < 1e-8


def test_ncl_taxonomy_multilabel_mean_unlabelled_zero_and_constant_prototypes():
    labels = [["A"], ["A"], ["A", "B"], [], ["B"], ["B"], ["B"]]     # A3 multi-label, A4 unlabelled
    m = _ncl("taxonomy", labels, batch_ref=1, tau=0.2)
    m.e_step(0)
    e = F.normalize(m.bb.layer0()[5 + 2], dim=0)
    logp = F.log_softmax(e @ m.item_c.T / 0.2, dim=0)
    assert abs(float(m.proto_loss_items(torch.tensor([2]))) - float(-(logp[0] + logp[1]) / 2)) < 1e-10
    assert float(m.proto_loss_items(torch.tensor([3]))) == 0.0
    assert not m.item_c.requires_grad
    m.proto_loss_items(torch.tensor([2, 6])).backward()
    assert m.bb.emb.weight.grad is not None


def test_ncl_taxonomy_rejects_label_without_train_member():
    bb = H.toy_backbone()
    pairs = torch.tensor([[0, 0], [1, 1]])
    with pytest.raises(AssertionError):
        NCL(bb, "taxonomy", pairs, 2, torch.tensor([True, False, True, True, True, True, True]), k=2)


def test_ncl_snapshot_reloads_after_e_step():
    m = _ncl()
    m.e_step(0)
    state = {k: v.clone() for k, v in m.state_dict().items()}
    fresh = _ncl()
    fresh.load_state_dict(state)
    assert torch.equal(fresh.item_c, m.item_c) and torch.equal(fresh.user_a, m.user_a)
