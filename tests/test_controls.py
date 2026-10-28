"""Prototype sources: shuffled-label properties and mask equality, K-means."""
import collections
import random

import torch

from models.HiTaxProtoCF import PrototypeMemory
from tests import helpers as H
from tools.toy import toy_example as toy
from utility.utility_data.controls import DEGREE_BINS, kmeans_labels, shuffle_labels
from utility.utility_data.taxonomy import pairs_from_labels, valid_labels


def _synthetic(n=300, seed=3):
    rnd = random.Random(seed)
    nodes = ["c%d" % k for k in range(15)]
    labels = {i: sorted(rnd.sample(nodes, rnd.choice([0, 1, 1, 1, 2, 2, 3]))) for i in range(n)}
    deg = {i: rnd.choice([0] + list(range(1, 12)) + [25, 60, 150]) for i in range(n)}
    labels, _ = valid_labels(labels, deg)
    return labels, deg


def _bin(d):
    return next(k for k, (lo, hi) in enumerate(DEGREE_BINS) if lo <= d <= hi)


def _node_sizes(labels, deg):
    return collections.Counter(n for i, ns in labels.items() if deg[i] >= 1 for n in ns)


def _has(labels, deg):
    names = sorted({n for ns in labels.values() for n in ns})
    pairs = pairs_from_labels(labels, {i: i for i in labels}, {n: k for k, n in enumerate(names)})
    is_tr = torch.tensor([deg[i] >= 1 for i in range(len(labels))])
    return PrototypeMemory(len(labels), max(1, len(names)), pairs, is_tr)(torch.randn(len(labels), 4))[1] & is_tr


def test_shuf_preserves_counts_sizes_and_strata_without_pooling():
    labels, deg = _synthetic()
    for seed in range(101, 106):
        out = shuffle_labels(labels, deg, seed, min_stratum=1)
        assert all(len(out[i]) == len(labels[i]) for i in labels)
        assert _node_sizes(out, deg) == _node_sizes(labels, deg)
        before = collections.Counter((_bin(deg[i]), tuple(labels[i])) for i in labels if deg[i] >= 1 and labels[i])
        after = collections.Counter((_bin(deg[i]), tuple(out[i])) for i in out if deg[i] >= 1 and out[i])
        assert before == after


def test_shuf_leaves_degree_zero_and_unlabelled_items_and_moves_labels():
    labels, deg = _synthetic()
    out = shuffle_labels(labels, deg, 101)
    assert all(out[i] == labels[i] for i in labels if deg[i] == 0 or not labels[i])
    assert sum(out[i] != labels[i] for i in labels) > 0


def test_shuf_mask_equals_tax_mask_on_train_items():
    labels, deg = _synthetic()
    tax = _has(labels, deg)
    for seed in range(101, 106):
        assert torch.equal(_has(shuffle_labels(labels, deg, seed), deg), tax)


def test_shuf_pooled_strata():
    labels, deg = _synthetic()
    info = {}
    out = shuffle_labels(labels, deg, 102, min_stratum=20, info=info)
    assert all(len(out[i]) == len(labels[i]) for i in labels)
    assert _node_sizes(out, deg) == _node_sizes(labels, deg)
    per_count = collections.Counter(info["pool_counts"])
    for size, count in zip(info["pool_sizes"], info["pool_counts"]):
        assert size >= 20 or per_count[count] == 1
    assert 0.0 <= info["share_unchanged"] < 1.0


def test_counterexample_masks_need_valid_labels():
    raw = {0: ["A"], 1: ["B"], 2: ["B"]}
    deg = {0: 1, 1: 1, 2: 1}
    differ = [not torch.equal(_has(shuffle_labels(raw, deg, s), deg), _has(raw, deg)) for s in range(10)]
    assert any(differ)
    resolved, _ = valid_labels(raw, deg)
    assert all(torch.equal(_has(shuffle_labels(resolved, deg, s), deg), _has(resolved, deg)) for s in range(10))


def test_kmeans_reaches_exact_toy_optimum_on_final_embeddings():
    e = H.toy_backbone()()[1]
    labels, sse, sses, k_eff = kmeans_labels(e, list(range(7)), 2, seed=0, n_init=10)
    assert abs(sse - 0.3673) < 5e-5 and k_eff == 2 and len(sses) == 10
    groups = {}
    for i, (c,) in labels.items():
        groups.setdefault(c, set()).add(toy.ITEMS[i])
    assert sorted(map(sorted, groups.values())) == [["A1", "A2", "A4", "N"], ["A3", "B1", "B2"]]


def test_kmeans_chunked_enforces_min_cluster_size():
    X = torch.cat([torch.randn(30, 2) * 0.1, torch.tensor([[50.0, 50.0]])])
    labels, _, _, _ = kmeans_labels(X, list(range(31)), 3, seed=1, n_init=3, chunk=7, min_size=2)
    counts = collections.Counter(c for (c,) in labels.values())
    assert min(counts.values()) >= 2


def test_kmeans_warm_start_returns_centres():
    X = torch.randn(50, 3)
    _, _, _, _, C = kmeans_labels(X, list(range(50)), 4, seed=0, n_init=2, return_centers=True)
    _, _, sses, _, C2 = kmeans_labels(X, list(range(50)), 4, seed=0, init=C, iters=20, return_centers=True)
    assert len(sses) == 1 and C2.shape == (4, 3)
