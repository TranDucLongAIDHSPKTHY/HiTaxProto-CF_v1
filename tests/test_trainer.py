"""Sampler, EarlyStopper, selection rule and the training loop."""
import numpy as np
import pytest
import torch

from tests import helpers as H
from utility.utility_data.sampler import make_sampler
from utility.utility_train.trainer import EarlyStopper, select_config, train


def test_forbidden_items_never_sampled_both_modes():
    edges = H.toy_edges().numpy()
    s = make_sampler(edges, 9, forbidden_items=(7, 8), seed=0)
    negs = np.concatenate([s(128)[2].numpy() for _ in range(100)])
    assert len(negs) == 12_800 and not np.isin(negs, [7, 8]).any()
    for _, _, n in s.epoch(4):
        assert not np.isin(n.numpy(), [7, 8]).any()


def test_negative_is_never_a_train_positive():
    edges = H.toy_edges().numpy()
    s = make_sampler(edges, 7, seed=1)
    pos = {(int(u), int(i)) for u, i in edges}
    for _ in range(50):
        u, _, n = s(64)
        assert not any((int(a), int(b)) in pos for a, b in zip(u, n))


def test_epoch_visits_every_train_edge_exactly_once():
    edges = H.toy_edges().numpy()
    s = make_sampler(edges, 7, seed=2)
    seen = [(int(u), int(p)) for bu, bp, _ in s.epoch(4) for u, p in zip(bu, bp)]
    assert sorted(seen) == sorted(map(tuple, edges.tolist()))


def test_sampler_raises_when_user_has_no_negative():
    with pytest.raises(ValueError):
        make_sampler(np.array([[0, 0], [0, 1]]), 2)


def test_selection_rule_and_fallback():
    cands = [{"name": "a", "val_LT": 0.30, "val_Overall": 0.0990},
             {"name": "b", "val_LT": 0.20, "val_Overall": 0.1000},
             {"name": "c", "val_LT": 0.40, "val_Overall": 0.0900}]
    chosen, feasible = select_config(cands, reference_overall=0.1)
    assert chosen["name"] == "b" and feasible                 # a violates (1 - 0.5%) * reference = 0.0995
    chosen, feasible = select_config(cands[2:], reference_overall=0.1)
    assert chosen["name"] == "c" and not feasible


def _run(stopper, overall, lt):
    for ep, (o, l) in enumerate(zip(overall, lt)):
        if stopper.step(ep, l, o, lambda ep=ep: {"ep": ep}):
            return ep
    return None


def test_early_stopper_late_feasible_run_not_stopped():
    overall = [0.052 + 0.002 * e for e in range(30)]            # crosses 0.0995 at epoch 24, keeps rising
    lt = [0.01] * 30
    st = EarlyStopper(reference_overall=0.1, patience=20)
    assert _run(st, overall, lt) is None
    state, ep, feas = st.result()
    assert feas and ep == 24 and state == {"ep": 24}


def test_early_stopper_never_feasible_returns_best_overall_flagged():
    overall = [0.05, 0.06, 0.07, 0.08, 0.085, 0.086, 0.087, 0.09] + [0.08] * 25
    st = EarlyStopper(reference_overall=0.1, patience=20)
    _run(st, overall, [0.2] * len(overall))
    state, ep, feas = st.result()
    assert ep == 7 and not feas


def test_early_stopper_reference_mode_uses_overall_only():
    st = EarlyStopper(reference_overall=None, patience=3)
    stop = _run(st, [0.1, 0.2, 0.15, 0.15, 0.15, 0.3], [0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert stop == 4 and st.result()[1] == 1 and st.result()[2]


def test_train_requires_explicit_stopper_with_validation():
    m = H.toy_model("degree")
    with pytest.raises(ValueError):
        train(m, make_sampler(H.toy_edges().numpy(), 7), 13, 1, 4, val_fn=lambda _: (0.0, 0.0))


def test_smoke_training_reduces_bpr_loss():
    torch.manual_seed(0)
    m = H.toy_model("degree")
    u, p, n = torch.tensor([2, 3, 4, 0]), torch.tensor([6, 4, 2, 1]), torch.tensor([0, 3, 1, 5])
    before = float(m.bpr_loss(u, p, n, reg=0.0))
    train(m, make_sampler(H.toy_edges().numpy(), 7, seed=0), 13, 60, 4, lr=0.05, reg=0.0, log=lambda *_: None)
    assert float(m.bpr_loss(u, p, n, reg=0.0)) < before


def test_resumed_run_past_its_stopping_epoch_trains_no_extra_epoch():
    m = H.toy_model("degree")
    before = {k: v.clone() for k, v in m.state_dict().items()}
    st = EarlyStopper(reference_overall=None, patience=3)
    st.best_over, st.bad = (0.1, 4, {k: v.clone() for k, v in before.items()}), 3     # stopped at epoch 7 (index 6)
    calls = []
    _, info = train(m, make_sampler(H.toy_edges().numpy(), 7), 13, 10, 4, stopper=st, start_epoch=7,
                    val_fn=lambda _: calls.append(1) or (0.0, 0.0), log=lambda *_: None)
    assert not calls and info["stopped_early"] and not info["hit_cap"] and info["last_epoch"] == 6
    assert info["epoch"] == 4 and all(torch.equal(before[k], v) for k, v in m.state_dict().items())
