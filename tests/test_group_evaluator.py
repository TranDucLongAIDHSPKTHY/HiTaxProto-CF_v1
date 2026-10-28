"""Group evaluation vs the worked example, tie handling, and cross-check with the TaxPro-CL repository evaluator."""
import importlib.util

import numpy as np
import pytest
import torch

from config_path.config_path import taxprocl_file
from tests import helpers as H
from tools.toy import toy_example as toy
from utility.utility_train.group_evaluator import evaluate, topk_stable

TOY_GROUPS = {"sparse": (1, 1), "warm": (2, 10**12)}
REF = toy.compute_all()


def _toy_eval(name):
    gate, source, loo, alpha = toy.CONFIGS[name]
    eu, ei, _ = H.toy_model(gate, source, loo, alpha).representations()
    seen = {u: items for u, items in toy.TRAIN.items()}
    return evaluate(eu.detach(), ei.detach(), seen, toy.TEST, H.toy_item_degree(), ks=(2,),
                    groups=("overall", "sparse", "warm"), group_ranges=TOY_GROUPS)


@pytest.mark.parametrize("name", ["lightgcn", "global_gate", "degree_gate", "degree_content_gate"])
def test_recall_ndcg_at_2_match_reference(name):
    out = _toy_eval(name)
    for g in ("overall", "sparse", "warm"):
        rec, ndcg = REF[name]["metrics_at_2"][g]
        assert abs(out[g]["mean"]["Recall@2"] - rec) < 1e-12 and abs(out[g]["mean"]["NDCG@2"] - ndcg) < 1e-12


def test_documented_values_lightgcn_global_degree():
    expect = {"lightgcn": (0.5, 0.0, 1.0), "global_gate": (1.0, 1.0, 0.6309), "degree_gate": (1.0, 0.6309, 1.0)}
    for name, (rec_all, ndcg_sparse, ndcg_warm) in expect.items():
        out = _toy_eval(name)
        assert abs(out["overall"]["mean"]["Recall@2"] - rec_all) < 5e-5
        assert abs(out["sparse"]["mean"]["NDCG@2"] - ndcg_sparse) < 5e-5
        assert abs(out["warm"]["mean"]["NDCG@2"] - ndcg_warm) < 5e-5


def test_output_is_per_user_and_depends_on_seen_masking():
    eu, ei, _ = H.toy_model("none").representations()
    out = evaluate(eu.detach(), ei.detach(), {3: toy.TRAIN[3]}, toy.TEST, H.toy_item_degree(), ks=(2,))
    assert out["overall"]["Recall@2"].shape == (1,) and list(out["overall"]["users"]) == [3]
    unmasked = evaluate(eu.detach(), ei.detach(), {}, toy.TEST, H.toy_item_degree(), ks=(2,))
    assert unmasked["overall"]["mean"]["Recall@2"] != out["overall"]["mean"]["Recall@2"]


def test_ties_index_order_and_exact_expectation():
    eu = torch.tensor([[1.0]]); ei = torch.tensor([[1.0], [1.0]])           # two candidates tied
    deg = torch.tensor([3, 3])
    det = evaluate(eu, ei, {}, {0: [1]}, deg, ks=(1,), groups=("overall",))
    exp = evaluate(eu, ei, {}, {0: [1]}, deg, ks=(1,), groups=("overall",), ties="expected")
    assert det["overall"]["mean"]["Recall@1"] == 0.0 and abs(exp["overall"]["mean"]["Recall@1"] - 0.5) < 1e-12


def test_candidate_mask_removes_non_candidates():
    eu = torch.tensor([[1.0]]); ei = torch.tensor([[3.0], [2.0], [1.0]])
    cand = torch.tensor([False, True, True])
    out = evaluate(eu, ei, {}, {0: [2]}, torch.tensor([3, 3, 3]), ks=(1,), groups=("overall",), candidates=cand)
    assert out["overall"]["mean"]["Recall@1"] == 0.0
    out = evaluate(eu, ei, {}, {0: [1]}, torch.tensor([3, 3, 3]), ks=(1,), groups=("overall",), candidates=cand)
    assert out["overall"]["mean"]["Recall@1"] == 1.0


def test_topk_stable_equals_stable_full_sort():
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        S = torch.randint(0, 4, (6, 30), generator=g).to(torch.float64)
        k = int(torch.randint(1, 30, (1,), generator=g))
        ref = torch.sort(-S, dim=1, stable=True).indices[:, :k]
        assert torch.equal(topk_stable(S, k), ref)


TAXPROCL_EVAL = taxprocl_file("utility/utility_train/group_evaluator.py")


@pytest.mark.skipif(TAXPROCL_EVAL is None or not TAXPROCL_EVAL.is_file(),
                    reason="TaxPro-CL repository copy not available (set HITAX_TAXPROCL_ROOT)")
def test_cross_check_with_taxprocl_evaluator_without_ties():
    """The TaxPro-CL evaluator (pure Python, read-only oracle) must agree when no scores are tied."""
    spec = importlib.util.spec_from_file_location("taxprocl_group_evaluator", TAXPROCL_EVAL)
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    g = torch.Generator().manual_seed(4)
    U, I = 40, 120
    eu, ei = torch.randn(U, 8, generator=g), torch.randn(I, 8, generator=g)
    deg = torch.randint(1, 30, (I,), generator=g)
    seen = {u: torch.randperm(I, generator=g)[:5].tolist() for u in range(U)}
    truth = {u: [i for i in torch.randperm(I, generator=g)[:6].tolist() if i not in seen[u]][:4] for u in range(U)}
    out = evaluate(eu, ei, seen, truth, deg, ks=(20, 10), groups=("overall", "long_tail"))
    S = eu @ ei.T
    rankings = {}
    for u in range(U):
        s = S[u].clone(); s[seen[u]] = -float("inf")
        rankings[u] = torch.argsort(-s).tolist()[:20]
    lt = {u: [i for i in its if 1 <= int(deg[i]) <= 10] for u, its in truth.items()}
    for group, targets in (("overall", truth), ("long_tail", lt)):
        out_ref = ref.evaluate_target_slice(rankings, targets, ks=(10, 20))
        for k in (10, 20):
            assert abs(out[group]["mean"][f"Recall@{k}"] - out_ref["recall"][str(k)]) < 1e-12
            assert abs(out[group]["mean"][f"NDCG@{k}"] - out_ref["ndcg"][str(k)]) < 1e-12


def test_within_group_ranking_equals_candidate_restricted_evaluation_and_a_dense_reference():
    from utility.utility_train.group_evaluator import evaluate, item_group_mask
    torch.manual_seed(5)
    U, I, d = 40, 90, 8
    eu, ei = torch.randn(U, d, dtype=torch.float64), torch.randn(I, d, dtype=torch.float64)
    deg = torch.randint(0, 25, (I,))
    rng = np.random.default_rng(0)
    seen = {u: sorted(rng.choice(I, 8, replace=False).tolist()) for u in range(U)}
    truth = {u: sorted(set(rng.choice(I, 6, replace=False).tolist()) - set(seen[u])) for u in range(U)}
    truth = {u: t for u, t in truth.items() if t}
    res = evaluate(eu, ei, seen, truth, deg, ks=(5, 20), groups=("overall", "long_tail"), within=("long_tail",))
    lt = item_group_mask(deg, "long_tail")
    ref = evaluate(eu, ei, seen, truth, deg, ks=(5, 20), groups=("long_tail",), candidates=lt)["long_tail"]
    w = res["long_tail_within"]
    assert list(w["users"]) == list(ref["users"])
    for m in ("Recall@5", "Recall@20", "NDCG@5", "NDCG@20"):
        assert np.array_equal(w[m], ref[m])
    assert not np.array_equal(w["Recall@20"], res["long_tail"]["Recall@20"])   # the candidate set matters
    S = (eu @ ei.T).numpy()                                                   # dense reference of Recall@20
    for j, u in enumerate(w["users"]):
        cands = [i for i in range(I) if lt[i] and i not in seen[u]]
        top = sorted(cands, key=lambda i: (-S[u, i], i))[:20]
        rel = [i for i in truth[u] if lt[i]]
        assert abs(w["Recall@20"][j] - len(set(top) & set(rel)) / len(rel)) < 1e-12
