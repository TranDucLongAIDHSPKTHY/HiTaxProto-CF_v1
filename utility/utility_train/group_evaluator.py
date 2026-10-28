"""Full-ranking evaluation with item groups defined by TRAIN degree.

Groups restrict the relevant set and the eligible users, NOT the candidate set (whole catalogue). Within-group ranking
('<group>_within', e.g. long_tail_within) also restricts the CANDIDATES to the items of the group, so that a group of
rarely recommended items is ranked among its own items. Vectorised per user batch; returns per-user vectors (input of
the paired statistics) and their means.
"""
import math

import numpy as np
import torch

GROUPS = {"strict_cold": (0, 0), "near_cold": (1, 5), "mid_tail": (6, 10), "long_tail": (1, 10),
          "warm": (11, 10**12)}
WITHIN_GROUPS = ("near_cold", "mid_tail", "long_tail")      # groups also ranked among their own items
ENDPOINT_GROUP, ENDPOINT_METRIC = "long_tail_within", "Recall@20"   # primary endpoint and in-run selection metric


def item_group_mask(train_degree, group, group_ranges=GROUPS):
    if group == "overall":
        return torch.ones_like(train_degree, dtype=torch.bool)
    lo, hi = group_ranges[group]
    return (train_degree >= lo) & (train_degree <= hi)


def _expected_hits_under_random_ties(scores, rel_mask, k):
    """Exact expected Recall/DCG contributions at cut-off k when ties are broken uniformly at random.
    scores: 1-D tensor over candidates (non-candidates = -inf); rel_mask: bool over the same items.
    Returns (expected #relevant in top-k, expected DCG@k). Used for inductive cold-start items (identical prototype-only scores)."""
    s = scores[torch.isfinite(scores)]
    r = rel_mask[torch.isfinite(scores)]
    vals, inv = torch.unique(s, sorted=True, return_inverse=True)
    exp_hits, exp_dcg, pos = 0.0, 0.0, 0                     # pos = number of items ranked above current block
    for v_idx in range(len(vals) - 1, -1, -1):              # descending score blocks
        if pos >= k:
            break
        blk = inv == v_idx
        t = int(blk.sum()); rr = int((r & blk).sum())
        slots = min(k - pos, t)
        if rr:
            exp_hits += rr * slots / t
            exp_dcg += (rr / t) * sum(1.0 / math.log2(pos + j + 2) for j in range(slots))
        pos += t
    return exp_hits, exp_dcg


def topk_stable(S, k):
    """Top-k item indices per row, ordered by score descending with ties broken by LOWER item index —
    identical to a stable full sort, but O(I) per row instead of O(I log I)."""
    k = min(k, S.shape[1])
    thr = torch.topk(S, k, dim=1).values[:, -1:]                 # k-th largest score per row
    gt = S > thr
    need = k - gt.sum(1, keepdim=True)                            # slots left for items tied at the threshold
    eq = S == thr
    take = gt | (eq & (torch.cumsum(eq.to(torch.int64), 1) <= need))
    idx = take.nonzero()[:, 1].view(S.shape[0], k)               # exactly k per row, ascending item index
    order = torch.sort(-S.gather(1, idx), dim=1, stable=True).indices
    return idx.gather(1, order)


@torch.no_grad()
def evaluate(eu, ei, seen, truth, train_degree, ks=(20, 10), groups=("overall", "near_cold", "long_tail", "warm"),
             batch_users=1024, candidates=None, ties="index", group_ranges=GROUPS, within=()):
    """eu [U,d], ei [I,d]; seen: dict u -> items to exclude (train at validation; train+val at test);
    truth: dict u -> held-out positives; candidates: optional BoolTensor [I] (e.g. only inductive cold-start items, or all
    items except inductive cold ones). ties: 'index' (deterministic: lower item index first) or 'expected'
    (exact expectation under uniform random tie-breaking; per-user loop, meant for small cold-start candidate sets).
    within: groups also evaluated with the candidates restricted to the group's items (keys '<group>_within').

    Returns {group: {'users': np.array of user ids, f'Recall@{k}': per-user np.array, f'NDCG@{k}': ...,
                     'mean': {metric: float}, 'n_users': int}}."""
    I = ei.shape[0]
    dev = ei.device
    kmax = max(ks)
    train_degree = train_degree.to(dev)
    if within and ties != "index":
        raise ValueError("within-group ranking is implemented for ties='index' only")
    gmask = {g: item_group_mask(train_degree, g, group_ranges).to(dev) for g in set(groups) | set(within)}
    cand = None if candidates is None else candidates.to(dev)
    disc = 1.0 / torch.log2(torch.arange(2, kmax + 2, dtype=torch.float64, device=dev))
    idcg_table = torch.cumsum(disc.cpu(), 0).to(dev)              # CPU: no deterministic CUDA cumsum for floats
    keys = list(groups) + [g + "_within" for g in within]
    per = {g: {"users": [], **{f"Recall@{k}": [] for k in ks}, **{f"NDCG@{k}": [] for k in ks}} for g in keys}
    users = sorted(truth)
    for s0 in range(0, len(users), batch_users):
        ub = users[s0:s0 + batch_users]
        B = len(ub)
        S = eu[ub] @ ei.T                                            # [B, I], native dtype (float32 in real runs)
        r_idx = [r for r, u in enumerate(ub) for _ in seen.get(u, [])]
        c_idx = [i for u in ub for i in seen.get(u, [])]
        if r_idx:
            S[r_idx, c_idx] = -float("inf")
        if cand is not None:
            S[:, ~cand] = -float("inf")
        T = torch.zeros(B, I, dtype=torch.bool, device=dev)
        t_r = [r for r, u in enumerate(ub) for _ in truth[u]]
        t_c = [i for u in ub for i in truth[u]]
        T[t_r, t_c] = True
        top = topk_stable(S, kmax) if ties == "index" else None      # ties -> lower item index first
        jobs = [(g, g, top) for g in groups]
        for g in within:                                             # candidates = unseen items of the group
            jobs.append((g + "_within", g, topk_stable(S.masked_fill(~gmask[g].unsqueeze(0), -float("inf")), kmax)))
        for key, g, top_g in jobs:
            Tg = T & gmask[g].unsqueeze(0)
            if cand is not None:
                Tg = Tg & cand.unsqueeze(0)
            nrel = Tg.sum(1)
            ok = nrel > 0
            if not bool(ok.any()):
                continue
            per[key]["users"] += [u for u, o in zip(ub, ok.tolist()) if o]
            if ties == "index":
                hits = Tg.gather(1, top_g).to(torch.float64)            # [B, kmax]
                for k in ks:
                    h = hits[:, :k]
                    rec = h.sum(1) / nrel.clamp(min=1)
                    idcg = idcg_table[(nrel.clamp(min=1, max=k) - 1)]
                    ndcg = (h * disc[:k]).sum(1) / idcg
                    per[key][f"Recall@{k}"] += rec[ok].tolist()
                    per[key][f"NDCG@{k}"] += ndcg[ok].tolist()
            else:
                for r in ok.nonzero().flatten().tolist():
                    n = int(nrel[r])
                    for k in ks:
                        eh, ed = _expected_hits_under_random_ties(S[r], Tg[r], k)
                        idcg = sum(1.0 / math.log2(j + 2) for j in range(min(k, n)))
                        per[g][f"Recall@{k}"].append(eh / n)
                        per[g][f"NDCG@{k}"].append(ed / idcg)
    out = {}
    for g in keys:
        d = {key: np.asarray(v) for key, v in per[g].items()}
        d["n_users"] = len(d["users"])
        d["mean"] = {key: (float(v.mean()) if len(v) else float("nan")) for key, v in d.items()
                     if key.startswith(("Recall", "NDCG"))}
        out[g] = d
    return out
