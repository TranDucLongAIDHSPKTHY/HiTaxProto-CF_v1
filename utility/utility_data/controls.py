"""Prototype sources compared with the taxonomy: global mean, shuffled labels, K-means clusters (a matched set of
controls, not a nested ladder)."""
import numpy as np
import torch

DEGREE_BINS = [(1, 5), (6, 10), (11, 20), (21, 100), (101, 10**12)]


def global_pairs(n_items):
    """Global source: every item belongs to a single node 0 (leave-one-out global mean)."""
    return torch.stack([torch.arange(n_items), torch.zeros(n_items, dtype=torch.long)], 1)


def shuffle_labels(labels, train_degree, seed, bins=DEGREE_BINS, by_count=True, min_stratum=20, info=None):
    """Shuffled source: permute whole LABEL SETS among labelled TRAIN items (d_train >= 1) inside the same stratum.

    Stratum = (degree bin, |Λ(i)|) when by_count=True (default), so every item keeps its OWN number of
    labels and every node keeps its size. Small strata: for each label count, ADJACENT degree bins are
    pooled in increasing order until the pool has >= min_stratum items (a too-small last pool joins the previous one),
    so few items keep their own set merely because their stratum is tiny; min_stratum=1 disables pooling.
    Precondition: labels resolved by valid_labels (every used node has >= 2 train items); then every train item that has
    a taxonomy prototype also has a shuffled prototype (equal masks — tested). Items with degree 0 or without labels are left
    unchanged. Iteration order is fixed (sorted ids). Ancestor sets A(i) must be recomputed from the shuffled Λ.
    If `info` is a dict it receives the pooled strata sizes and the share of shuffled items whose set is unchanged."""
    rng = np.random.default_rng(seed)
    out = {i: list(ns) for i, ns in labels.items()}
    strata = {}
    for i in sorted(labels):
        ns = labels[i]
        d = train_degree[i]
        if not ns or d < 1:
            continue
        b = next(k for k, (lo, hi) in enumerate(bins) if lo <= d <= hi)
        strata.setdefault((len(ns) if by_count else 0, b), []).append(i)
    pools = []
    for c in sorted({k[0] for k in strata}):            # pools never mix label counts
        cp, cur = [], []
        for b in range(len(bins)):
            cur += strata.get((c, b), [])
            if len(cur) >= min_stratum:
                cp.append(cur); cur = []
        if cur:
            if cp:
                cp[-1] = cp[-1] + cur
            else:
                cp.append(cur)
        pools.extend(cp)
    same = moved = 0
    for members in pools:
        perm = rng.permutation(len(members))
        for i, j in zip(members, perm):
            out[i] = list(labels[members[j]])
            same += out[i] == list(labels[i]); moved += 1
    if info is not None:
        info.update(pool_sizes=[len(m) for m in pools], pool_counts=[len(labels[m[0]]) for m in pools],
                    share_unchanged=(same / moved if moved else float("nan")))
    return out


def kmeans_labels(item_emb, train_items, k, seed, n_init=10, iters=100, device=None, chunk=8192, min_size=2,
                  init=None, return_centers=False):
    """K-means source: Lloyd + k-means++ on TRAIN items only, in torch (runs on GPU when device='cuda').
    Memory is O(chunk * k): distances are never materialised for all items at once.
    Guards: k-means++ falls back to uniform sampling if all remaining distances are 0; an empty cluster is
    re-seeded at the farthest point. After convergence every cluster is forced to have >= min_size members
    (a singleton is moved to its second-nearest centre), so every train item has a leave-one-out K-means prototype and the
    shared fuse mask holds for K-means as for the taxonomy. Returns (labels dict item -> ['K<c>'], best SSE,
    list of SSEs, effective K of the best run) [+ centres [k, d] if return_centers].
    Warm start (K-means source from epoch 1 on): init = previous epoch's centres [k, d] -> one run from init,
    no k-means++ (n_init ignored); pass iters=20 for the pre-registered schedule.
    The K-means source clusters FINAL item embeddings, the space the taxonomy prototypes live in."""
    dev = torch.device(device) if device else item_emb.device
    X = item_emb.detach().to(dev, torch.float64)[torch.as_tensor(train_items, device=dev)]
    n = len(X)
    g = torch.Generator(device="cpu").manual_seed(int(seed))

    def assign(C):
        """chunked nearest centre: returns (argmin [n], min squared distance [n])."""
        a = torch.empty(n, dtype=torch.long, device=dev); dmin = torch.empty(n, dtype=X.dtype, device=dev)
        cn = (C * C).sum(1)
        for s0 in range(0, n, chunk):
            x = X[s0:s0 + chunk]
            D = ((x * x).sum(1, keepdim=True) - 2 * x @ C.T + cn).clamp_min_(0.0)
            dmin[s0:s0 + chunk], a[s0:s0 + chunk] = D.min(1)
        return a, dmin

    def d_to(c):
        return ((X - c) ** 2).sum(1)

    best, sses = None, []
    for _ in range(1 if init is not None else n_init):
        if init is not None:
            C = init.detach().to(dev, torch.float64).clone()
            assert C.shape == (k, X.shape[1]), "init must be [k, d]"
        else:
            idx = [int(torch.randint(n, (1,), generator=g))]
            d2 = d_to(X[idx[0]])
            for _ in range(1, k):
                tot = float(d2.sum())
                probs = (d2 / tot) if tot > 0 else torch.full_like(d2, 1.0 / n)
                idx.append(int(torch.multinomial(probs.cpu(), 1, generator=g)))
                d2 = torch.minimum(d2, d_to(X[idx[-1]]))
            C = X[idx].clone()
        for _ in range(iters):
            a, dmin = assign(C)
            cnt = torch.bincount(a, minlength=k)
            C_new = torch.zeros_like(C).index_add_(0, a, X) / cnt.clamp(min=1).unsqueeze(1).to(X.dtype)
            empty = (cnt == 0).nonzero().flatten()
            if len(empty):
                C_new[empty] = X[dmin.argsort(descending=True)[: len(empty)]]
            else:
                C_new = torch.where(cnt.unsqueeze(1) > 0, C_new, C)
            if torch.allclose(C_new, C):
                C = C_new
                break
            C = C_new
        a, dmin = assign(C)
        for _ in range(n):                                   # enforce min cluster size (bounded loop)
            cnt = torch.bincount(a, minlength=k)
            small = ((cnt > 0) & (cnt < min_size)).nonzero().flatten()
            if len(small) == 0:
                break
            c0 = int(small[0]); members = (a == c0).nonzero().flatten()
            for t in members.tolist():
                d = ((C - X[t]) ** 2).sum(1); d[c0] = float("inf"); d[cnt == 0] = float("inf")
                a[t] = int(d.argmin())
        sse = float(((X - C[a]) ** 2).sum())
        sses.append(sse)
        if best is None or sse < best[0]:
            best = (sse, a.cpu().clone(), int((torch.bincount(a, minlength=k) > 0).sum()), C.clone())
    labels = {int(train_items[t]): ["K%d" % int(best[1][t])] for t in range(n)}
    if return_centers:
        return labels, best[0], sses, best[2], best[3]
    return labels, best[0], sses, best[2]
