"""Taxonomy label operations: (item, node) pairs, Λ reduction, valid-label resolution with lifting, ancestor sets."""
from collections import deque

import torch

def pairs_from_labels(labels, item_index, node_index):
    """labels: dict item_id -> list[node_id]  ->  LongTensor [P, 2] (item, node)."""
    rows = [(item_index[i], node_index[n]) for i, ns in labels.items() for n in ns]
    return torch.tensor(rows, dtype=torch.long).reshape(-1, 2)


def _ancestors_of(n, parents, cache):
    """all strict ancestors of node n (DAG-safe, memoised)."""
    if n not in cache:
        acc = set()
        for p in parents.get(n, []):
            acc.add(p); acc |= _ancestors_of(p, parents, cache)
        cache[n] = acc
    return cache[n]


def valid_labels(labels, train_degree, min_members=2, parents=None, root=None, info=None):
    """Resolve Λ(i) to VALID labels. Runs AFTER the split (needs train degrees).

    item train  = item with d_train >= 1 (train_degree[i] >= 1) — the SAME definition as PrototypeMemory/fuse mask.
    valid node  = node that is a label of >= min_members TRAIN items (membership = Λ(i), the set the prototype uses).
    Input labels must already be reduced to the most specific labels (reduce_to_most_specific).

    An invalid label is NOT simply deleted: it is LIFTED to its parents (ALL parents in a DAG; the super-root `root`
    is never used as a label), the set is reduced to its most specific labels again, and the counts are recomputed.
    Each round lifts only the MINIMAL invalid nodes (no invalid descendant in use), so a node that becomes valid by
    receiving lifted members is not lifted itself (no over-lifting). Labels only move up a finite DAG, so the loop
    terminates; valid nodes never lose members (an existing label of i cannot be an ancestor of a lifted parent, since
    Λ(i) is reduced), so the result is a fixed point in which EVERY label used by ANY item has >= min_members train
    members. Flat taxonomy (parents=None): a lift has no target and the label is dropped — the plain filter.
    Items may end up with an empty set ("unlabelled" for every arm). Returns (labels, set of valid nodes);
    if `info` is a dict it receives n_lifted, n_dropped, rounds."""
    parents = parents or {}
    cache = {}
    is_tr = lambda i: train_degree[i] >= 1
    L = {i: set(ns) for i, ns in labels.items()}
    n_lift = n_drop = rounds = 0
    while True:
        size = {}
        for i, ns in L.items():
            if is_tr(i):
                for n in ns:
                    size[n] = size.get(n, 0) + 1
        bad = {n for ns in L.values() for n in ns if size.get(n, 0) < min_members}
        if not bad:
            break
        rounds += 1
        above = set().union(*[_ancestors_of(m, parents, cache) for m in bad])
        minimal = bad - above
        for i in sorted(L, key=str):
            hit = L[i] & minimal
            if not hit:
                continue
            new = L[i] - hit
            for n in sorted(hit, key=str):
                ps = [p for p in parents.get(n, []) if p != root]
                if ps:
                    n_lift += 1; new |= set(ps)
                else:
                    n_drop += 1
            anc = set().union(*[_ancestors_of(m, parents, cache) for m in new]) if new else set()
            L[i] = {m for m in new if m not in anc}
    if info is not None:
        info.update(n_lifted=n_lift, n_dropped=n_drop, rounds=rounds)
    valid = {n for ns in L.values() for n in ns}
    return {i: sorted(ns, key=str) for i, ns in L.items()}, valid


def reduce_to_most_specific(direct, parents):
    """Λ(i) reduction: drop a direct label if it is an ANCESTOR of another direct label of the same
    item (e.g. Yelp 'Restaurants' + 'Pizza'), so a branch is not averaged twice. Returns (reduced, n_removed)."""
    def ancestors(n, acc):
        for p in parents.get(n, []):
            if p not in acc:
                acc.add(p); ancestors(p, acc)
        return acc
    out, removed = {}, 0
    for i, ns in direct.items():
        anc = set().union(*[ancestors(n, set()) for n in ns]) if ns else set()
        keep = sorted(n for n in set(ns) if n not in anc)
        removed += len(set(ns)) - len(keep)
        out[i] = keep
    return out, removed


def ancestor_sets(direct, parents, train_items, tau, root):
    """A(i): direct labels + all ancestors, DEDUPLICATED (each node once even with several paths, i.e. DAG),
    the pipeline super-root removed, nodes covering > tau of TRAIN items removed (denominator: ALL train items).
    hop(i, n) = shortest number of child->parent steps from any direct label of i to n (0 for direct labels).
    Returns (raw sets, coverage, kept sets, hop dict {(i, n): h})."""
    raw, hop = {}, {}
    for i, ns in direct.items():
        dist = {n: 0 for n in ns}
        q = deque(ns)
        while q:
            n = q.popleft()
            for p in parents.get(n, []):
                if p not in dist:
                    dist[p] = dist[n] + 1; q.append(p)
        raw[i] = set(dist)
        for n, h in dist.items():
            hop[(i, n)] = h
    cover = {}
    for i in train_items:
        for n in raw[i]:
            cover[n] = cover.get(n, 0) + 1
    cover = {n: c / len(train_items) for n, c in cover.items()}
    kept = {i: sorted(n for n in raw[i] if n != root and cover.get(n, 0.0) <= tau) for i in raw}
    return raw, cover, kept, hop
