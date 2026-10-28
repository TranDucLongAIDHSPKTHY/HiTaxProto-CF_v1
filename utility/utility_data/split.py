"""Deterministic per-user 70/10/20 split.

Determinism rules:
  * users are identified by their remapped integer id, assigned in ascending order of the RAW id (string sort);
  * each user's items are sorted by remapped item id (ascending order of RAW item id) before shuffling;
  * every user gets an independent generator np.random.default_rng([seed, user_id]) (SeedSequence entropy),
    so changing one user never shifts the split of any other user;
  * users with fewer than 3 interactions are dropped (declared filter) and counted in the manifest.
NumPy does not promise identical Generator streams across versions: record numpy.__version__ in the manifest.
"""
import numpy as np

RATIOS = (0.7, 0.1, 0.2)


def split_user(items_sorted, seed, user_id):
    n = len(items_sorted)
    rng = np.random.default_rng([seed, user_id])
    idx = rng.permutation(n)
    n_test = max(1, int(0.2 * n))          # int() floors; never Python round() (banker's rounding)
    n_val = max(1, int(0.1 * n))
    assert n - n_test - n_val >= 1, "user needs >= 3 interactions"
    test = [items_sorted[k] for k in idx[:n_test]]
    val = [items_sorted[k] for k in idx[n_test:n_test + n_val]]
    train = [items_sorted[k] for k in idx[n_test + n_val:]]
    return train, val, test


def split_all(user_items, seed, min_interactions=3):
    """user_items: dict user_id(int) -> iterable of item_id(int). Returns (rows, dropped_users) with
    rows = list of (user_id, item_id, split)."""
    rows, dropped = [], []
    for u in sorted(user_items):
        items = sorted(set(user_items[u]))
        if len(items) < min_interactions:
            dropped.append(u)
            continue
        tr, va, te = split_user(items, seed, u)
        rows += [(u, i, "train") for i in tr] + [(u, i, "val") for i in va] + [(u, i, "test") for i in te]
    return rows, dropped
