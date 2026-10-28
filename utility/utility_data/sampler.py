"""BPR triple sampler with forbidden (inductive cold-start) items."""
import numpy as np
import torch

class BPRSampler:
    """Uniform BPR triples. Negatives are drawn from items that are NOT the user's train positives and NOT in
    `forbidden_items` (inductive cold-start items must never be sampled).
    Collisions are resampled in vectorised rounds; a user whose positives cover every allowed item raises an
    error instead of looping forever.

    Two ways of drawing positives:
      * epoch(batch_size): one pass over a fresh permutation of ALL train edges (default of train());
      * __call__(batch_size): edges drawn WITH replacement.
    """

    def __init__(self, edges, n_items, forbidden_items=(), seed=0, max_rounds=100):
        self.rng = np.random.default_rng(seed)
        self.max_rounds = max_rounds
        self.edges = np.asarray(edges, dtype=np.int64)
        self.allowed = np.setdiff1d(np.arange(n_items), np.asarray(list(forbidden_items), dtype=np.int64))
        self.n_items_all = int(n_items)
        # sorted unique (user * I + item) keys of positives -> O(log E) membership test
        self.pos_keys = np.unique(self.edges[:, 0] * self.n_items_all + self.edges[:, 1])
        allowed_set = np.zeros(self.n_items_all, dtype=bool); allowed_set[self.allowed] = True
        n_pos_allowed = {}
        for u, cnt in zip(*np.unique(self.edges[allowed_set[self.edges[:, 1]], 0], return_counts=True)):
            n_pos_allowed[int(u)] = int(cnt)
        full = [u for u, c in n_pos_allowed.items() if c >= len(self.allowed)]
        if full:
            raise ValueError(f"users {full[:5]} have no valid negative item")

    def __len__(self):
        return len(self.edges)

    def _is_pos(self, u, i):
        k = u * self.n_items_all + i
        j = np.searchsorted(self.pos_keys, k)
        return (j < len(self.pos_keys)) & (self.pos_keys[np.minimum(j, len(self.pos_keys) - 1)] == k)

    def _triples(self, idx):
        users, pos = self.edges[idx, 0], self.edges[idx, 1]
        neg = self.allowed[self.rng.integers(len(self.allowed), size=len(idx))]
        bad = self._is_pos(users, neg)
        rounds = 0
        while bad.any():
            rounds += 1
            if rounds > self.max_rounds:
                raise RuntimeError("negative resampling did not converge")
            neg[bad] = self.allowed[self.rng.integers(len(self.allowed), size=int(bad.sum()))]
            bad = self._is_pos(users, neg)
        return torch.as_tensor(users), torch.as_tensor(pos), torch.as_tensor(neg)

    def __call__(self, batch_size):
        return self._triples(self.rng.integers(len(self.edges), size=batch_size))

    def epoch(self, batch_size):
        perm = self.rng.permutation(len(self.edges))
        for s0 in range(0, len(perm), batch_size):
            yield self._triples(perm[s0:s0 + batch_size])


def make_sampler(edges, n_items, forbidden_items=(), seed=0, max_rounds=100):
    """Guidance-document entry point; returns a BPRSampler (callable with replacement, .epoch() without)."""
    return BPRSampler(edges, n_items, forbidden_items, seed, max_rounds)
