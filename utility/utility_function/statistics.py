"""Statistical procedures fixed in advance of the test evaluation. Unit of analysis: the user.

Inputs are per-seed, per-user metric matrices of shape [S, U] (S paired seeds, U users eligible for the group),
with arms paired by seed index (for the shuffled source: shuffle k is paired with training seed k).
"""
import numpy as np


def rel_delta(mA, mB):
    """Relative difference of MEANS: mean(A) / mean(B) - 1 (never a mean of per-user ratios,
    which is undefined whenever m_B(u) = 0, common for long-tail users)."""
    return float(np.mean(mA) / np.mean(mB) - 1.0)


def signflip_p(D, B=100_000, seed=0, max_elems=20_000_000):
    """Paired two-sided sign-flip test on user-level differences D[u] (already averaged over seeds).
    p = (b + 1) / (B + 1), b = #{|mean(s * D)| >= |mean(D)|}.
    Sign matrices are generated in blocks of at most max_elems entries, whatever the number of users."""
    D = np.asarray(D, dtype=float)
    chunk = max(1, int(max_elems // max(1, len(D))))
    rng = np.random.default_rng(seed)
    obs = abs(D.mean())
    b, done = 0, 0
    while done < B:
        n = min(chunk, B - done)
        s = rng.choice([-1.0, 1.0], size=(n, len(D)))
        b += int((np.abs((s * D).mean(1)) >= obs - 1e-15).sum())
        done += n
    return (b + 1) / (B + 1)


def two_level_bootstrap(A, Bm, n_boot=5000, seed=0, level=0.95, one_sided=False):
    """Resample seeds (paired across arms), then users; returns dict with the absolute and relative
    differences and their percentile intervals. A, Bm: arrays [S, U]."""
    A, Bm = np.asarray(A, float), np.asarray(Bm, float)
    S, U = A.shape
    rng = np.random.default_rng(seed)
    abs_d, rel_d = np.empty(n_boot), np.empty(n_boot)
    for t in range(n_boot):
        s = rng.integers(0, S, S)
        u = rng.integers(0, U, U)
        a, b = A[np.ix_(s, u)].mean(), Bm[np.ix_(s, u)].mean()
        abs_d[t], rel_d[t] = a - b, a / b - 1.0
    if one_sided:
        q = [(1 - level) * 100]
        lo_a, lo_r = np.percentile(abs_d, q)[0], np.percentile(rel_d, q)[0]
        ci_abs, ci_rel = (lo_a, np.inf), (lo_r, np.inf)
    else:
        q = [(1 - level) / 2 * 100, (1 + level) / 2 * 100]
        ci_abs, ci_rel = tuple(np.percentile(abs_d, q)), tuple(np.percentile(rel_d, q))
    return {"abs": float(A.mean() - Bm.mean()), "rel": rel_delta(A, Bm), "ci_abs": ci_abs, "ci_rel": ci_rel,
            "boot_abs": abs_d, "boot_rel": rel_d}


def noninferiority(A, ref, delta=0.01, n_boot=5000, seed=0):
    """Non-inferiority of arm A vs the reference LightGCN on Overall: H1 is mean(A) / mean(ref) > 1 - delta.
    Implemented on D'_u = m_A(u) - (1 - delta) * m_ref(u): exactly equivalent to the ratio statement
    (mean(ref) > 0), so no ratio CI is needed. Passes if the one-sided 95% lower bound of mean(D') > 0."""
    A, ref = np.asarray(A, float), np.asarray(ref, float)
    res = two_level_bootstrap(A, (1 - delta) * ref, n_boot=n_boot, seed=seed, one_sided=True)
    return {"lower_bound_Dprime": res["ci_abs"][0], "pass": bool(res["ci_abs"][0] > 0), "rel_vs_ref": rel_delta(A, ref)}


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for k, idx in enumerate(order):
        running = max(running, (m - k) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def iut(*pvals):
    """Intersection-union test for a conjunctive claim: p = max of component p-values
    (each component difference must also be positive)."""
    return max(pvals)


def split_validation_users(user_ids, seed):
    """Pilot: split validation users into a SELECTION half (used to choose the configuration)
    and an ASSESSMENT half (used once to judge the pilot), to avoid the winner's curse."""
    ids = np.array(sorted(user_ids))
    perm = np.random.default_rng(seed).permutation(len(ids))
    h = len(ids) // 2
    return set(ids[perm[:h]].tolist()), set(ids[perm[h:]].tolist())


def claim_positive(p_holm, ci, alpha=0.05):
    """Decision rule for a POSITIVE claim: Holm-adjusted p < alpha AND the two-level
    (seed x user) CI excludes 0 on the positive side. The sign-flip test treats the trained checkpoints as fixed,
    so it is not sufficient on its own."""
    return bool(p_holm < alpha and ci[0] > 0)
