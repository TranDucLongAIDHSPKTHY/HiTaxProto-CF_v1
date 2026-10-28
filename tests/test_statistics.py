"""Pre-declared statistical procedures."""
import numpy as np

from utility.utility_function.statistics import (claim_positive, holm, noninferiority, rel_delta, signflip_p,
                                                 split_validation_users, two_level_bootstrap)


def test_holm_textbook_example():
    p = [0.01, 0.04, 0.03, 0.005]
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])


def test_signflip_p_is_one_when_all_differences_are_zero():
    assert signflip_p(np.zeros(50), B=2000) == 1.0


def test_signflip_detects_a_clear_shift():
    rng = np.random.default_rng(0)
    assert signflip_p(rng.normal(0.5, 1.0, 400), B=5000) < 0.01


def test_relative_delta_is_ratio_of_means_with_zero_baseline_users():
    a = np.array([[0.2, 0.0, 0.4]]); b = np.array([[0.0, 0.1, 0.2]])
    assert abs(rel_delta(a, b) - (0.2 / 0.1 - 1)) < 1e-12


def test_noninferiority_on_dprime_matches_ratio_statement():
    rng = np.random.default_rng(1)
    m0 = rng.uniform(0.05, 0.15, (5, 800))
    same = noninferiority(m0, m0, delta=0.01, n_boot=500)
    worse = noninferiority(0.95 * m0, m0, delta=0.01, n_boot=500)
    assert same["pass"] and not worse["pass"]
    assert abs(same["rel_vs_ref"]) < 1e-12


def test_two_level_bootstrap_ci_contains_true_difference():
    rng = np.random.default_rng(2)
    base = rng.uniform(0, 1, (5, 600))
    res = two_level_bootstrap(base + 0.05, base, n_boot=800)
    assert res["ci_abs"][0] <= 0.05 + 1e-9 <= res["ci_abs"][1] + 1e-9


def test_claim_rule_requires_holm_and_ci():
    assert claim_positive(0.01, (0.001, 0.02))
    assert not claim_positive(0.01, (-0.001, 0.02))
    assert not claim_positive(0.2, (0.001, 0.02))


def test_validation_users_split_into_disjoint_halves():
    sel, ass = split_validation_users(range(101), seed=2026)
    assert not (sel & ass) and len(sel | ass) == 101 and abs(len(sel) - len(ass)) <= 1
