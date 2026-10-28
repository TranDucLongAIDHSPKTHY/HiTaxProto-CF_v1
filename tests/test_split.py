"""Deterministic per-user 70/10/20 split."""
from utility.utility_data.split import split_all, split_user


def test_split_sizes_floor_with_minimum_one():
    for n, (tr, va, te) in ((3, (1, 1, 1)), (10, (7, 1, 2)), (16, (12, 1, 3)), (20, (14, 2, 4))):
        a, b, c = split_user(list(range(n)), 2026, 0)
        assert (len(a), len(b), len(c)) == (tr, va, te)
        assert sorted(a + b + c) == list(range(n))


def test_users_below_three_interactions_are_dropped():
    rows, dropped = split_all({0: [1, 2], 1: [1, 2, 3]}, 2026)
    assert dropped == [0] and {u for u, _, _ in rows} == {1}


def test_one_user_change_does_not_shift_others_and_is_deterministic():
    base = {u: list(range(u, u + 12)) for u in range(5)}
    changed = dict(base); changed[2] = list(range(40, 60))
    r1, _ = split_all(base, 2026); r2, _ = split_all(changed, 2026); r3, _ = split_all(base, 2026)
    assert r1 == r3
    assert [r for r in r1 if r[0] != 2] == [r for r in r2 if r[0] != 2]
