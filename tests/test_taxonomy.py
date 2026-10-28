"""Label resolution: Λ reduction, valid labels with lifting, ancestor sets."""
import random

from utility.utility_data.taxonomy import ancestor_sets, reduce_to_most_specific, valid_labels


def test_flat_taxonomy_drops_nodes_with_fewer_than_two_train_items():
    labels = {0: ["A"], 1: ["A"], 2: ["B"], 3: ["B"], 4: ["C"]}
    deg = {0: 3, 1: 1, 2: 1, 3: 0, 4: 2}           # item 3 has train degree 0: does not count as a member of B
    out, valid = valid_labels(labels, deg)
    assert valid == {"A"} and out[2] == [] and out[4] == [] and out[0] == ["A"]


def test_invalid_label_is_lifted_not_deleted():
    """Counterexample of the independent review: Pizza, Sushi children of Restaurants."""
    parents = {"Pizza": ["Restaurants"], "Sushi": ["Restaurants"], "Restaurants": ["Food"], "Food": ["ROOT"]}
    raw = {"i": ["Restaurants", "Pizza"], "j": ["Restaurants", "Sushi"], "k": ["Restaurants", "Sushi"],
           "m": ["Restaurants"]}
    reduced, _ = reduce_to_most_specific(raw, parents)
    assert reduced == {"i": ["Pizza"], "j": ["Sushi"], "k": ["Sushi"], "m": ["Restaurants"]}
    deg = {x: 1 for x in raw}
    info = {}
    out, valid = valid_labels(reduced, deg, parents=parents, root="ROOT", info=info)
    assert out == {"i": ["Restaurants"], "j": ["Sushi"], "k": ["Sushi"], "m": ["Restaurants"]}
    assert "Food" not in valid                     # no over-lifting: Restaurants became valid, not lifted to Food
    assert info["n_lifted"] == 1


def test_lift_to_root_is_a_drop():
    parents = {"A": ["ROOT"], "B": ["ROOT"]}
    out, valid = valid_labels({0: ["A"], 1: ["B"], 2: ["B"]}, {0: 1, 1: 1, 2: 1}, parents=parents, root="ROOT")
    assert out[0] == [] and "ROOT" not in valid and valid == {"B"}


def test_node_with_one_train_item_and_one_degree_zero_item_is_invalid():
    out, valid = valid_labels({0: ["A"], 1: ["A"]}, {0: 1, 1: 0})
    assert valid == set() and out[0] == []


def test_e2_dag_label_lifted_to_both_parents():
    """Worked example, DAG case: AB (two parents A, B) has only A3 -> Λ(A3) = {A, B}."""
    parents = {"AB": ["A", "B"], "A": ["F"], "B": ["F"], "F": ["S"]}
    labels = {"A1": ["A"], "A2": ["A"], "A3": ["AB"], "A4": ["A"], "B1": ["B"], "B2": ["B"], "N": ["B"]}
    deg = {i: 1 for i in labels}
    out, _ = valid_labels(labels, deg, parents=parents, root="S")
    assert out["A3"] == ["A", "B"]


def test_reduce_drops_ancestor_labels():
    out, removed = reduce_to_most_specific({0: ["A", "AB"]}, {"AB": ["A", "B"]})
    assert out[0] == ["AB"] and removed == 1


def test_ancestor_sets_dedup_hop_coverage_tau():
    parents = {"AB": ["A", "B"], "A": ["F"], "B": ["F"], "F": ["S"]}
    direct = {"A1": ["A"], "A2": ["A"], "A3": ["A", "B"], "A4": ["A"], "B1": ["B"], "B2": ["B"], "N": ["B"]}
    raw, cover, kept, hop = ancestor_sets(direct, parents, list(direct), tau=0.9, root="S")
    assert raw["A3"] == {"A", "B", "F", "S"}                                  # F, S once despite two paths
    assert (hop[("A3", "A")], hop[("A3", "B")], hop[("A3", "F")], hop[("A3", "S")]) == (0, 0, 1, 2)
    assert cover["S"] == 1.0 and cover["F"] == 1.0 and abs(cover["A"] - 4 / 7) < 1e-12
    assert kept["A3"] == ["A", "B"]
    raw2, _, _, hop2 = ancestor_sets({"A3": ["AB"]}, parents, ["A3"], tau=1.0, root="S")
    assert [hop2[("A3", n)] for n in ("AB", "A", "B", "F", "S")] == [0, 1, 1, 2, 3]


def test_valid_labels_on_random_dags():
    rnd = random.Random(7)
    for _ in range(200):
        n_nodes = rnd.randint(3, 12)
        nodes = ["n%d" % k for k in range(n_nodes)]
        parents = {}
        for k in range(1, n_nodes):                                          # parents have smaller index: acyclic
            ps = rnd.sample(nodes[:k], rnd.randint(1, min(2, k)))
            parents[nodes[k]] = ps
        parents[nodes[0]] = ["ROOT"]
        labels = {i: rnd.sample(nodes, rnd.randint(0, 2)) for i in range(rnd.randint(2, 15))}
        labels, _ = reduce_to_most_specific(labels, parents)
        deg = {i: rnd.choice([0, 1, 1, 2]) for i in labels}
        out, valid = valid_labels(labels, deg, parents=parents, root="ROOT")

        def anc_or_self(n):
            acc, stack = {n}, [n]
            while stack:
                for p in parents.get(stack.pop(), []):
                    if p not in acc:
                        acc.add(p); stack.append(p)
            return acc
        for i, ns in out.items():
            for n in ns:
                assert n != "ROOT"
                assert sum(1 for j, ms in out.items() if deg[j] >= 1 and n in ms) >= 2
                assert any(n in anc_or_self(o) for o in labels[i])            # ancestor-or-self of an original label
                assert not any(n in anc_or_self(m) - {m} for m in ns)         # most specific
