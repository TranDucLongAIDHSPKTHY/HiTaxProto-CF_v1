"""Data pipeline helpers: release choice, Kindle paths, Yelp hierarchy, metadata parsing."""
from tools.data.build_taxonomy import apply_kindle, choose_paths, node_depths, rejoin_titles
from tools.data.parse_yelp_categories import parse
from tools.data.prepare_metadata import paths_2014, single_path

REL = ("2014", "2018", "2023")


def test_release_choice_policies():
    by = {"2014": [["Books"]], "2018": [["Books", "Fiction"]], "2023": [["Books", "Fiction", "Sci-Fi"]]}
    assert choose_paths(by, REL, "first_informative") == ("2018", [["Books", "Fiction"]])
    assert choose_paths(by, REL, "deepest") == ("2023", [["Books", "Fiction", "Sci-Fi"]])
    assert choose_paths(by, REL, "earliest_release") == ("2014", [["Books"]])
    assert choose_paths({"2014": [["Books"]]}, REL, "first_informative") == ("2014", [["Books"]])
    assert choose_paths({}, REL, "first_informative") == (None, [])


def test_kindle_paths_dropped_only_when_books_paths_exist():
    both = [["Books", "Romance"], ["Kindle Store", "Kindle eBooks", "Romance"]]
    assert apply_kindle(both, "drop_if_books") == ([["Books", "Romance"]], True)
    only_kindle = [["Kindle Store", "Kindle eBooks", "Romance"]]
    assert apply_kindle(only_kindle, "drop_if_books") == (only_kindle, False)
    assert apply_kindle(both, "keep") == (both, False)


def test_yelp_comma_titles_rejoined():
    known = {"beer, wine & spirits", "books, mags, music & video", "pizza"}
    frags = ["Pizza", "Beer", "Wine & Spirits", "Books", "Mags", "Music & Video", "Bars"]
    assert rejoin_titles(frags, known) == ["Pizza", "Beer, Wine & Spirits", "Books, Mags, Music & Video", "Bars"]


def test_node_depths_take_longest_path_to_root():
    parents = {"a": ["ROOT"], "b": ["a"], "c": ["b", "ROOT"]}
    assert node_depths(["a", "b", "c"], parents) == {"a": 1, "b": 2, "c": 3}


def test_yelp_docs_parser_nesting_and_dag():
    page = ('<ul class="bullet-list-round category-list"><li> Restaurants (restaurants, All) </li><ul>'
            '<li> Pizza (pizza, All) </li><li> Tex-Mex (tex-mex, [US, CA]) </li></ul>'
            '<li> Food (food, All) </li><ul><li> Pizza (pizza, All) </li></ul></ul></body>')
    titles, locales, parents = parse(page)
    assert titles["tex-mex"] == "Tex-Mex" and locales["tex-mex"] == "[US, CA]"
    assert parents["pizza"] == ["food", "restaurants"] and "restaurants" not in parents


def test_amazon_path_parsers():
    assert paths_2014([["Books", "A &amp; B"], ["Books", "A &amp; B"], []]) == [["Books", "A & B"]]
    assert paths_2014(["Books", "Fiction"]) == [["Books", "Fiction"]]
    assert single_path(["Books", "Literature"], "Books") == [["Books", "Literature"]]
    assert single_path(["Books", "Literature"], "Buy a Kindle") == [["Books", "Literature"]]   # no false root
    assert single_path([], "Books") == [["Books"]]
    assert single_path([], None) == []
