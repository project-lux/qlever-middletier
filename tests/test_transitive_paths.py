"""Transitive search terms are bounded to a number of hops, not the full closure."""

from qleverlux.query.predicates import SCOPE_FIELDS, bound_path, get_predicate


def test_plus_becomes_a_bounded_range():
    assert bound_path("lux:broader+", 8) == "lux:broader{1,8}"
    assert bound_path("^lux:placePartOf+", 3) == "^lux:placePartOf{1,3}"


def test_zero_leaves_it_unbounded_and_other_paths_alone():
    assert bound_path("lux:broader+", 0) == "lux:broader+"
    assert bound_path("lux:broader", 8) == "lux:broader"
    assert bound_path("^lux:setMemberOfSet", 8) == "^lux:setMemberOfSet"


def test_every_plus_term_maps_to_a_transitive_path():
    plus = [(scope, term) for scope, terms in SCOPE_FIELDS.items() for term in terms if term.endswith("+")]
    assert sorted(plus) == sorted([("concept", "broader+"), ("concept", "narrower+"), ("place", "partOf+"),
                                   ("place", "contains+"), ("set", "memberOf+"), ("set", "containingSet+")])
    for scope, term in plus:
        assert bound_path(get_predicate(term, scope), 8).endswith("{1,8}")
