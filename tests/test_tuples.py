"""The per-tuple rules behind the paper's metrics (gsi.analysis.tuples)."""
from itertools import permutations

from gsi.analysis.tuples import (comparison_value, divergence_site, failure_step, reading_tag, spread,
                                 tuple_factors, tuple_members, tuple_reference)


def test_divergence_site_does_not_depend_on_serialization_order():
    # two correct copies disagree 1-1 and a wrong copy agrees with one of them: no majority to diverge from
    vals, bad = ["CORRECT", "7", "7"], [False, False, True]
    sites = {divergence_site([vals[i] for i in p], [bad[i] for i in p], True) for p in permutations(range(3))}
    assert sites == {"program"}


def test_divergence_site_blames_a_wrong_copy_that_left_the_majority():
    assert divergence_site(["CORRECT", "CORRECT", "5"], [False, False, True], True) == "copy"
    assert divergence_site(["CORRECT", "CORRECT", "CORRECT", "5"], [False, False, True, False], True) == "program"
    assert divergence_site(["CORRECT", "NA"], [False, False], False) == "crash"
    assert divergence_site(["5", "6"], [True, True], True) == "copy"  # no correct copy to compare against


def test_failure_step_assigns_the_first_step_that_failed():
    assert failure_step("execution", False, False) == "copy"  # wrong copy, then a crash
    assert failure_step("execution", True, False) == "run"
    assert failure_step("execution", None, False) == "run"
    assert failure_step("transcription", False, False) == "copy"
    assert failure_step("logic", True, False) == "solve"
    assert failure_step("construction", True, False) == "solve"
    assert failure_step("format", True, False) == "other"
    assert failure_step("no_computation", True, True) is None  # a correct answer is not a failure
    assert failure_step("ok", False, True) is None


def test_spread():
    assert spread([3.0, 6.0], 3.0) == (1.0, True)
    assert spread([0.0, 2.0], 0.0) == (2.0, False)
    assert spread([4.0, 4.0], 4.0) == (None, None)
    assert spread([1.0, 2.0], None) == (None, None)


def test_reading_tag():
    base, other = {1, 2}, {1, 2, 3}
    assert reading_tag("connected_nodes", [{1, 2}, {1, 2, 3}], base, other) == "takes the other reading"
    assert reading_tag("connected_nodes", [{1, 2}, {1}], base, other) == "other"
    assert reading_tag("disconnected_nodes", [{4}, {5}], {4}, {4}) == "readings coincide"
    assert reading_tag("neighbor", [{1}, {1, 2}], {1, 2}, None) == "strict subset of true neighbours"
    assert reading_tag("neighbor", [{1, 2}, {1, 2, 3}], {1, 2}, None) == "other"


def _answer(axis, vid, identity=False):
    return {"axis": axis, "variant_id": vid, "is_identity": identity}


def test_tuple_members_follow_the_paper():
    canon, ident = _answer("canonical", "c"), _answer("relabel", "r:identity", True)
    s0, o0 = _answer("relabel", "r:s0"), _answer("order", "o:0")
    answers = [canon, ident, s0, o0]
    assert tuple_members(answers, "relabel") == [ident, s0]  # compared with the sorted identity, not canonical
    assert tuple_members(answers, "order") == [canon, o0]
    assert tuple_members([canon, s0], "relabel") == [canon, s0]  # no identity reference: canonical
    assert tuple_members(answers, "syntax") is None  # canonical alone is no tuple
    assert tuple_reference([ident, s0]) is ident and tuple_reference([canon, s0]) is canon
    assert tuple_reference([canon, o0]) is canon
    assert tuple_factors("graph_as_code/native") == ("relabel", "order")
    assert tuple_factors("code/native") == ("relabel", "order", "structure", "syntax")
    assert [comparison_value(*a) for a in [(True, True, "3"), (False, True, "3"), (False, False, "null")]] == \
        ["CORRECT", "3", "NA"]
