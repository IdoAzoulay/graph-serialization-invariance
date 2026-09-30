"""E1/E2 permutation detection and voting (mitigation/scripts/permutation_signal.py, loaded by path)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


permutation_signal = load("permutation_signal")

def _row(axis, key, correct, vid, seed=None):
    return {"mode": "code", "library": "native", "instance_id": "i", "axis": axis, "variant_id": vid,
            "answer_key": key, "correct": correct, "params": {"seed": seed}}


def test_identity_is_not_a_relabeling(tmp_path):
    rows = [_row("canonical", "1", True, "i::canonical"),
            _row("relabel", "1", True, "i::relabel:identity", "identity"),
            _row("relabel", "2", False, "i::relabel:s0", 0),
            _row("order", "1", True, "i::order:x")]
    (tmp_path / "scored").mkdir()
    (tmp_path / "scored" / "m.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    groups = permutation_signal.load(SimpleNamespace(results_dir=tmp_path), "m")
    d = groups[("code/native", "i")]
    assert [r["variant_id"] for r in d["relabel"]] == ["i::relabel:s0"]
    assert len(d["order"]) == 1


def test_recall_on_answered_errors_excludes_crashes():
    canon_crash = {"canon": _row("canonical", "null", False, "a::canonical"),
                   "relabel": [_row("relabel", "null", False, "a::relabel:s0", 0)], "order": []}
    silent_caught = {"canon": _row("canonical", "5", False, "b::canonical"),
                     "relabel": [_row("relabel", "4", True, "b::relabel:s0", 0)], "order": []}
    silent_missed = {"canon": _row("canonical", "5", False, "c::canonical"),
                     "relabel": [_row("relabel", "5", False, "c::relabel:s0", 0)], "order": []}
    right = {"canon": _row("canonical", "4", True, "d::canonical"),
             "relabel": [_row("relabel", "4", True, "d::relabel:s0", 0)], "order": []}
    groups = {("code/native", x): g for x, g in zip("abcd", (canon_crash, silent_caught, silent_missed, right))}
    e = permutation_signal.evaluate(groups, {"relabel"}, None)
    assert e["errors"] == 3 and e["caught"] == 2 and abs(e["recall"] - 2 / 3) < 1e-9
    assert e["errors_answered"] == 2 and e["caught_answered"] == 1 and e["recall_answered"] == 0.5


def test_flip_rule_counts_correct_answers_as_one_and_missing_answers_as_one():
    two_correct = {"canon": _row("canonical", "[0, 1, 2]", True, "a::canonical"),
                   "relabel": [_row("relabel", "[0, 3, 2]", True, "a::relabel:s0", 0)], "order": []}
    all_crash = {"canon": _row("canonical", "null", False, "b::canonical"),
                 "relabel": [_row("relabel", "null", False, "b::relabel:s0", 0)], "order": []}
    crash_vs_answer = {"canon": _row("canonical", "null", False, "c::canonical"),
                       "relabel": [_row("relabel", "4", True, "c::relabel:s0", 0)], "order": []}
    silent_caught = {"canon": _row("canonical", "5", False, "d::canonical"),
                     "relabel": [_row("relabel", "4", True, "d::relabel:s0", 0)], "order": []}
    same_wrong = {"canon": _row("canonical", "5", False, "e::canonical"),
                  "relabel": [_row("relabel", "5", False, "e::relabel:s0", 0)], "order": []}
    cases = (two_correct, all_crash, crash_vs_answer, silent_caught, same_wrong)
    flags = {rule: [permutation_signal.disagree([g["canon"]] + g["relabel"], rule) for g in cases]
             for rule in permutation_signal.RULES}
    assert flags == {"literal": [True, True, True, True, False], "flip": [False, False, True, True, False]}
    groups = {("code/native", x): g for x, g in zip("abcde", cases)}
    lit = permutation_signal.evaluate(groups, {"relabel"}, None)
    flip = permutation_signal.evaluate(groups, {"relabel"}, None, "flip")
    assert lit["flag_rate"] == 0.8 and flip["flag_rate"] == 0.4
    # the flags on answered errors do not depend on the rule; the vote compares keys under both
    assert lit["caught_answered"] == flip["caught_answered"] == 1 and lit["errors_answered"] == 2
    assert lit["vote_acc"] == flip["vote_acc"]
    # the all-crash instance is no longer flagged: abstaining keeps it, and it is wrong
    assert lit["abstain_coverage"] == 0.2 and flip["abstain_coverage"] == 0.6
    assert flip["abstain_acc"] == 1 / 3 and lit["abstain_acc"] == 0.0
