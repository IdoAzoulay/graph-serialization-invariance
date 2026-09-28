"""E6 program voting (scripts/program_voting.py, loaded by path)."""
import importlib.util
from pathlib import Path

from gsi.data.base import GraphInstance
from gsi.exec import sandbox
from gsi.experiment.run import score_record
from gsi.serial.variant import canonical, relabel

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


program_voting = load("program_voting")

# node 3 has degree 2; the relabeling below renames 3 -> 1, colliding with the `+= 1` constant
INST = GraphInstance(id="t-node_degree", dataset="graphqa", task="node_degree", directed=False, weighted=False,
                     nodes=[0, 1, 2, 3], edges=[[0, 3], [1, 2], [2, 3]], query_args={"node": 3},
                     answer_type="int", ground_truth=2)
CANON = canonical(INST, structure="edge_list", syntax="plain")


def renamed(mapping):
    v = relabel(CANON, 0)
    v.label_map = dict(mapping)
    v.node_seq = sorted(mapping[u] for u in CANON.node_seq)
    v.edge_seq = sorted([mapping[a], mapping[b]] for a, b in CANON.edge_seq)
    return v


def test_declarations_are_stripped_and_nothing_else_is_rewritten():
    code = "nodes = [0, 1]\nedges = [(0, 1)]\ndeg = {}\nfor u, v in edges:\n    deg[u] = deg.get(u, 0) + 1\nans = deg[1]\n"
    out = program_voting.strip_declarations(code)
    assert "nodes =" not in out and "edges =" not in out
    assert "+ 1" in out and "deg[1]" in out


def test_a_query_label_equal_to_a_constant_is_not_corrupted():
    """The first E6 run rewrote every 1 to 3 here and scored 3x the degree."""
    v = renamed({0: 0, 1: 2, 2: 3, 3: 1})            # the question asks about node 1 in this naming
    code = ("degree = {node: 0 for node in nodes}\nfor u, w in edges:\n"
            "    degree[u] += 1\n    degree[w] += 1\nans = degree[1]\n")
    data = program_voting.canonical_data_as(CANON, v)
    assert sorted(map(tuple, data.edge_seq)) == sorted(map(tuple, v.edge_seq))  # same graph, v's naming
    ex = sandbox.run(program_voting.strip_declarations(code), data, "graph_as_code", timeout=5).to_dict()
    row = {"record_id": "r", "mode": "graph_as_code", "library": "native", "model": "m",
           "raw_text": "", "code": code, "usage": {}, "prompt_hash": None}
    rec = score_record(INST, data, row, ex)
    assert rec["parsed"] == 2 and rec["correct"]


def test_canonical_data_keeps_the_canonical_edge_order():
    v = renamed({0: 3, 1: 2, 2: 1, 3: 0})
    data = program_voting.canonical_data_as(CANON, v)
    back = {x: o for o, x in v.label_map.items()}
    assert [[back[a], back[b]] for a, b in data.edge_seq] == [list(e) for e in CANON.edge_seq]
    assert data.label_map == v.label_map


def test_vote_ties_never_depend_on_execution_order():
    assert program_voting.vote(['"a"', '"b"', '"c"'], '"b"') == '"b"'       # tie -> canonical answer
    assert program_voting.vote(['"b"', '"a"', "null"], "null") == '"a"'      # tie, no canonical -> smallest
    assert program_voting.vote(['"b"', '"a"', '"a"'], '"b"') == '"a"'        # a majority wins
    assert program_voting.vote(["null", None], "null") is None
