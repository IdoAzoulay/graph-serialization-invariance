"""E5 solver prompts (scripts/solver_per_task.py, loaded by path)."""
import importlib.util
from pathlib import Path

from gsi.data.base import GraphInstance
from gsi.exec import sandbox
from gsi.serial.variant import canonical, relabel

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


solver_per_task = load("solver_per_task")


def inst(directed: bool) -> GraphInstance:
    return GraphInstance(id=f"t-{directed}-neighbor", dataset="erdos", task="neighbor", directed=directed,
                         weighted=False, nodes=[0, 1, 2], edges=[[0, 1], [1, 2]], query_args={"node": 1},
                         answer_type="node_set", ground_truth=[0, 2])


def test_mixed_type_solver_takes_the_graph_type_as_an_argument():
    p = solver_per_task.solver_prompt(inst(False), "Which nodes neighbour node {node}?", "native", mixed_types=True)
    assert "solve(nodes, edges, directed, node)" in p.user
    assert solver_per_task.DIRECTED_ARG[:-1] in p.user
    # the prompt is built from one instance; it must not state that instance's type for all of them
    assert "The graph is directed" not in p.user
    assert p.user == solver_per_task.solver_prompt(inst(True), "Which nodes neighbour node {node}?", "native",
                                                   mixed_types=True).user


def test_single_type_solver_prompt_is_unchanged():
    p = solver_per_task.solver_prompt(inst(True), "Which nodes neighbour node {node}?", "native")
    assert "solve(nodes, edges, node)" in p.user
    assert "The graph is directed: (u, v) is an edge from u to v." in p.user
    assert "`directed`" not in p.user


def test_mixed_type_tasks_are_the_tasks_with_both_graph_types():
    insts = [inst(False), inst(True),
             GraphInstance(id="t-deg", dataset="erdos", task="degree", directed=False, weighted=False, nodes=[0, 1],
                           edges=[[0, 1]], query_args={"node": 0}, answer_type="int", ground_truth=1)]
    assert solver_per_task.mixed_type_tasks(insts) == {"neighbor"}


# out-degree of node 1 in 0->1, 1->2, 1->3, 2->3 is 2; a solver written the way five Qwen programs were,
# ending with its own call of `solve`
DIRECTED = GraphInstance(id="t-degree", dataset="erdos", task="degree", directed=True, weighted=False,
                         nodes=[0, 1, 2, 3], edges=[[0, 1], [1, 2], [1, 3], [2, 3]], query_args={"node": 1},
                         answer_type="int", ground_truth=2)
OWN_CALL = ("def solve(nodes, edges, directed, node):\n"
            "    if directed:\n"
            "        return sum(1 for u, v in edges if u == node)\n"
            "    return sum(1 for e in edges if node in e)\n"
            "\n"
            "ans = solve(nodes, edges, directed, node)\n")


def test_harness_call_passes_each_variants_type_and_relabelled_arguments():
    v = relabel(canonical(DIRECTED, structure="edge_list", syntax="plain"), seed=0)
    assert solver_per_task.solver_call(DIRECTED, v, True) == f"ans = solve(nodes, edges, directed=True, node={v.label_map[1]!r})"
    assert solver_per_task.solver_call(DIRECTED, v, False) == f"ans = solve(nodes, edges, node={v.label_map[1]!r})"


def test_a_programs_own_call_is_removed_and_nothing_else():
    code = ("import math\nLIMIT = 10\n\n"
            "def solve(nodes, edges, directed, node):\n"
            "    return 0 if not edges else 1 + solve(nodes, edges[1:], directed, node)\n\n"
            "ans = solve(nodes, edges, directed, node)\n")
    # the module-level call goes; the constant, the import and the call inside the function stay
    assert solver_per_task.without_own_calls(code) == code.replace("ans = solve(nodes, edges, directed, node)\n", "")
    assert solver_per_task.without_own_calls(OWN_CALL.split("\nans =")[0]) == OWN_CALL.split("\nans =")[0]
    assert solver_per_task.without_own_calls("def solve(:\n") == "def solve(:\n"  # unparsable: left to fail as it is


def test_a_program_with_its_own_call_answers_through_the_harness():
    v = relabel(canonical(DIRECTED, structure="edge_list", syntax="plain"), seed=0)
    call = solver_per_task.solver_call(DIRECTED, v, True)
    # as the programs were run before: their own call names `directed` and `node`, which are not globals
    assert sandbox.run(OWN_CALL + f"\n\n{call}\n", v, "graph_as_code", timeout=5).ans is None
    ex = sandbox.run(solver_per_task.without_own_calls(OWN_CALL) + f"\n\n{call}\n", v, "graph_as_code", timeout=5)
    assert ex.ans == "2"


# the same conflict obeyed inside the function: `ans` is set and never returned; the nested helper's
# own return must not count as the function's
NO_RETURN = ("def solve(nodes, edges, directed, node):\n"
             "    def count(pairs):\n"
             "        return sum(1 for u, v in pairs if u == node)\n"
             "    if directed:\n"
             "        ans = count(edges)\n"
             "    else:\n"
             "        ans = sum(1 for e in edges if node in e)\n")


def test_a_function_that_only_sets_ans_returns_it():
    fixed = solver_per_task.returning_ans(NO_RETURN)
    assert fixed == NO_RETURN + "    return ans\n"
    v = relabel(canonical(DIRECTED, structure="edge_list", syntax="plain"), seed=0)
    call = solver_per_task.solver_call(DIRECTED, v, True)
    assert sandbox.run(NO_RETURN + f"\n\n{call}\n", v, "graph_as_code", timeout=5).ans == "None"
    assert sandbox.run(fixed + f"\n\n{call}\n", v, "graph_as_code", timeout=5).ans == "2"


def test_a_function_with_a_return_of_its_own_is_left_as_written():
    early = NO_RETURN.replace("    if directed:\n", "    if not edges:\n        return 0\n    if directed:\n")
    assert solver_per_task.returning_ans(early) == early
    assert solver_per_task.returning_ans(OWN_CALL) == OWN_CALL


def test_ans_bound_only_in_a_comprehension_or_a_one_line_def_is_left_alone():
    comp = "def solve(nodes, edges):\n    print([ans for ans in nodes])\n"
    one_line = "def solve(nodes, edges): ans = len(nodes)\n"
    assert solver_per_task.returning_ans(comp) == comp
    assert solver_per_task.returning_ans(one_line) == one_line


def test_a_resumed_run_re_executes_records_made_with_other_code_or_another_prompt():
    v = relabel(canonical(DIRECTED, structure="edge_list", syntax="plain"), seed=0)
    now = solver_per_task.code_sha(solver_per_task.harness_code(NO_RETURN, DIRECTED, v, True))
    before = solver_per_task.code_sha(NO_RETURN + f"\n\n{solver_per_task.solver_call(DIRECTED, v, True)}\n")
    assert now != before  # the harness fix changes what runs, not the prompt
    expected = {"a": (now, "p"), "b": (now, "p"), "c": (now, "p"), "d": (now, "p")}
    records = [{"record_id": "a", "exec_sha": now, "prompt_hash": "p"},        # current: kept
               {"record_id": "b", "exec_sha": before, "prompt_hash": "p"},     # older harness
               {"record_id": "c", "prompt_hash": "p"},                         # older script, no hash
               {"record_id": "d", "exec_sha": now, "prompt_hash": "old"},      # older prompt
               {"record_id": "e", "exec_sha": now, "prompt_hash": "p"}]        # no longer a job
    kept, n_stale = solver_per_task.split_stale(records, expected)
    assert [r["record_id"] for r in kept] == ["a"] and n_stale == 4
