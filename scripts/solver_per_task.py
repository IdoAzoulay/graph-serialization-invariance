"""E5: one solver per task. Decouple the program from the instance.

The model is asked once per (task, library) for a function `solve(nodes, edges, ...)` with no
graph in the prompt. That single program is then executed on every variant of every instance of
the task through the existing sandbox. Permutation invariance of the *prompt* is 1.0 by
construction; what is measured is (a) the accuracy cost of removing the instance from the prompt
and (b) whether the model's algorithm itself is invariant when run on permuted data.

    python scripts/solver_per_task.py --config configs/graphqa.yaml --models m1 m2

Writes results/<cfg>/scored_solver/<model>.jsonl (one row per variant x library, same schema as
scored/) and prints accuracy on canonical forms and invariance on the permutation axes.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from gsi.analysis.tables import invariance_table
from gsi.data.base import GraphInstance, append_jsonl, read_jsonl
from gsi.exec import sandbox
from gsi.exec.sandbox import atomic_write_text
from gsi.experiment.config import load_config, load_models
from gsi.experiment.run import load_dotenv, load_instances, load_variants, make_client, score_record
from gsi.prompts.modes import (ANS_HINT, CODEGRAPH_INSTRUCTION, LIBRARIES, LIBRARY_INSTRUCTION, ROLE, Prompt,
                               _graph_notes, task_line)
from gsi.serial.variant import Variant


def question_template(inst: GraphInstance, q: str) -> str:
    """Turn 'What is the degree of node 13?' into 'What is the degree of node {node}?'."""
    for k, v in sorted(inst.query_args.items(), key=lambda kv: -len(str(kv[1]))):
        q = q.replace(str(v), "{" + k + "}")
    return q


DIRECTED_ARG = ("`directed` is True when the graph is directed, so (u, v) is an edge from u to v, and "
                "False when it is undirected, so (u, v) and (v, u) denote the same edge; the function "
                "is called on graphs of both kinds.")


def solver_prompt(inst: GraphInstance, q_template: str, library: str, mixed_types: bool = False) -> Prompt:
    """`mixed_types`: the task has directed and undirected instances. One program then serves both, so
    it receives the graph type as an argument instead of reading it from one instance's prompt."""
    keys = list(inst.query_args)
    params = ", ".join(["nodes", "edges", *(["directed"] if mixed_types else []), *keys])
    pair = "(u, v, w) triples, w being the edge weight" if inst.weighted else "(u, v) pairs"
    line = task_line(inst)
    if mixed_types and not inst.meta.get("preamble"):
        # the generic task line names this instance's graph type; the dataset preambles do not
        line = line.replace(f"the {'directed' if inst.directed else 'undirected'} graph G", "the graph G")
    user = "\n\n".join([
        line + ("" if mixed_types else _graph_notes(inst)),
        f"Write a Python function `solve({params})` that returns the answer for ANY graph of this "
        f"kind. `nodes` is a list of node labels and `edges` a list of {pair}"
        + (f"; {DIRECTED_ARG[:-1]}" if mixed_types else "")
        + (f"; {', '.join(keys)} " + ("is the question's argument" if len(keys) == 1 else "are the question's arguments")
           if keys else "") + ". The function will be called on many different graphs, so it must not "
        "assume anything about the labels, their order, or the order of the edges.",
        f"The question it answers, for a graph G: {q_template}",
        f"Return {ANS_HINT[inst.answer_type]}. Define only the function; do not call it. "
        + CODEGRAPH_INSTRUCTION,
        LIBRARY_INSTRUCTION[library],
    ])
    h = hashlib.sha256((ROLE + "\n\x00\n" + user).encode()).hexdigest()
    return Prompt(prompt_id=f"solver::{inst.dataset}::{inst.task}::{library}", instance_id=inst.id,
                  variant_id="solver", mode="code", library=library, system=ROLE, user=user,
                  answer_type=inst.answer_type, prompt_hash=h)


def mixed_type_tasks(instances) -> set[str]:
    """Tasks whose instances mix directed and undirected graphs. One program serves both kinds, so the
    solver takes the graph type as an argument and is called with each graph's type."""
    kinds: dict[str, set[bool]] = {}
    for i in instances:
        kinds.setdefault(i.task, set()).add(i.directed)
    return {t for t, k in kinds.items() if len(k) > 1}


def solver_call(inst: GraphInstance, v: Variant, mixed_types: bool) -> str:
    """The harness's call: the variant's graph, its type for a mixed-type task, and the question's
    arguments in the variant's labels."""
    q = {k: v.label_map.get(val, val) for k, val in inst.query_args.items()}
    args = ["nodes", "edges"] + ([f"directed={v.directed!r}"] if mixed_types else []) + [f"{k}={val!r}" for k, val in q.items()]
    return f"ans = solve({', '.join(args)})"


def without_own_calls(code: str) -> str:
    """Drop the program's own module-level calls of `solve`. The prompt asks for a function that returns
    the answer, but it also carries the generic instruction to leave the answer in `ans`, and a program
    that obeys that with `ans = solve(nodes, edges, directed, node)` fails before the harness's call:
    `directed` and the question's arguments are arguments, not globals. Only those statements go; the
    rest of the text is kept as written."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    drop = set()
    for st in tree.body:
        if isinstance(st, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "solve" for x in ast.walk(st)):
            drop.update(range(st.lineno, st.end_lineno + 1))
    if not drop:
        return code
    return "\n".join(line for n, line in enumerate(code.split("\n"), 1) if n not in drop)


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
           ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _own_nodes(fn: ast.FunctionDef):
    """The nodes of a function's own scope: not those of functions, classes or comprehensions inside it."""
    stack = list(fn.body)
    while stack:
        n = stack.pop()
        yield n
        if not isinstance(n, _SCOPES):
            stack.extend(ast.iter_child_nodes(n))


def returning_ans(code: str) -> str:
    """The same instruction conflict, obeyed inside the function: a `solve` that assigns `ans` and has
    no `return` at all hands the harness None. Such a function gets a final `return ans`; a function
    with any `return` of its own is left as written."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "solve"]
    if not fns:
        return code
    own = list(_own_nodes(fns[-1]))
    if fns[-1].body[0].lineno == fns[-1].lineno:  # `def solve(...): ...` on one line
        return code
    if any(isinstance(n, ast.Return) for n in own) or not any(
            isinstance(n, ast.Name) and n.id == "ans" and isinstance(n.ctx, ast.Store) for n in own):
        return code
    lines = code.split("\n")
    first = fns[-1].body[0]
    indent = lines[first.lineno - 1][:first.col_offset]
    lines.insert(fns[-1].end_lineno, indent + "return ans")
    return "\n".join(lines)


def harness_code(program: str, inst: GraphInstance, v: Variant, mixed_types: bool) -> str:
    """Exactly what the sandbox executes for one variant: the program as the harness adapts it, then
    the harness's call."""
    return returning_ans(without_own_calls(program)) + f"\n\n{solver_call(inst, v, mixed_types)}\n"


def code_sha(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def split_stale(records: list[dict], expected: dict[str, tuple[str, str]]) -> tuple[list[dict], int]:
    """Resuming skips records that are done, so each record carries the hash of the code it executed
    and its prompt's hash. A record is kept only if both still match what this run would execute
    (`expected`: record_id -> (exec_sha, prompt_hash)); a change to a prompt, a program or the harness
    itself therefore re-executes the records it affects instead of keeping their old scores."""
    kept = [r for r in records
            if r["record_id"] in expected and (r.get("exec_sha"), r.get("prompt_hash")) == expected[r["record_id"]]]
    return kept, len(records) - len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    load_dotenv(cfg.root)
    specs = load_models(cfg.models_file)
    instances = load_instances(cfg)
    variants = load_variants(cfg)
    by_task: dict[str, list[GraphInstance]] = {}
    for i in instances:
        by_task.setdefault(i.task, []).append(i)
    mixed = mixed_type_tasks(instances)
    from gsi.prompts.modes import question_for

    summary_rows = []
    for name in args.models:
        spec = specs[name]
        client = make_client(spec, cfg.cache_dir)
        out_path = cfg.results_dir / "scored_solver" / f"{name}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        programs = {}
        for task, insts in by_task.items():
            first = insts[0]
            canon = next(v for v in variants[first.id] if v.axis == "canonical")
            q_t = question_template(first, question_for(first, canon.label_map))
            for lib in LIBRARIES:
                p = solver_prompt(first, q_t, lib, mixed_types=task in mixed)
                r = client.complete(p, context={"inst": first, "variant": canon})
                programs[(task, lib)] = r
                print(f"[{name}] {task}/{lib}: {'program' if r.code else 'NO CODE'} "
                      f"({len(r.code or '')} chars, {'cached' if r.cached else 'new'})", flush=True)
        jobs, expected = [], {}
        for inst in instances:
            for v in variants[inst.id]:
                for lib in LIBRARIES:
                    r = programs[(inst.task, lib)]
                    if r.code:
                        rid = f"{v.variant_id}::solver::{lib}::{name}"
                        code = harness_code(r.code, inst, v, inst.task in mixed)
                        expected[rid] = (code_sha(code), r.prompt_hash)
                        jobs.append((rid, inst, v, lib, code))
        records = list(read_jsonl(out_path)) if out_path.exists() else []
        kept, n_stale = split_stale(records, expected)
        if n_stale:
            print(f"[{name}] {n_stale} records were executed with other code or another prompt: re-executing them",
                  flush=True)
            atomic_write_text(out_path, "".join(json.dumps(r) + "\n" for r in kept))
        done = {r["record_id"] for r in kept}
        jobs = [j for j in jobs if j[0] not in done]
        print(f"[{name}] executing the task solvers on {len(jobs)} variants ({len(done)} already done)", flush=True)

        def run(job):
            rid, inst, v, lib, code = job
            r = programs[(inst.task, lib)]
            ex = sandbox.run(code, v, "graph_as_code", timeout=cfg.exec.get("timeout", 20),
                             mem_mb=cfg.exec.get("mem_mb", 1024), cache_dir=cfg.cache_dir / "exec")
            resp = {"record_id": rid, "mode": "graph_as_code", "library": lib, "model": name,
                    "raw_text": r.raw_text, "code": code, "usage": r.usage, "prompt_hash": r.prompt_hash}
            row = score_record(inst, v, resp, ex.to_dict())
            row["mode"] = "solver"
            row["exec_sha"] = code_sha(code)
            append_jsonl(out_path, row)

        with ThreadPoolExecutor(max_workers=cfg.exec.get("workers", 4)) as pool:
            for n, f in enumerate(as_completed([pool.submit(run, j) for j in jobs]), 1):
                f.result()
                if n % 2000 == 0:
                    print(f"  [{name}] {n}/{len(jobs)}", flush=True)

        def prep(frame: pd.DataFrame) -> pd.DataFrame:
            # what load_scored() adds before invariance_table() can run
            frame["library"] = frame["library"].fillna("-")
            frame["parsed_num"] = pd.to_numeric(
                frame["parsed_canonical"].where(frame["answer_type"].isin(["int", "float"])), errors="coerce")
            return frame

        rows = list(read_jsonl(out_path))
        df = pd.DataFrame(rows)
        if "library" not in df.columns:
            continue
        df = prep(df)
        base = prep(pd.DataFrame(list(read_jsonl(cfg.results_dir / "scored" / f"{name}.jsonl"))))
        for lib in LIBRARIES:
            s = df[(df.library == lib)]
            b = base[(base["mode"] == "code") & (base.library == lib)]
            acc_s = s[s.axis == "canonical"].correct.mean()
            acc_b = b[b.axis == "canonical"].correct.mean()
            inv = invariance_table(s.assign(model=name))
            inv_perm = inv[inv.axis.isin(["relabel", "order"])]
            inv_b = invariance_table(b.assign(model=name))
            inv_bp = inv_b[inv_b.axis.isin(["relabel", "order"])]
            # a task whose single program never returns a parsable answer (a crash on every graph)
            # fails the all-parsed rule of the invariance metric on every instance: that, not an
            # order-dependent algorithm, is what pulls solver invariance down for such a model
            perm = s[s.axis.isin(["canonical", "relabel", "order"])]
            dead = perm.groupby("task")["answer_key"].apply(lambda k: k.isin([None, "null"]).all())
            summary_rows.append({
                "dataset": cfg.name, "model": name, "library": lib,
                "solver_acc_canonical": acc_s, "M2_acc_canonical": acc_b, "delta_acc": acc_s - acc_b,
                "tasks_no_answer": int(dead.sum()),
                "solver_perm_invariance": (inv_perm.frac_identical * inv_perm.n_instances).sum() / inv_perm.n_instances.sum(),
                "M2_perm_invariance": (inv_bp.frac_identical * inv_bp.n_instances).sum() / inv_bp.n_instances.sum(),
                "n_tasks": len(by_task),
            })
    out = pd.DataFrame(summary_rows)
    pd.set_option("display.width", 200)
    print("\n=== E5: one solver per task vs per-instance M2 ===")
    print(out.round(3).to_string(index=False))
    out.to_csv(cfg.results_dir / "tables" / "solver_per_task.csv", index=False)


if __name__ == "__main__":
    main()
