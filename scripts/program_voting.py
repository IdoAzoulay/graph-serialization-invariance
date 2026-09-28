"""E6: program voting. Do the *different* programs the model writes for permuted inputs compute
the same thing when given the same graph, and can they vote?

For each instance, take the program written for every relabel/order variant, remove its
`nodes`/`edges` declarations and execute it with the canonical graph injected -- the same edges in
the same order -- written in the labels that program expects (the variant's own naming). The
program text is not rewritten: its query literals already name the right nodes in that naming,
and a blanket rewrite of "the query label" also rewrites unrelated constants that happen to be
equal to it (`degree[u] += 1` when the question is about node 1). Node-valued answers are mapped
back to the original labels by the scorer, exactly as for the variant's own run.

For an `order` variant the naming is canonical, so the program runs on the canonical data
itself. For a relabeling it runs on the canonical data renamed; a program whose behaviour
depends on the label values (e.g. breaking ties by the smallest label) can still differ there,
which is part of what is measured. The sorted-identity form is a reference, not a perturbation,
and is skipped. Agreement on one graph does not prove two programs equivalent on every graph.

Then vote over the outputs; ties go to the canonical program's answer, as in answer voting
(permutation_signal.py). Local execution only; no inference.

    python scripts/program_voting.py --config configs/graphqa.yaml --models m1 m2 [--limit 300]

Writes results/<cfg>/tables/program_voting.csv.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from gsi.data.base import read_jsonl
from gsi.exec import sandbox
from gsi.experiment.config import load_config
from gsi.experiment.run import load_instances, load_variants, score_record
from gsi.serial.variant import Variant

NULL = (None, "null")
DECLARATIONS = ("nodes", "edges")


def _is_declaration(node: ast.stmt) -> bool:
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in DECLARATIONS
                                                for t in node.targets)


def strip_declarations(code: str) -> str | None:
    """The program without its top-level `nodes`/`edges` declarations, so the harness can inject
    the data. Nothing else is changed."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    tree.body = [n for n in tree.body if not _is_declaration(n)]
    return ast.unparse(tree)


def canonical_data_as(canon: Variant, v: Variant) -> Variant:
    """The canonical graph, in canonical node and edge order, written in `v`'s labels. Its
    label_map is `v`'s, so the scorer maps node-valued answers back to the original labels."""
    to_v = {canon.label_map[o]: v.label_map[o] for o in canon.label_map}
    return Variant(
        variant_id=f"{v.variant_id}::on_canonical", instance_id=canon.instance_id, axis=v.axis,
        params=dict(canon.params), label_map=dict(v.label_map),
        node_seq=[to_v[u] for u in canon.node_seq],
        edge_seq=[[to_v[e[0]], to_v[e[1]], *e[2:]] for e in canon.edge_seq],
        directed=canon.directed, weighted=canon.weighted)


def vote(keys: list, canonical_key) -> object | None:
    """Majority over parsable answer keys; a tie goes to the canonical answer if it is among the
    tied, otherwise to the smallest key, so the result never depends on execution order."""
    votes = Counter(k for k in keys if k not in NULL)
    if not votes:
        return None
    best = max(votes.values())
    tied = sorted(k for k, c in votes.items() if c == best)
    return canonical_key if canonical_key in tied else tied[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    cfg = load_config(args.config)
    instances = {i.id: i for i in load_instances(cfg)}
    picked = list(instances.values())[:: max(1, len(instances) // args.limit)][: args.limit]
    variants = load_variants(cfg)
    rows = []
    for name in args.models:
        resp = {r["record_id"]: r for r in read_jsonl(cfg.results_dir / "responses" / f"{name}.jsonl")}
        scored = {r["record_id"]: r for r in read_jsonl(cfg.results_dir / "scored" / f"{name}.jsonl")}
        jobs = []
        for inst in picked:
            canon = next(v for v in variants[inst.id] if v.axis == "canonical")
            for lib in ("native", "networkx"):
                for v in variants[inst.id]:
                    if v.axis not in ("relabel", "order") or v.is_reference:
                        continue
                    rid = f"{v.variant_id}::code::{lib}::{name}"
                    r = resp.get(rid)
                    if not r or not r.get("code"):
                        continue
                    code = strip_declarations(r["code"])
                    if code:
                        jobs.append((inst, canonical_data_as(canon, v), lib, rid, code))
        print(f"[{name}] executing {len(jobs)} programs on the canonical data", flush=True)
        results = defaultdict(list)

        def run(job):
            inst, data, lib, rid, code = job
            ex = sandbox.run(code, data, "graph_as_code", timeout=cfg.exec.get("timeout", 20),
                             cache_dir=cfg.cache_dir / "exec").to_dict()
            row = {"record_id": rid, "mode": "graph_as_code", "library": lib, "model": name,
                   "raw_text": "", "code": code, "usage": {}, "prompt_hash": None}
            rec = score_record(inst, data, row, ex)
            return (inst.id, lib), rid, rec["answer_key"], rec["correct"]

        with ThreadPoolExecutor(max_workers=cfg.exec.get("workers", 4)) as pool:
            for n, f in enumerate(as_completed([pool.submit(run, j) for j in jobs]), 1):
                key, rid, ans, ok = f.result()
                results[key].append((rid, ans, ok))
                if n % 1000 == 0:
                    print(f"  [{name}] {n}/{len(jobs)}", flush=True)

        for lib in ("native", "networkx"):
            n = n_prog = no_answer = agree_all = disagree = vote_ok = canon_ok = 0
            for (iid, l), outs in sorted(results.items()):
                if l != lib:
                    continue
                canon_rec = scored.get(f"{iid}::canonical::code::{lib}::{name}")
                if canon_rec is None:
                    continue
                outs = sorted(outs)
                n += 1
                n_prog += len(outs)
                no_answer += sum(a in NULL for _, a, _ in outs)
                canon_ok += canon_rec["correct"]
                keys = [a for _, a, _ in outs] + [canon_rec["answer_key"]]
                agree_all += all(k not in NULL for k in keys) and len(set(keys)) == 1
                disagree += len({k for k in keys if k not in NULL}) > 1
                top = vote(keys, canon_rec["answer_key"])
                if top is not None:
                    vote_ok += any(ok for _, a, ok in outs if a == top) or (
                        canon_rec["answer_key"] == top and canon_rec["correct"])
            if n:
                rows.append({"dataset": cfg.name, "model": name, "library": lib, "n_instances": n,
                             "n_programs": n_prog,
                             # every program (canonical included) returned the same parsable answer
                             "programs_agree_on_canonical_data": agree_all / n,
                             # at least two programs returned different parsable answers
                             "programs_disagree": disagree / n,
                             # share of the variants' programs that returned no parsable answer
                             "programs_no_answer": no_answer / n_prog if n_prog else float("nan"),
                             "canonical_acc": canon_ok / n, "program_vote_acc": vote_ok / n})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"\n=== E6: the variants' programs, run on the CANONICAL data ({cfg.name}) ===")
    print(out.round(3).to_string(index=False))
    out.to_csv(cfg.results_dir / "tables" / "program_voting.csv", index=False)


if __name__ == "__main__":
    main()
