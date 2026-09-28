"""E3: optimise the input by canonical labeling.

Relabel every graph with a deterministic, structure-derived labeling before rendering it. If the
labeling is canonical, every relabeling of an instance renders to the same prompt and the relabel
axis vanishes; the question left is whether that labeling costs or gains accuracy against the
dataset's native labels and against random relabelings.

Schemes:
  canonical  -- exact canonical labeling of the (graph, query) pair (gsi.serial.canonical): low
                degree first, remaining ties resolved by search, undirected edges written (low,
                high). Every relabeling renders to the same prompt; `--invariance-only` checks it.
  degree_wl  -- HEURISTIC: nodes sorted by (degree, 1-WL colour); remaining ties by the INPUT
                label, so a relabeled input can come out differently. Kept to reproduce the first
                E3 run; its residual sensitivity is measured, not assumed away.
  bfs_min    -- HEURISTIC: BFS from the smallest (degree, WL) key; ties by the input label.

    python scripts/canonical_labeling.py --config configs/graphqa.yaml --models m1 m2 [--limit N]
    python scripts/canonical_labeling.py --config configs/graphqa.yaml --invariance-only

Writes results/<cfg>/scored_canon/<scheme>/<model>.jsonl and tables/canonical_labeling.csv
(paired accuracy deltas); --invariance-only writes tables/canonical_labeling_invariance.csv (no
inference): for each scheme, the share of relabelings that render to the same prompt as the
canonical form does.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from gsi.data.base import GraphInstance, append_jsonl, read_jsonl
from gsi.exec import sandbox
from gsi.experiment.config import load_config, load_models
from gsi.experiment.run import iter_arms, load_dotenv, load_instances, load_variants, make_client, score_record
from gsi.prompts.modes import build_prompt
from gsi.serial.canonical import canonical_order as exact_order
from gsi.serial.render import render
from gsi.serial.variant import Variant, _derive, _sort_in_place

SCHEMES = ("canonical", "degree_wl", "bfs_min")


def wl_colours(nodes, adj, rounds=4):
    col = {u: len(adj[u]) for u in nodes}
    for _ in range(rounds):
        sig = {u: (col[u], tuple(sorted(col[w] for w in adj[u]))) for u in nodes}
        keys = {s: i for i, s in enumerate(sorted(set(sig.values())))}
        new = {u: keys[sig[u]] for u in nodes}
        if new == col:
            break
        col = new
    return col


def canonical_order(scheme: str, nodes: list[int], edges: list[list]) -> list[int]:
    adj = defaultdict(list)
    for e in edges:
        adj[e[0]].append(e[1]); adj[e[1]].append(e[0])
    col = wl_colours(nodes, adj)
    key = lambda u: (len(adj[u]), col[u], u)
    if scheme == "degree_wl":
        return sorted(nodes, key=key)
    if scheme == "bfs_min":
        seen, order = set(), []
        for start in sorted(nodes, key=key):
            if start in seen:
                continue
            q = deque([start]); seen.add(start)
            while q:
                u = q.popleft(); order.append(u)
                for w in sorted(adj[u], key=key):
                    if w not in seen:
                        seen.add(w); q.append(w)
        return order
    raise ValueError(scheme)


def canonical_variant(canon: Variant, scheme: str, query_args: dict | None = None) -> Variant:
    """Relabel `canon` so that the canonical order gets labels in the dataset's own label range.
    `canon` may be any variant of the instance; `query_args` (original labels) is needed by the
    exact scheme, which makes the question part of the canonical form."""
    old_labels = [canon.label_map[c] for c in sorted(canon.label_map)]
    target = sorted(old_labels)                       # keep the label set (0..n-1 or 1..n)
    if scheme == "canonical":
        query = [canon.label_map[v] for _, v in sorted((query_args or {}).items())]
        order = exact_order(list(canon.node_seq), [list(e) for e in canon.edge_seq], canon.directed, query)
    else:
        order = canonical_order(scheme, list(canon.node_seq), [list(e) for e in canon.edge_seq])
    new_of_old = {u: target[i] for i, u in enumerate(order)}
    label_map = {c: new_of_old[cur] for c, cur in canon.label_map.items()}
    node_seq = [new_of_old[u] for u in canon.node_seq]
    edge_seq = [[new_of_old[e[0]], new_of_old[e[1]], *e[2:]] for e in canon.edge_seq]
    if scheme == "canonical" and not canon.directed:
        # an undirected edge's orientation carries no meaning but does reach the text
        edge_seq = [[min(e[0], e[1]), max(e[0], e[1]), *e[2:]] for e in edge_seq]
    _sort_in_place(node_seq, edge_seq)
    v = _derive(canon, "relabel", f"canon_{scheme}", params={"seed": f"canon_{scheme}", "ordering": "sorted_st"},
                label_map=label_map, node_seq=node_seq, edge_seq=edge_seq)
    v.text = render(v)
    return v


def invariance(cfg, instances, variants, schemes) -> pd.DataFrame:
    """For each scheme and permutation axis: does every variant of an instance, once relabeled by
    the scheme, render to the same prompt as the canonical form does? Direct and code prompts are
    both compared, so the graph text and the question are covered. No inference."""
    rows = []
    for scheme in schemes:
        for axis in ("relabel", "order"):
            n_var = same_var = n_inst = same_inst = 0
            for inst in instances:
                vs = variants[inst.id]
                canon = next(v for v in vs if v.axis == "canonical")
                others = [v for v in vs if v.axis == axis]
                if not others:
                    continue

                def text(v):
                    cv = canonical_variant(v, scheme, inst.query_args)
                    return tuple(build_prompt(m, inst, cv, lib).user for m, lib in (("direct", None), ("code", "native")))
                ref = text(canon)
                same = [text(v) == ref for v in others]
                n_var += len(same); same_var += sum(same)
                n_inst += 1; same_inst += all(same)
            rows.append({"dataset": cfg.name, "scheme": scheme, "axis": axis, "n_instances": n_inst,
                         "n_variants": n_var, "variants_same_prompt": same_var / n_var if n_var else float("nan"),
                         "instances_all_same": same_inst / n_inst if n_inst else float("nan")})
    return pd.DataFrame(rows)


def merge_rows(old: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """Replace only the (model, scheme) rows this run computed, where they stand, and append the rest:
    a new model can be added with the exact scheme alone without dropping the other models' rows or
    the heuristic schemes, and a rerun rewrites the file byte for byte."""
    new = {key: g for key, g in out.groupby(["model", "scheme"], sort=False)}
    parts = [new.pop(key, g) for key, g in old.groupby(["model", "scheme"], sort=False)]
    return pd.concat(parts + list(new.values()), ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--limit", type=int, default=None, help="instances per dataset (strided)")
    ap.add_argument("--schemes", nargs="*", default=list(SCHEMES))
    ap.add_argument("--invariance-only", action="store_true",
                    help="measure each scheme's relabel/order invariance on every instance; no inference")
    args = ap.parse_args()
    cfg = load_config(args.config)
    instances = load_instances(cfg)
    variants = load_variants(cfg)
    if args.invariance_only:
        out = invariance(cfg, instances, variants, args.schemes)
        pd.set_option("display.width", 200)
        print(out.round(4).to_string(index=False))
        out.to_csv(cfg.results_dir / "tables" / "canonical_labeling_invariance.csv", index=False)
        return
    if not args.models:
        ap.error("--models is required unless --invariance-only")
    load_dotenv(cfg.root)
    specs = load_models(cfg.models_file)
    if args.limit:
        instances = instances[:: max(1, len(instances) // args.limit)][: args.limit]
    canon = {i.id: next(v for v in variants[i.id] if v.axis == "canonical") for i in instances}

    summary = []
    for name in args.models:
        spec = specs[name]
        client = make_client(spec, cfg.cache_dir)
        base = {r["record_id"]: r for r in read_jsonl(cfg.results_dir / "scored" / f"{name}.jsonl")}
        for scheme in args.schemes:
            out_path = cfg.results_dir / "scored_canon" / scheme / f"{name}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            done = {r["record_id"]: r.get("prompt_hash") for r in read_jsonl(out_path)} if out_path.exists() else {}
            jobs, stale = [], []
            for inst in instances:
                v = canonical_variant(canon[inst.id], scheme, inst.query_args)
                for mode, lib in iter_arms(cfg, inst):
                    p = build_prompt(mode, inst, v, lib)
                    rid = f"{p.prompt_id}::{name}"
                    if rid not in done:
                        jobs.append((rid, p, inst, v))
                    elif done[rid] != p.prompt_hash:
                        stale.append(rid)
            # resuming skips records by id, so records from an earlier version of a prompt would be kept
            if stale:
                raise SystemExit(f"{out_path}: {len(stale)} records come from another version of their prompt "
                                 f"(e.g. {stale[0]}); remove them and rerun")
            print(f"[{name}/{scheme}] {len(jobs)} prompts ({len(done)} done)", flush=True)

            def work(job):
                rid, p, inst, v = job
                r = client.complete(p, context={"inst": inst, "variant": v})
                ex = None
                if p.mode != "direct" and r.code:
                    ex = sandbox.run(r.code, v, p.mode, timeout=cfg.exec.get("timeout", 20),
                                     mem_mb=cfg.exec.get("mem_mb", 1024), cache_dir=cfg.cache_dir / "exec").to_dict()
                resp = {"record_id": rid, "mode": p.mode, "library": p.library, "model": name,
                        "raw_text": r.raw_text, "code": r.code, "usage": r.usage, "prompt_hash": r.prompt_hash}
                append_jsonl(out_path, score_record(inst, v, resp, ex))

            errors = 0
            with ThreadPoolExecutor(max_workers=max(1, spec.max_concurrency)) as pool:
                for n, f in enumerate(as_completed([pool.submit(work, j) for j in jobs]), 1):
                    try:
                        f.result()
                    except Exception as e:
                        errors += 1
                        if errors <= 3:
                            print(f"  error: {type(e).__name__}: {str(e)[:120]}", flush=True)
                    if n % 500 == 0:
                        print(f"  [{name}/{scheme}] {n}/{len(jobs)} ({errors} errors)", flush=True)
            if errors:
                print(f"  [{name}/{scheme}] {errors} errors (credits? re-run to resume)", flush=True)

            # paired comparison per instance x arm: canonical labeling vs native labels, and vs the
            # mean of the random relabelings the base run already has
            rows = list(read_jsonl(out_path))
            for arm in sorted({(r["mode"], r["library"] or "-") for r in rows}):
                mode, lib = arm
                d_native, d_random = [], []
                for r in rows:
                    if (r["mode"], r["library"] or "-") != arm:
                        continue
                    inst_id = r["instance_id"]
                    nat = base.get(f"{inst_id}::canonical::{mode}" + (f"::{lib}" if lib != "-" else "") + f"::{name}")
                    rnd = [b for k, b in base.items() if k.startswith(f"{inst_id}::relabel:s") and b["mode"] == mode
                           and (b["library"] or "-") == lib]
                    if nat is None:
                        continue
                    d_native.append(r["correct"] - nat["correct"])
                    if rnd:
                        d_random.append(r["correct"] - np.mean([b["correct"] for b in rnd]))
                def ci(x):
                    x = np.array(x, float)
                    if len(x) < 2:
                        return (float("nan"),) * 3
                    m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x))
                    return m, m - 1.96 * se, m + 1.96 * se
                mn, lo, hi = ci(d_native); mr, rlo, rhi = ci(d_random)
                summary.append({"dataset": cfg.name, "model": name, "scheme": scheme, "arm": f"{mode}/{lib}",
                                "n": len(d_native), "acc": np.mean([r["correct"] for r in rows
                                                                    if (r["mode"], r["library"] or "-") == arm]),
                                "delta_vs_native": mn, "ci_lo": lo, "ci_hi": hi,
                                "delta_vs_random_relabel": mr, "r_ci_lo": rlo, "r_ci_hi": rhi})
    out = pd.DataFrame(summary)
    pd.set_option("display.width", 220)
    print(f"\n=== E3: canonical labeling, paired accuracy deltas ({cfg.name}) ===")
    print(out.round(3).to_string(index=False))
    path = cfg.results_dir / "tables" / "canonical_labeling.csv"
    if path.exists() and not out.empty:
        out = merge_rows(pd.read_csv(path, float_precision="round_trip"), out)  # kept rows digit for digit
    out.to_csv(path, index=False)


if __name__ == "__main__":
    main()
