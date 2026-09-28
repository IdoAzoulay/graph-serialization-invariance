"""Stages: data -> variants -> llm -> exec -> score. Every stage is idempotent:
LLM responses are cached by prompt hash, executions by code hash, and each
JSONL sink skips record_ids it already holds."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gsi.data.base import GraphInstance, append_jsonl, read_jsonl, write_jsonl
from gsi.exec import sandbox
from gsi.experiment.config import ExperimentConfig, load_config, load_models
from gsi.llm.client import LLMClient, ModelSpec
from gsi.llm.stub import StubLLM
from gsi.prompts.modes import CODE_MODES, build_prompt
from gsi.score.classify import classify
from gsi.score.compare import compare, normalize_answer, relabel_answer
from gsi.score.parse_answer import find_boxed_answer, parse_value
from gsi.serial.render import render
from gsi.serial.variant import Variant, canonical, make_variants

STAGES = ("data", "variants", "llm", "exec", "score")
ERROR_ABORT_MIN, ERROR_ABORT_FRACTION = 25, 0.5      # abort the llm stage past this error rate


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_dotenv(root: Path) -> None:
    p = root / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def arm_label(mode: str, library: str | None) -> str:
    return mode if library is None else f"{mode}/{library}"


def iter_arms(cfg: ExperimentConfig, inst: GraphInstance):
    """(mode, library) pairs to run for one instance. `library` is crossed with the code
    modes rather than baked into the mode name; `direct` yields library None."""
    for mode in cfg.modes:
        for library in cfg.arms_for(mode, inst.task):
            yield mode, library


def estimate(cfg: ExperimentConfig, instances, variants, model_names: list[str]) -> None:
    """Prompt counts and rough token volume (chars/4) per arm; M3 prompts dedupe by hash."""
    by_arm: dict[str, dict] = {}
    for inst in instances:
        for v in variants[inst.id]:
            for mode, library in iter_arms(cfg, inst):
                p = build_prompt(mode, inst, v, library, inject_graph_type=cfg.inject_graph_type)
                d = by_arm.setdefault(arm_label(mode, library), {"prompts": 0, "unique": set(), "chars": 0})
                d["prompts"] += 1
                if p.prompt_hash not in d["unique"]:
                    d["unique"].add(p.prompt_hash)
                    d["chars"] += len(p.system) + len(p.user)
    total_unique = sum(len(d["unique"]) for d in by_arm.values())
    total_tokens = sum(d["chars"] for d in by_arm.values()) / 4
    print(f"config {cfg.name}: {len(instances)} instances, {sum(len(v) for v in variants.values())} variants")
    for arm, d in by_arm.items():
        print(f"  {arm:24s} {d['prompts']:7d} prompts, {len(d['unique']):7d} unique, ~{d['chars'] / 4 / 1e6:.2f}M input tokens")
    print(f"  per model: {total_unique} API calls, ~{total_tokens / 1e6:.2f}M input tokens (+ outputs); "
          f"x {len(model_names)} models = {total_unique * len(model_names)} calls")


# ---------------- data & variants ----------------

def build_instances(cfg: ExperimentConfig) -> list[GraphInstance]:
    out: list[GraphInstance] = []
    for name, ds in cfg.datasets.items():
        if name == "graphqa":
            from gsi.data.graphqa import GENERATORS, TASKS, build_graphqa
            out += build_graphqa(ds["n_graphs"], seed=ds.get("seed", 0),
                                 generators=tuple(ds.get("generators", GENERATORS)),
                                 tasks=tuple(ds.get("tasks", TASKS)))
        elif name == "erdos":
            from gsi.data.erdos import build_erdos
            out += build_erdos(**{k: v for k, v in ds.items() if k not in ("canonical", "variants")})
        else:
            raise ValueError(f"unknown dataset {name}")
    write_jsonl(cfg.data_dir / "instances.jsonl", (i.to_dict() for i in out))
    return out


def build_variants(cfg: ExperimentConfig, instances: list[GraphInstance]) -> dict[str, list[Variant]]:
    out: dict[str, list[Variant]] = {}
    rows = []
    for inst in instances:
        canon = canonical(inst, **cfg.canonical_for(inst.dataset))
        vs = make_variants(inst, canon, cfg.variants_for(inst.dataset))
        for v in vs:
            v.text = render(v)
            rows.append(v.to_dict())
        out[inst.id] = vs
    write_jsonl(cfg.data_dir / "variants.jsonl", rows)
    return out


def load_instances(cfg: ExperimentConfig) -> list[GraphInstance]:
    return [GraphInstance.from_dict(d) for d in read_jsonl(cfg.data_dir / "instances.jsonl")]


def load_variants(cfg: ExperimentConfig) -> dict[str, list[Variant]]:
    out: dict[str, list[Variant]] = {}
    for d in read_jsonl(cfg.data_dir / "variants.jsonl"):
        v = Variant.from_dict(d)
        out.setdefault(v.instance_id, []).append(v)
    return out


# ---------------- sinks ----------------

class Sink:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.ids = {r["record_id"] for r in read_jsonl(path)}

    def has(self, record_id: str) -> bool:
        return record_id in self.ids

    def write(self, row: dict) -> None:
        with self.lock:
            if row["record_id"] in self.ids:
                return
            append_jsonl(self.path, row)
            self.ids.add(row["record_id"])


def make_client(spec: ModelSpec, cache_dir: Path):
    return StubLLM(spec) if spec.provider == "stub" else LLMClient(spec, cache_dir)


# ---------------- llm stage ----------------

def stage_llm(cfg: ExperimentConfig, spec: ModelSpec, instances, variants, limit: int | None = None) -> None:
    client = make_client(spec, cfg.cache_dir)
    sink = Sink(cfg.results_dir / "responses" / f"{spec.name}.jsonl")
    jobs = []
    for inst in instances:
        for v in variants[inst.id]:
            for mode, library in iter_arms(cfg, inst):
                p = build_prompt(mode, inst, v, library, inject_graph_type=cfg.inject_graph_type)
                rid = f"{p.prompt_id}::{spec.name}"
                if not sink.has(rid):
                    jobs.append((rid, p, inst, v))
    if limit:
        jobs = jobs[:limit]
    log(f"[llm:{spec.name}] {len(jobs)} prompts to run ({len(sink.ids)} already done)")
    errors = 0

    def work(job):
        rid, p, inst, v = job
        r = client.complete(p, context={"inst": inst, "variant": v})
        row = {"record_id": rid, "instance_id": inst.id, "variant_id": v.variant_id, "mode": p.mode,
               "library": p.library, **r.to_dict()}
        sink.write(row)
        return r.cached

    with ThreadPoolExecutor(max_workers=max(1, spec.max_concurrency)) as pool:
        futs = {pool.submit(work, j): j for j in jobs}
        done = cached = 0
        for fut in as_completed(futs):
            try:
                cached += bool(fut.result())
            except Exception as e:
                errors += 1
                log(f"[llm:{spec.name}] error on {futs[fut][0]}: {e!r}")
            done += 1
            if done % 50 == 0 or done == len(jobs):
                log(f"[llm:{spec.name}] {done}/{len(jobs)} done, {cached} from cache, {errors} errors")
            # A sustained error rate means something systemic -- exhausted credits (402), a dead
            # endpoint, a bad key -- and spinning through the remaining jobs only fills the log.
            # Stop; the sink has everything that succeeded and a re-run resumes from there.
            if errors >= ERROR_ABORT_MIN and errors >= ERROR_ABORT_FRACTION * done:
                pool.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(f"[llm:{spec.name}] aborting: {errors} errors in {done} completions "
                                   f"({len(jobs) - done} jobs cancelled); fix the cause and re-run to resume")


# ---------------- exec stage ----------------

def stage_exec(cfg: ExperimentConfig, spec: ModelSpec, variants) -> None:
    by_id = {v.variant_id: v for vs in variants.values() for v in vs}
    responses = [r for r in read_jsonl(cfg.results_dir / "responses" / f"{spec.name}.jsonl") if r["mode"] in CODE_MODES]
    sink = Sink(cfg.results_dir / "exec" / f"{spec.name}.jsonl")
    jobs = [r for r in responses if r.get("code") and not sink.has(r["record_id"])]
    log(f"[exec:{spec.name}] {len(jobs)} programs to run ({len(sink.ids)} already done)")
    timeout, mem = cfg.exec.get("timeout", 20), cfg.exec.get("mem_mb", 1024)

    def work(r):
        ex = sandbox.run(r["code"], by_id[r["variant_id"]], r["mode"], timeout=timeout, mem_mb=mem,
                         cache_dir=cfg.cache_dir / "exec")
        sink.write({"record_id": r["record_id"], **ex.to_dict()})

    with ThreadPoolExecutor(max_workers=cfg.exec.get("workers", os.cpu_count() or 2)) as pool:
        futs = [pool.submit(work, r) for r in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if i % 50 == 0 or i == len(jobs):
                log(f"[exec:{spec.name}] {i}/{len(jobs)} done")


# ---------------- score stage ----------------

def score_record(inst: GraphInstance, v: Variant, resp: dict, ex: dict | None) -> dict:
    mode, t = resp["mode"], inst.answer_type
    if mode in CODE_MODES:
        # CodeGraph contract: the answer is the variable `ans`; nothing else is consulted.
        line = ex.get("ans") if (ex and resp.get("code")) else None
        parsed = parse_value(line, t)
    else:
        line = find_boxed_answer(resp["raw_text"])
        parsed = parse_value(line, t)
    pred = relabel_answer(parsed, t, v.inverse_label_map)
    correct = compare(pred, inst.ground_truth, t, G=inst.to_nx()) if parsed is not None else False
    if mode in CODE_MODES and not resp.get("code"):
        fclass = "format"
    else:
        fclass = classify(mode, correct, parsed, ex)
    key = normalize_answer(pred, t, inst.directed) if parsed is not None else None
    return {
        "record_id": resp["record_id"], "dataset": inst.dataset, "task": inst.task,
        "graph_id": inst.meta.get("graph_id", inst.id), "instance_id": inst.id, "variant_id": v.variant_id,
        "axis": v.axis, "params": v.params, "mode": mode, "library": resp.get("library"),
        "model": resp["model"], "answer_type": t,
        # Rows sharing a prompt_hash came from ONE model call. M3 prompts for a task whose
        # question names no node are identical across every graph and every variant, so a
        # cell can hold 100 rows backed by a single response; the tables report both counts.
        "prompt_hash": resp.get("prompt_hash"),
        # "length" = the reply was cut off at max_tokens. For a degenerate reply (Qwen3-8B in
        # non-thinking mode loops on some prompts at temperature 0) this is a model defect, not a
        # serialization effect; the analysis needs to be able to see it to say so.
        "finish_reason": (resp.get("usage") or {}).get("finish_reason"),
        "ground_truth": inst.ground_truth, "answer_line": line, "parsed": parsed, "parsed_canonical": pred,
        "answer_key": json.dumps(key), "correct": bool(correct), "failure_class": fclass,
        "graph_match": ex.get("graph_match") if ex else None, "n_graphs": len(ex["graphs"]) if ex else None,
        "declared_ok": ex.get("declared_ok") if ex else None,
        "construction_ok": ex.get("construction_ok") if ex else None,
        "ans_is_literal": ex.get("ans_is_literal") if ex else None,
        "reads_graph_vars": ex.get("reads_graph_vars") if ex else None,
        "timed_out": ex.get("timed_out") if ex else None, "exit_code": ex.get("exit_code") if ex else None,
        "directedness_mismatch": ex.get("directedness_mismatch") if ex else None,
        "stub_behaviour": (resp.get("usage") or {}).get("stub_behaviour"),
    }


def stage_score(cfg: ExperimentConfig, spec: ModelSpec, instances, variants) -> None:
    inst_by_id = {i.id: i for i in instances}
    var_by_id = {v.variant_id: v for vs in variants.values() for v in vs}
    execs = {r["record_id"]: r for r in read_jsonl(cfg.results_dir / "exec" / f"{spec.name}.jsonl")}
    rows = []
    for resp in read_jsonl(cfg.results_dir / "responses" / f"{spec.name}.jsonl"):
        inst, v = inst_by_id.get(resp["instance_id"]), var_by_id.get(resp["variant_id"])
        if inst is None or v is None:
            continue
        ex = execs.get(resp["record_id"])
        if resp["mode"] in CODE_MODES and resp.get("code") and ex is None:
            continue
        rows.append(score_record(inst, v, resp, ex))
    write_jsonl(cfg.results_dir / "scored" / f"{spec.name}.jsonl", rows)
    n_ok = sum(r["correct"] for r in rows)
    log(f"[score:{spec.name}] {len(rows)} records scored, accuracy {n_ok / max(1, len(rows)):.3f}")


# ---------------- main ----------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", help="override the model list from the config")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--limit", type=int, help="max new prompts per model (debugging)")
    ap.add_argument("--estimate", action="store_true", help="build data/variants, print prompt counts, and exit")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    load_dotenv(cfg.root)
    stages = [s.strip() for s in args.stages.split(",")]
    specs = load_models(cfg.models_file)
    model_names = args.models or cfg.models

    if "data" in stages or args.estimate:
        instances = build_instances(cfg)
        log(f"[data] {len(instances)} instances")
    else:
        instances = load_instances(cfg)
    if "variants" in stages or args.estimate:
        variants = build_variants(cfg, instances)
        log(f"[variants] {sum(len(v) for v in variants.values())} variants")
    else:
        variants = load_variants(cfg)

    if args.estimate:
        estimate(cfg, instances, variants, model_names)
        return

    for name in model_names:
        spec = specs[name]
        if "llm" in stages:
            stage_llm(cfg, spec, instances, variants, limit=args.limit)
        if "exec" in stages:
            stage_exec(cfg, spec, variants)
        if "score" in stages:
            stage_score(cfg, spec, instances, variants)


if __name__ == "__main__":
    main()
