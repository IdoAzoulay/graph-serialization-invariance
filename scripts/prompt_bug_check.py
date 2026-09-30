"""Offline impact check of the two defects in the published prompts (README, "Known issues in the published prompts").

    python scripts/prompt_bug_check.py [--out PATH]   ->  results/review/prompt_bug_check.json

The defects
  (a) A relabeling renames both ends of every edge and re-sorts the list, but never writes an undirected edge
      smaller label first. About half of a relabeling's edges come out larger label first, against none in the
      GraphQA references and about a quarter in the Erdős ones, so the relabel factor also reorients edges.
  (b) graphqa-er-0025 has no edges; its GraphQA edge list reads "The edges in G are: ." (7 instances).
  `prompt_fixes: true` in a config fixes both. It changes published prompts and was not run.

Inputs: the scored records, results/{graphqa,erdos,erdos_m3fix}/scored/<model>.jsonl (scripts/restore_from_hf.py),
the instances in data/processed/<dataset>/instances.jsonl and the configs; the variants are rebuilt in memory.
No model is called. Tuples follow the paper (gsi.analysis.tuples); every M1/M2 flip rate is checked against
results/review/paper_metrics.json before anything is written.

Sections
  prompt_hashes     every published record's prompt rebuilt: identical with prompt_fixes off (asserted); with it
                    on, per dataset and mode, the records whose prompt changes, the distinct new prompts, and the
                    Graph-as-Code records that keep their prompt but would run on re-oriented edges
  orientation       (i) undirected edges written larger label first, in the references and in the relabelings, and
                    per relabeling the share of the reference's edges whose written orientation it flips
  renamed_list      (ii) read as ordered pairs, each relabeled edge list is the renamed reference list: a program
                    that treats the pairs as a graph, directed or not, gives the renamed answer. The lists are
                    re-sorted, though, so list order and label values still differ. The smaller-first fix would
                    break this for a program that reads undirected pairs as directed.
  m3_one_program    (iii) M3 relabel tuples whose serializations share one prompt (the question names no node, so
                    one program runs on every relabeling), and those that still flip, which only list order or
                    label values can cause
  dose_response     (iv) within relabel tuples (M1/M2, undirected graphs), a conditional logit of "the answer
                    differs from the reference's" on the share of reoriented edges; odds ratio per +10 points
  relabel_vs_order  (v) per pair (reference, one variant), how often the answer differs, relabel vs order, on
                    undirected graphs; split by whether the question names a node (its wording then changes too)
  empty_graph       (vi) accuracy, robust accuracy and flip rates with and without graphqa-er-0025
Rates are percentages. Nothing is sampled, so the output is deterministic.
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from statsmodels.discrete.conditional_models import ConditionalLogit

from gsi.analysis.tuples import comparison_value, tuple_factors, tuple_members, tuple_reference
from gsi.experiment.config import load_config
from gsi.experiment.run import load_instances, make_all_variants
from gsi.prompts.modes import build_prompt, question_for

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["graphqa", "erdos", "erdos_m3fix"]  # erdos_m3fix: the Erdős M3 rerun, same instances and variants as erdos
MODELS = ["hf-qwen3-8b-nscale", "hf-deepseek-v3.1-novita", "hf-gemma-4-31b", "hf-deepseek-v4-flash"]
ARMS_M12 = ["direct", "code/native", "code/networkx"]
EMPTY_GRAPH = "graphqa-er-0025"

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--out", default=str(ROOT / "results/review/prompt_bug_check.json"))
args = ap.parse_args()


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def pct(a, b, nd=2):
    return round(100 * a / b, nd) if b else None


def missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def larger_first(edges):
    return sum(e[0] > e[1] for e in edges)


def flipped_share(ref_edges, label_map):
    """Share of the reference's edges whose written orientation (smaller or larger label first) the renaming flips."""
    return float(np.mean([(e[0] < e[1]) != (label_map[e[0]] < label_map[e[1]]) for e in ref_edges]))


out = {"notes": __doc__}

# ---------------------------------------------------------------- data: configs, instances, variants (off / on)
cfgs, inst, var_off, var_on = {}, {}, {}, {}
for name in CONFIGS:
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert not cfg.prompt_fixes, f"configs/{name}.yaml: the published runs had prompt_fixes off"
    cfgs[name] = cfg
    insts = load_instances(cfg)
    inst[name] = {i.id: i for i in insts}
    var_off[name] = {v.variant_id: v for vs in make_all_variants(cfg, insts).values() for v in vs}
    fixed = make_all_variants(replace(cfg, prompt_fixes=True), insts)
    var_on[name] = {v.variant_id: v for vs in fixed.values() for v in vs}

# ---------------------------------------------------------------- scored records, one pass
# answers[config][(instance, model, arm)] = the answers of a tuple group, in record order
answers = {name: defaultdict(list) for name in CONFIGS}
records = {name: [] for name in CONFIGS}  # (model, variant_id, mode, library, prompt_hash)
for name in CONFIGS:
    for m in MODELS:
        p = ROOT / "results" / name / "scored" / f"{m}.jsonl"
        if not p.exists():
            raise SystemExit(f"no scored records at {p}; run scripts/restore_from_hf.py")
        with open(p) as fh:
            for r in map(json.loads, fh):
                arm = "direct" if r["mode"] == "direct" else f"{r['mode']}/{r['library']}"
                answered = not missing(r["parsed"])
                answers[name][(r["instance_id"], m, arm)].append(dict(
                    axis=r["axis"], variant_id=r["variant_id"],
                    is_identity=(r["params"] or {}).get("seed") == "identity",
                    val=comparison_value(bool(r["correct"]), answered, r["answer_key"]), correct=bool(r["correct"]),
                    prompt_hash=r["prompt_hash"]))
                records[name].append((m, r["variant_id"], r["mode"], r["library"], r["prompt_hash"]))
    log(f"[load] {name}: {len(records[name])} records")

# ---------------------------------------------------------------- prompt hashes, off and on
hashes = {}
for name in CONFIGS:
    cfg, built = cfgs[name], {}
    per_mode, by_axis = defaultdict(Counter), defaultdict(Counter)
    new_prompts, published = defaultdict(set), defaultdict(set)
    for m, vid, mode, lib, h in records[name]:
        k = (vid, mode, lib)
        if k not in built:
            i = inst[name][var_off[name][vid].instance_id]
            off = build_prompt(mode, i, var_off[name][vid], lib, inject_graph_type=cfg.inject_graph_type).prompt_hash
            on = build_prompt(mode, i, var_on[name][vid], lib, inject_graph_type=cfg.inject_graph_type).prompt_hash
            a, b = var_off[name][vid], var_on[name][vid]
            built[k] = (off, on, (a.node_seq, a.edge_seq) != (b.node_seq, b.edge_seq))
        off, on, regraphed = built[k]
        c = per_mode[(m, mode)]
        c["records"] += 1
        c["identical_off"] += off == h
        c["changed_on"] += on != h
        c["m3_reexecuted_on"] += mode == "graph_as_code" and on == h and regraphed
        published[m].add(h)
        if on != h:
            new_prompts[(m, mode)].add(on)
            by_axis[(m, mode)][var_off[name][vid].axis] += 1
    for m in MODELS:
        for mode in {md for mm, md in per_mode if mm == m}:
            per_mode[(m, mode)]["new_distinct_prompts_on"] = len(new_prompts[(m, mode)])
            per_mode[(m, mode)]["new_prompts_not_published_on"] = len(new_prompts[(m, mode)] - published[m])
    bad = sum(c["records"] - c["identical_off"] for c in per_mode.values())
    assert bad == 0, f"{name}: {bad} published prompts do not rebuild with prompt_fixes off"
    # every model was asked the same prompts, so the counts are per model
    by_mode = {}
    for mode in sorted({md for _, md in per_mode}):
        cs = [dict(per_mode[(m, mode)], changed_on_by_axis=dict(sorted(by_axis[(m, mode)].items()))) for m in MODELS]
        assert all(c == cs[0] for c in cs), (name, mode, cs)
        by_mode[mode] = cs[0]
    hashes[name] = by_mode
    log(f"[hashes] {name}: {sum(c['records'] for c in by_mode.values())} records per model, all rebuild")
total = Counter()
for by_mode in hashes.values():
    for c in by_mode.values():
        total.update({k: v for k, v in c.items() if k != "changed_on_by_axis"})
out["prompt_hashes"] = {"per_model": hashes, "per_model_total": dict(total), "models": len(MODELS),
                        "calls_for_all_models_on": total["new_prompts_not_published_on"] * len(MODELS)}

# ---------------------------------------------------------------- tuples (paper rules)
tuples = {name: [] for name in CONFIGS}
for name in CONFIGS:
    for (iid, model, arm), ans in answers[name].items():
        for axis in tuple_factors(arm):
            sub = tuple_members(ans, axis)
            if sub is None:
                continue
            tuples[name].append(dict(instance_id=iid, model=model, arm=arm, axis=axis, members=sub,
                                     ref=tuple_reference(sub), flip=len({a["val"] for a in sub}) > 1,
                                     robust=all(a["correct"] for a in sub),
                                     one_prompt=len({a["prompt_hash"] for a in sub}) == 1))

# the M1/M2 flip rates must be the paper's (results/review/paper_metrics.json, one decimal)
pm = json.load(open(ROOT / "results/review/paper_metrics.json"))["invariance"]
for name in ["graphqa", "erdos"]:
    rates = defaultdict(list)
    for t in tuples[name]:
        if t["arm"] in ARMS_M12:
            rates[(t["arm"], t["axis"])].append(t["flip"])
    for (arm, axis), f in rates.items():
        mine, paper = round(100 * float(np.mean(f)), 1), pm[f"{name}|{arm}|{axis}"]["flip_rate"]
        assert mine == paper, f"{name} {arm} {axis}: flip rate {mine} here, {paper} in paper_metrics.json"

# ---------------------------------------------------------------- (i) orientation, (ii) renamed list
orientation, renamed = {}, {}
flipped = {}  # relabel variant_id -> share of the reference's edges it reorients (undirected, with edges)
for name in ["graphqa", "erdos"]:
    counts = defaultdict(lambda: [0, 0])
    per_variant, n_rel, same_pairs, same_order = [], 0, 0, 0
    ref_of = {v.instance_id: v for v in var_off[name].values()
              if v.axis == "relabel" and v.params["seed"] == "identity"}
    for v in var_off[name].values():
        if v.directed:
            continue
        kind = ("reference (sorted identity)" if v.params.get("seed") == "identity" else "relabeling") \
            if v.axis == "relabel" else ("canonical" if v.axis == "canonical" else "order, structure, syntax")
        counts[kind][0] += larger_first(v.edge_seq)
        counts[kind][1] += len(v.edge_seq)
        if v.axis == "relabel" and v.params.get("seed") != "identity":
            ref, lm = ref_of[v.instance_id], v.label_map  # reference label -> new label
            renamed_ref = [[lm[e[0]], lm[e[1]], *e[2:]] for e in ref.edge_seq]
            n_rel += 1
            same_pairs += sorted(v.edge_seq) == sorted(renamed_ref)
            same_order += v.edge_seq == renamed_ref
            if ref.edge_seq:
                flipped[v.variant_id] = flipped_share(ref.edge_seq, lm)
                per_variant.append(flipped[v.variant_id])
    pv = np.array(per_variant)
    orientation[name] = {
        "larger_first_pct_of_edges": {k: pct(a, b) for k, (a, b) in sorted(counts.items())},
        "edges": {k: b for k, (a, b) in sorted(counts.items())},
        "reoriented_vs_reference_pct_per_relabeling": dict(
            mean=round(100 * float(pv.mean()), 2), p5=round(100 * float(np.percentile(pv, 5)), 1),
            p95=round(100 * float(np.percentile(pv, 95)), 1), relabelings=int(len(pv)))}
    renamed[name] = {"relabelings": n_rel, "renamed_reference_list_as_ordered_pairs": same_pairs,
                     "also_in_the_renamed_reference_order": same_order}
out["orientation"] = orientation
out["renamed_list"] = renamed

# ---------------------------------------------------------------- (iii) M3 relabel tuples answered by one program
m3 = {}
for name in CONFIGS:
    ts = [t for t in tuples[name] if t["arm"].startswith("graph_as_code") and t["axis"] == "relabel"]
    one = [t for t in ts if t["one_prompt"]]
    where = Counter(f"{t['model']}|{inst[name][t['instance_id']].task}|{t['arm']}" for t in one if t["flip"])
    m3[name] = {"relabel_tuples": len(ts), "one_prompt_tuples": len(one),
                "one_prompt_flips": sum(t["flip"] for t in one),
                "one_prompt_flips_by_model_task_arm": dict(sorted(where.items()))}
out["m3_one_program"] = m3

# ---------------------------------------------------------------- pairs for (iv) and (v)
pairs = {}
for name in ["graphqa", "erdos"]:
    # a question that names a node changes under relabeling (and with it the M3 prompt)
    by_inst = defaultdict(set)
    for v in var_off[name].values():
        if v.axis == "relabel":
            by_inst[v.instance_id].add(question_for(inst[name][v.instance_id], v.label_map))
    named = {iid: len(qs) > 1 for iid, qs in by_inst.items()}
    rows = []
    for t in tuples[name]:
        if t["arm"] not in ARMS_M12 or t["axis"] not in ("relabel", "order") or inst[name][t["instance_id"]].directed:
            continue
        for a in t["members"]:
            if a is t["ref"]:
                continue
            rows.append(dict(arm=t["arm"], axis=t["axis"], tup=(t["instance_id"], t["model"], t["arm"]),
                             differs=int(a["val"] != t["ref"]["val"]), named=named[t["instance_id"]],
                             reo=flipped.get(a["variant_id"])))
    pairs[name] = rows

# ---------------------------------------------------------------- (iv) dose-response, conditional logit
dose = {}
for name in ["graphqa", "erdos"]:
    dose[name] = {}
    for arm in ARMS_M12:
        rs = [r for r in pairs[name] if r["arm"] == arm and r["axis"] == "relabel" and r["reo"] is not None]
        by_tup = defaultdict(list)
        for r in rs:
            by_tup[r["tup"]].append(r)
        informative = [g for _, g in sorted(by_tup.items()) if len({r["differs"] for r in g}) > 1]
        y = np.array([r["differs"] for g in informative for r in g])
        x = np.array([[10 * r["reo"]] for g in informative for r in g])
        groups = np.array([k for k, g in enumerate(informative) for _ in g])
        fit = ConditionalLogit(y, x, groups=groups).fit(disp=0)
        b, se = float(fit.params[0]), float(fit.bse[0])
        within_sd = float(np.mean([np.std([r["reo"] for r in g], ddof=1) for g in informative]))
        dose[name][arm] = dict(odds_ratio_per_10_points=round(math.exp(b), 2),
                               ci95=[round(math.exp(b - 1.96 * se), 2), round(math.exp(b + 1.96 * se), 2)],
                               p=round(float(fit.pvalues[0]), 3), informative_tuples=len(informative),
                               pairs=int(len(y)), mean_within_tuple_sd_of_share=round(within_sd, 3))
out["dose_response"] = dose

# ---------------------------------------------------------------- (v) relabel vs order, per pair
rvo = {}
for name in ["graphqa", "erdos"]:
    rvo[name] = {}
    for subset, keep in [("all", lambda r: True), ("question_names_no_node", lambda r: not r["named"]),
                         ("question_names_a_node", lambda r: r["named"])]:
        cell = {}
        for arm in ARMS_M12:
            rate = {}
            for axis in ("relabel", "order"):
                d = [r["differs"] for r in pairs[name] if r["arm"] == arm and r["axis"] == axis and keep(r)]
                rate[axis] = pct(sum(d), len(d))
                rate[f"{axis}_pairs"] = len(d)
            rate["relabel_minus_order"] = round(rate["relabel"] - rate["order"], 2)
            cell[arm] = rate
        rvo[name][subset] = cell
out["relabel_vs_order"] = rvo

# ---------------------------------------------------------------- (vi) the empty graph
name = "graphqa"
drop = {iid for iid, i in inst[name].items() if i.meta.get("graph_id") == EMPTY_GRAPH}


def with_without(values):
    """values: (instance_id, 0/1) pairs -> [rate with, rate without the empty graph], unrounded percent."""
    a = [x for _, x in values]
    b = [x for iid, x in values if iid not in drop]
    return [100 * sum(a) / len(a), 100 * sum(b) / len(b)]


cells = {"accuracy": defaultdict(list), "accuracy_by_serialization": defaultdict(list),
         "robust_accuracy": defaultdict(list), "flip_rate": defaultdict(list)}
for (iid, model, arm), ans in answers[name].items():
    for a in ans:
        kind = "relabel (sorted identity)" if a["is_identity"] else a["axis"]
        cells["accuracy"][arm].append((iid, a["correct"]))
        cells["accuracy_by_serialization"][f"{arm}|{kind}"].append((iid, a["correct"]))
for t in tuples[name]:
    for fac in [t["axis"]] + (["relabel+order"] if t["axis"] in ("relabel", "order") else []):
        cells["robust_accuracy"][f"{t['arm']}|{fac}"].append((t["instance_id"], t["robust"]))
        cells["flip_rate"][f"{t['arm']}|{fac}"].append((t["instance_id"], t["flip"]))
rates = {k: {cell: with_without(v) for cell, v in sorted(c.items())} for k, c in cells.items()}


def biggest(d, keep=lambda cell: True):
    return round(max(abs(a - b) for cell, (a, b) in d.items() if keep(cell)), 3)


table2 = lambda cell: cell.split("|")[0] in ARMS_M12 and cell.split("|")[1] != "relabel+order"
pooled = lambda cell: cell.endswith("|relabel+order")
et = [t for t in tuples[name] if t["instance_id"] in drop]
out["empty_graph"] = {
    "graph": EMPTY_GRAPH, "instances": len(drop),
    "answers": sum(iid in drop for c in cells["accuracy"].values() for iid, _ in c),
    "tuples": len(et), "flipped_tuples": sum(t["flip"] for t in et),
    "empty_edge_list_texts": sum(v.text.endswith("The edges in G are: .") for v in var_off[name].values()),
    "note": "[with, without] graphqa-er-0025, percent. accuracy: all of an arm's answers; by serialization: the "
            "answers to one kind of serialization; robust accuracy and flip rate per tuple, by factor (Table 2's "
            "cells are the M1/M2 arms' four factors). max_abs_change: points, from unrounded rates",
    **{k: {cell: [round(a, 2), round(b, 2)] for cell, (a, b) in d.items()} for k, d in rates.items()},
    "max_abs_change": {
        "accuracy": biggest(rates["accuracy"]),
        "accuracy_by_serialization": biggest(rates["accuracy_by_serialization"]),
        "robust_accuracy_relabel+order": biggest(rates["robust_accuracy"], pooled),
        "robust_accuracy_any_cell": biggest(rates["robust_accuracy"]),
        "flip_rate_table2": biggest(rates["flip_rate"], table2),
        "flip_rate_relabel+order": biggest(rates["flip_rate"], pooled),
        "flip_rate_any_cell": biggest(rates["flip_rate"])}}

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
    fh.write("\n")
print(f"-> {args.out}")
