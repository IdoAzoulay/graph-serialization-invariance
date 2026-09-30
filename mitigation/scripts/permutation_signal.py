"""E1 + E2: use serialization sensitivity as a signal, on data already on disk.

E1 detection  -- does disagreement between the canonical answer and its permuted variants
                 predict that the canonical answer is wrong?
E2 voting     -- majority over the canonical + permuted answers, with and without abstention.

    python mitigation/scripts/permutation_signal.py --config configs/graphqa.yaml [--models ...] [--rule flip]

Writes mitigation/results/<cfg>/permutation_signal.csv (per model x arm x probe x k; with
`--rule flip`, permutation_signal_fliprule.csv) and prints the headline rows. No inference: reads
scored/<model>.jsonl only. Rows follow the order of --models (default: the scored files, sorted);
the committed tables use hf-qwen3-8b-nscale hf-deepseek-v3.1-novita hf-gemma-4-31b hf-deepseek-v4-flash.

Rules. When do the answers of an instance disagree (and the instance get flagged)?
  * literal (default) -- the invariance metric's rule: unless every answer parsed and all answer
    keys are equal. Two different correct answers (two shortest paths) raise a flag, and so does
    an unparsable answer, even when every answer failed alike. This is the detector as it could be
    deployed: it needs no true answer.
  * flip -- the paper's flip rule (Section 4.2; scripts/paper_metrics.py): unless all answers are the
    same, where every correct answer (the scored record's `correct`) counts as the same answer and
    every missing answer counts as one shared value. It uses the true answer, so it is not a
    deployable detector. It applies the paper's same-answer rule to the canonical answer and the
    probes: on GraphQA its flags are exactly the paper's flips, while on Erdos the relabel probe
    compares with the canonical answer, not the sorted identity of the paper's tuples, so about
    1% of flags and flips differ (116 and 113 of 14,400). It can only remove
    flags, and only on instances whose canonical answer is correct or missing: the flags on
    answered errors (`caught_answered`, `recall_answered`) are the same under both rules. The
    majority vote (`vote_acc`, `vote_delta`) compares answer keys under both rules.

Probes. `relabel` uses the seeded relabelings only: the sorted-identity form renames nothing (on
GraphQA it is byte-identical to the canonical prompt, on Erdos it re-sorts the edges), so it is
not a relabeling. On Erdos a relabeling is also re-sorted after renaming, so against the
canonical answer the relabel probe changes labels and edge order together. `both` lists the
order variants first, then the relabelings; `k` is the number of answers used, canonical
included, taking each probe's variants in variant-id order. `relabel@matched` / `order@matched`
compare the two probes at the same k on the same instances (those with at least k-1 variants
of each kind), which is the only fair way to say which probe is better.

Recall. An unparsable canonical answer (a crash, a missing `ans`) is always flagged and always
wrong, but it is visible without any detector. `recall_answered` is the recall on the errors
that returned a parsable answer -- the silent errors a detector is for.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from gsi.data.base import read_jsonl
from gsi.experiment.config import load_config


def out_dir(cfg) -> Path:
    """mitigation/results/<dataset>/, where this experiment's committed tables live."""
    d = Path(__file__).resolve().parents[1] / "results" / cfg.results_dir.name
    d.mkdir(parents=True, exist_ok=True)
    return d


PROBES = {"relabel": {"relabel"}, "order": {"order"}, "both": {"relabel", "order"}}
NULL = (None, "null")
RULES = ("literal", "flip")
OUT_NAME = {"literal": "permutation_signal.csv", "flip": "permutation_signal_fliprule.csv"}


def disagree(rows, rule: str = "literal") -> bool:
    """Do these answers disagree (see Rules above)?"""
    if rule == "literal":
        keys = [r["answer_key"] for r in rows]
        # the invariance metric treats an unparsable answer as non-identical; so do we
        return len(set(keys)) > 1 or any(x in NULL for x in keys)
    if rule == "flip":
        return len({"CORRECT" if r["correct"] else ("NA" if r["answer_key"] in NULL else r["answer_key"])
                    for r in rows}) > 1
    raise ValueError(rule)


def load(cfg, model):
    """{(arm, instance): {"canon": row, "relabel": [rows], "order": [rows]}}"""
    g = defaultdict(lambda: {"canon": None, "relabel": [], "order": []})
    for r in read_jsonl(cfg.results_dir / "scored" / f"{model}.jsonl"):
        arm = "direct" if r["mode"] == "direct" else f"{r['mode']}/{r['library']}"
        k = (arm, r["instance_id"])
        if r["axis"] == "canonical":
            g[k]["canon"] = r
        elif r["axis"] == "relabel" and (r.get("params") or {}).get("seed") == "identity":
            continue  # a reference, not a perturbation
        elif r["axis"] in ("relabel", "order"):
            g[k][r["axis"]].append(r)
    for v in g.values():
        v["relabel"].sort(key=lambda r: r["variant_id"])
        v["order"].sort(key=lambda r: r["variant_id"])
    return {k: v for k, v in g.items() if v["canon"] is not None}


def evaluate(groups, probe: set[str], k: int | None, rule: str = "literal"):
    """One row of metrics for a probe (which axes supply the permutations) and a budget k
    (total answers used, canonical included; None = all available); `rule` decides disagreement."""
    n = flagged = errors = caught = errors_answered = caught_answered = 0
    vote_ok = base_ok = 0
    kept = kept_ok = 0
    for (arm, _), d in groups.items():
        canon = d["canon"]
        perms = [r for ax in sorted(probe) for r in d[ax]]
        if k is not None:
            perms = perms[: max(k - 1, 0)]
        if not perms:
            continue
        n += 1
        base_ok += canon["correct"]
        wrong = not canon["correct"]
        errors += wrong
        keys = [r["answer_key"] for r in [canon] + perms]
        flag = disagree([canon] + perms, rule)
        flagged += flag
        caught += flag and wrong
        if wrong and canon["answer_key"] not in NULL:
            errors_answered += 1
            caught_answered += flag
        # majority vote over parsable answers; ties resolve to the canonical answer
        votes = Counter(x for x in keys if x not in NULL)
        if votes:
            top, cnt = votes.most_common(1)[0]
            if list(votes.values()).count(cnt) > 1 and canon["answer_key"] not in NULL:
                top = canon["answer_key"]
            vote_ok += any(r["correct"] for r in [canon] + perms if r["answer_key"] == top)
        # abstain when flagged: accuracy on what is answered, and how much is answered
        if not flag:
            kept += 1
            kept_ok += canon["correct"]
    if n == 0:
        return None
    return {
        "n": n, "base_acc": base_ok / n, "vote_acc": vote_ok / n, "vote_delta": (vote_ok - base_ok) / n,
        "flag_rate": flagged / n, "errors": errors, "caught": caught,
        "recall": caught / errors if errors else float("nan"),
        "errors_answered": errors_answered, "caught_answered": caught_answered,
        "recall_answered": caught_answered / errors_answered if errors_answered else float("nan"),
        "precision": caught / flagged if flagged else float("nan"),
        "p_wrong_given_agree": (errors - caught) / (n - flagged) if n > flagged else float("nan"),
        "p_wrong_given_disagree": caught / flagged if flagged else float("nan"),
        "abstain_coverage": kept / n, "abstain_acc": kept_ok / kept if kept else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", help="row order follows this list (default: scored files, sorted)")
    ap.add_argument("--rule", choices=RULES, default="literal",
                    help="literal (default): answer keys compared as they are, writes permutation_signal.csv; "
                         "flip: the paper's flip rule, writes permutation_signal_fliprule.csv")
    args = ap.parse_args()
    cfg = load_config(args.config)
    models = args.models or sorted(p.stem for p in (cfg.results_dir / "scored").glob("*.jsonl"))
    rows = []
    for m in models:
        groups = load(cfg, m)
        if not groups:
            print(f"[{m}] no scored file; skipped")
            continue
        for arm in sorted({a for a, _ in groups}):
            sub = {k: v for k, v in groups.items() if k[0] == arm}
            for probe, axes in PROBES.items():
                for k in (2, 3, 4, None):
                    e = evaluate(sub, axes, k, args.rule)
                    if e:
                        rows.append({"dataset": cfg.name, "model": m, "arm": arm, "probe": probe,
                                     "k": k or "all", **e})
            for k in (2, 3, 4):
                both = {key: d for key, d in sub.items()
                        if len(d["relabel"]) >= k - 1 and len(d["order"]) >= k - 1}
                for probe in ("relabel", "order"):
                    e = evaluate(both, PROBES[probe], k, args.rule)
                    if e:
                        rows.append({"dataset": cfg.name, "model": m, "arm": arm, "probe": f"{probe}@matched",
                                     "k": k, **e})
    df = pd.DataFrame(rows)
    out = out_dir(cfg) / OUT_NAME[args.rule]
    df.to_csv(out, index=False)
    pd.set_option("display.width", 230, "display.max_rows", 300)
    show = df[(df.probe == "both") & (df.k == "all")]
    cols = ["model", "arm", "n", "base_acc", "vote_acc", "vote_delta", "flag_rate", "errors", "recall",
            "errors_answered", "recall_answered", "abstain_coverage", "abstain_acc"]
    print(f"\n=== {cfg.name}: all permutations as the probe ({args.rule} rule) ===")
    print(show[cols].round(3).to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
