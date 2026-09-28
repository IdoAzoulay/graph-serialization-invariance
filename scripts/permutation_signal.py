"""E1 + E2: use serialization sensitivity as a signal, on data already on disk.

E1 detection  -- does disagreement between the canonical answer and its permuted variants
                 predict that the canonical answer is wrong?
E2 voting     -- majority over the canonical + permuted answers, with and without abstention.

    python scripts/permutation_signal.py --config configs/graphqa.yaml [--models ...]

Writes results/<cfg>/tables/permutation_signal.csv (per model x arm x probe x k) and prints the
headline rows. No inference: reads scored/<model>.jsonl only.

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
import itertools
import json
from collections import Counter, defaultdict

import pandas as pd

from gsi.data.base import read_jsonl
from gsi.experiment.config import load_config

PROBES = {"relabel": {"relabel"}, "order": {"order"}, "both": {"relabel", "order"}}
NULL = (None, "null")


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


def evaluate(groups, probe: set[str], k: int | None):
    """One row of metrics for a probe (which axes supply the permutations) and a budget k
    (total answers used, canonical included; None = all available)."""
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
        # the invariance metric treats an unparsable answer as non-identical; so do we
        disagree = len(set(keys)) > 1 or any(x in NULL for x in keys)
        flagged += disagree
        caught += disagree and wrong
        if wrong and canon["answer_key"] not in NULL:
            errors_answered += 1
            caught_answered += disagree
        # majority vote over parsable answers; ties resolve to the canonical answer
        votes = Counter(x for x in keys if x not in NULL)
        if votes:
            top, cnt = votes.most_common(1)[0]
            if list(votes.values()).count(cnt) > 1 and canon["answer_key"] not in NULL:
                top = canon["answer_key"]
            vote_ok += any(r["correct"] for r in [canon] + perms if r["answer_key"] == top)
        # abstain when not unanimous: accuracy on what is answered, and how much is answered
        if not disagree:
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
    ap.add_argument("--models", nargs="*")
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
                    e = evaluate(sub, axes, k)
                    if e:
                        rows.append({"dataset": cfg.name, "model": m, "arm": arm, "probe": probe,
                                     "k": k or "all", **e})
            for k in (2, 3, 4):
                both = {key: d for key, d in sub.items()
                        if len(d["relabel"]) >= k - 1 and len(d["order"]) >= k - 1}
                for probe in ("relabel", "order"):
                    e = evaluate(both, PROBES[probe], k)
                    if e:
                        rows.append({"dataset": cfg.name, "model": m, "arm": arm, "probe": f"{probe}@matched",
                                     "k": k, **e})
    df = pd.DataFrame(rows)
    out = cfg.results_dir / "tables" / "permutation_signal.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 230, "display.max_rows", 300)
    show = df[(df.probe == "both") & (df.k == "all")]
    cols = ["model", "arm", "n", "base_acc", "vote_acc", "vote_delta", "flag_rate", "errors", "recall",
            "errors_answered", "recall_answered", "abstain_coverage", "abstain_acc"]
    print(f"\n=== {cfg.name}: all permutations as the probe ===")
    print(show[cols].round(3).to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
