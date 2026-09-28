"""Actual and projected spend per model, from the response cache.

Uses usage.estimated_cost where the provider reports it (deepinfra, together) and RATES
otherwise (novita, featherless, nscale do not report). Run at any point during a run:

    python scripts/cost.py                    # every model in the cache
    python scripts/cost.py hf-deepseek-v3.1-novita
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict

# $/M tokens (input, output), from each provider's public price list. Verified 2026-09-17.
RATES = {
    "hf-deepseek-v3.1-novita": (0.27, 1.00),
    "hf-deepseek-v3": (0.32, 0.89),          # deepinfra; also reported per call, so this is a check
    "hf-qwen3-8b": (0.117, 0.455),           # featherless
    "hf-qwen3-8b-nscale": (0.04, 0.14),      # nscale list price for an 8B model; confirm on the billing page
    "hf-gemma-4-31b": (0.14, 0.40),          # novita, HF router catalog 2026-09-19; novita does not report cost
    "hf-deepseek-v4-flash": (0.09, 0.18),    # deepinfra, HF router catalog 2026-09-19; also reported per call
}
# unique API calls for a full run, from `run.py --estimate`
UNIQUE_TOTAL = 65624


def arm_of(prompt_id: str) -> str:
    p = prompt_id.split("::")
    return f"{p[-2]}/{p[-1]}" if p[-1] in ("networkx", "native") else p[-1]


def main(models: list[str]) -> None:
    for model in models:
        files = glob.glob(f"results/cache/{model}/*.json")
        if not files:
            continue
        rep = 0.0
        tin = tout = n = 0
        by = defaultdict(lambda: [0, 0, 0])
        for f in files:
            d = json.load(open(f))
            u = d.get("usage") or {}
            if "prompt_tokens" not in u:
                continue
            n += 1
            tin += u["prompt_tokens"]
            tout += u["completion_tokens"]
            rep += u.get("estimated_cost") or 0.0
            a = by[arm_of(d["prompt_id"])]
            a[0] += 1
            a[1] += u["prompt_tokens"]
            a[2] += u["completion_tokens"]
        rin, rout = RATES.get(model, (0.0, 0.0))
        est = (tin * rin + tout * rout) / 1e6
        spent = rep or est
        src = "provider-reported" if rep else f"estimated at ${rin}/${rout} per M"
        print(f"\n=== {model}: {n} calls cached ===")
        print(f"  tokens: {tin/1e6:.2f}M in, {tout/1e6:.2f}M out")
        print(f"  spend so far: ${spent:.2f}  ({src})")
        if n:
            print(f"  per call: ${spent/n:.5f}  ->  full run of {UNIQUE_TOTAL} calls: ${spent/n*UNIQUE_TOTAL:.2f}")
        for arm, (c, i, o) in sorted(by.items()):
            print(f"    {arm:24s} {c:6d} calls  {i/c:6.0f} in  {o/c:6.0f} out")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted({p.split("/")[-2] for p in glob.glob("results/cache/*/")} |
                                {p.split("/")[2] for p in glob.glob("results/cache/*/*.json")}))
