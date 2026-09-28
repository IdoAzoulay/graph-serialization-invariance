"""The paper's metric set, computed from the scored records of all four models.

    python scripts/paper_metrics.py [--out PATH]   ->  results/review/paper_metrics.json

Inputs
  scored records, results/<dataset>/scored/<model>.jsonl (git-ignored; scripts/restore_from_hf.py writes them).
  data/processed/<dataset>/instances.jsonl gives the true answers, edge counts, graphs and graph ids.

Units
  answer = one serialization of one instance, one model, one mode
  tuple  = the reference serialization + one factor's variants (as in scripts/audit_results.py):
           relabel -> the sorted identity + the relabelings; other factors -> canonical + that factor's variants
Same-answer rules: node answers are compared after mapping back (answer_key); two correct answers are the
same answer; "no answer" (crash, timeout, unparseable) is an answer of its own.

Metrics
  performance  accuracy (no answer = wrong), robust accuracy (every serialization correct), drop vs reference
  invariance   six buckets; flip rate (D > 0); D = share of serialization pairs that differ; severity = mean D
               among flips; weighted flip rate = mean D; S = (max - min) / |truth| over a flip's answered values
               (flipped tuples of numeric tasks)
  localization (a) wrong M2 answers by the first step that failed (copy -> solve -> run), as shares among
                   the wrong answers, with their rate
               (b) flipped M2 tuples by where they diverge: copy / crash / program
               (c) flip rate of M2 vs M3 on the same tuples, relabel and order only
  also         flip rate by graph size, and whether flipped node-list answers take the other reading
Statistics: McNemar (flip, code vs direct; M2 vs M3), Wilcoxon signed-rank (D, code vs direct), cluster
bootstrap over graphs (headline rates, and the M1 / pooled-M2 rates per model drawn in the figures), logistic
regressions clustered by graph, one per dataset. Datasets are never pooled. McNemar and Wilcoxon treat a
model's tuples as independent pairs; tuples of one graph are not.
"""
import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import binomtest, wilcoxon
from statsmodels.stats.multitest import multipletests

from gsi.analysis.tuples import READING_TASKS, divergence_site, failure_step, reading_tag, spread

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["graphqa", "erdos"]
MODELS = ["hf-qwen3-8b-nscale", "hf-deepseek-v3.1-novita", "hf-gemma-4-31b", "hf-deepseek-v4-flash"]
COLS = ["dataset", "task", "instance_id", "variant_id", "axis", "params", "mode", "library", "model",
        "answer_key", "parsed", "correct", "failure_class", "declared_ok"]
RNG = np.random.default_rng(0)
ARMS_M12 = ["direct", "code/native", "code/networkx"]
FACTORS = ["relabel", "order", "structure", "syntax"]
NUMERIC = {"node_degree", "node_count", "edge_count", "triangles", "degree", "edge_number",
           "connected_component_number", "diameter", "density"}

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--out", default=str(ROOT / "results/review/paper_metrics.json"))
args = ap.parse_args()

rows = []
for ds in DATASETS:
    for m in MODELS:
        p = ROOT / "results" / ds / "scored" / f"{m}.jsonl"
        if not p.exists():
            raise SystemExit(f"no scored records at {p}; run scripts/restore_from_hf.py")
        with open(p) as fh:
            rows += [{k: r.get(k) for k in COLS} for r in map(json.loads, fh)]
j = pd.DataFrame(rows)
j["arm"] = np.where(j["mode"] == "direct", "direct", j["mode"] + "/" + j["library"].astype(str))
miss = lambda v: v is None or (isinstance(v, float) and math.isnan(v))
j["answered"] = ~j["parsed"].map(miss)
j["correct"] = j["correct"].astype(bool)
j["is_identity"] = j["params"].map(lambda p: (p or {}).get("seed") == "identity")

inst = {}
for ds in DATASETS:
    for line in open(ROOT / "data/processed" / ds / "instances.jsonl"):
        r = json.loads(line)
        inst[r["id"]] = r
truth = {i: r["ground_truth"] for i, r in inst.items()}
cluster = {i: r["meta"].get("source_file") or r["meta"]["graph_id"] for i, r in inst.items()}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- tuples
rows, reading = [], []  # reading: flipped code tuples of the node-list tasks (the "other reading" analysis)
for (iid, model, arm), g in j.groupby(["instance_id", "model", "arm"], sort=False):
    canon = g[g["axis"] == "canonical"]
    ds, task = g["dataset"].iat[0], g["task"].iat[0]
    for axis in FACTORS:
        if arm.startswith("graph_as_code") and axis not in ("relabel", "order"):
            continue
        var = g[g["axis"] == axis]
        sub = var if (axis == "relabel" and var["is_identity"].any()) else pd.concat([canon, var])
        if sub["variant_id"].nunique() < 2:
            continue
        keys = [k if a else "NA" for k, a in zip(sub["answer_key"], sub["answered"])]
        corr = sub["correct"].tolist()
        vals = ["CORRECT" if c else k for k, c in zip(keys, corr)]
        n = len(vals)
        pairs = list(combinations(vals, 2))
        D = sum(a != b for a, b in pairs) / len(pairs)
        flip = D > 0
        all_answered = "NA" not in keys
        literal_same = all_answered and len(set(keys)) == 1
        if literal_same and all(corr):
            bucket = "same_correct"
        elif literal_same:
            bucket = "same_wrong"
        elif all(corr):
            bucket = "different_all_correct"
        elif all(k == "NA" for k in keys):
            bucket = "no_answer_all"
        elif flip and all_answered:
            bucket = "flip_all_answered"
        else:
            bucket = "flip_with_crash"
        assert flip == (bucket in ("flip_all_answered", "flip_with_crash")), (iid, arm, axis, vals)
        # S: how far apart a flip's answers are; flipped tuples of numeric tasks, answered values only
        S = doubled = None
        if flip and task in NUMERIC:
            xs = [x for x in (num(k) for k in keys if k != "NA") if x is not None]
            S, doubled = spread(xs, num(truth.get(iid)))
        # (b) where a flipped M2 tuple diverges
        where = None
        if flip and arm.startswith("code/"):
            bad = (sub["declared_ok"] == False).tolist()  # noqa: E712  (None = not known to be wrong)
            where = divergence_site(vals, bad, all_answered)
        if flip and arm.startswith("code/") and axis in ("relabel", "order") and task in READING_TASKS:
            reading.append(dict(dataset=ds, task=task, model=model, instance_id=iid, with_crash=not all_answered,
                                answers=[json.loads(k) for k in keys if k != "NA"]))
        rows.append(dict(dataset=ds, task=task, instance_id=iid, cluster=cluster.get(iid, iid), model=model,
                         arm=arm, axis=axis, n=n, bucket=bucket, flip=flip, D=D, robust=all(corr),
                         S=S, doubled=doubled, where=where))
T = pd.DataFrame(rows)
M12 = T[T.arm.isin(ARMS_M12)]
PERM = ["relabel", "order"]


def fam(df):
    """Rows for each factor plus the relabel+order pool."""
    return pd.concat([df.assign(factor=df.axis), df[df.axis.isin(PERM)].assign(factor="relabel+order")])


out = {"notes": __doc__}

# ---------------------------------------------------------------- invariance
F = fam(M12)
inv = {}
for (ds, arm, fac), d in F.groupby(["dataset", "arm", "factor"]):
    flips = d[d.flip]
    b = d.bucket.value_counts(normalize=True).mul(100).round(1).to_dict()
    s = d[d.S.notna()]
    inv[f"{ds}|{arm}|{fac}"] = dict(
        tuples=int(len(d)), flip_rate=round(100 * d.flip.mean(), 1), severity=round(flips.D.mean(), 2) if len(flips) else None,
        weighted_flip=round(100 * d.D.mean(), 1), robust_accuracy=round(100 * d.robust.mean(), 1), buckets=b,
        S_n=int(len(s)), S_median=round(s.S.median(), 2) if len(s) else None,
        S_doubling_pct=round(100 * s.doubled.mean(), 1) if len(s) else None,
        one_off_pct_of_4member_flips=round(100 * ((flips.n == 4) & (flips.D == 0.5)).sum() / max((flips.n == 4).sum(), 1), 1),
        all_different_pct_of_4member_flips=round(100 * ((flips.n == 4) & (flips.D == 1)).sum() / max((flips.n == 4).sum(), 1), 1))
out["invariance"] = inv

# per model flip rates (the spread across models), relabel+order
pm = F[F.factor == "relabel+order"].groupby(["dataset", "arm", "model"]).flip.mean().mul(100).round(1)
out["flip_rate_per_model_relabel+order"] = {f"{a}|{b}|{c}": v for (a, b, c), v in pm.items()}

# ---------------------------------------------------------------- performance
R = j[j["mode"].isin(["direct", "code"])]
perf = {}
for (ds, arm), g in R.groupby(["dataset", "arm"]):
    ref_canon = g[g.axis == "canonical"].correct.mean()
    ref_ident = g[(g.axis == "relabel") & g.is_identity].correct.mean()
    for fac in FACTORS:
        if fac == "relabel":
            var, ref = g[(g.axis == "relabel") & ~g.is_identity], ref_ident
        else:
            var, ref = g[g.axis == fac], ref_canon
        perf[f"{ds}|{arm}|{fac}"] = dict(answers=int(len(var)), accuracy=round(100 * var.correct.mean(), 1),
                                         reference_accuracy=round(100 * ref, 1),
                                         drop=round(100 * (var.correct.mean() - ref), 1))
    perf[f"{ds}|{arm}|all"] = dict(answers=int(len(g)), accuracy=round(100 * g.correct.mean(), 1))
out["performance"] = perf

# ---------------------------------------------------------------- localization (a)
CODE = j[j["mode"] == "code"].copy()
# A failure is a wrong answer (as in accuracy), assigned the first step that failed (copy -> solve -> run)
CODE["step"] = [failure_step(fc, dok, c) for fc, dok, c in zip(CODE.failure_class, CODE.declared_ok, CODE.correct)]
CODE["factor"] = CODE.axis.replace({"canonical": "reference", "relabel": "relabel+order", "order": "relabel+order"})
loc_a = {}
for (ds, lib, fac), g in CODE.groupby(["dataset", "library", "factor"]):
    fails = g[g.step.notna()]
    shares = fails.step.value_counts(normalize=True).mul(100).round(1).to_dict()
    loc_a[f"{ds}|{lib}|{fac}"] = dict(answers=int(len(g)), failures=int(len(fails)),
                                      failure_rate=round(100 * len(fails) / len(g), 1),
                                      shares_among_failures={k: shares.get(k, 0.0) for k in ["copy", "solve", "run", "other"]},
                                      counts={k: int((fails.step == k).sum()) for k in ["copy", "solve", "run", "other"]})
out["localization_a"] = loc_a

# ---------------------------------------------------------------- localization (b)
FB = fam(T[T.arm.str.startswith("code/") & T.flip])
loc_b = {}
for (ds, arm, fac), g in FB.groupby(["dataset", "arm", "factor"]):
    loc_b[f"{ds}|{arm}|{fac}"] = dict(flips=int(len(g)), **{k: round(100 * (g["where"] == k).mean(), 1)
                                                         for k in ["program", "crash", "copy"]})
bm = FB[FB.factor == "relabel+order"].groupby(["dataset", "arm", "model"])["where"].value_counts().unstack(fill_value=0)
out["localization_b_by_model_relabel+order"] = {f"{a}|{b}|{c}": {k: int(v) for k, v in r.items()} for (a, b, c), r in bm.iterrows()}
out["localization_b"] = loc_b

# ---------------------------------------------------------------- localization (c)
loc_c, mc_rows = {}, []
for lib in ["native", "networkx"]:
    a = T[T.arm == f"code/{lib}"].set_index(["dataset", "model", "instance_id", "axis"]).flip
    b = T[T.arm == f"graph_as_code/{lib}"].set_index(["dataset", "model", "instance_id", "axis"]).flip
    P = pd.concat([a.rename("m2"), b.rename("m3")], axis=1, join="inner").reset_index()
    for (ds, ax), g in P.groupby(["dataset", "axis"]):
        loc_c[f"{ds}|{lib}|{ax}"] = dict(tuples=int(len(g)), m2_flip=round(100 * g.m2.mean(), 1), m3_flip=round(100 * g.m3.mean(), 1))
    for (ds, model), g in P.groupby(["dataset", "model"]):
        m2o, m3o = int((g.m2 & ~g.m3).sum()), int((~g.m2 & g.m3).sum())
        p = binomtest(m2o, m2o + m3o, 0.5).pvalue if m2o + m3o else 1.0
        mc_rows.append(dict(dataset=ds, model=model, library=lib, only_m2=m2o, only_m3=m3o, p=p))
MC = pd.DataFrame(mc_rows)
MC["p_holm"] = multipletests(MC.p, method="holm")[1]
out["localization_c"] = loc_c
out["mcnemar_m2_vs_m3"] = MC.round(4).to_dict("records")

# ---------------------------------------------------------------- statistics: code vs direct
RO = M12[M12.axis.isin(PERM)]
key = ["dataset", "model", "instance_id", "axis"]
dirx = RO[RO.arm == "direct"].set_index(key)[["flip", "D"]]
mcn, wil = [], []
for lib in ["native", "networkx"]:
    code = RO[RO.arm == f"code/{lib}"].set_index(key)[["flip", "D"]]
    P = dirx.join(code, lsuffix="_dir", rsuffix="_code", how="inner").reset_index()
    for (ds, model), g in P.groupby(["dataset", "model"]):
        od, oc = int((g.flip_dir & ~g.flip_code).sum()), int((~g.flip_dir & g.flip_code).sum())
        mcn.append(dict(dataset=ds, model=model, library=lib, only_direct=od, only_code=oc,
                        p=binomtest(od, od + oc, 0.5).pvalue if od + oc else 1.0))
        diff = g.D_dir - g.D_code
        nz = diff[diff != 0]
        p = wilcoxon(g.D_dir, g.D_code, zero_method="wilcox").pvalue if len(nz) else 1.0
        wil.append(dict(dataset=ds, model=model, library=lib, pairs=int(len(g)), direct_worse=int((nz > 0).sum()),
                        code_worse=int((nz < 0).sum()), p=p))
    for ds, g in P.groupby("dataset"):
        diff = g.D_dir - g.D_code
        out[f"wilcoxon_pooled_models|{ds}|{lib}"] = dict(pairs=int(len(g)), direct_worse=int((diff > 0).sum()),
                                                         code_worse=int((diff < 0).sum()),
                                                         p=float(wilcoxon(g.D_dir, g.D_code, zero_method="wilcox").pvalue))
for name, rows_ in [("mcnemar_code_vs_direct", mcn), ("wilcoxon_D_code_vs_direct", wil)]:
    df = pd.DataFrame(rows_)
    df["p_holm"] = multipletests(df.p, method="holm")[1]
    better = "only_direct" if name.startswith("mcnemar") else "direct_worse"
    worse = "only_code" if name.startswith("mcnemar") else "code_worse"
    out[name] = df.round(4).to_dict("records")
    out[name + "_summary"] = dict(cells=len(df), code_better_sig=int(((df.p_holm < 0.05) & (df[better] > df[worse])).sum()),
                                  direct_better_sig=int(((df.p_holm < 0.05) & (df[better] < df[worse])).sum()))

# ---------------------------------------------------------------- bootstrap CIs (headline, relabel+order)
boot = {}
for ds, g in RO.groupby("dataset"):
    cl = g.cluster.unique()
    agg = g.groupby(["cluster", "arm"]).agg(flips=("flip", "sum"), dsum=("D", "sum"), n=("flip", "size")).unstack("arm")
    reps = {a: {"flip": [], "weighted": []} for a in ARMS_M12}
    reps.update({f"diff|{a}": {"flip": [], "weighted": []} for a in ARMS_M12[1:]})
    for _ in range(2000):
        s = agg.loc[RNG.choice(cl, len(cl), replace=True)].sum()
        vals = {a: (s[("flips", a)] / s[("n", a)], s[("dsum", a)] / s[("n", a)]) for a in ARMS_M12}
        for a in ARMS_M12:
            reps[a]["flip"].append(vals[a][0]); reps[a]["weighted"].append(vals[a][1])
        for a in ARMS_M12[1:]:
            reps[f"diff|{a}"]["flip"].append(vals["direct"][0] - vals[a][0])
            reps[f"diff|{a}"]["weighted"].append(vals["direct"][1] - vals[a][1])
    for a, r in reps.items():
        boot[f"{ds}|{a}"] = {m: [round(100 * float(np.percentile(v, 2.5)), 1), round(100 * float(np.percentile(v, 97.5)), 1)]
                             for m, v in r.items()}
out["bootstrap_95ci_relabel+order"] = boot

# ---------------------------------------------------------------- logistic regression (flip), clustered by graph
# One model per dataset (datasets are never pooled). Main effects only: a factor's odds ratio is averaged
# over the modes, and a mode's over the factors.
out["logit_flip"], out["logit_flip_n"] = {}, {}
for ds, L in M12.assign(y=M12.flip.astype(int)).groupby("dataset", sort=False):
    fit = smf.glm("y ~ C(arm, Treatment('direct')) + C(axis, Treatment('relabel')) + C(model) + C(task)", data=L,
                  family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(L.cluster)[0]})
    ci = fit.conf_int()
    out["logit_flip"][ds] = {k: [round(float(np.exp(fit.params[k])), 2), round(float(np.exp(ci.loc[k, 0])), 2),
                                 round(float(np.exp(ci.loc[k, 1])), 2)] for k in fit.params.index if "arm" in k or "axis" in k}
    out["logit_flip_n"][ds] = dict(tuples=int(len(L)), clusters=int(L.cluster.nunique()))

# ---------------------------------------------------------------- charts: how flips split (D) and how far apart (S)
prof = {}
F4 = M12[M12.axis.isin(PERM) & M12.flip & (M12.n == 4)]
for (ds, arm), g in F4.groupby(["dataset", "arm"]):
    vc = g.D.round(2).value_counts(normalize=True).mul(100)
    prof[f"{ds}|{arm}"] = dict(flips=int(len(g)), mean_D=round(float(g.D.mean()), 2),
                               **{f"D={k:.2f}": round(float(vc.get(k, 0.0)), 1) for k in [0.5, 0.67, 0.83, 1.0]})
out["severity_profile_4member_relabel+order"] = prof
sq = {}
for (ds, arm, fac), g in fam(M12).groupby(["dataset", "arm", "factor"]):
    s = g.S.dropna()
    if len(s):
        sq[f"{ds}|{arm}|{fac}"] = dict(n=int(len(s)), q1=round(float(s.quantile(.25)), 2),
                                       median=round(float(s.median()), 2), q3=round(float(s.quantile(.75)), 2))
out["S_quartiles"] = sq

# ---------------------------------------------------------------- graph size (Figure 1)
P = M12[M12.axis.isin(PERM)].copy()
P["m"] = P.instance_id.map(lambda i: len(inst[i]["edges"]))
P["bin"] = pd.cut(P.m, bins=[-1, 10, 20, 40, 10**6], labels=["up to 10", "11 to 20", "21 to 40", "over 40"])
G_ = P.groupby(["dataset", "bin", "arm"], observed=True).flip.agg(["sum", "size"]).reset_index()
out["flip_rate_by_edges_relabel+order"] = {f"{r.dataset}|{r.bin}|{r.arm}": dict(flips=int(r["sum"]), tuples=int(r["size"]),
                                            flip_rate=round(100 * r["sum"] / r["size"], 1)) for _, r in G_.iterrows()}
P["log2m"] = np.log2(P.m + 1)  # log2(edges + 1): one unit = a doubling of (edges + 1)
P["y"] = P.flip.astype(int)
P["arm"] = pd.Categorical(P.arm, ARMS_M12)
size_or = {}
for ds, g in P.groupby("dataset", sort=False):  # one model per dataset
    fit = smf.glm("y ~ C(arm) * log2m + C(task) + C(model)", data=g, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(g.cluster)[0]})
    size_or[ds] = {}
    for arm, terms in [("direct", ["log2m"]), ("code/native", ["log2m", "C(arm)[T.code/native]:log2m"]),
                       ("code/networkx", ["log2m", "C(arm)[T.code/networkx]:log2m"])]:
        v = pd.Series(0.0, index=fit.params.index)
        v[terms] = 1.0
        est, se = float(v @ fit.params), float(np.sqrt(v @ fit.cov_params() @ v))
        size_or[ds][arm] = [round(math.exp(est), 2), round(math.exp(est - 1.96 * se), 2), round(math.exp(est + 1.96 * se), 2)]
out["flip_odds_ratio_per_doubling_of_edges"] = size_or
wt = []
for (ds, task), g in P.groupby(["dataset", "task"]):
    if g.m.nunique() < 3:
        continue
    lo, hi = g.m.quantile(1 / 3), g.m.quantile(2 / 3)
    for arm in ["direct", "code/native"]:
        a = g[g.arm == arm]
        wt.append(dict(arm=arm, rise=100 * (a[a.m >= hi].flip.mean() - a[a.m <= lo].flip.mean())))
W = pd.DataFrame(wt)
out["within_task_rise_largest_vs_smallest_third"] = {a: dict(tasks=int((W.arm == a).sum()), rising=int((W[W.arm == a].rise > 0).sum()),
                                                            median_points=round(float(W[W.arm == a].rise.median()), 1))
                                                     for a in ["direct", "code/native"]}

# ---------------------------------------------------------------- other reading (Figure 6)
# Flipped M2 tuples (relabel and order, both libraries) of the node-list questions: does some answer in the
# tuple equal the other reading's answer? GraphQA's "connected to" questions (undirected graphs) read as either
# adjacency or reachability; Erdős `neighbor` means the successors on a directed graph and the neighbours on an
# undirected one. Tags use the answers that exist; `with_crash` counts the flips in which some serialization
# gave no answer (they are tagged on the rest).
alt = {}
for iid, r in inst.items():
    if r["task"] in READING_TASKS and r["query_args"].get("node") is not None:
        Gx = nx.DiGraph() if r["directed"] else nx.Graph()
        Gx.add_nodes_from(r["nodes"])
        Gx.add_edges_from(tuple(e[:2]) for e in r["edges"])
        q = r["query_args"]["node"]
        N, V = set(Gx.successors(q) if r["directed"] else Gx.neighbors(q)), set(Gx.nodes)
        R = None if r["directed"] else nx.node_connected_component(Gx, q) - {q}
        alt[iid] = {"connected_nodes": (N, R), "disconnected_nodes": (V - N - {q}, V - R - {q}) if R is not None else None,
                    "neighbor": (N, None)}[r["task"]]
TG = pd.DataFrame([dict(t, tag=reading_tag(t["task"], [set(a) for a in t["answers"] if isinstance(a, list)],
                                             *alt[t["instance_id"]])) for t in reading])
reading_out = {}
for (d, t, m), g in TG.groupby(["dataset", "task", "model"]):
    reading_out[f"{d}|{t}|{m}"] = dict(flips=int(len(g)), with_crash=int(g.with_crash.sum()),
                                       **{k: int(v) for k, v in g.tag.value_counts().items()})
out["other_reading_code_flips"] = reading_out

# ---------------------------------------------------------------- bootstrap CIs for the figures (relabel+order)
# Figures 2-3 pool M2 over its two libraries and the models; Figures B1-B2 keep the models apart. Same
# resampling as the headline CIs (a dataset's graphs with replacement, 2,000 times, a graph's tuples kept
# together), with its own generator so that every number above stays as it was.
RNG_FIG = np.random.default_rng(1)


def fig_rates(s):
    """Flip rate and weighted flip rate per (model or "all", mode) from summed cluster counts."""
    res = {}
    for md in ("M1", "M2"):
        f, d, n = (sum(s[(c, m, md)] for m in MODELS) for c in ("flips", "dsum", "n"))
        res[("all", md)] = (f / n, d / n)
        for m in MODELS:
            n_m = s[("n", m, md)]
            res[(m, md)] = (s[("flips", m, md)] / n_m, s[("dsum", m, md)] / n_m) if n_m else (np.nan, np.nan)
    return res


fig_ci = {}
for ds, g in RO.assign(mode2=np.where(RO.arm == "direct", "M1", "M2")).groupby("dataset"):
    cl = g.cluster.unique()
    agg = (g.groupby(["cluster", "model", "mode2"]).agg(flips=("flip", "sum"), dsum=("D", "sum"), n=("flip", "size"))
           .unstack(["model", "mode2"]).fillna(0))
    point = fig_rates(agg.sum())
    reps = {k: ([], []) for k in point}
    for _ in range(2000):
        for k, (fl, wt) in fig_rates(agg.loc[RNG_FIG.choice(cl, len(cl), replace=True)].sum()).items():
            reps[k][0].append(fl)
            reps[k][1].append(wt)
    for (who, md), (fl, wt) in reps.items():
        fig_ci[f"{ds}|{who}|{md}"] = {
            name: [round(100 * float(point[(who, md)][i]), 1), round(100 * float(np.nanpercentile(v, 2.5)), 1),
                   round(100 * float(np.nanpercentile(v, 97.5)), 1)]
            for i, (name, v) in enumerate([("flip", fl), ("weighted", wt)])}
out["figure_intervals_relabel+order"] = fig_ci  # {dataset|model or all|M1 or M2: {flip|weighted: [rate, lo, hi]}}

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1, default=float)
print(f"{len(T)} tuples ({len(M12)} direct and code) -> {args.out}")
