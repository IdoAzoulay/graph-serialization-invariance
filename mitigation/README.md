# Mitigation experiments (separable)

This folder holds the experiments built on top of the main measurement: the repeat-noise floor, disagreement across serializations used as an error detector, voting, two ways to remove the serialization from the prompt, and two prompt hints. They live apart from the main pipeline so that they can be dropped as one unit. `scripts/reproduce.sh` does not run them, and nothing outside this folder imports from it.

| Id | Script | Committed result | What it measures | Needs |
|---|---|---|---|---|
| E0 | `scripts/nondeterminism_floor.py`, `scripts/floor_stats.py` | `results/<ds>/noise_floor.json`, `results/<ds>/noise_floor_instances.jsonl`; `results/floor_stats_{literal,flip}_{cells,pooled,modes}.csv` | How often the identical prompt, asked again, changes its answer: the floor for every flip rate. `floor_stats.py` tests each flip rate against it: per cell, per model, and per dataset x mode | the floor: 4 repeats per prompt (model calls, or the replies in the HF config `noise_floor_replies`); `floor_stats.py`: the committed per-instance rows only, offline |
| E1/E2 | `scripts/permutation_signal.py` | `results/<ds>/permutation_signal.csv`; `results/<ds>/permutation_signal_fliprule.csv` (`--rule flip`) | Disagreement across permutations as an error detector (E1), and majority voting with and without abstention (E2) | scored records only; offline |
| E3 | `scripts/canonical_labeling.py` | `results/<ds>/canonical_labeling.csv`, `canonical_labeling_invariance.csv` | Rendering every relabeling and reordering to one canonical prompt; `--invariance-only` checks the schemes without a model | new prompts (model calls, or the HF config `canonical_labeling`); `--invariance-only` is offline |
| E5 | `scripts/solver_per_task.py` | `results/<ds>/solver_per_task.csv` | One program per task, with no graph in the prompt, run on every instance | one model call per task and arm (HF config `solver_per_task`), then the sandbox |
| E6 | `scripts/program_voting.py` | `results/<ds>/program_voting.csv` | Running every variant's M2 program on the canonical graph and voting over their outputs | `results/<ds>/responses/<model>.jsonl` (the M2 programs; the HF records hold them in their `code` column, but no script here restores that file) and the sandbox; no model calls |
| hints | none (not ported) | `results/graphqa/prompt_hint/{dedup,dedup_full,disconnected}/` | A one-sentence hint to count duplicate edges once, and one defining "connected"; two models, GraphQA, code arms | tables only; see `results/graphqa/prompt_hint/README.md` |

`<ds>` is `graphqa` or `erdos`. The scripts write their tables straight into `mitigation/results/<ds>/` (`floor_stats.py`, which covers both datasets, into `mitigation/results/`). Intermediate records (`results/<ds>/scored_canon/`, `results/<ds>/scored_solver/`, the reply cache `results/cache/`) are git-ignored.

E3's accuracy table covers a subset: `--limit 400` on GraphQA, which is the first 400 instances (58 of the 100 graphs), and `--limit 150` on Erdős, every 8th instance. Its intervals are normal intervals over instances, not clustered by graph. The docstring of `scripts/canonical_labeling.py` has the details.

Run the tests with `python -m pytest mitigation/tests`. The main suite (`python -m pytest`) does not collect them.

## Metric conventions

E0 (`nondeterminism_floor.py`, `noise_floor.json`) and E1/E2 (`permutation_signal.csv`) compare answers literally, unless run with `--rule flip`. Literally, a group of answers agrees only if every answer parsed and all answer keys are equal. So two different correct answers (two shortest paths) count as a change, and so does the same crash on every serialization.

The paper's flip rule (Section 4.2) counts every correct answer as the same answer and every missing answer as one shared value. `floor_stats.py --rule flip` and `permutation_signal.py --rule flip` apply it and write separate files. The flip-rule detector uses the true answer, so it is a measurement convention, not a detector anyone could deploy. The flags on answered errors (`caught_answered`, `recall_answered`) and the vote are the same under both rules. Only the flag rate, precision, recall with crashes counted, and the abstention columns change.

E5's invariance columns, E6's agreement and the prompt hints' `I` are literal as well. Accuracy is the scored `correct` field everywhere.

## Where the paper uses these results

| Paper location | Experiment | File and columns |
|---|---|---|
| Section 4.2 "Statistics", last sentence: identical prompts asked again, as a floor for the flip rate | E0 | `results/<ds>/noise_floor.json` |
| Section 5.1, the repeat-noise paragraph | E0, flip rule | `results/floor_stats_flip_modes.csv`: rows `model = all` (four models counting equally) and per model; per-cell tests in `floor_stats_flip_cells.csv` |
| Appendix D, Table D2: floor, flip rate on the same instances, excess, graph-bootstrap 95% interval, per model x dataset x mode | E0, flip rule | `results/floor_stats_flip_modes.csv`: columns `floor`, `flip`, `excess`, `boot_lo`, `boot_hi` |
| Section 5.4 "What the sensitivity is good for"; Conclusion | E1/E2 | `results/<ds>/permutation_signal.csv` |
| Appendix C, Table C1 | E1/E2 | `permutation_signal.csv`, rows `probe = both`, `k = all`, arm `code/native`: Flag = `flag_rate`, Caught = `recall_answered`, Abst. = `abstain_acc` (`abstain_coverage`). The NetworkX columns proposed for Table C1 are the same rows with arm `code/networkx`. These are literal-rule values; the flip-rule ones are in `permutation_signal_fliprule.csv`, with the same rows. "15 of 18 at 4.4%" is GraphQA, Qwen3-8B, `code/native`, `probe = relabel`, `k = 2` (`caught_answered` / `errors_answered`, `flag_rate`). |
| Appendix C, answer voting ("−2.4 to +6.4 points") | E2 | `permutation_signal.csv`, `vote_delta`, rows `probe = both`, `k = all`, code arms |
| Appendix C, program voting | E6 | `results/<ds>/program_voting.csv` |
| Appendix C, canonical labeling | E3 | `results/<ds>/canonical_labeling.csv` (scheme `canonical`), `canonical_labeling_invariance.csv` |
| Appendix C, one solver per task | E5 | `results/<ds>/solver_per_task.csv` |
| Appendix C, the deduplication hint (two models) | hints | `results/graphqa/prompt_hint/dedup/`, `dedup_full/` |
| Limitations and Conclusion: pre-registered detection, per-task-solver and canonical-labeling targets | E1, E5, E3 | the section below |

## Pre-registered criteria

`plans/robust-serialization.md` §5 in the original development repository, commit `d529e85` (2026-09-21), "written before any of E1–E5 is run, so they cannot be moved afterwards":

> 1. **Detection is real** if E1 catches ≥70% of wrong answers at ≤15% flag rate on ≥3 of 4 models in the native code arm.
> 2. **Voting is safe** if E2 never lowers code-arm accuracy by more than the floor interval, and improves it on ≥2 models.
> 3. **Canonical labeling is free** if E3's best scheme is within 1 pp of the random-label baseline on both datasets (better is a bonus, not required).
> 4. **A hint is adopted** only if it improves its target failure on ≥3 models with no regression on any other axis beyond the floor interval.
> 5. **Decoupling works** if E5 is within 5 pp of M2 on both datasets.
> 6. **The recipe is a result** if it reduces the floor-corrected flip rate by ≥50% on ≥3 models while accuracy does not drop.

Outcomes, checked against the committed tables:

| # | Outcome | Evidence |
|---|---|---|
| 1 | **Not met**: 2 of 4 models on each dataset. GraphQA: Qwen3-8B (16 of 18 errors caught at 9.1% flagged) and Gemma 4 31B (12 of 16 at 5.1%) pass; DeepSeek-V3.1 (1 of 90) and DeepSeek-V4-Flash (40 of 152) fail. Erdős: DeepSeek-V3.1 (2 of 2 at 8.1%) and DeepSeek-V4-Flash (17 of 23, 74%, at 9.5%) pass, but only if crashes count as caught: on silent errors V4-Flash catches 7 of 13. Qwen3-8B catches 75% but flags 17.1%; only a probe chosen afterwards, three relabelings, reaches 70% at 14.0%. Gemma makes no plain-Python errors on Erdős. Under the flip rule Qwen3-8B's Erdős flag rate is 11.8%, but that rule uses the true answer, and GraphQA stays at 2 of 4. | `permutation_signal.csv`, arm `code/native`, `probe = both`, `k = all`: `recall`, `recall_answered`, `flag_rate` |
| 2 | **Met if "the floor interval" means the floor's 95% upper bound; missed by the point estimate in one arm.** Voting moves code-arm accuracy by −2.4 to +6.4 points and gains on three models (Qwen3-8B, DeepSeek-V3.1, DeepSeek-V4-Flash). Every loss is within its arm's floor, the largest being DeepSeek-V4-Flash's GraphQA plain Python (−2.4 against a floor of 4 of 100), except DeepSeek-V3.1's GraphQA plain Python: −0.3 (2 of 700), where the floor is 0 of 100 (95% upper bound 3.6%). | `permutation_signal.csv`, `vote_delta` (`probe = both`, `k = all`); `noise_floor.json`, `floor` |
| 3 | **Met for DeepSeek-V3.1 and Gemma 4 31B only**, with the exact canonical scheme, the only one run on all four models and the only one that is a canonical labeling. Across both datasets and all five arms, their worst difference from random labels is 0.0 and −0.4 points. Qwen3-8B loses 2.0 and DeepSeek-V4-Flash 1.25 points against random labels, both in GraphQA plain Python. The pre-registered heuristic schemes (`degree_wl`, `bfs_min`, run on Qwen3-8B and DeepSeek-V3.1 only) are not canonical: they give 54–58% (GraphQA) and 25–27% (Erdős) of instances one prompt across relabelings. Qwen3-8B's `bfs_min` would stay within 1 point (worst −0.9). | `canonical_labeling.csv`, `delta_vs_random_relabel`; `canonical_labeling_invariance.csv`, `instances_all_same` |
| 4 | **Not run.** E4, three hints on four models, was not carried out. The one hint that was run, deduplication on two models (`results/graphqa/prompt_hint/`), could not have met it. It fixes its target on only those two models, fewer than three. And it lowers Qwen3-8B's invariance under relabeling, order and syntax by 6.3–13.1 points, with every interval excluding 0. | `prompt_hint/dedup/accuracy_by_task_form.csv`, `prompt_hint/dedup_full/axis_invariance.csv` |
| 5 | **Missed**: no model is within 5 points of per-instance M2 in both arms on both datasets. On Erdős, DeepSeek-V3.1, Gemma 4 31B and DeepSeek-V4-Flash are within 1.2 points. On GraphQA each of them loses about 10–11 points in at least one arm: DeepSeek-V3.1 NetworkX −11.0, Gemma −10.1 and −10.4, V4-Flash NetworkX −10.1. Qwen3-8B is within 5 on GraphQA (−4.4, +2.6) but loses 6.6 in its Erdős NetworkX arm. | `solver_per_task.csv`, `delta_acc` |
| 6 | **Not run** (the combined recipe). | — |

## Rerunning

Every command runs from the repository root. Only E1/E2, E0's statistics (`floor_stats.py`), E3 `--invariance-only` and E6 can run without model calls. E1/E2 need the scored records (`scripts/restore_from_hf.py`). `floor_stats.py` needs only the committed per-instance rows; with `--rule flip` it also checks its correctness against the scored records when they are restored. E3 needs the variants, which are git-ignored and rebuilt offline by `scripts/run.py --stages variants`. E6 needs `results/<ds>/responses/<model>.jsonl`, which `scripts/restore_from_hf.py` does not restore.

```bash
python mitigation/scripts/permutation_signal.py --config configs/graphqa.yaml \
    --models hf-qwen3-8b-nscale hf-deepseek-v3.1-novita hf-gemma-4-31b hf-deepseek-v4-flash
python mitigation/scripts/permutation_signal.py --config configs/graphqa.yaml --rule flip \
    --models hf-qwen3-8b-nscale hf-deepseek-v3.1-novita hf-gemma-4-31b hf-deepseek-v4-flash
python mitigation/scripts/floor_stats.py --rule literal   # asserts it reproduces noise_floor.json
python mitigation/scripts/floor_stats.py --rule flip
python scripts/run.py --config configs/graphqa.yaml --stages variants
python mitigation/scripts/canonical_labeling.py --config configs/graphqa.yaml --invariance-only
```

Repeat the per-dataset commands with `configs/erdos.yaml`. With the scored records restored, these commands reproduce the committed tables byte for byte. The model order sets the row order of `permutation_signal*.csv`. `floor_stats.py` draws its graph bootstrap with a fixed seed: 10,000 resamples, seed 20260928, the same draws for both rules.

The committed tables of E0 (`noise_floor.json` and the per-instance rows), E3, E5 and E6 come from the runs described in the paper. Rerunning E0, E3 or E5 calls the models for every reply that is not in `results/cache/`; E6 only runs programs. The prompt-hint tables cannot be regenerated here (see their README).

## Removing the mitigation experiments

1. `rm -rf mitigation/`. Nothing else in the repository imports from it, and the main tests and `scripts/reproduce.sh` do not use it.
2. Optional: `src/gsi/serial/canonical.py` and `tests/test_canonical.py` are used only by E3 and can go too.
3. Remove the paper text that cites these results (the table above), and the "mitigation/" line in the top-level README.
