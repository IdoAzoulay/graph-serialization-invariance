# Mitigation experiments (separable)

This folder holds the experiments built on top of the main measurement: the noise floor, using disagreement across serializations as an error detector, voting, and two ways to remove the serialization from the prompt. They live apart from the main pipeline so that they can be dropped as one unit. `scripts/reproduce.sh` does not run them, and nothing outside this folder imports from it.

| Id | Script | Committed result | What it measures | Needs |
|---|---|---|---|---|
| E0 | `scripts/nondeterminism_floor.py` | `results/<ds>/noise_floor.json` | How often the identical prompt, asked again, changes its answer: the floor for every flip rate | 4 repeats per prompt (model calls, or the replies in the HF config `noise_floor_replies`) |
| E1/E2 | `scripts/permutation_signal.py` | `results/<ds>/permutation_signal.csv` | Disagreement across permutations as an error detector (E1), and majority voting with and without abstention (E2) | scored records only; offline |
| E3 | `scripts/canonical_labeling.py` | `results/<ds>/canonical_labeling.csv`, `canonical_labeling_invariance.csv` | Rendering every relabeling and reordering to one canonical prompt; `--invariance-only` checks the schemes without a model | new prompts (model calls, or the HF config `canonical_labeling`); `--invariance-only` is offline |
| E5 | `scripts/solver_per_task.py` | `results/<ds>/solver_per_task.csv` | One program per task, with no graph in the prompt, run on every instance | one model call per task and arm (HF config `solver_per_task`), then the sandbox |
| E6 | `scripts/program_voting.py` | `results/<ds>/program_voting.csv` | Running every variant's M2 program on the canonical graph and voting over their outputs | the M2 programs (the `code` column of the HF records) and the sandbox; no model calls |

`<ds>` is `graphqa` or `erdos`. The scripts write their tables straight into `mitigation/results/<ds>/`. Intermediate records (`results/<ds>/scored_canon/`, `results/<ds>/scored_solver/`, the reply cache `results/cache/`) are git-ignored.

Run the tests with `python -m pytest mitigation/tests`. The main suite (`python -m pytest`) does not collect them.

## Where the paper uses these results

| Paper location | Experiment |
|---|---|
| Section 4.2 "Statistics", last sentence, and Section 5.1: the resampling floor ("exceeds that floor in all 8 cells") | E0 |
| Abstract, second paragraph ("flagging instances whose answers change ... catches more than half of the silent errors"); Section 5.4 "What the sensitivity is good for"; contribution (iv) in the Introduction; Conclusion | E1/E2 |
| Appendix C "Mitigation": detector table, answer voting, program voting, canonical labeling, solver per task | E1/E2, E6, E3, E5 |
| Appendix C, the deduplication hint (two models) | not in this repository: its replies and tables are only in the HF dataset (`prompt_hint_replies`, `results/mitigation/prompt_hint/`) |
| Conclusion and Limitations: "the repeat baseline", pre-registered detection, per-task-solver and canonical-labeling targets | E0, E1, E5, E3 |

## Rerunning

Every command runs from the repository root. Only E1/E2, E3 `--invariance-only` and E6 can run without model calls. E6 needs `results/<ds>/responses/<model>.jsonl`, which is not restored by `scripts/restore_from_hf.py`.

```bash
python mitigation/scripts/permutation_signal.py --config configs/graphqa.yaml \
    --models hf-qwen3-8b-nscale hf-deepseek-v3.1-novita hf-gemma-4-31b hf-deepseek-v4-flash
python mitigation/scripts/canonical_labeling.py --config configs/graphqa.yaml --invariance-only
```

With the scored records restored (`scripts/restore_from_hf.py`), both commands reproduce the committed tables byte for byte. The model order sets the row order.

The committed tables of E0, E3, E5 and E6 come from the runs described in the paper. Rerunning them calls the models for every reply that is not in `results/cache/`.

## Removing the mitigation experiments

1. `rm -rf mitigation/`. Nothing else in the repository imports from it, and the main tests and `scripts/reproduce.sh` do not use it.
2. Optional: `src/gsi/serial/canonical.py` and `tests/test_canonical.py` are used only by E3 and can go too.
3. Remove the paper text that cites these results (the table above), and the "mitigation/" line in the top-level README.
