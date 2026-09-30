# Prompt-hint experiments (GraphQA, two models)

Two one-sentence prompt hints, run on GraphQA with DeepSeek-V3.1 (`hf-deepseek-v3.1-novita`) and Qwen3-8B (`hf-qwen3-8b-nscale`), code from text (M2) only, in both library arms. The paper reports the deduplication hint in Appendix C. These tables are data only: the code that ran the hints and wrote the tables is not ported to this repository (see "Not ported" below).

## The hints

From `PROMPT_HINTS` in `src/gsi/prompts/modes.py` of the original development repository, at commit `9e17199^`. A hint is added only to undirected graphs. In a code prompt it follows the graph notes of the instruction.

| Key | Sentence |
|---|---|
| `dedup_undirected` | An edge may be listed in both directions; (u, v) and (v, u) are the same edge and must be counted once. |
| `direct_connection` | A node is connected to another node only if there is an edge directly between them. |

## The runs and their tables

| Folder | Hint | Config | Records per model and arm | Tables |
|---|---|---|---|---|
| `dedup/` | `dedup_undirected` | `configs/mitigation_dedup.yaml` | structure factor only: 100 graphs x 7 tasks x 3 forms (reference, adjacency list, each edge in both directions) = 2,100 | `accuracy_by_task`, `accuracy_by_task_form`, `doubled_degree`, `other_tasks`, `structure_invariance` |
| `dedup_full/` | `dedup_undirected` | `configs/mitigation_dedup_full.yaml` | every GraphQA variant (reference, relabel, order, structure, syntax) = 8,925 | `accuracy_overall`, `accuracy_by_axis`, `accuracy_by_task`, `axis_invariance` |
| `disconnected/` | `direct_connection` | `configs/mitigation_disconnected.yaml` | `disconnected_nodes`, reference form only = 100 | `accuracy`, `interpretation` |

The original notes (`MITIGATION.md`) call the `disconnected/` run a diagnostic, not a mitigation: it is specific to one task and checks the reading of "disconnected" rather than anything about the serialization.

The three configs at `9e17199^` differ from `configs/graphqa.yaml` in `name` and in the settings below (`variants` and `keep_tasks` sit under `datasets.graphqa` in the files). `prompt_hint` and `keep_tasks` are not config keys in this repository, so the current loader rejects these files.

```yaml
# mitigation_dedup.yaml
prompt_hint: dedup_undirected
variants: {structure: {toggle: true, replicated: true}}
modes: [code]
libraries: [networkx, native]
models: [hf-deepseek-v3.1-novita, hf-qwen3-8b-nscale]

# mitigation_dedup_full.yaml: as above, with every factor of graphqa.yaml
prompt_hint: dedup_undirected
variants: {relabel: {seeds: 3}, order: {kinds: [s_sorted_t_shuffled, t_sorted_s_shuffled, shuffle_all], seeds: 1},
           structure: {toggle: true, replicated: true}, syntax: {kinds: [plain, json, networkx_code]}}

# mitigation_disconnected.yaml
prompt_hint: direct_connection
keep_tasks: [disconnected_nodes]   # all seven tasks are generated so that the graphs match the original run
variants: {}                       # reference form only
```

All three use the same 100 GraphQA graphs (`n_graphs: 100`, `seed: 0`) and `exec: {timeout: 20, mem_mb: 1024, workers: 4}`.

## Baseline, and why the comparison is not clean

Each hinted record is paired, by `record_id`, with the same record of the published GraphQA run (`results/graphqa/scored/`, restored by `scripts/restore_from_hf.py`). The pairs share the graph, task, form, arm and model; the prompts differ by the hint sentence. We checked that every baseline column matches the published records: all 84 rows of `dedup/accuracy_by_task_form.csv`, the four rows of `dedup_full/accuracy_overall.csv`, and `disconnected/accuracy.csv` (11, 12, 93 and 46 of 100).

The hint runs were made later than the baseline (2026-09-18/19) and on another machine: Windows for the hints, Linux for the baseline. Program execution is deterministic, but provider-side drift between the two dates is not excluded. A difference between baseline and hint is therefore the hint plus whatever re-asking alone changes. For the size of that effect, see the repeat floor in `../noise_floor.json` and `../../floor_stats_*`.

## Metric conventions

- Accuracy is the share of records whose `correct` is true.
- `I` (in `structure_invariance` and `axis_invariance`) is the share of instances whose answers are literally identical on every form of the factor, with every answer parsed. This is the old `audit_results.per_instance` rule, not the paper's flip rule: two different correct answers count as a change. `R` is the share of instances where every form is correct.
- `axis_invariance` averages over tasks. Its `permutation` rows pool relabeling and order.
- Intervals are paired bootstrap intervals over graphs, conditional on the collected responses.

The numbers in Appendix C come from these tables:
- node degree in plain Python, from `dedup/accuracy_by_task_form.csv`: DeepSeek-V3.1 goes from 34 (adjacency) and 19 (both directions) to 100; Qwen3-8B goes to 87, and its reference-form degree falls from 99 to 86;
- Qwen3-8B's `I`, from `dedup_full/axis_invariance.csv`: it falls by 6.3 to 13.1 points under relabeling, order and syntax, in both arms.

## Provenance

The tables come from the original development repository (`ml_graphs`) at `9e17199^` (= `c1e7f99`, 2026-09-21), the parent of `9e17199`, "Remove mitigation experiments and results from main". They were copied from `results/mitigation_dedup/tables/`, `results/mitigation_dedup_full/tables/` and `results/mitigation_disconnected/tables/`, and are byte-identical to those files (same git blob hashes).

| Commit | Date | What |
|---|---|---|
| `01ff5c7` | 2026-09-18 | prompt hints (`PROMPT_HINTS`, config key `prompt_hint`) and `configs/mitigation_dedup.yaml` |
| `68783c7` | 2026-09-18 | the `direct_connection` hint, `keep_tasks`, `configs/mitigation_disconnected.yaml` |
| `ca65af1` | 2026-09-18 | `scripts/mitigation_compare.py` (pairs hinted and baseline records, writes the tables); `disconnected/` tables |
| `b41055a` | 2026-09-18 | `dedup/` tables |
| `bd2d029` | 2026-09-19 | `configs/mitigation_dedup_full.yaml`, `dedup_full/` tables |
| `12e3504`, `e0b0444` | 2026-09-18/19 | `MITIGATION.md`, the write-up |

The model replies of the hint runs are in the Hugging Face dataset [`Dolevabudi/graph-serialization-invariance-full`](https://huggingface.co/datasets/Dolevabudi/graph-serialization-invariance-full), config `prompt_hint_replies`. The dataset also holds these tables under `results/mitigation/prompt_hint/`.

## Not ported

The hint code is not in this repository: `PROMPT_HINTS`, the `hint` argument of `build_prompt`, the config keys `prompt_hint` and `keep_tasks`, and `scripts/mitigation_compare.py`. The tables cannot be regenerated here. They are kept as the record of runs the paper cites.
