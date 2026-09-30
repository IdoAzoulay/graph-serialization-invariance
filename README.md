# How Invariant is Code Generation to Graph Serialization?

Code and data for the paper *How Invariant is Code Generation to Graph Serialization? Measuring and Localizing the Fragility* ([paper/main.pdf](paper/main.pdf)).

Ram Kedem, Yehonatan Barel, Ido Azoulay, Dolev Abudi

Language models often answer graph questions better by writing code than by answering directly. We ask whether that code is also *invariant*: does it give the same answer when the same graph is written differently? Each graph is rewritten in equivalent serializations that change one factor at a time (node relabeling, edge order, structure, syntax; a relabeling also reorients about half of the undirected edges, see [Known issues in the published prompts](#known-issues-in-the-published-prompts)), and every serialization is posed in three modes:

| Mode | The model sees | It returns |
|---|---|---|
| M1, direct answering | the serialized graph and the question | an answer |
| M2, code from text | the serialized graph and the question | a program that copies the graph into `nodes`/`edges` and solves the question; we run it |
| M3, Graph-as-Code | only the task and the question | a program over `nodes`/`edges` that we fill with the graph at execution |

The code modes run in two arms: plain Python only, or NetworkX allowed. Four models (Qwen3-8B, Gemma 4 31B, DeepSeek-V4-Flash, DeepSeek-V3.1) answer every serialization on two datasets: GraphQA (700 instances) and 12 tasks of Erdős (1,200 instances). That makes 463,420 scored answers, plus 113,968 in the Erdős M3 rerun that states the graph type.

## Quick start: reproduce the results offline

Every model reply is published in the Hugging Face dataset [`Dolevabudi/graph-serialization-invariance-full`](https://huggingface.co/datasets/Dolevabudi/graph-serialization-invariance-full). Reproducing the tables, the paper's metrics and its figures needs no API key and calls no model.

```bash
git clone https://github.com/dolev98/graph-serialization-invariance.git
cd graph-serialization-invariance
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,reproduce]"
scripts/reproduce.sh          # about 5 minutes; add --pdf to rebuild paper/main.pdf with latexmk
git status                    # CSV/JSON outputs: no changes expected
```

`reproduce.sh` downloads the scored records (pinned dataset revision), then regenerates:

| Output | Script |
|---|---|
| `results/{graphqa,erdos}/tables/*.csv` (all models) and `tables/<model>/*.csv`, `results/*/figs/*.png` | `scripts/analyze.py` |
| `results/erdos_m3fix/tables/*.csv`: the Erdős M3 rerun that states the graph type | `scripts/analyze.py` |
| `results/erdos_m3fix/tables/compare/*.csv`: before/after the graph-type sentence | `scripts/m3fix_compare.py` |
| `results/erdos_m3fix/tables/ablation/*.csv`: graph-type omission ablation | `scripts/graph_type_ablation.py` |
| `results/review/{summary.json,paired_intervals.csv}`: descriptive audit, paired graph bootstrap | `scripts/audit_results.py` |
| `results/review/paper_metrics.json`: the paper's metric set for the main experiment | `scripts/paper_metrics.py` |
| `results/review/prompt_bug_check.json`: offline impact of the two prompt defects | `scripts/prompt_bug_check.py` |
| `mitigation/results/<dataset>/permutation_signal{,_fliprule}.csv`: the disagreement detector and voting | `mitigation/scripts/permutation_signal.py` |
| `mitigation/results/floor_stats_*.csv`: the repeat floor against the flip rates (Table D2) | `mitigation/scripts/floor_stats.py` |
| `paper/figures/*.pdf` | `paper/figures/make_figures.py` |

Every CSV and JSON file comes out byte-identical to the committed one, and so do the figures' contents (their bytes carry the Matplotlib version and a timestamp). Checked with Python 3.14.6, pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, networkx 3.7 and matplotlib 3.11.2.

Every figure reads its numbers from `paper_metrics.json`. The figure files keep their original names; in the paper they are Figure 1 (`fig2_flip_by_mode`), Figure 2 (`fig4_graph_size`), Figures B1-B5 (`fig3_weighted_by_mode`, `figB1_flip_by_model`, `figB2_weighted_by_model`, `fig5_where_flips_start`, `fig6_other_reading`) and Figure D1 (`fig1_tuple`). The LaTeX source in `paper/` is still the version of the snapshot, written before commit d987c4f fixed four metric computations, so some of its printed numbers differ from `paper_metrics.json`; the final paper replaces it.

Tests: `.venv/bin/python -m pytest` (the main suite, about 35 s, no network) and `.venv/bin/python -m pytest mitigation/tests`.

## Repository layout

```
src/gsi/            the pipeline package
  data/             GraphQA generation (Fatemi et al. recipe) and the Erdős loader; ground truth
  serial/           serialization variants (relabel, order, structure, syntax) and rendering
  prompts/          the M1/M2/M3 prompts and library arms
  llm/              OpenAI-compatible client with an on-disk response cache; a stub model for tests
  exec/             sandbox that runs generated programs and recovers the graph they built
  score/            answer parsing, mapping node answers back, comparison, failure classes
  experiment/       config loading and the stages: data, variants, llm, exec, score
  analysis/         invariance, accuracy and failure tables, figures
scripts/            command-line entry points (see the table above and "Running the pipeline")
configs/            experiment configs (graphqa, erdos, erdos_m3fix, smoke) and models.yaml
data/processed/     the sampled instances (instances.jsonl), pinned so the graphs never drift
results/            committed tables and figures; scored records are restored here (git-ignored)
paper/              LaTeX source, figures and the compiled PDF
mitigation/         separable mitigation experiments (noise floor, detection, voting, canonical
                    labeling, one solver per task); see mitigation/README.md
tests/              pytest suite for the main pipeline
tmp/                files probably no longer needed (the pilot config, unused model entries, a cost
                    monitor); see tmp/README.md
```

## Running the pipeline from scratch

This calls paid model APIs. The published run used the Hugging Face Inference Providers router. Copy `.env.example` to `.env` and set `HF_TOKEN`; the models and providers are in `configs/models.yaml`. Install with `pip install -e ".[dev,erdos]"`. `scripts/run_model.sh <model>` runs every stage of both datasets and of the Erdős M3 rerun for one model.

```bash
python scripts/run.py --config configs/smoke.yaml --models stub     # offline end-to-end check with a stub model
python scripts/run.py --config configs/graphqa.yaml --estimate      # prompt and token counts, no calls
python scripts/run.py --config configs/graphqa.yaml --models hf-qwen3-8b-nscale --stages variants,llm,exec,score
python scripts/analyze.py --config configs/graphqa.yaml
```

Replies are cached under `results/cache/<model>/<prompt_hash>.json`, and every stage resumes where it stopped. A stored reply whose prompt has changed is asked again. The `data` stage keeps the committed `data/processed/<dataset>/instances.jsonl`. `--rebuild-data` regenerates it, but GraphQA's SBM graphs come out different under networkx 3.7 or later, and Erdős is downloaded from `PKU-ML/Erdos` at a pinned revision.

The published records were produced by this pipeline before the fixes listed in the commit history (execution retries a timeout, gives programs an empty stdin, and keys its cache on the limits). None of them changes a prompt: every published record's prompt rebuilds with the same hash.

## Known issues in the published prompts

Two defects remain in the prompts that every published record answers. They are documented, not fixed in the data: fixing them changes the prompts of 19,347 records per model (19,068 distinct prompts, 18,996 of them never asked: about 76,000 calls for the four models) and re-executes 21,552 Graph-as-Code programs per model, over both Erdős runs, on re-oriented edges. That rerun has not been done.

- **A relabeling also reorients edges.** It renames both ends of every edge and re-sorts the list, but does not write an undirected edge smaller label first, as Herbst et al.'s relabeled example does. About half of a relabeling's edges come out larger label first (50.4% on GraphQA, 50.2% on Erdős), against 0% of the GraphQA reference edges and 25.7% of the Erdős ones. Each relabeling flips the written orientation of 50% of its reference's edges on average (5-95%: 20-80% on GraphQA, 27-73% on Erdős), so the relabel factor changes the labels and the orientation together.
- **The edgeless graph has an empty edge list.** graphqa-er-0025 has no edges, and its GraphQA edge lists read "The edges in G are: ." (42 serializations of its 7 instances; the graph has 1,400 answers).

`prompt_fixes: true` in a config applies the fixes of the unmerged branch `fix/prompt-bugs` (073f2f0): relabelings and their sorted identity reference write undirected edges smaller label first, and an empty GraphQA edge list reads "(none)". It defaults to `false`, under which every published prompt rebuilds byte for byte. To run it, copy a config under a new `name` with `prompt_fixes: true`, copy `data/processed/<dataset>/instances.jsonl` to `data/processed/<name>/` (otherwise the data stage regenerates the instances), and run the stages from `variants` on. Such variants carry `prompt_fixes: true` in their params, and a `variants.jsonl` built under the other setting is refused.

`scripts/prompt_bug_check.py` (run by `reproduce.sh`) measures the impact offline and writes `results/review/prompt_bug_check.json`. It also checks that the prompts of all 144,347 records per model rebuild with their published hash.

- Read as ordered pairs, every relabeled edge list is the renamed reference list (2,099 of 2,099 on GraphQA, 3,369 of 3,369 on Erdős). A program that treats the pairs as a graph, directed or not, therefore gives the renamed answer. The lists are re-sorted, though, so list order and label values still change. Graph-as-Code relabel tuples that one program answers (the question names no node) flip 0 of 2,408 times on GraphQA and 19 of 4,808 on Erdős, in the first run and in the rerun alike. In each run all 19 come from two programs that start at the first listed node: Qwen3-8B's `has_cycle` and DeepSeek-V3.1's `diameter`.
- Within relabel tuples, an answer is not more likely to differ from the reference when more edges are reoriented. The odds ratio per +10 points is 0.97-1.05 for direct answers and both code arms on both datasets, and every 95% interval includes 1.
- On questions that name no node, where relabeling leaves the wording unchanged, code differs from the reference about as often under relabeling as under reordering. Per answer pair on undirected graphs, plain Python gives 0.14 vs 0.18% on GraphQA and 2.55 vs 2.47% on Erdős, and NetworkX 1.52 vs 1.72% and 1.55 vs 1.34%. Direct answers differ 0.4 points (GraphQA) and 0.9 points (Erdős) more often.
- Dropping graphqa-er-0025 moves an arm's accuracy by at most 0.075 points (0.15 for one kind of serialization) and robust accuracy by at most 0.05 on the relabel+order pool (0.12 for a single factor). Table 2's flip rates move by at most 0.19 points: syntax NetworkX goes from 13.61 to 13.42, syntax plain Python from 4.82 to 4.65, and relabel NetworkX from 7.21 to 7.11.

A caveat before rerunning with the fix: writing each edge smaller label first breaks the renamed-list property above. A program that reads undirected pairs as directed edges then sees a different directed graph under each relabeling. Programs like the Erdős `neighbor` ones that read undirected graphs as directed (paper, Section 5.5) could then flip under relabeling because of the fix itself.

## Data sources

- GraphQA: graphs generated with the recipe of Fatemi et al., *Talk like a Graph: Encoding Graphs for Large Language Models* (ICLR 2024).
- Erdős: the published suite of Guo et al. (`PKU-ML/Erdos` on Hugging Face), as used by Herbst et al.
