# How Invariant is Code Generation to Graph Serialization?

Code and data for the paper *How Invariant is Code Generation to Graph Serialization? Measuring and Localizing the Fragility* ([paper/main.pdf](paper/main.pdf)).

Ram Kedem, Yehonatan Barel, Ido Azoulay, Dolev Abudi

Language models often answer graph questions better by writing code than by answering directly. We ask whether that code is also *invariant*: does it give the same answer when the same graph is written differently? Each graph is rewritten in equivalent serializations that change one factor at a time (node relabeling, edge order, structure, syntax), and every serialization is posed in three modes:

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
.venv/bin/pip install -e ".[dev,stats,erdos]"
scripts/reproduce.sh          # about 4 minutes; add --pdf to rebuild paper/main.pdf with latexmk
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
| `results/review/paper_metrics.json`: every number the paper reports for the main experiment | `scripts/paper_metrics.py` |
| `paper/figures/*.pdf` | `paper/figures/make_figures.py` |

Every CSV and JSON file comes out byte-identical to the committed one. The figures can differ in bytes (Matplotlib version and timestamps) but not in content.

Tests: `.venv/bin/python -m pytest` (the main suite, about 20 s, no network) and `.venv/bin/python -m pytest mitigation/tests`.

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
```

## Running the pipeline from scratch

This calls paid model APIs. The published run used the Hugging Face Inference Providers router. Copy `.env.example` to `.env` and set `HF_TOKEN`; the models and providers are in `configs/models.yaml`.

```bash
python scripts/run.py --config configs/smoke.yaml --models stub     # offline end-to-end check with a stub model
python scripts/run.py --config configs/graphqa.yaml --estimate      # prompt and token counts, no calls
python scripts/run.py --config configs/graphqa.yaml --models hf-qwen3-8b-nscale --stages variants,llm,exec,score
python scripts/analyze.py --config configs/graphqa.yaml
```

Replies are cached under `results/cache/<model>/<prompt_hash>.json`, and every stage resumes where it stopped. The `data` stage rebuilds `data/processed/<dataset>/instances.jsonl`. It is not needed, because the instances are committed; for Erdős it downloads `PKU-ML/Erdos`.

## Data sources

- GraphQA: graphs generated with the recipe of Fatemi et al., *Talk like a Graph: Encoding Graphs for Large Language Models* (ICLR 2024).
- Erdős: the published suite of Guo et al. (`PKU-ML/Erdos` on Hugging Face), as used by Herbst et al.
