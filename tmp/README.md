# tmp/: probably unused, kept for a second look

Everything here was moved out of the main tree because nothing in the published pipeline uses it, but it may still be worth keeping. Nothing outside `tmp/` imports or reads these files; the tests and `scripts/reproduce.sh` pass without the folder. Delete it with `rm -rf tmp/` once you have checked.

| File | Came from | Why it is here | Depends on it |
|---|---|---|---|
| `configs/pilot.yaml` | `configs/` | The pilot run (a few graphs of both datasets with real models) that checked prompt compliance and cost before the full run. None of its numbers are results. | nothing |
| `data/processed/pilot/instances.jsonl` | `data/processed/pilot/` | The pilot's sampled instances. | `configs/pilot.yaml` |
| `configs/models_unused.yaml` | `configs/models.yaml` | Model entries no published run used: `deepseek-chat`, `deepseek-reasoner`, `qwen-7b` (old config defaults), `hf-deepseek-v3` and `hf-qwen3-8b` (the pilot's models), `hf-qwen2.5-7b`, `hf-llama3.1-8b`. | `configs/pilot.yaml` |
| `scripts/cost.py` | `scripts/` | Spend monitor for a live run: reads the reply cache `results/cache/`, which is neither published nor restored, so it cannot run on the published data. Known quirks: it counts the execution cache folder as a model called "exec", and the noise-floor re-asks (`.r1`-`.r3`) as an arm called "r1"; it must run from the repository root. | nothing |

To use the pilot again, move `configs/pilot.yaml` back to `configs/`, `data/processed/pilot/` back to `data/processed/`, and copy `hf-deepseek-v3` and `hf-qwen3-8b` into `configs/models.yaml`.
