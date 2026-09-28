#!/bin/bash
# Full experiment for one model: both datasets, all stages, analysis. Resumable at every stage,
# so re-running after any failure continues where it stopped. Keeps the machine awake.
#
#   nohup scripts/run_model.sh hf-qwen3-8b > results/hf-qwen3-8b.log 2>&1 &
#   tail -f results/hf-qwen3-8b.log
set -u
MODEL="${1:?usage: run_model.sh <model-name-from-configs/models.yaml>}"
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
[ -x "$PY" ] || PY=.venv/Scripts/python   # Windows venv layout
# caffeinate keeps macOS awake. Elsewhere, hold ES_CONTINUOUS|ES_SYSTEM_REQUIRED for the life of the
# command on Windows (a PC that slept mid-run once turned 4 programs into timeouts); no-op otherwise.
command -v caffeinate >/dev/null 2>&1 || caffeinate() {
  [ "$1" = "-i" ] && shift
  "$PY" -c 'import ctypes, shutil, subprocess, sys
k = getattr(ctypes, "windll", None)
if k: k.kernel32.SetThreadExecutionState(0x80000001)
cmd = sys.argv[1:]
cmd[0] = shutil.which(cmd[0]) or cmd[0]   # CreateProcess does not add .exe to a relative path
sys.exit(subprocess.call(cmd))' "$@"
}
echo "=== $(date '+%F %T') START model=$MODEL host=$(hostname) ==="
for cfg in graphqa erdos; do
  echo "=== $(date '+%F %T') $cfg: llm ==="
  caffeinate -i $PY scripts/run.py --config configs/$cfg.yaml --models "$MODEL" --stages data,variants,llm || { echo "=== llm stage failed for $cfg; stopping ==="; exit 2; }
  echo "=== $(date '+%F %T') $cfg: exec,score ==="
  caffeinate -i $PY scripts/run.py --config configs/$cfg.yaml --models "$MODEL" --stages exec,score || { echo "=== exec/score failed for $cfg; stopping ==="; exit 3; }
  echo "=== $(date '+%F %T') $cfg: analyze ==="
  $PY scripts/analyze.py --config configs/$cfg.yaml --models "$MODEL" > "results/$cfg/analyze_$MODEL.txt" 2>&1 || echo "=== analyze failed for $cfg (non-fatal) ==="
done
echo "=== $(date '+%F %T') ALL DONE model=$MODEL ==="
