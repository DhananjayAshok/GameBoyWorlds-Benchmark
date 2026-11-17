#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
source setup/.venv/bin/activate
source configs/config.env
export TMPDIR="${TMPDIR:-/scratch1/$USER/tmp}"
mkdir -p "$TMPDIR"
MODEL=Qwen/Qwen3-VL-2B-Instruct
LOGDIR=logs/qwen/lanes/qwen3-vl-2b-instruct_baseline
mkdir -p "$LOGDIR"
stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
stamp "2b baseline gaps start host=$(hostname -s)"
for g in sword_of_hope_1 sword_of_hope_2; do
    (
        stamp "bench $g start"
        bash scripts/benchmark/run_benchmark.sh \
            --game "$g" --supervisor subgoal --executor single_visual --controller_variant low_level \
            --executor_vlm_model "$MODEL" --executor_vlm_kind vllm \
            --max_steps 75 \
            > "$LOGDIR/bench_${g}.out" 2>&1
        stamp "bench $g exit=$?"
    ) &
done
wait
stamp "2b baseline gaps done"
