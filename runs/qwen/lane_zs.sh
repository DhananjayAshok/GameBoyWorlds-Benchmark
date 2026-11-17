#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
source setup/.venv/bin/activate
source configs/config.env
source scripts/core/utils.sh
export TMPDIR="${TMPDIR:-/scratch1/$USER/tmp}"
mkdir -p "$TMPDIR"

MODEL="$1"
GAME="$2"
EXECUTOR=single_visual
CV=low_level
TAG="$(model_save_name "$MODEL")_${GAME}"
LOGDIR=logs/qwen/lanes/$TAG
mkdir -p "$LOGDIR"
stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
stamp "zeroshot lane start host=$(hostname -s) model=$MODEL game=$GAME base_url=${VLLM_BASE_URL:-default}"

base=$(path_of zeroshot_dir --game "$GAME" --model_name "$MODEL")
stem="$base/zeroshot_tasks_${EXECUTOR}_${CV}_attempts/success_trajectories"
if [[ -f "$stem.json" && -f "$stem.pkl" ]]; then
    stamp "zeroshot already complete at $stem"
else
    stamp "zeroshot start"
    bash scripts/pipeline/curiosity_and_zeroshot_all.sh \
        --game "$GAME" --model_name "$MODEL" --vlm_kind vllm \
        --executor "$EXECUTOR" --controller_variant "$CV" --run_name my_run \
        > "$LOGDIR/zeroshot.out" 2>&1
    stamp "zeroshot exit=$?"
fi

bash runs/qwen/lane.sh "$MODEL" "$GAME" none practice
stamp "zeroshot lane done"
