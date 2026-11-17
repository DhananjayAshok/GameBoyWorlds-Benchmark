#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
source setup/.venv/bin/activate
source configs/config.env
source scripts/core/utils.sh
export TMPDIR="${TMPDIR:-/scratch1/$USER/tmp}"
mkdir -p "$TMPDIR"

MODEL="$1"
TRAIN_GAME="$2"
BENCH_GAMES="$3"
EXECUTOR=single_visual
CV=low_level
TAG="$(model_save_name "$MODEL")_${TRAIN_GAME}"
LOGDIR=logs/qwen/lanes/$TAG
mkdir -p "$LOGDIR"

stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
stamp "curiosity lane start host=$(hostname -s) model=$MODEL train=$TRAIN_GAME bench=$BENCH_GAMES base_url=${VLLM_BASE_URL:-default}"

stamp "curiosity doc start"
bash scripts/vlm/create_all_info_docs.sh \
    --game "$TRAIN_GAME" --model_name "$MODEL" --vlm_kind vllm \
    --mode curiosity_only --run_name my_run \
    --executor "$EXECUTOR" --controller_variant "$CV" \
    > "$LOGDIR/curiosity_doc.out" 2>&1
rc=$?
stamp "curiosity doc exit=$rc"
[[ $rc -ne 0 ]] && exit $rc

IFS=',' read -ra games <<< "$BENCH_GAMES"
for g in "${games[@]}"; do
    (
        stamp "bench $g start"
        bash scripts/benchmark/run_benchmark_info_retrieval.sh \
            --game "$g" --executor "$EXECUTOR" --controller_variant "$CV" \
            --executor_vlm_model "$MODEL" --executor_vlm_kind vllm \
            --docs_mode curiosity_only --docs_model "$MODEL" --docs_run_name my_run \
            --docs_executor "$EXECUTOR" --docs_controller_variant "$CV" \
            --max_steps 75 --executor_max_new_tokens 16000 --supervisor_max_new_tokens 10000 \
            --max_concurrency 16 \
            > "$LOGDIR/bench_${g}.out" 2>&1
        stamp "bench $g exit=$?"
    ) &
done
wait
stamp "curiosity lane done"
