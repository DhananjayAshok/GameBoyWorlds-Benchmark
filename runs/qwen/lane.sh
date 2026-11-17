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
STAGES="${4:-collect,bench,practice}"
EXECUTOR=single_visual
CV=low_level
TAG="$(model_save_name "$MODEL")_${TRAIN_GAME}"
LOGDIR=logs/qwen/lanes/$TAG
mkdir -p "$LOGDIR"

stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
stamp "lane start host=$(hostname -s) model=$MODEL train=$TRAIN_GAME bench=$BENCH_GAMES stages=$STAGES"

if [[ ",$STAGES," == *",collect,"* ]]; then
    stamp "collect start"
    bash scripts/pipeline/collect_and_info_all.sh \
        --game "$TRAIN_GAME" --model_name "$MODEL" --vlm_kind vllm \
        --executor "$EXECUTOR" --controller_variant "$CV" --run_name my_run \
        > "$LOGDIR/collect.out" 2>&1
    stamp "collect exit=$?"
fi

if [[ ",$STAGES," == *",bench,"* ]]; then
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
                > "$LOGDIR/bench_${g}.out" 2>&1
            stamp "bench $g exit=$?"
        ) &
    done
fi

if [[ ",$STAGES," == *",practice,"* ]]; then
    base=$(path_of zeroshot_dir --game "$TRAIN_GAME" --model_name "$MODEL")
    stem="$base/zeroshot_tasks_${EXECUTOR}_${CV}_attempts/success_trajectories"
    if [[ -f "$stem.json" && -f "$stem.pkl" ]]; then
        stamp "practice start stem=$stem"
        bash scripts/pipeline/guidance_and_practice.sh \
            --game "$TRAIN_GAME" --model_name "$MODEL" --vlm_kind vllm \
            --trajectory_path "$stem" --executor "$EXECUTOR" --controller_variant "$CV" \
            --do_clean true \
            > "$LOGDIR/practice.out" 2>&1
        stamp "practice exit=$?"
    else
        stamp "practice SKIPPED: no zeroshot stem at $stem"
    fi
fi

wait
stamp "lane done"
