#!/usr/bin/env bash
# The gemma no-doc subgoal sweep followed by the info-parametric sweep, with one vLLM server
# held for both.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }
source configs/config.env || { echo "Could not source configs/config.env"; exit 1; }
source runs/games.sh || { echo "Could not source runs/games.sh"; exit 1; }

MODEL="google/gemma-4-31b-it"
VLM_KIND="vllm"
EXECUTOR="single_visual"
CONTROLLER_VARIANT="low_level"
MAX_STEPS=175

bash scripts/core/serve_vllm.sh "$MODEL" -tp 4 || { echo "Could not start vLLM"; exit 1; }

for GAME in "${BENCH_GAMES[@]}"; do
    echo ""
    echo "############################################################"
    echo "# $GAME  —  no-doc subgoal benchmark"
    echo "############################################################"
    if ! bash scripts/benchmark/run_benchmark.sh \
            --game "$GAME" \
            --supervisor subgoal \
            --executor "$EXECUTOR" \
            --controller_variant "$CONTROLLER_VARIANT" \
            --executor_vlm_model "$MODEL" \
            --executor_vlm_kind "$VLM_KIND" \
            --max_steps "$MAX_STEPS"; then
        echo "FAILED: $GAME | subgoal"
    fi
done

for GAME in "${BENCH_GAMES[@]}"; do
    echo ""
    echo "############################################################"
    echo "# $GAME  —  info parametric benchmark"
    echo "############################################################"
    if ! bash scripts/benchmark/run_benchmark.sh \
            --game "$GAME" \
            --supervisor info_subgoal_parametric \
            --executor "$EXECUTOR" \
            --controller_variant "$CONTROLLER_VARIANT" \
            --executor_vlm_model "$MODEL" \
            --executor_vlm_kind "$VLM_KIND" \
            --parametric_categories 10 \
            --max_steps "$MAX_STEPS"; then
        echo "FAILED: $GAME | parametric"
    fi
done

echo "DONE ALL"

bash scripts/core/stop_vllm.sh 8000
