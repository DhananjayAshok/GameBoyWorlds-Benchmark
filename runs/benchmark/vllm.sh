#!/usr/bin/env bash
# Benchmark sweep over a locally served (vLLM) model. One log file per arm under logs/.
# Does NOT start the server — bring it up first with scripts/core/serve_vllm.sh.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }
source configs/config.env || { echo "Could not source configs/config.env"; exit 1; }
source runs/games.sh || { echo "Could not source runs/games.sh"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=("model")

ARGS["games"]=none
ARGS["executor"]="single_visual"
ARGS["supervisor"]="subgoal"
ARGS["controller_variant"]="low_level"
ARGS["max_steps"]=75
ARGS["executor_max_new_tokens"]=16000
ARGS["supervisor_max_new_tokens"]=10000
ARGS["log_dir"]="logs/benchmark_vllm"

ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")
USAGE_STR="Usage: $0"
for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done
for opt in "${!ARGS[@]}"; do
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"; exit 1
        fi
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done
function usage() { echo "$USAGE_STR"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then VALID=true; break; fi
            done
            if [ "$VALID" = false ]; then echo "Error: Unknown flag --$FLAG"; usage; fi
            ARGS["$FLAG"]="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then echo "Error: --$req is required."; FAILED=true; fi
done
if [ "$FAILED" = true ]; then usage; fi

echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

mapfile -t games < <(resolve_games BENCH_GAMES "${ARGS["games"]}") || exit 1

model="${ARGS["model"]}"
log_dir="${ARGS["log_dir"]}"
mkdir -p "$log_dir"

FAILED_ARMS=""
for game in "${games[@]}"; do
    echo ""
    echo "=== game: $game ==="
    log_file="$log_dir/${game}_$(model_save_name "$model")_${ARGS["executor"]}_${ARGS["supervisor"]}.out"
    echo "--- $game | $model | ${ARGS["executor"]} | ${ARGS["supervisor"]} -> $log_file"
    if ! bash scripts/benchmark/run_benchmark.sh \
            --game "$game" \
            --supervisor "${ARGS["supervisor"]}" \
            --executor "${ARGS["executor"]}" \
            --controller_variant "${ARGS["controller_variant"]}" \
            --executor_vlm_model "$model" \
            --executor_vlm_kind vllm \
            --max_steps "${ARGS["max_steps"]}" \
            --executor_max_new_tokens "${ARGS["executor_max_new_tokens"]}" \
            --supervisor_max_new_tokens "${ARGS["supervisor_max_new_tokens"]}" \
            > "$log_file" 2>&1; then
        echo "FAILED: $game | $model | ${ARGS["executor"]} | ${ARGS["supervisor"]}"
        FAILED_ARMS+="$game "
    fi
done

echo ""
[[ -n "$FAILED_ARMS" ]] && echo "Failed games: $FAILED_ARMS"
echo "DONE ALL: $model"
