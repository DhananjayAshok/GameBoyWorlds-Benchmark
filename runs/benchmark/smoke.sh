#!/usr/bin/env bash
# Two-step smoke test of the benchmark path on a cheap API model.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }
source configs/config.env || { echo "Could not source configs/config.env"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

ARGS["model"]="gpt-4o-mini"
ARGS["games"]="harry_potter_philosophers_stone,harry_potter_chamber_of_secrets"
ARGS["executor"]="single_actions"
ARGS["max_steps"]=2

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

IFS=',' read -ra games <<< "${ARGS["games"]}"

for game in "${games[@]}"; do
    echo ""
    echo "=== smoke: $game | ${ARGS["model"]} | ${ARGS["executor"]} ==="
    if ! bash scripts/benchmark/run_benchmark.sh \
            --game "$game" \
            --executor "${ARGS["executor"]}" \
            --executor_vlm_model "${ARGS["model"]}" \
            --executor_vlm_kind openrouter \
            --max_steps "${ARGS["max_steps"]}" \
            --regenerate true; then
        echo "FAILED: $game"
    fi
done

echo "DONE ALL"
