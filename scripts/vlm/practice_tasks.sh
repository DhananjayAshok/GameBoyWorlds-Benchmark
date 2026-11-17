#!/usr/bin/env bash
# Runs the VLM agent through guided practice sessions using pre-inferred guidance
# (--guidance_path from infer_guidance.sh). The agent is given n_attempts tries,
# starting from n_random_actions random steps to vary the starting state.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array PRACTICE_TASKS_ESSENTIALS REQUIRED_ARGS
populate_dict PRACTICE_TASKS_DEFAULTS ARGS

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

if [[ "${ARGS["overwrite"]}" == "true" || "${ARGS["overwrite"]}" == "yes" || "${ARGS["overwrite"]}" == "y" ]]; then
    overwrite_flag="--overwrite"
else
    overwrite_flag=""
fi

if [[ "${ARGS["verbose"]}" == "true" || "${ARGS["verbose"]}" == "yes" || "${ARGS["verbose"]}" == "y" ]]; then
    verbose_flag="--verbose"
else
    verbose_flag=""
fi

python vlm.py \
    --game "${ARGS["game"]}" \
    --model_name "${ARGS["model_name"]}" \
    --vlm_kind "${ARGS["vlm_kind"]}" \
    --max_new_tokens "${ARGS["max_new_tokens"]}" \
    $overwrite_flag $verbose_flag \
    practice_tasks \
    --guidance_path "${ARGS["guidance_path"]}" \
    --n_attempts "${ARGS["n_attempts"]}" \
    --max_total_practice_runs "${ARGS["max_total_practice_runs"]}" \
    --n_random_actions "${ARGS["n_random_actions"]}" \
    --max_steps "${ARGS["max_steps"]}" \
    --lookback "${ARGS["lookback"]}" \
    --executor "${ARGS["executor"]}" \
    --controller_variant "${ARGS["controller_variant"]}" \
    --checker_max_new_tokens "${ARGS["checker_max_new_tokens"]}" \
    --max_concurrency "${ARGS["max_concurrency"]}"
