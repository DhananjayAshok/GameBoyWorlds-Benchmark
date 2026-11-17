#!/usr/bin/env bash
# Uses the VLM to infer task labels from the grouped trajectories (--trajectory_path)
# produced by group_trajectories.sh, for a given --run_name.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array INFER_TASKS_ESSENTIALS REQUIRED_ARGS
populate_dict INFER_TASKS_DEFAULTS ARGS

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

if [[ "${ARGS["describe_pairs"]}" == "true" || "${ARGS["describe_pairs"]}" == "yes" || "${ARGS["describe_pairs"]}" == "y" ]]; then
    describe_pairs_flag="--describe_pairs"
else
    describe_pairs_flag=""
fi

python vlm.py \
    --game "${ARGS["game"]}" \
    --model_name "${ARGS["model_name"]}" \
    --vlm_kind "${ARGS["vlm_kind"]}" \
    --max_new_tokens "${ARGS["max_new_tokens"]}" \
    $overwrite_flag $verbose_flag \
    infer_tasks \
    --trajectory_path "${ARGS["trajectory_path"]}" \
    --run_name "${ARGS["run_name"]}" \
    --lookback "${ARGS["lookback"]}" \
    --max_trajectories_per_group "${ARGS["max_trajectories_per_group"]}" \
    $describe_pairs_flag
