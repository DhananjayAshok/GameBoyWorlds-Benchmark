#!/usr/bin/env bash
# Runs both data-collection verticals for a game: curiosity (RL exploration -> infer_tasks)
# and zeroshot (propose -> attempt).

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }
python "$PROJECT_ROOT/python_funcs.py" task_dictionary || { echo "Could not regenerate train states"; exit 1; }
source scripts/core/all_train_states.sh

declare -A ARGS
REQUIRED_ARGS=()

populate_array VLM_ESSENTIALS REQUIRED_ARGS
populate_dict PROPOSE_AND_ATTEMPT_DEFAULTS ARGS

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

model_save_name=$(model_save_name "${ARGS["model_name"]}")
curiosity_dir=$(path_of curiosity_dir --game "${ARGS["game"]}" \
                  --model_name "${ARGS["model_name"]}" --run_name "${ARGS["run_name"]}")

# overwrite=false so existing curiosity output is reused rather than regenerated.
saved_overwrite="${ARGS["overwrite"]}"
ARGS["overwrite"]="false"
curiosity_flags=$(args_to_flags_subset ARGS CURIOSITY_TASKS_ARG_KEYS)
bash scripts/pipeline/curiosity_all_tasks.sh $curiosity_flags || exit 1
ARGS["overwrite"]="$saved_overwrite"

if [[ ! -f "$curiosity_dir/trajectory_annotation.json" || ! -f "$curiosity_dir/trajectory_annotation.pkl" ]]; then
    echo "Error: curiosity annotation still missing at $curiosity_dir after curiosity_all_tasks. Aborting."
    exit 1
fi

ARGS["propose_only"]="false"
flags=$(args_to_flags_subset ARGS PROPOSE_AND_ATTEMPT_ARG_KEYS)
bash scripts/pipeline/propose_and_attempt_all.sh $flags || exit 1

executor="${ARGS["executor"]}"
zeroshot_base=$(path_of zeroshot_dir --game "${ARGS["game"]}" \
                  --model_name "${ARGS["model_name"]}")
echo "Trajectory stems:"
echo "  curiosity: $curiosity_dir/trajectory_annotation"
echo "  zeroshot:  $zeroshot_base/zeroshot_tasks_${executor}_attempts/success_trajectories"
