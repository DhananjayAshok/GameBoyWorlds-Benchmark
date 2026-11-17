#!/usr/bin/env bash
# Batch version of curiosity_tasks: loops over all init_states for a game,
# running create_traj then infer_tasks for each. Regenerates TRAIN_STATES
# before iterating.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }
python "$PROJECT_ROOT/python_funcs.py" task_dictionary || { echo "Could not regenerate train states"; exit 1; }
source scripts/core/all_train_states.sh

declare -A ARGS
REQUIRED_ARGS=()

populate_array VLM_ESSENTIALS REQUIRED_ARGS
REQUIRED_ARGS+=("run_name")
populate_dict CURIOSITY_TASKS_DEFAULTS ARGS
ARGS["overwrite_annotation"]=false

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

if [[ "${ARGS["run_name"]}" == "none" ]]; then
    echo "Error: --run_name cannot be 'none'."; usage
fi

echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

game="${ARGS["game"]}"
IFS=',' read -ra init_states_arr <<< "${TRAIN_STATES[$game]}"
grouped_pkls=()
for init_state in "${init_states_arr[@]}"; do
    ARGS["init_state"]=$init_state

    create_flags=$(args_to_flags_subset ARGS CREATE_TRAJ_ARG_KEYS)
    bash scripts/rl/create_traj.sh $create_flags || exit 1

    init_state_group="${ARGS["init_state_group"]}"
    if [[ "$init_state_group" == "none" ]]; then
        init_state_group="$init_state"
    fi
    grouped_pkls+=("$(path_of grouped_file --game "$game" --run_name "${ARGS["run_name"]}" \
                              --init_state "${init_state_group}")")
done

# Combine every init_state's grouped trajectories before a single infer_tasks call:
# infer_tasks keys its output only on run_name, so a per-init_state loop would overwrite.
# The combined dir is "all", which collides with a real init_state named "all".
combined_dir="$(path_of grouped_dir --game "$game" --run_name "${ARGS["run_name"]}")/all"
input_paths=$(IFS=,; echo "${grouped_pkls[*]}")
python "$PROJECT_ROOT/python_funcs.py" combine_trajectories \
    --input_paths "$input_paths" \
    --save_path "$combined_dir" \
    --z_kind global || exit 1

ARGS["trajectory_path"]="$combined_dir/grouped_global_high_reward_trajectories.pkl"
saved_overwrite="${ARGS["overwrite"]}"
ARGS["overwrite"]="${ARGS["overwrite_annotation"]}"
infer_flags=$(args_to_flags_subset ARGS INFER_TASKS_ARG_KEYS)
ARGS["overwrite"]="$saved_overwrite"
bash scripts/vlm/infer_tasks.sh $infer_flags || exit 1

model_save_name=$(model_save_name "${ARGS["model_name"]}")
echo "curiosity trajectory stem: $(path_of info_source_stem --game "$game" \
        --model_name "${ARGS["model_name"]}" --run_name "${ARGS["run_name"]}" \
        --executor single_actions --controller_variant low_level --source curiosity)"
