#!/usr/bin/env bash
# Trajectory creation for a single init_state: runs iterative RL agents to collect a replay
# buffer, then group_trajectories.sh to cluster it. The grouped trajectories feed
# infer_tasks.sh. Use create_all_traj.sh to run this across all init_states of a game.

source scripts/core/utils.sh

declare -A ARGS
REQUIRED_ARGS=()

populate_dict CREATE_TRAJ_DEFAULTS ARGS
populate_array  CREATE_TRAJ_ESSENTIALS REQUIRED_ARGS


ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")

USAGE_STR="Usage: $0"

for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done

for opt in "${!ARGS[@]}"; do
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done

function usage() {
    echo "$USAGE_STR"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then
                    VALID=true
                    break
                fi
            done
            if [ "$VALID" = false ]; then
                echo "Error: Unknown flag --$FLAG"
                usage
            fi            
            ARGS["$FLAG"]="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then
        echo "Error: Argument --$req is required."
        FAILED=true
    fi
done

if [ "$FAILED" = true ]; then usage; fi

if [[ -z "${ARGS["run_name"]}" || "${ARGS["run_name"]}" == "none" ]]; then
    echo "Error: --run_name is required and cannot be 'none'."
    usage
fi

echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

sweeping=false
if [ "${ARGS["sweep"]}" == "true" ]; then
    sweep_run_name="iterative_sweep"
    sweeping=true
else
    sweep_run_name="iterative_agent"
fi

if [[ -z "${ARGS["init_state_group"]}" || "${ARGS["init_state_group"]}" == "none" ]]; then
    ARGS["init_state_group"]="${ARGS["init_state"]}"
fi

init_state_group=${ARGS["init_state_group"]}
replay_buffer_save_folder=${ARGS["run_name"]}/${init_state_group}/$sweep_run_name/
echo "Setting replay_buffer_save_folder to $replay_buffer_save_folder for iterative training"

# z_kind is not an arg here; group_trajectories defaults to global, so the output is always the global file.
grouped_file=$(path_of grouped_file --game "${ARGS["game"]}" \
                 --run_name "${ARGS["run_name"]}" --init_state "${init_state_group}")
# Only skip when grouping: with --group_trajectories false the grouped file is not this run's output.
if [[ "${ARGS["group_trajectories"]}" == "true" \
      && "${ARGS["overwrite"]}" != "true" && -f "$grouped_file" ]]; then
    echo "Grouped trajectories already exist at $grouped_file. Skipping (pass --overwrite true to regenerate)."
    exit 0
fi

if [ "${ARGS["skip_training"]}" == "true" ]; then
    echo "Skipping training as per --skip_training flag."
else
    echo "Clearing replay buffer for iterative training..."
    rm -rf $storage_dir/replay_buffers/${ARGS["game"]}/$replay_buffer_save_folder/* # clear replay buffer to ensure we don't have old trajectories lying around.
    run_name=${ARGS["run_name"]}
    echo "Run name: $run_name | Setting model_dir to run_name for iterative training"
    ARGS["model_dir"]="${run_name}"
    argstring=$(args_to_flags_subset ARGS ITERATIVE_TRAINING_ARG_KEYS)
    bash scripts/rl/iterative_training.sh $argstring
fi

if [ "${ARGS["group_trajectories"]}" != "true" ]; then
    echo "Skipping grouping as per --group_trajectories false. Replay buffers retained at $storage_dir/replay_buffers/${ARGS["game"]}/$replay_buffer_save_folder"
    exit 0
fi

bash scripts/rl/group_trajectories.sh --game ${ARGS["game"]} --replay_buffer_folder $replay_buffer_save_folder --save_path "$(path_of grouped_dir --game "${ARGS["game"]}" --run_name "${ARGS["run_name"]}")/${init_state_group}/" --z_min ${ARGS["z_min"]}
group_status=$?

# Only reclaim disk once grouping has succeeded and the grouped file is on disk.
if [[ $group_status -ne 0 || ! -f "$grouped_file" ]]; then
    echo "Grouping failed (exit $group_status) or output missing at $grouped_file; NOT clearing replay buffers/models."
    exit $group_status
fi

if [ "${ARGS["keep_buffers"]}" == "true" ]; then
    echo "Grouping succeeded. Retaining replay buffers at $storage_dir/replay_buffers/${ARGS["game"]}/$replay_buffer_save_folder (--keep_buffers true)."
    exit 0
fi

echo "Grouping succeeded. Clearing replay buffers for ${ARGS["game"]}/${ARGS["run_name"]}/${init_state_group}..."
rm -rf "$storage_dir/replay_buffers/${ARGS["game"]}/$replay_buffer_save_folder"