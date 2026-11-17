#!/usr/bin/env bash
# Runs multiple rounds of RL training where each agent's curiosity buffer seeds the next.
# Retrains the world model between rounds when curiosity_module=world_model; --sweep true
# sweeps each round via sweep.sh. Called by create_traj.sh.

source scripts/core/utils.sh

declare -A ARGS
REQUIRED_ARGS=()
populate_dict ITERATIVE_TRAINING_DEFAULTS ARGS
populate_array ITERATIVE_TRAINING_ESSENTIALS REQUIRED_ARGS

combination_buffer_ocr_alphas=(0.75)


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

echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done


sweeping=false
init_state_name=${ARGS["init_state"]}
if [ "${ARGS["sweep"]}" == "true" ]; then
    run_name="iterative_sweep"
    sweeping=true
else
    run_name="iterative_agent"
fi
ocr_alpha_sweep=false
if [ "${ARGS["combination_buffer_sweep"]}" == "true" ] && [ "${ARGS["curiosity_module"]}" == "combinationbuffer" ]; then
    ocr_alpha_sweep=true
fi
if [ "$ocr_alpha_sweep" = true ] && [ "${ARGS["ocr_alpha"]}" != "0.0" ]; then
    ARGS["ocr_alpha"]=0.0
    echo "Warning: combination_buffer_sweep is true but ocr_alpha is not 0.0. Setting ocr_alpha to 0.0 for sweeping."
fi
if [ "${ARGS["init_state_group"]}" != "none" ]; then
    init_state_name="${ARGS["init_state_group"]}"
fi

replay_buffer_save_folder=${ARGS["run_name"]}/${init_state_name}/$run_name/
ARGS["latest_replay_buffer_folder"]=$replay_buffer_save_folder
ARGS["replay_buffer_save_folder"]=$replay_buffer_save_folder



prev_buffer_load_path="none"
log_folder="${ARGS["game"]}/${ARGS["run_name"]}/$run_name/$init_state_name/"

function call_agent(){
    local buffer_load_path="$1"
    local buffer_save_path="$2"
    ARGS["buffer_save_path"]=$buffer_save_path
    ARGS["buffer_load_path"]=$buffer_load_path
    if [ "$sweeping" = true ]; then
        argstring=$(args_to_flags_subset ARGS SWEEP_ARG_KEYS)
        bash scripts/rl/sweep.sh $argstring || exit 1
        if [ "$ocr_alpha_sweep" = true ]; then
            for ocr_alpha in "${combination_buffer_ocr_alphas[@]}"; do
                ARGS["ocr_alpha"]=$ocr_alpha
                argstring=$(args_to_flags_subset ARGS SWEEP_ARG_KEYS)
                bash scripts/rl/sweep.sh $argstring || exit 1
            done
            ARGS["ocr_alpha"]=0.0 # reset to default after sweep
        fi
    else
        argstring=$(args_to_flags_subset ARGS TRAINING_ARG_KEYS)
        bash scripts/core_rl/default_rl.sh $argstring || exit 1
        if [ "$ocr_alpha_sweep" = true ]; then
            for ocr_alpha in "${combination_buffer_ocr_alphas[@]}"; do
                ARGS["ocr_alpha"]=$ocr_alpha
                argstring=$(args_to_flags_subset ARGS TRAINING_ARG_KEYS)
                bash scripts/core_rl/default_rl.sh $argstring || exit 1
            done
            ARGS["ocr_alpha"]=0.0 # reset to default after sweep
        fi
    fi
}

function train_world_model(){
    local buffer_save_path="$1"
    ARGS["buffer_save_path"]=$buffer_save_path
    ARGS["buffer_load_path"]="none" # world model will train on all trajectories in the replay buffer folder, so no need to specify a single buffer to load from.
    argstring=$(args_to_flags_subset ARGS WORLD_MODEL_ARG_KEYS)
    bash scripts/core_rl/train_world_model.sh $argstring 
}


if [ "$sweeping" = true ]; then
    all_buffer_save_paths=${ARGS["run_name"]}/${ARGS["init_state"]}/${ARGS["observation_embedder"]}/sweep_
else
    all_buffer_save_paths=${ARGS["run_name"]}/${ARGS["init_state"]}/${ARGS["observation_embedder"]}/${ARGS["algorithm"]}_agent_
fi


buffer_save_path=${all_buffer_save_paths}0
prev_buffer_save_path=$buffer_save_path

call_agent "none" $buffer_save_path

if [ "${ARGS["curiosity_module"]}" == "world_model" ]; then
    train_world_model $buffer_save_path    
fi

prev_buffer_load_path=$prev_buffer_save_path
buffer_save_path=${all_buffer_save_paths}1

for ((i=0; i<${ARGS["n_agents"]}; i++)); do
    echo "Starting iteration $i with buffer load path '$prev_buffer_load_path' and buffer save path '$buffer_save_path'"

    call_agent $prev_buffer_load_path $buffer_save_path

    # No world model training on the last iteration: its buffer is not used again.
    if [ $i -lt $((${ARGS["n_agents"]}-1)) ]; then
        if [ "${ARGS["curiosity_module"]}" == "world_model" ]; then
            train_world_model $buffer_save_path    
        fi
    fi

    prev_buffer_load_path=$buffer_save_path
    buffer_save_path="${all_buffer_save_paths}$((i+2))"
done