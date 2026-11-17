#!/usr/bin/env bash
# Runs a single RL training job with the specified curiosity module via cleanrl.
# Skips the run if a model checkpoint already exists, unless --overwrite_model true.

source scripts/core/utils.sh

declare -A ARGS

REQUIRED_ARGS=()

populate_dict TRAINING_DEFAULTS ARGS
populate_array TRAINING_ESSENTIALS REQUIRED_ARGS

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

if [[ "${ARGS["curiosity_module"]}" != "combinationbuffer" ]]; then
    ARGS["ocr_alpha"]=0.0
fi

cd cleanrl


train_env_id=$(get_string_from_args "env_id" ARGS)
if [[ -z "$train_env_id" ]]; then
    echo "Error: Failed to get train_env_id"
    exit 1
fi
train_env_id=$train_env_id-False

exp_name=$(get_string_from_args "exp_name" ARGS)
if [[ -z "$exp_name" ]]; then
    echo "Error: Failed to generate exp_name"
    exit 1
fi

if [[ "${ARGS["model_dir"]}" != "none" ]]; then
    model_save_path="$storage_dir/models/${ARGS["game"]}/${ARGS["model_dir"]}/$exp_name/"
else
    model_save_path="$storage_dir/models/${ARGS["game"]}/$exp_name/"
fi


extra_arg_part=""
if [[ "${ARGS["replay_buffer_save_folder"]}" != "none" ]]; then
    extra_arg_part+="--replay_buffer_save_folder $storage_dir/replay_buffers/${ARGS["game"]}/${ARGS["replay_buffer_save_folder"]} "
fi

if [[ "${ARGS["buffer_save_path"]}" != "none" ]]; then
    extra_arg_part+="--buffer_save_path $storage_dir/curiosity_buffers/${ARGS["curiosity_module"]}/${ARGS["game"]}/${ARGS["buffer_save_path"]} "
fi

if [[ "${ARGS["buffer_load_path"]}" != "none" ]]; then
    extra_arg_part+="--buffer_load_path $storage_dir/curiosity_buffers/${ARGS["curiosity_module"]}/${ARGS["game"]}/${ARGS["buffer_load_path"]} "
fi
if [[ "${ARGS["embedder_load_path"]}" != "none" ]]; then
    extra_arg_part+="--embedder_load_path $storage_dir/observation_embedders/${ARGS["game"]}/${ARGS["embedder_load_path"]} "
fi
if [[ "${ARGS["capture_video"]}" == "true" || "${ARGS["capture_video"]}" == "True" || "${ARGS["capture_video"]}" == "yes" || "${ARGS["capture_video"]}" == "y" ]]; then
    extra_arg_part+="--capture_video "
fi

log_file="../logs/$exp_name.out"
if [[ "${ARGS["log_folder"]}" != "none" ]]; then
    log_file="$storage_dir/logs/${ARGS["log_folder"]}/$exp_name.out"
fi
mkdir -p "$(dirname "$log_file")"

echo "Starting Experiment: $exp_name logging to $log_file"

if [[ -d "$model_save_path" ]] && [[ ${ARGS["overwrite_model"]} != "true" ]]; then
    echo "Model $model_save_path already exists. Skipping Run. Use --overwrite_model true to overwrite."
    exit 0
fi


python cleanrl/${ARGS["algorithm"]}_curiosity.py --exp_name $exp_name --seed ${ARGS["seed"]} --gamma ${ARGS["gamma"]} --env-id $train_env_id --total-timesteps ${ARGS["timesteps"]} --track \
    --wandb-project-name $WANDB_PROJECT --model_save_path $model_save_path --save_model \
    --observation_embedder ${ARGS["observation_embedder"]} --similarity_metric ${ARGS["similarity_metric"]} \
    --curiosity-module ${ARGS["curiosity_module"]} --ocr_alpha ${ARGS["ocr_alpha"]} --reset-curiosity-module $extra_arg_part &> $log_file || exit 1 #

cd ..