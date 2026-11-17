#!/usr/bin/env bash
# Hyperparameter sweep over seeds, gammas and algorithms using default_rl.sh. Prunes to the
# best-k models by test reward, then re-runs only the winners to collect replay buffers.
# Called by iterative_training.sh when --sweep true.

# GameBoyWorlds' own $storage_dir, needed for video deletion.
source GameBoyWorlds/configs/config.env || { echo "GameBoyWorlds/configs/config.env not found"; exit 1; }
gameboy_worlds_storage_dir="$storage_dir"

source scripts/core/utils.sh

declare -A ARGS
REQUIRED_ARGS=()
populate_array SWEEP_ESSENTIALS REQUIRED_ARGS
populate_dict SWEEP_DEFAULTS ARGS



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
if [[ "${ARGS["model_dir"]}" != "none" ]]; then
    model_save_path="$storage_dir/models/${ARGS["game"]}/${ARGS["model_dir"]}/"
    rm -rf $model_save_path/* # clear the model save path to ensure we don't have old models lying around.
else
    echo "Sweep requires you to specify a --model_dir"
    exit 1
fi

video_session_dir="$gameboy_worlds_storage_dir/sessions/${ARGS["game"]}/"

keep_arg_str=""
if [[ "${ARGS["replay_buffer_save_folder"]}" != "none" ]]; then
    keep_arg_str+="--replay_buffer_save_folder $storage_dir/replay_buffers/${ARGS["game"]}/${ARGS["replay_buffer_save_folder"]} "
else
    ARGS["clear_loser_replay_buffer"]="false" # force false if replay buffer isn't saved in the first place. 
fi


keep_arg_str+="--best_k ${ARGS["best_k"]} --model_dir $model_save_path --video_session_dir $video_session_dir"
if [[ "${ARGS["clear_loser_replay_buffer"]}" == "true" ]]; then
    keep_arg_str+=" --clear_loser_replay_buffer"
fi


echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done


SEEDS=(0)
GAMMAS=(0.99 0.995 0.999)
ALGORITHMS=(ppo)


true_buffer_save_path=${ARGS["buffer_save_path"]}
ARGS["buffer_save_path"]="none" # the first eval of the sweep is just to assess test reward, not to save to buffers. 
for seed in "${SEEDS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
        for algorithm in "${ALGORITHMS[@]}"; do
            ARGS["seed"]=$seed
            ARGS["gamma"]=$gamma
            ARGS["algorithm"]=$algorithm
            argstring=$(args_to_flags_subset ARGS TRAINING_ARG_KEYS)
            bash scripts/core_rl/default_rl.sh $argstring
        done
    done
done
ARGS["buffer_save_path"]="$true_buffer_save_path"



cd cleanrl
python cleanrl_utils/keep_only_best_models.py $keep_arg_str || exit 1
cd ..


for seed in "${SEEDS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
        for algorithm in "${ALGORITHMS[@]}"; do
            ARGS["seed"]=$seed
            ARGS["gamma"]=$gamma
            ARGS["algorithm"]=$algorithm
            argstring=$(args_to_flags_subset ARGS TRAINING_ARG_KEYS)
            bash scripts/core_rl/default_rl.sh $argstring --eval_only true
        done
    done
done