#!/usr/bin/env bash
# Fine-tunes a VLM with LoRA via the llm-utils training CLI on --train_file and saves the
# adapter to storage under --run_name. --push_to_hub optionally pushes to HuggingFace Hub.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
ARGS["batch_size"]="4"
ARGS["num_train_epochs"]="2"
ARGS["lora_rank"]="64"
ARGS["lora_alpha"]="128"
ARGS["learning_rate"]="2e-4"
ARGS["weight_decay"]="0.01"
ARGS["overwrite"]=false
ARGS["push_to_hub"]=false
ARGS["validation_file"]=none
ARGS["action_loss_weight"]=none

REQUIRED_ARGS=("train_file" "model_name" "run_name")

ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")

USAGE_STR="Usage: $0"

for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done

for opt in "${!ARGS[@]}"; do
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"
            exit 1
        fi
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
    echo $req : "${ARGS[$req]}"
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




model_name="${ARGS["model_name"]}"
model_save_name=$(model_save_name "$model_name")

if [[ "${ARGS["overwrite"]}" == "true" || "${ARGS["overwrite"]}" == "yes" || "${ARGS["overwrite"]}" == "y" ]]; then
    overwrite_flag="--resume_from_checkpoint False --overwrite_final True"
else
    overwrite_flag=""
fi

if [[ "${ARGS["push_to_hub"]}" == "true" || "${ARGS["push_to_hub"]}" == "yes" || "${ARGS["push_to_hub"]}" == "y" ]]; then
    push_to_hub_flag="--push_to_hub True --hub_model_id ${huggingface_repo_namespace}/${ARGS["run_name"]}-$model_save_name"
else
    push_to_hub_flag=""
fi

# Without a validation file, fall back to an internal 0.85 train/validation split.
if [[ "${ARGS["validation_file"]}" != "none" ]]; then
    validation_flag="--validation_file ${ARGS["validation_file"]}"
else
    validation_flag="--train_validation_split 0.85"
fi

# Upweights the "Action:" tokens in the completion loss.
if [[ "${ARGS["action_loss_weight"]}" != "none" ]]; then
    action_loss_flag="--action_loss_weight ${ARGS["action_loss_weight"]}"
else
    action_loss_flag=""
fi




bash scripts/core/llm-utils.sh python train.py --training_kind sft --modality vlm --model_name ${ARGS["model_name"]} \
        --output_dir $storage_dir/models/${ARGS["run_name"]}/$model_save_name \
        --train_file ${ARGS["train_file"]}  \
        --run_name vlm-sft-${ARGS["run_name"]}-$model_save_name \
        --per_device_train_batch_size ${ARGS["batch_size"]} --per_device_eval_batch_size ${ARGS["batch_size"]} \
        $validation_flag \
        --logging_strategy steps --logging_steps 200 \
        --save_strategy epoch --save_steps 0.5 \
        --eval_strategy epoch --eval_steps 0.5 \
        --early_stopping_patience 2 --load_best_model_at_end \
        --num_train_epochs ${ARGS["num_train_epochs"]} \
        --lora_r ${ARGS["lora_rank"]} \
        --lora_alpha ${ARGS["lora_alpha"]} \
        --learning_rate ${ARGS["learning_rate"]} \
        --weight_decay ${ARGS["weight_decay"]} $action_loss_flag $overwrite_flag $push_to_hub_flag || exit 1
