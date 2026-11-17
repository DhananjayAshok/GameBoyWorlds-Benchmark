#!/usr/bin/env bash
# Loads a trained model by --exp_name and evaluates it on a test environment, which may
# differ from the training one. Also the evaluation phase of default_rl.sh.

source scripts/core/utils.sh

declare -A ARGS
declare -A ARGS

REQUIRED_ARGS=()

populate_dict EVALUATION_DEFAULTS ARGS
populate_array EVALUATION_ESSENTIALS REQUIRED_ARGS

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
                echo "Error: Unknown flag --$FLAG. Allowed flags are: ${ALLOWED_FLAGS[*]}"
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

echo "Script $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

if [[ "${ARGS["curiosity_module"]}" != "combinationbuffer" ]]; then
    ARGS["ocr_alpha"]=0.0
fi


exp_name="${ARGS["exp_name"]}"
test_env_id=$(get_string_from_args "env_id" ARGS)
if [[ -z "$test_env_id" ]]; then
    echo "Error: Failed to construct test environment ID. Please check your input parameters."
    exit 1
fi
test_env_id="${test_env_id}-True"

if [[ "${ARGS["model_dir"]}" != "none" ]]; then
    model_save_path="$storage_dir/models/${ARGS["game"]}/${ARGS["model_dir"]}/$exp_name/"
else
    model_save_path="$storage_dir/models/${ARGS["game"]}/$exp_name/"
fi

extra_arg_part=""
if [[ "${ARGS["buffer_save_path"]}" != "none" ]]; then
    extra_arg_part+="--buffer_save_path $storage_dir/curiosity_buffers/${ARGS["curiosity_module"]}/${ARGS["game"]}/${ARGS["buffer_save_path"]} "
fi

if [[ "${ARGS["buffer_load_path"]}" != "none" ]]; then
    extra_arg_part+="--buffer_load_path $storage_dir/curiosity_buffers/${ARGS["curiosity_module"]}/${ARGS["game"]}/${ARGS["buffer_load_path"]} "
fi
if [[ "${ARGS["embedder_load_path"]}" != "none" ]]; then
    extra_arg_part+="--embedder_load_path $storage_dir/observation_embedders/${ARGS["game"]}/${ARGS["embedder_load_path"]} "
fi


cd cleanrl

python cleanrl_utils/enjoy.py --exp-name ${ARGS["algorithm"]}_curiosity --model_path $model_save_path \
    --env-id $test_env_id --save-name $exp_name \
    --similarity_metric ${ARGS["similarity_metric"]} --observation_embedder ${ARGS["observation_embedder"]} \
    --curiosity_module ${ARGS["curiosity_module"]} --ocr_alpha ${ARGS["ocr_alpha"]} $extra_arg_part

cd ..