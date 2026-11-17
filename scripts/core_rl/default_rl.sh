#!/usr/bin/env bash
# Train-then-evaluate cycle: calls train.sh then enjoy.sh on the resulting model.
# --train_only / --eval_only run one half; --test_env / --test_init_state override the test env.

source scripts/core/utils.sh

declare -A ARGS
ARGS["test_env"]="none"
ARGS["test_init_state"]="none"
ARGS["train_only"]="false"
ARGS["eval_only"]="false"


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

if [[ "${ARGS["eval_only"]}" == "true" || "${ARGS["eval_only"]}" == "yes" || "${ARGS["eval_only"]}" == "y" ]]; then
    echo "Eval only flag is set. Skipping training and going to evaluation."
else
    arg_string=$(args_to_flags_subset ARGS TRAINING_ARG_KEYS)
    bash scripts/core_rl/train.sh $arg_string || exit 1
fi

if [[ -z "${ARGS["test_env"]}" ]]; then
    ARGS["test_env"]="${ARGS["env"]}"
fi
if [[ -z "${ARGS["test_init_state"]}" ]]; then
    ARGS["test_init_state"]="${ARGS["init_state"]}"
fi

if [[ "${ARGS["train_only"]}" == "true" || "${ARGS["train_only"]}" == "yes" || "${ARGS["train_only"]}" == "y" ]]; then
    echo "Train only flag is set. Exiting after training."
    exit 0
fi


ARGS["exp_name"]=$exp_name
arg_string=$(args_to_flags_subset ARGS EVALUATION_ARG_KEYS)
bash scripts/core_rl/enjoy.sh $arg_string || exit 1