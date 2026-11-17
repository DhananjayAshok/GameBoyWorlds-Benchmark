#!/usr/bin/env bash
# Clusters the trajectories of a collected replay buffer by observation similarity
# (z-score gating via --z_min) and writes the groups to --save_path. Called by create_traj.sh.

source scripts/core/utils.sh

declare -A ARGS
REQUIRED_ARGS=("game" "replay_buffer_folder" "save_path")

ARGS["observation_embedder"]=random_patch
ARGS["z_min"]=2.0
ARGS["z_kind"]="global"


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

if [[ "${ARGS["replay_buffer_folder"]}" == "none" ]]; then
    echo "Error: Argument --replay_buffer_folder cannot be none."
    exit 1
fi

replay_buffer_save_folder="${ARGS["replay_buffer_folder"]}"
cd cleanrl
python cleanrl_utils/group_trajectories.py --replay_buffer_folder $storage_dir/replay_buffers/${ARGS["game"]}/$replay_buffer_save_folder --save_path ${ARGS["save_path"]} --z_min ${ARGS["z_min"]} --z_kind ${ARGS["z_kind"]}
cd ..