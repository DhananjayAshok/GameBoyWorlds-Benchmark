#!/usr/bin/env bash
# Short RL run per title, to check a game's environment starts and steps at all.
# Disable train logging before running this over many titles.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }
source configs/config.env || { echo "Could not source configs/config.env"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

ARGS["games"]="harry_potter_philosophers_stone,harry_potter_chamber_of_secrets"
ARGS["timesteps"]=1000
ARGS["log_dir"]="logs/smoke_games"

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

IFS=',' read -ra games <<< "${ARGS["games"]}"
log_dir="${ARGS["log_dir"]}"
mkdir -p "$log_dir"

FAILED_GAMES=""
for game in "${games[@]}"; do
    echo "Testing game: $game -> $log_dir/${game}.out"
    if ! bash scripts/core_rl/default_rl.sh \
            --game "$game" \
            --timesteps "${ARGS["timesteps"]}" &> "$log_dir/${game}.out"; then
        echo "FAILED: $game"
        FAILED_GAMES+="$game "
    fi
done

echo ""
[[ -n "$FAILED_GAMES" ]] && echo "Failed games: $FAILED_GAMES"
echo "DONE ALL"
