#!/usr/bin/env bash
# Every playthrough, in sequence, on one model. The four Pokemon titles the strategist
# plays; see runs/games.sh POKEMON_GAMES.
#
# Sequential on purpose: each playthrough owns an emulator and a long wall clock, so these
# are normally one slurm job each (slurm/playthrough_*.sh). This driver is for running the
# set on one node.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }
source configs/config.env || { echo "Could not source configs/config.env"; exit 1; }
source runs/games.sh || { echo "Could not source runs/games.sh"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=("model")

ARGS["games"]=none
ARGS["vlm_kind"]="openrouter"
ARGS["max_episodes"]=75
ARGS["name_suffix"]="none"
ARGS["init_state"]="initial"
ARGS["resume"]=true

# --- Argument parsing (copy verbatim) ---
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
# --- End argument parsing ---

echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

mapfile -t games < <(resolve_games POKEMON_GAMES "${ARGS["games"]}") || exit 1

# A playthrough's --name keys its state directory, so it has to be stable across resumes and
# distinct per (game, model). "<title>_<suffix>" — red_flash, brown_gemma — is the convention
# the existing runs use.
suffix="${ARGS["name_suffix"]}"
if [[ "$suffix" == "none" ]]; then
    suffix=$(model_save_name "${ARGS["model"]}")
    suffix="${suffix%%-*}"
fi

FAILED_RUNS=""
for game in "${games[@]}"; do
    name="${game#pokemon_}_${suffix}"
    echo ""
    echo "############################################################"
    echo "# $game  —  playthrough '$name'"
    echo "############################################################"
    if ! bash runs/playthrough/playthrough.sh \
            --game "$game" \
            --name "$name" \
            --model "${ARGS["model"]}" \
            --vlm_kind "${ARGS["vlm_kind"]}" \
            --max_episodes "${ARGS["max_episodes"]}" \
            --init_state "${ARGS["init_state"]}" \
            --resume "${ARGS["resume"]}"; then
        echo "FAILED: $game | $name"
        FAILED_RUNS+="$name "
    fi
done

echo ""
[[ -n "$FAILED_RUNS" ]] && echo "Failed playthroughs: $FAILED_RUNS"
echo "DONE ALL"
