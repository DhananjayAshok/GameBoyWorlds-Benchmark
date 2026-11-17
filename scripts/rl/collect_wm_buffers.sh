#!/usr/bin/env bash
# Collects the replay buffers the world-model baseline trains on: for every train game of
# --game and every init_state, runs a random-action agent then a curiosity agent and keeps
# both buffers. The buffers are large; see the size estimate printed at startup.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }
python "$PROJECT_ROOT/python_funcs.py" task_dictionary || { echo "Could not regenerate train states"; exit 1; }
source scripts/core/all_train_states.sh

declare -A ARGS
REQUIRED_ARGS=()
populate_array CREATE_TRAJ_ESSENTIALS REQUIRED_ARGS
# init_state comes from the TRAIN_STATES loop below, not the CLI.
for i in "${!REQUIRED_ARGS[@]}"; do
    [[ "${REQUIRED_ARGS[$i]}" == "init_state" ]] && unset 'REQUIRED_ARGS[$i]'
done
REQUIRED_ARGS=("${REQUIRED_ARGS[@]}")
populate_dict CREATE_TRAJ_DEFAULTS ARGS


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

# These two names collide with paths the iterative training machinery composes for itself.
if [[ -z "${ARGS["run_name"]}" ]] || [[ "${ARGS["run_name"]}" == "default" ]] || [[ "${ARGS["run_name"]}" == "iterative" ]]; then
    echo "Error: --run_name is required and cannot be 'default' or 'iterative'."
    usage
fi

echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

game="${ARGS["game"]}"
run_name="${ARGS["run_name"]}"

mapfile -t train_games < <(path_of train_games --game "$game")
if [[ ${#train_games[@]} -eq 0 ]]; then
    echo "Error: no train games for $game (path_of train_games returned nothing)."
    exit 1
fi
echo "Train games for $game: ${train_games[*]}"

ARGS["n_agents"]=1                  # no iterative self-improvement; still 2 agent rounds
ARGS["sweep"]=false                 # not the seed/gamma/algorithm sweep
ARGS["combination_buffer_sweep"]=false
ARGS["keep_buffers"]=true           # the buffers ARE the output
ARGS["group_trajectories"]=false    # nothing downstream of grouping is wanted here

curiosity_timesteps="${ARGS["timesteps"]}"
random_timesteps=$(( curiosity_timesteps / 10 ))
if [[ $random_timesteps -lt 1 ]]; then
    echo "Error: --timesteps ${curiosity_timesteps} is too small; the random run would get $random_timesteps steps."
    exit 1
fi
curiosity_algorithm="${ARGS["algorithm"]}"

# 46080 = FRAME_STACK(2) * 144 * 160 uint8, the on-disk cost of one stored step.
bytes_per_step=46080
total_states=0
for train_game in "${train_games[@]}"; do
    IFS=',' read -ra _states <<< "${TRAIN_STATES[$train_game]}"
    total_states=$(( total_states + ${#_states[@]} ))
done
# 2 agent rounds, doubled by combination_buffer_sweep, plus the one random run.
est_steps=$(( total_states * (4 * curiosity_timesteps + random_timesteps) ))
est_gb=$(( est_steps * bytes_per_step / 1024 / 1024 / 1024 ))
echo ""
echo "Estimated replay buffer size: ~${est_gb} GB across ${total_states} init states"
echo "  (${total_states} states x (4 curiosity runs x ${curiosity_timesteps} + ${random_timesteps} random) steps x ${bytes_per_step} B)"
echo "Check your quota with myquota before letting this run to completion."
echo ""

collected=""
failed=""
for train_game in "${train_games[@]}"; do
    IFS=',' read -ra init_states_arr <<< "${TRAIN_STATES[$train_game]}"
    if [[ ${#init_states_arr[@]} -eq 0 ]]; then
        echo "WARNING: $train_game declares no train states; skipping."
        continue
    fi

    ARGS["game"]="$train_game"
    ARGS["model_dir"]="$run_name"

    echo ""
    echo "############################################################"
    echo "# $train_game  —  random buffers (${#init_states_arr[@]} init states)"
    echo "############################################################"

    ARGS["algorithm"]="random"
    ARGS["timesteps"]="$random_timesteps"
    for init_state in "${init_states_arr[@]}"; do
        ARGS["init_state"]="$init_state"
        ARGS["replay_buffer_save_folder"]="${run_name}/${init_state}/random/"
        argstring=$(args_to_flags_subset ARGS TRAINING_ARG_KEYS)
        if ! bash scripts/core_rl/train.sh $argstring; then
            echo "WARNING: random collection failed for $train_game/$init_state; continuing."
            failed+="${failed:+ }$train_game/$init_state:random"
        fi
    done

    ARGS["algorithm"]="$curiosity_algorithm"
    ARGS["timesteps"]="$curiosity_timesteps"
    # create_all_traj.sh rejects --init_state, so drop the one the random loop left behind.
    unset ARGS["replay_buffer_save_folder"]
    unset ARGS["init_state"]

    echo ""
    echo "############################################################"
    echo "# $train_game  —  curiosity buffers (${#init_states_arr[@]} init states)"
    echo "############################################################"

    argstring=$(args_to_flags_subset ARGS CREATE_TRAJ_ARG_KEYS)
    if bash scripts/rl/create_all_traj.sh $argstring; then
        collected+="${collected:+ }$train_game"
    else
        echo "WARNING: curiosity collection failed for $train_game; continuing."
        failed+="${failed:+ }$train_game:curiosity"
    fi
done
ARGS["game"]="$game"

echo ""
echo "############################################################"
echo "# $game  —  collection summary"
echo "############################################################"
echo "  collected: ${collected:-<none>}"
echo "  failed:    ${failed:-<none>}"
echo "  buffers:   $storage_dir/replay_buffers/<train_game>/$run_name/"

if [[ -z "$collected" ]]; then
    echo "Error: no train game collected for $game."
    exit 1
fi
