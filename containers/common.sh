# Sourced by the sandbox scripts. Site-specific locations come from the environment:
#   SANDBOX_SIF           the built image (default <storage_dir>/containers/sandbox.sif)
#   SANDBOX_RUNTIME_DIR   where sockets and agent homes go (default /dev/shm, else $TMPDIR, else /tmp)

ENV_SERVER_PORT=8710
MODEL_GATEWAY_PORT=8720
SANDBOX_FLAGS=(--containall --cleanenv --net --network none)

die() { echo "error: $*" >&2; exit 1; }

require_apptainer() {
    if ! command -v apptainer >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
        module load apptainer 2>/dev/null || true
    fi
    command -v apptainer >/dev/null 2>&1 \
        || die "apptainer is not on PATH. Install it, or load it (e.g. 'module load apptainer') and retry."
}

# Unix socket paths are limited to ~100 characters, so this directory must have a short path.
sandbox_runtime_dir() {
    local dir
    for dir in "${SANDBOX_RUNTIME_DIR:-}" /dev/shm "${TMPDIR:-}" /tmp; do
        if [[ -n "$dir" && -d "$dir" && -w "$dir" ]]; then echo "$dir"; return; fi
    done
    die "no writable runtime directory; set SANDBOX_RUNTIME_DIR"
}

default_sif() {
    echo "${SANDBOX_SIF:-${storage_dir:?source configs/config.env first}/containers/sandbox.sif}"
}

# wait_for_socket <path> <pid> <log> [seconds]: fails if the process dies or the socket never appears.
wait_for_socket() {
    local path="$1" pid="$2" log="$3" tries=$(( ${4:-120} * 2 ))
    while (( tries-- > 0 )); do
        [[ -S "$path" ]] && return 0
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    echo "--- $log" >&2
    tail -n 30 "$log" >&2
    die "$(basename "$path") did not come up"
}
