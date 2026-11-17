#!/usr/bin/env bash
# Check a built sandbox image: pinned contents, no baked-in agent state, and isolation from the
# host filesystem and network. Exits non-zero if any check fails.
#
#     bash containers/verify_sandbox.sh [<image.sif>]

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source configs/config.env || { echo "Could not source configs/config.env"; exit 1; }
source containers/common.sh
source containers/versions.env

SIF="${1:-$(default_sif)}"
[[ -f "$SIF" ]] || die "no image at $SIF; build it first (see containers/sandbox.def)"
require_apptainer
REPO_ROOT="$(pwd -P)"
RUNTIME="$(sandbox_runtime_dir)"

PASS=0
FAIL=0
record() {
    local ok="$1" label="$2"
    if [[ "$ok" == 0 ]]; then
        printf '  PASS  %s\n' "$label"; PASS=$((PASS + 1))
    else
        printf '  FAIL  %s\n' "$label"; FAIL=$((FAIL + 1))
    fi
}
run() {
    local label="$1"; shift
    "$@" >/dev/null 2>&1
    record $? "$label"
}

AGENT_HOME="$(mktemp -d "$RUNTIME/agenthome.XXXXXX")"
NETDIR="$(mktemp -d "$RUNTIME/sandboxnet.XXXXXX")"
SEEDHOME="$(mktemp -d "$RUNTIME/seedhome.XXXXXX")"
WS="$(mktemp -d "$RUNTIME/ws.XXXXXX")"
HELPERS=()
cleanup() {
    if [[ ${#HELPERS[@]} -gt 0 ]]; then
        kill "${HELPERS[@]}" 2>/dev/null
        wait "${HELPERS[@]}" 2>/dev/null
    fi
    rm -rf "$AGENT_HOME" "$NETDIR" "$SEEDHOME" "$WS"
}
trap cleanup EXIT

# The flags run_sandbox.sh launches the agent with.
sandbox() {
    apptainer exec "${SANDBOX_FLAGS[@]}" \
        --home "$AGENT_HOME":/home/agent --bind "$NETDIR":/sandboxnet "$@"
}
# A host path that exists outside must not exist inside. A path absent on the host proves nothing.
absent() {
    local label="$1" path="$2"
    if [[ -z "$path" || ! -e "$path" ]]; then
        printf '  SKIP  %s (not present on this host)\n' "$label"
        return
    fi
    sandbox "$SIF" test -e "$path" >/dev/null 2>&1
    record $(( $? == 0 )) "$label"
}
in_image_absent() {
    local label="$1" path="$2"
    apptainer exec --containall --cleanenv --no-home "$SIF" test -e "$path" >/dev/null 2>&1
    record $(( $? == 0 )) "$label"
}
LABELS="$(apptainer inspect "$SIF" 2>/dev/null)"
has_label() {
    grep -qF "$1: $2" <<< "$LABELS"
}

echo "image: $SIF"
echo
echo "== provenance =="
run "label prime_agent.version is $PRIME_AGENT_VERSION" has_label prime_agent.version "$PRIME_AGENT_VERSION"
run "label installer.sha256 matches versions.env"      has_label installer.sha256 "$INSTALLER_SHA256"
run "label kernel.python is $KERNEL_PYTHON_VERSION"     has_label kernel.python "$KERNEL_PYTHON_VERSION"
run "the vendored installer matches its pinned hash" \
    sh -c 'echo "$0  containers/prime_agent_install.sh" | sha256sum -c -' "$INSTALLER_SHA256"

echo
echo "== prime-agent =="
run "prime-agent on PATH is the seeding wrapper" \
    sandbox "$SIF" sh -c '[ "$(command -v prime-agent)" = /opt/sandbox/bin/prime-agent ]'
run "reports the pinned version $PRIME_AGENT_VERSION" \
    sandbox "$SIF" sh -c 'prime-agent --version 2>&1 | grep -qF "$0"' "$PRIME_AGENT_VERSION"
run "headless --print is accepted" \
    sandbox "$SIF" sh -c 'prime-agent --help 2>&1 | grep -q -- "--print"'
run "telemetry is off" \
    sandbox "$SIF" sh -c '[ "$PRIME_AGENT_TELEMETRY" = 0 ] && [ "$DO_NOT_TRACK" = 1 ]'

echo
echo "== the Python kernel =="
run "uv $UV_VERSION is on PATH" \
    sandbox "$SIF" sh -c 'uv --version | grep -qF "uv $0"' "$UV_VERSION"
run "the seed kernel is CPython $KERNEL_PYTHON_VERSION" \
    sandbox "$SIF" sh -c '/opt/pa-kernel/seed-venv/bin/python --version 2>&1 | grep -qF "$0"' "$KERNEL_PYTHON_VERSION"
run "the seed kernel's base interpreter is in the image" \
    sandbox "$SIF" grep -qx "home = /opt/pa-kernel/data/uv/python/.*" /opt/pa-kernel/seed-venv/pyvenv.cfg
run "the bundled skills are installed and recorded" \
    sandbox "$SIF" /opt/pa-kernel/seed-venv/bin/python -c "
import json, agent_message, attach_image, compact, edit, goal, refine
v = json.load(open('/opt/pa-kernel/seed-venv/.bootstrap-version'))
assert len(v.get('python_skills') or []) == 6, v.get('python_skills')"
run "the seed kernel imports its runtime and default packages" \
    sandbox "$SIF" /opt/pa-kernel/seed-venv/bin/python -c "import rlm, dill, requests, numpy, pandas, scipy, pydantic, yaml, bs4, lxml, tyro, dotenv, tomli, httpx"
run "every constraint pin is what got installed" \
    sandbox "$SIF" sh -c 'grep -v "^#" /dev/stdin | while read -r pin; do grep -qix "$pin" /opt/pa-kernel/freeze.txt || exit 1; done' \
    < containers/kernel_constraints.txt
run "a fresh agent home is seeded with a kernel on first use" \
    apptainer exec "${SANDBOX_FLAGS[@]}" --home "$SEEDHOME":/home/agent "$SIF" sh -c \
    'prime-agent --version >/dev/null && ~/.prime/agent/kernel-venv/bin/python -c "import rlm"'
run "the seeded kernel is writable" \
    test -w "$SEEDHOME/.prime/agent/kernel-venv/lib"
run "no half-written seed copies are left behind" \
    sh -c '! ls -d "$0"/.prime/agent/.kernel-venv.seeding.* >/dev/null 2>&1' "$SEEDHOME"

echo
echo "== no agent state baked into the image =="
in_image_absent "no ~/.prime under /root"          /root/.prime
in_image_absent "no build HOME (/home/agent)"      /home/agent
in_image_absent "no installer or constraints left" /opt/build
in_image_absent "no uv build cache left"           /tmp/uv-build-cache
run "the seed holds a kernel and nothing else of an agent home" \
    sandbox "$SIF" sh -c '[ ! -e /opt/pa-kernel/seed-venv/sessions ] && [ ! -e /opt/pa-kernel/seed-venv/auth.json ] && [ -f /opt/pa-kernel/seed-venv/.bootstrap-version ]'

echo
echo "== isolation: the host filesystem =="
ROM_STORAGE="$(sed -n 's/^export storage_dir=//p' GameBoyWorlds/configs/config.env 2>/dev/null | tr -d "\"'")"
absent "the user's home is not visible"            "$HOME"
absent "this repository is not visible"            "$REPO_ROOT"
absent "the project storage is not visible"        "$storage_dir"
absent "the ROM and save-state storage is not visible" "$ROM_STORAGE"
absent "the split file is not visible"             "$REPO_ROOT/configs/splits/benchmark_splits.json"
run "HOME inside is the separate agent home" \
    sandbox "$SIF" sh -c '[ "$HOME" = /home/agent ]'

echo
echo "== isolation: the network =="
# A listener on the host's loopback stands in for any other service on the machine.
HOSTPORT="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
python3 -c 'import socket, sys
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(8)
while True: s.accept()[0].close()' "$HOSTPORT" &
HELPERS+=($!)
# A Unix socket in the bind-mounted channel directory stands in for the env server and the gateway.
python3 -c 'import os, socket, sys
p = sys.argv[1]; s = socket.socket(socket.AF_UNIX); s.bind(p); os.chmod(p, 0o600); s.listen(8)
while True:
    c = s.accept()[0]; c.sendall(b"pong"); c.close()' "$NETDIR/echo.sock" &
HELPERS+=($!)
for _ in $(seq 50); do [[ -S "$NETDIR/echo.sock" ]] && break; sleep 0.1; done
[[ -S "$NETDIR/echo.sock" ]] || die "the test socket did not come up"
connect='import socket, sys
try: socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=4).close()
except OSError: sys.exit(1)'
run "the host-loopback listener is reachable from the host (the probe works)" \
    python3 -c "$connect" 127.0.0.1 "$HOSTPORT"
run "a service on the host's loopback is unreachable" \
    sandbox "$SIF" sh -c '! python -c "$0" 127.0.0.1 "$1"' "$connect" "$HOSTPORT"
run "the internet is unreachable directly" \
    sandbox "$SIF" sh -c '! python -c "$0" 1.1.1.1 443' "$connect"
run "DNS does not resolve" \
    sandbox "$SIF" sh -c '! python -c "import socket; socket.gethostbyname(\"pypi.org\")"'
run "the host's listening sockets are not visible" \
    sandbox "$SIF" sh -c '[ "$(awk "NR>1 && \$4==\"0A\"" /proc/net/tcp /proc/net/tcp6 | wc -l)" = 0 ]'
run "sandbox-run bridges a named channel, and only that one" \
    env APPTAINERENV_SANDBOX_BRIDGE="9000=/sandboxnet/echo.sock" \
    apptainer exec "${SANDBOX_FLAGS[@]}" --home "$AGENT_HOME":/home/agent \
        --bind "$NETDIR":/sandboxnet "$SIF" sandbox-run python -c 'import socket
s = socket.create_connection(("127.0.0.1", 9000), timeout=4); assert s.recv(16) == b"pong"
try:
    socket.create_connection(("127.0.0.1", 9001), timeout=2); raise SystemExit("an unbridged port answered")
except ConnectionRefusedError: pass'

echo
echo "== the workspace =="
echo "hello" > "$WS/marker.txt"
run "workspace readable on /work" \
    sandbox --bind "$WS":/work "$SIF" sh -c 'grep -q hello /work/marker.txt'
run "workspace writable on /work" \
    sandbox --bind "$WS":/work "$SIF" sh -c 'touch /work/written-from-inside'
run "/work holds only the workspace" \
    sandbox --bind "$WS":/work "$SIF" sh -c \
    '[ "$(ls -A /work | sort | tr "\n" " ")" = "marker.txt written-from-inside " ]'

echo
echo "$PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]
