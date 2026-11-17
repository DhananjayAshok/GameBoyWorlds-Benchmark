#!/usr/bin/env bash
# Assert what the sandbox image must be true of, against a built .sif.
#
#     bash containers/verify_sandbox.sh <storage>/containers/sandbox-pa.sif
#
# Every check is a claim that would otherwise be an assumption. The isolation ones are
# re-run on every rebuild rather than inherited: a new base digest reasserts them, it does
# not preserve them.

set -uo pipefail
SIF="${1:?usage: verify_sandbox.sh <image.sif>}"
module load apptainer 2>/dev/null
export APPTAINER_CACHEDIR="/scratch1/$USER/apptainer_cache"
export TMPDIR="/scratch1/$USER/tmp"

PASS=0
FAIL=0
run() {
    local label="$1"; shift
    if "$@" >/dev/null 2>&1; then
        printf '  PASS  %s\n' "$label"; PASS=$((PASS + 1))
    else
        printf '  FAIL  %s\n' "$label"; FAIL=$((FAIL + 1))
    fi
}
# The checks run with the flags the harness launches the agent with, not bare --containall:
#   - bare --containall mounts an EMPTY tmpfs at the invoking user's home path, so
#     `test -e /home1/$USER` is true there while nothing of the real home is visible; with the
#     harness's --home <agent home>:/home/agent the path is not there at all;
#   - --net --network none: an empty network namespace. Without it the sandbox shares the node's
#     loopback and can connect to every service listening on it -- measured: the VS Code server.
# Agent homes go in node-local memory, as the harness should place them: seeding the kernel's ~8.8k
# files onto BeeGFS takes ~40 s, into /dev/shm ~3 s.
AGENT_HOME="$(mktemp -d /dev/shm/agenthome.XXXXXX)"
NETDIR="$(mktemp -d /dev/shm/sandboxnet.XXXXXX)"
HELPERS=()
trap 'kill "${HELPERS[@]}" 2>/dev/null; rm -rf "$AGENT_HOME" "$NETDIR"' EXIT
sandbox() {
    apptainer exec --containall --cleanenv --net --network none \
        --home "$AGENT_HOME":/home/agent --bind "$NETDIR":/sandboxnet "$@"
}
absent() {
    local label="$1" path="$2"
    if sandbox "$SIF" test -e "$path" >/dev/null 2>&1; then
        printf '  FAIL  %s (%s is visible)\n' "$label" "$path"; FAIL=$((FAIL + 1))
    else
        printf '  PASS  %s\n' "$label"; PASS=$((PASS + 1))
    fi
}

echo "image: $SIF"
echo
echo "== provenance =="
apptainer inspect "$SIF" 2>/dev/null | grep -E "prime_agent|base\.|installer|kernel\.|telemetry" | sed 's/^/  /'

echo
echo "== prime-agent =="
echo -n "  version: "
sandbox "$SIF" prime-agent --version 2>&1 | head -1
run "prime-agent is on PATH" \
    sandbox "$SIF" sh -c 'command -v prime-agent'
run "it lives under the pinned prefix" \
    sandbox "$SIF" sh -c '[ -x /opt/pa/bin/prime-agent ]'
run "reports the pinned version 0.10.3" \
    sandbox "$SIF" sh -c 'prime-agent --version 2>&1 | grep -q 0\.10\.3'
run "headless --print is accepted (-p in --help)" \
    sandbox "$SIF" sh -c 'prime-agent --help 2>&1 | grep -q -- "--print"'
run "the seeding wrapper is what PATH resolves" \
    sandbox "$SIF" sh -c '[ "$(command -v prime-agent)" = /opt/sandbox/bin/prime-agent ]'
run "telemetry is off" \
    sandbox "$SIF" sh -c '[ "$PRIME_AGENT_TELEMETRY" = 0 ] && [ "$DO_NOT_TRACK" = 1 ]'

echo
echo "== the Python kernel: pinned, in the image, seeded writable =="
run "uv 0.13.8 is on PATH" \
    sandbox "$SIF" sh -c 'uv --version | grep -q "^uv 0\.13\.8"'
run "the seed kernel is CPython 3.11.18" \
    sandbox "$SIF" sh -c '/opt/pa-kernel/seed-venv/bin/python --version 2>&1 | grep -q "3\.11\.18"'
run "the seed kernel's base interpreter is in the image, not under a build HOME" \
    sandbox "$SIF" grep -qx "home = /opt/pa-kernel/data/uv/python/.*" /opt/pa-kernel/seed-venv/pyvenv.cfg
run "the bundled skills are installed and recorded, so a session installs nothing" \
    sandbox "$SIF" /opt/pa-kernel/seed-venv/bin/python -c "
import json, agent_message, attach_image, compact, edit, goal, refine
v = json.load(open('/opt/pa-kernel/seed-venv/.bootstrap-version'))
assert len(v.get('python_skills') or []) == 6, v.get('python_skills')"
run "the seed kernel imports its runtime and default packages" \
    sandbox "$SIF" /opt/pa-kernel/seed-venv/bin/python -c "import rlm, dill, requests, numpy, pandas, scipy, pydantic, yaml, bs4, lxml, tyro, dotenv, tomli, httpx"
run "every constraint pin is what got installed" \
    sandbox "$SIF" sh -c 'grep -v "^#" /dev/stdin | while read -r pin; do grep -qix "$pin" /opt/pa-kernel/freeze.txt || { echo "missing $pin"; exit 1; }; done' \
    < "$(dirname "$0")/kernel_constraints.txt"
# A fresh agent home gets a writable copy on first use, and a second start reuses it.
SEEDHOME="$(mktemp -d /dev/shm/seedhome.XXXXXX)"
run "a fresh agent home is seeded with a kernel on first use" \
    apptainer exec --containall --cleanenv --net --network none --home "$SEEDHOME":/home/agent "$SIF" sh -c \
    'prime-agent --version >/dev/null && ~/.prime/agent/kernel-venv/bin/python -c "import rlm"'
run "the seeded kernel is writable (so /refine skills can install)" \
    test -w "$SEEDHOME/.prime/agent/kernel-venv/lib"
run "no half-written seed copies are left behind" \
    sh -c '! ls -d "$0"/.prime/agent/.kernel-venv.seeding.* >/dev/null 2>&1' "$SEEDHOME"
rm -rf "$SEEDHOME"

echo
echo "== no agent state baked into the image =="
# Checked without the agent-home mount, which would hide the image's own /home/agent.
in_image_absent() {
    local label="$1" path="$2"
    if apptainer exec --containall --cleanenv --no-home "$SIF" test -e "$path" >/dev/null 2>&1; then
        printf '  FAIL  %s (%s is in the image)\n' "$label" "$path"; FAIL=$((FAIL + 1))
    else
        printf '  PASS  %s\n' "$label"; PASS=$((PASS + 1))
    fi
}
in_image_absent "no ~/.prime under /root"          /root/.prime
in_image_absent "no build HOME (/home/agent)"      /home/agent
in_image_absent "no legacy build HOME"             /tmp/build-home
in_image_absent "no installer or constraints left" /opt/build
in_image_absent "no uv build cache left"           /tmp/uv-build-cache
run "the seed holds a kernel and nothing else of an agent home" \
    sandbox "$SIF" sh -c '[ ! -e /opt/pa-kernel/seed-venv/sessions ] && [ ! -e /opt/pa-kernel/seed-venv/auth.json ] && [ -f /opt/pa-kernel/seed-venv/.bootstrap-version ]'

echo
echo "== isolation: nothing of this project is reachable =="
absent "user home absent"        /home1/ashokd
absent "GameBoyRL absent"        /home1/ashokd/projects/GameBoyRL
absent "project storage absent"  /project2/jessetho_1732
absent "ROM data absent"         /project2/jonmay_1426
absent "scratch absent"          /scratch1/ashokd

echo
echo "== runtime shape =="
echo -n "  glibc: "
sandbox "$SIF" ldd --version 2>/dev/null | head -1
echo -n "  python: "
sandbox "$SIF" python --version 2>&1
run "HOME is the separate agent home" \
    sandbox "$SIF" sh -c '[ "$HOME" = /home/agent ]'

echo
echo "== isolation: the network =="
# A listener on the host's loopback, standing in for whatever else runs on the node (it was the
# VS Code server when this was measured). It must be unreachable from inside.
HOSTPORT="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
python3 -c 'import socket, sys
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(8)
while True: s.accept()[0].close()' "$HOSTPORT" &
HELPERS+=($!)
# A Unix socket in the bind-mounted channel directory, standing in for the env server and the gateway.
python3 -c 'import os, socket, sys
p = sys.argv[1]; s = socket.socket(socket.AF_UNIX); s.bind(p); os.chmod(p, 0o600); s.listen(8)
while True:
    c = s.accept()[0]; c.sendall(b"pong"); c.close()' "$NETDIR/echo.sock" &
HELPERS+=($!)
for _ in $(seq 50); do [ -S "$NETDIR/echo.sock" ] && break; sleep 0.1; done
connect='import socket, sys
try: socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=4).close()
except OSError: sys.exit(1)'
run "a service on the host's loopback is unreachable" \
    sandbox "$SIF" sh -c '! python -c "$0" 127.0.0.1 "$1"' "$connect" "$HOSTPORT"
run "the internet is unreachable directly" \
    sandbox "$SIF" sh -c '! python -c "$0" 1.1.1.1 443' "$connect"
run "DNS does not resolve" \
    sandbox "$SIF" sh -c '! python -c "import socket; socket.gethostbyname(\"pypi.org\")"'
run "the node's listening sockets are not visible" \
    sandbox "$SIF" sh -c '[ "$(awk "NR>1 && \$4==\"0A\"" /proc/net/tcp /proc/net/tcp6 | wc -l)" = 0 ]'
run "sandbox-run bridges a named channel, and only that one" \
    env APPTAINERENV_SANDBOX_BRIDGE="9000=/sandboxnet/echo.sock" \
    apptainer exec --containall --cleanenv --net --network none --home "$AGENT_HOME":/home/agent \
        --bind "$NETDIR":/sandboxnet "$SIF" sandbox-run python -c 'import socket
s = socket.create_connection(("127.0.0.1", 9000), timeout=4); assert s.recv(16) == b"pong"
try:
    socket.create_connection(("127.0.0.1", 9001), timeout=2); raise SystemExit("an unbridged port answered")
except ConnectionRefusedError: pass'

echo
echo "== a bind-mounted workspace is writable and is all that /work holds =="
WS="$(mktemp -d "${TMPDIR:-/tmp}/ws.XXXXXX")"
echo "hello" > "$WS/marker.txt"
run "workspace readable on /work" \
    sandbox --bind "$WS":/work "$SIF" sh -c 'grep -q hello /work/marker.txt'
run "workspace writable on /work" \
    sandbox --bind "$WS":/work "$SIF" sh -c 'touch /work/written-from-inside'
run "/work holds only the workspace" \
    sandbox --bind "$WS":/work "$SIF" sh -c \
    '[ "$(ls -A /work | sort | tr "\n" " ")" = "marker.txt written-from-inside " ]'
rm -rf "$WS"

echo
echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
