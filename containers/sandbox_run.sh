#!/bin/sh
# /opt/sandbox/bin/sandbox-run in the image. Starts netbridge.py for the port=socket pairs in
# SANDBOX_BRIDGE, waits until it is listening, then becomes the given command.
set -eu
if [ -n "${SANDBOX_BRIDGE:-}" ]; then
    log=/tmp/netbridge.log
    # shellcheck disable=SC2086
    python /opt/sandbox/bin/netbridge.py $SANDBOX_BRIDGE > "$log" 2>&1 &
    tries=0
    until grep -q "netbridge ready" "$log" 2>/dev/null; do
        tries=$((tries + 1))
        if [ "$tries" -gt 100 ]; then
            echo "sandbox-run: netbridge did not start: $(cat "$log" 2>/dev/null)" >&2
            exit 1
        fi
        sleep 0.1
    done
fi
exec "$@"
