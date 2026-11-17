#!/bin/sh
# Installed in the sandbox image as /opt/sandbox/bin/sandbox-run: the entry point for an isolated run.
#
#     apptainer exec --containall --cleanenv --net --network none \
#         --bind <dir with the sockets>:/sandboxnet ... sandbox.sif \
#         sandbox-run prime-agent -p "..."
#
# with SANDBOX_BRIDGE="8710=/sandboxnet/envserver.sock 8720=/sandboxnet/model.sock" in the environment.
# Starts netbridge.py for those channels on the sandbox's private loopback, waits until it is
# listening, then becomes the given command. Without SANDBOX_BRIDGE it only runs the command.
#
# Under --network none these are the only addresses that lead anywhere: the sandbox has no
# interface but its own loopback, and no DNS.
if [ -n "$SANDBOX_BRIDGE" ]; then
    log=/tmp/netbridge.log
    # shellcheck disable=SC2086  # SANDBOX_BRIDGE is a space-separated list of port=socket pairs.
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
