#!/bin/sh
# Installed in the sandbox image as /opt/sandbox/bin/prime-agent, ahead of /opt/pa/bin on PATH.
#
# prime-agent's Python kernel lives at $HOME/.prime/agent/kernel-venv and must be writable:
# skills that /refine creates are installed into it, and a kernel handed over read-only
# (PRIME_AGENT_KERNEL_PYTHON) silently disables them. So the image carries a kernel built at
# exactly that path and copies it into each fresh agent home on first use. The harness always
# mounts the agent home at /home/agent, which keeps the venv's absolute paths valid.
#
# The copy goes to a temporary sibling and is renamed into place, so two processes starting in
# the same fresh home -- a session and its subagent -- cannot both half-write it; the loser
# discards its copy.
seed=/opt/pa-kernel/seed-venv
dest="$HOME/.prime/agent/kernel-venv"
if [ ! -e "$dest" ] && [ -d "$seed" ]; then
    mkdir -p "$HOME/.prime/agent"
    tmp="$(mktemp -d "$HOME/.prime/agent/.kernel-venv.seeding.XXXXXX")"
    if cp -R -P --preserve=mode,timestamps "$seed/." "$tmp/" && mv -T "$tmp" "$dest" 2>/dev/null; then
        :
    else
        rm -rf "$tmp"
    fi
fi
exec /opt/pa/bin/prime-agent "$@"
