#!/bin/sh
# /opt/sandbox/bin/prime-agent in the image. Copies the image's pinned Python kernel into a fresh
# agent home on first use (it must be writable there), then runs the real prime-agent.
seed=/opt/pa-kernel/seed-venv
dest="$HOME/.prime/agent/kernel-venv"
if [ ! -e "$dest" ] && [ -d "$seed" ]; then
    mkdir -p "$HOME/.prime/agent"
    # Copy to a sibling and rename, so two processes starting in one fresh home cannot half-write it.
    tmp="$(mktemp -d "$HOME/.prime/agent/.kernel-venv.seeding.XXXXXX")"
    if ! { cp -R -P --preserve=mode,timestamps "$seed/." "$tmp/" && mv -T "$tmp" "$dest" 2>/dev/null; }; then
        rm -rf "$tmp"
    fi
fi
exec /opt/pa/bin/prime-agent "$@"
