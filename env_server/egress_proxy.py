"""
The sandbox's only way out: an HTTP CONNECT proxy on a Unix socket, with a host allowlist.

The sandbox runs in an empty network namespace -- no interfaces but its own loopback, no DNS -- so
it reaches nothing on the node or the internet. This proxy's socket file is bind-mounted in, and
a relay inside the sandbox (containers/netbridge.py) exposes it as 127.0.0.1:<port> there, which
prime-agent uses through HTTPS_PROXY. A Unix socket is a file, not a network endpoint, so it
crosses the namespace boundary while nothing else does.

Only CONNECT to an allowlisted host:port is tunnelled; TLS stays end to end, so the proxy sees the
host name and the byte counts, never the content. Anything else -- another host, another port, a
plain-HTTP request -- is refused. Every request is logged, allowed or not: the log is the record
of what an agent tried to reach.

    python -m env_server.egress_proxy --socket <path> --allow openrouter.ai:443 --log <file>
"""
import argparse
import asyncio
import json
import os
import time

MAX_HEADER_BYTES = 16384
CONNECT_TIMEOUT = 15


def _log(handle, **event) -> None:
    event["t"] = round(time.time(), 3)
    handle.write(json.dumps(event) + "\n")
    handle.flush()


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, counter: list) -> None:
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            counter[0] += len(chunk)
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _refuse(writer: asyncio.StreamWriter, status: str) -> None:
    writer.write(f"HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode())
    await writer.drain()
    writer.close()


def _handler(allow: set, log):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=CONNECT_TIMEOUT)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError):
            _log(log, verdict="malformed")
            writer.close()
            return
        request_line = head.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        parts = request_line.split()
        if len(parts) != 3 or parts[0] != "CONNECT":
            _log(log, verdict="refused", reason="not CONNECT", request=request_line[:200])
            await _refuse(writer, "405 Method Not Allowed")
            return
        target = parts[1].lower()
        if target not in allow:
            _log(log, verdict="refused", reason="not allowlisted", target=target)
            await _refuse(writer, "403 Forbidden")
            return
        host, _, port = target.rpartition(":")
        try:
            up_reader, up_writer = await asyncio.wait_for(asyncio.open_connection(host, int(port)),
                                                          timeout=CONNECT_TIMEOUT)
        except Exception as error:
            _log(log, verdict="upstream failed", target=target, error=type(error).__name__)
            await _refuse(writer, "502 Bad Gateway")
            return
        writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await writer.drain()
        sent, received = [0], [0]
        await asyncio.gather(_pipe(reader, up_writer, sent), _pipe(up_reader, writer, received))
        _log(log, verdict="allowed", target=target, bytes_out=sent[0], bytes_in=received[0])
    return handle


async def serve(*, socket_path: str, allow: set, log_path: str) -> None:
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    with open(log_path, "a") as log:
        server = await asyncio.start_unix_server(_handler(allow, log), path=socket_path,
                                                 limit=MAX_HEADER_BYTES)
        os.chmod(socket_path, 0o600)
        _log(log, verdict="listening", socket=socket_path, allow=sorted(allow))
        async with server:
            await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--allow", action="append", required=True, help="host:port, repeatable")
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    allow = {a.lower() for a in args.allow}
    for entry in allow:
        host, _, port = entry.rpartition(":")
        if not host or not port.isdigit():
            parser.error(f"--allow takes host:port, got {entry!r}")
    asyncio.run(serve(socket_path=args.socket, allow=allow, log_path=args.log))


if __name__ == "__main__":
    main()
