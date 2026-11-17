"""
Runs inside the sandbox: exposes bind-mounted Unix sockets as TCP ports on the sandbox's private loopback.

    python netbridge.py 8710=/sandboxnet/envserver.sock 8720=/sandboxnet/model.sock
"""
import asyncio
import sys


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


def _forward(socket_path: str):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            up_reader, up_writer = await asyncio.open_unix_connection(socket_path)
        except OSError:
            writer.close()
            return
        await asyncio.gather(_pipe(reader, up_writer), _pipe(up_reader, writer))
    return handle


async def main(pairs: list) -> None:
    servers = []
    for pair in pairs:
        port, _, path = pair.partition("=")
        servers.append(await asyncio.start_server(_forward(path), host="127.0.0.1", port=int(port)))
    print("netbridge ready", flush=True)
    await asyncio.gather(*(s.serve_forever() for s in servers))


if __name__ == "__main__":
    if len(sys.argv) < 2 or any("=" not in a for a in sys.argv[1:]):
        sys.exit("usage: netbridge.py <port>=<unix socket path> ...")
    asyncio.run(main(sys.argv[1:]))
