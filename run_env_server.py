import json

import click

from utils import load_parameters


@click.group()
def main():
    """The environment server: emulator sessions over HTTP, under a dev or test policy."""


@main.command("generate_splits")
@click.option("--show", is_flag=True, help="Print the per-game counts.")
def generate_splits(show):
    """Write configs/splits/benchmark_splits.json from split_config.yaml."""
    from env_server import splits
    parameters = load_parameters()
    table = splits.generate(parameters=parameters)
    if show:
        per_game = {}
        for task_id, entry in table.items():
            game = splits.game_of(task_id=task_id)
            per_game.setdefault(game, {}).setdefault(entry["split"], 0)
            per_game[game][entry["split"]] += 1
        for game in sorted(per_game):
            click.echo(f"{game:45s} {per_game[game]}")


@main.command("check")
def check():
    """Assert the dev and test policies against live sessions."""
    from env_server import check as checks
    parameters = load_parameters()
    if not checks.run(parameters=parameters):
        raise SystemExit(1)


@main.command("serve")
@click.option("--kind", required=True, type=click.Choice(["dev", "test"]), help="dev (train + val) or test.")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=None, type=int, help="TCP port. Give this or --unix_socket.")
@click.option("--unix_socket", default=None,
              help="Listen on this Unix socket instead of TCP: how an isolated sandbox reaches the server.")
@click.option("--token_file", required=True, help="A file holding the bearer token clients must send.")
@click.option("--max_sessions", default=4, show_default=True)
@click.option("--idle_seconds", default=1800, show_default=True, help="Close a session after this long unused.")
@click.option("--max_steps", default=200, show_default=True, type=int,
              help="The per-episode step budget; 200 is what run_benchmark.py scores a task against.")
@click.option("--record_dir", default=None, help="Record every session's full state here. Off by default.")
def serve(kind, host, port, unix_socket, token_file, max_sessions, idle_seconds, max_steps, record_dir):
    """Serve one policy on host:port or on a Unix socket."""
    from env_server.policy import Policy
    from env_server.server import serve as serve_forever
    from env_server.sessions import Sessions
    if (port is None) == (unix_socket is None):
        raise click.ClickException("give exactly one of --port or --unix_socket")
    parameters = load_parameters()
    sessions = Sessions(policy=Policy.of(kind=kind), max_sessions=max_sessions, idle_seconds=idle_seconds,
                        record_dir=record_dir, max_steps=max_steps, parameters=parameters)
    serve_forever(sessions=sessions, host=host, port=port, token=_read_token(token_file),
                  parameters=parameters, unix_socket=unix_socket)


def _read_token(token_file: str) -> str:
    with open(token_file) as f:
        token = f.read().strip()
    if not token:
        raise click.ClickException(f"{token_file} is empty")
    return token


def _request(*, unix_socket: str, token_file: str, method: str, path: str, body=None) -> dict:
    from env_server.unix_http import unix_request
    status, reply = unix_request(socket_path=unix_socket, method=method, path=path,
                                 token=_read_token(token_file), body=body)
    if status != 200:
        raise click.ClickException(f"{method} {path} returned {status}: {reply.get('error', reply)}")
    return reply


@main.command("open_session")
@click.option("--unix_socket", required=True)
@click.option("--token_file", required=True)
@click.option("--task_id", required=True)
def open_session(unix_socket, token_file, task_id):
    """Open a task session on a running server and print it, with its session token, as JSON."""
    click.echo(json.dumps(_request(unix_socket=unix_socket, token_file=token_file, method="POST",
                                   path="/sessions", body={"task_id": task_id})))


@main.command("session_info")
@click.option("--unix_socket", required=True)
@click.option("--token_file", required=True)
@click.option("--session", required=True)
def session_info(unix_socket, token_file, session):
    """Print a session's status (steps, episode_over, success) as JSON."""
    click.echo(json.dumps(_request(unix_socket=unix_socket, token_file=token_file, method="GET",
                                   path=f"/sessions/{session}")))


if __name__ == "__main__":
    main()
