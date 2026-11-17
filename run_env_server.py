"""
The environment server: one policy (dev or test), on host:port or on a Unix socket.

    python run_env_server.py generate_splits
    python run_env_server.py serve --kind dev --port 8710 --token_file <file>
    python run_env_server.py serve --kind test --unix_socket <dir>/envserver.sock --token_file <file>

dev lists and opens train + val tasks, allows free play, the full tracker state and save/load. test opens test +
test_env only, lists nothing, and returns only task success and subgoal progress. The policy is this flag; no request
can change it. Serve an isolated sandbox over --unix_socket: on TCP, anything on the node can reach the server.
"""
import click

from utils import load_parameters


@click.group()
def main():
    pass


@main.command("generate_splits")
@click.option("--show", is_flag=True, help="Print the per-game counts.")
def generate_splits(show):
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
              help="The per-episode step budget. 200 is run_benchmark.py's budget, which is what a task is scored "
                   "against; gameboy_worlds' own default is 10000, which is not a benchmark budget at all.")
@click.option("--record_dir", default=None, help="Record every session's full state here. Off by default.")
def serve(kind, host, port, unix_socket, token_file, max_sessions, idle_seconds, max_steps, record_dir):
    from env_server.policy import Policy
    from env_server.server import serve as serve_forever
    from env_server.sessions import Sessions
    if (port is None) == (unix_socket is None):
        raise click.ClickException("give exactly one of --port or --unix_socket")
    parameters = load_parameters()
    with open(token_file) as f:
        token = f.read().strip()
    if not token:
        raise click.ClickException(f"{token_file} is empty")
    sessions = Sessions(policy=Policy.of(kind=kind), max_sessions=max_sessions, idle_seconds=idle_seconds,
                        record_dir=record_dir, max_steps=max_steps, parameters=parameters)
    serve_forever(sessions=sessions, host=host, port=port, token=token, parameters=parameters,
                  unix_socket=unix_socket)


if __name__ == "__main__":
    main()
