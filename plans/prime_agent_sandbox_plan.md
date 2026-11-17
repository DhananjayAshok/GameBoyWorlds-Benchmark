# prime-agent sandbox: env server, client CLI, solution repo

> **Partly superseded — see `plans/prime_agent_port_end_state.md`.** Out of date here: "The firewall" (the
> allowlist was replaced by dev = full state, test = success + subgoals, plus `obs.signals`), "CARC specifics" (no
> `debug` partition on this cluster; build in an allocation you hold), and "Status" (prime-agent is installed and
> runs in the sandbox). The rest still holds.

Ported from CUSI (`../CUSI/plans/user_sandbox_overview.md`, `user_sandbox_plan.md`), GameBoy only, as a
subpackage of GameBoyRL.

Goal: prime-agent develops a repo-as-policy on train and val tasks, through the environments only, without ever
seeing test tasks, held-out games, GameBoyRL, or our results. At test time a harness on our side runs that repo —
or prime-agent itself — on the test tasks.

## Why a server and a sandbox, not a library

The agent must be unable to reach what it must not see, rather than merely instructed not to look. Two boundaries:

- **Process boundary (the server).** The agent speaks HTTP to a server that owns the split and the emulators. What
  it may see is a launch flag of that server, never a request field. A `dev` server cannot be talked into showing a
  test task.
- **Filesystem boundary (the sandbox).** Apptainer `--containall`: only the agent's workspace (`/work`) and its own
  home are mounted. GameBoyRL, `storage_dir`, `$HOME` and the HF cache are not. Outbound network stays open, so it
  can call API models and reach the server on 127.0.0.1.

prime-agent's own README is explicit that it is not a sandbox and that untrusted code belongs in an external one, so
this is a requirement of the agent, not a precaution of ours.

## The firewall: obs vs state

`gameboy_worlds.Environment.get_info()` returns `self._emulator.state_tracker.report()` — the full RAM-derived state
tracker report. That is the single most dangerous object in this design and it must never cross to a client.

The server therefore projects it. The client receives only an allowlist:

| Served to the client | Never served |
|---|---|
| frame (PNG) | the state tracker report |
| `texts` (screen OCR regions) | task/subgoal progress trackers, success probes |
| `actions` (from `get_action_strings`) | `previous_action_details` transition states, passed frames |
| `goal` | emulator save-state blobs (dev: only as opaque handles) |
| `action_success`, `changed`, `steps`, `max_steps` | anything RAM-derived, on dev or test |
| `success`, at episode end only | |
| `state`, **dev only and only if the session asked** | |

This is the existing standing rule of the project — the agent is vision-only; RAM parsers exist for our analysis,
never as agent input — made structural rather than conventional. Enforced two ways: the codec allowlists keys
rather than denylisting them, and a live check asserts that no `test`-server response carries a state key.

**This is a deliberate deviation from CUSI, not a port of it.** CUSI denylists (`HIDDEN_INFO`, `DROPPED_INFO` in
`sessions.py`), which is safe there because its `TextActionEnv` had already curated `info`. Here the raw object *is*
the tracker report, so a denylist would leak every tracker key added in future. Allowlist or nothing.

`ocr_regions` was checked and is **not** a leak: `TrackerMixin.step` builds it with
`state_parser.capture_named_region(current_frame=...)`, and every `can_read_kind` implementation takes
`current_screen=current_frame`. They are crops of the frame, not memory reads, and not decoded text either — the
"OCR" here is image regions, where CUSI's `obs["texts"]` was strings.

They are therefore legal but redundant: the agent already has the full frame. Default is not to serve them, behind a
server flag, so the served surface stays as small as it can be.

Two further facts that shape the codec:

- `get_observation()` returns the screen and nothing else, so the obs side needs no filtering at all, only PNG
  encoding. The whole risk is `info`.
- `TestEnvironmentMixin` puts success in `terminated` (the task's own check) and the budget in `truncated`, so
  `success = terminated` at episode end and a truncated episode scores 0 — the same rule CUSI used for GameBoy.

## Splits — "customizable train and test sets"

One flat file, `configs/splits/benchmark_splits.json`: `{task_id: {"split", "text"}}`, with
`task_id = gb.<game>.<index>` over `get_benchmark_tasks(game)` rows.

Splits: `train`, `val` (both listed on dev), `test` (held-out tasks in training games), `test_env` (every task of a
held-out game). A generator builds the file from a small editable config naming which games are training games and
how their tasks divide, so changing the train/test sets is editing that config and regenerating — not touching code.

A start state belongs to exactly one split: free play must not start the agent where a test task starts.

Resolver: a task id becomes a row, and a row becomes an env via `get_test_environment(row)` — `game`, `init_state`
and `state_tracker_class` are already on the row. Free play uses `get_environment(game, variant, init_state)` over
`get_available_init_states`, restricted to train/val start states.

GameBoy needs no in-episode containment: games are separate ROMs, so a held-out game is unreachable by construction.
This is the one place the port gets simpler than CUSI, which had to undo steps into held-out apps and websites.

## Layout

```
env_server/            ours, never mounted in the sandbox
  policy.py            dev | test, fixed at launch
  splits.py            the split file, loader, generator
  sessions.py          one live emulator per session; step count, episode end, success
  codec.py             the allowlist projection of obs/info
  server.py            stdlib ThreadingHTTPServer, bearer tokens
  recorder.py          server-side full-state recording (diagnostics, off by default)
  sandbox.py           apptainer exec --containall argv
  harness.py           the test-time schedule
  check.py             live assertion that test responses carry no state
run_env_server.py      python run_env_server.py --kind dev --port 8710 --token_file F
run_user_eval.py       the test harness
user_template/         the seed workspace: copied out, becomes the solution repo
  envclient.py         stdlib only, the only code of ours the agent gets
  envctl               the CLI
  skills/envctl.md     a prime-agent repo skill teaching the CLI
  guides/ tools/ prompts/
```

Living inside GameBoyRL is safe only because the sandbox mounts neither the project root nor `storage_dir`. The
residual risk is config: a `private_vars.yaml` path or a `project_root` reference reaching into the workspace. The
copy-out list is explicit (CUSI's `SKIPPED` pattern: `.envctl`, `obs`, `envs.json`, `__pycache__`).

## Tokens

The server token lists tasks and opens sessions. Each session also gets its own token that reaches only that
session. At test time the harness hands the agent only the session token, so it cannot list tasks, open another test
task, or read another goal.

## prime-agent integration

Headless, per `crates/pa-cli/src/args.rs`:

```
prime-agent -p "<prompt>" --cwd /work --autonomous \
  --autonomous-max-turns N --autonomous-max-tokens N \
  --autonomous-gate "<cmd>" --append-system-prompt "<env contract>" \
  --session-dir <agent home>/sessions
```

- Durable state is `~/.prime` (and `~/.config`), written by `/refine`: supplemental prompts, memories, skill and
  subagent specs, with the base system prompt immutable and snapshots for rollback. Carrying that directory across
  tasks is the Continual Harness actually improving; discarding it is the control.
- Repo skills are Markdown instructions with no auto-execution, so `user_template/skills/envctl.md` is the right
  place to teach the environment contract. Python user skills are imported into the kernel — we do not ship any, so
  nothing of ours executes inside the agent's kernel.
- It is daemon-backed and needs a writable `$HOME`, which is why the sandbox mounts a separate agent home instead of
  letting it near the real one.

## Two evaluation modes

- `fixed_repo`: runs the entrypoint named by the workspace's `policy.json`. The agent is gone; only the code it
  wrote is judged. Reproducible, no API spend at test time.
- `live_agent`: prime-agent runs per test task with budgets and its carried `~/.prime`.

`--memory carry` keeps one workspace (and one agent home) through the schedule, so what task k wrote is present at
task k+1; `isolated` starts every task from the same snapshot. Nothing is discarded either way: every task's
workspace is committed, so the git history *is* the solution repo and its development record.

Budgets in both modes: the task's step budget (enforced by the server) and a wall-clock limit per task (enforced by
the harness). No model is provided — the agent brings its own.

## Step budget

`run_benchmark.py --max_steps` defaults to **200**, and that is the budget a benchmark task is scored against.
`gameboy_worlds`' own default is 10000, which is not a benchmark budget — a server left on it gives the agent 50x
the real allowance. The server therefore defaults to 200 and says so on the flag.

`check_if_done` truncates at `step_count >= max_steps - 1`, so a budget of N ends on step N-1. That is the
benchmark's own arithmetic and is deliberately left alone: changing it would make our numbers incomparable with
every run already recorded.

Verified: a 5-step budget ends the episode at step 4 with `success: 0.0`, no `state` key, and a further step is
refused with "the episode is over; reset first".

## save/load resets the step counter

Measured, over HTTP: `load` returns with `steps` back to 0. So save/load is not budget-neutral — an episode can be
rewound to spend the budget again. It is dev-only, where the budget is not the binding constraint, and the test
policy refuses save/load outright, so this does not affect scoring. It does mean dev step counts are not a
measure of effort.

## CARC specifics

- `module load apptainer/1.5.5`. Set `APPTAINER_CACHEDIR` and `TMPDIR` under `/scratch1/$USER` — the default cache
  is `~/.apptainer`, which eats the home inode quota.
- **The official prebuilt binary cannot run on the host.** `make glibc-gate` holds releases to a GLIBC_2.35
  (Ubuntu 22.04) baseline; the login nodes are Rocky 8.10 with glibc 2.28. It runs fine *inside* the sandbox —
  the built image reports glibc 2.41 (Debian 13) — which is where we want it anyway. Host-side use would mean
  building from source, and the `rust` modules here are 1.41/1.60, so that means rustup via the vendored
  `install-rust.sh`.
- Building the image on a login node took ~9 minutes, nearly all of it `mksquashfs` on one core. Build it in an
  allocation next time.
- Emulators and the agent belong in a SLURM allocation, `--partition=debug` for smoke tests.

## Steps

1. Split generator + loader + the config that names train/test games and task division. Check: loader round-trips
   every task; a reworded task fails loudly.
2. `env_server/` over `gameboy_worlds` (sessions, policy, codec, recorder) + `run_env_server.py`. Check: live dev
   session on a train task — reset, 20 steps, save/load, close.
3. Client + `envctl`. Check: drive a train task by hand from the CLI.
4. Sandbox image + launcher. Check from inside: `ls /`, `$HOME`, the GameBoyRL path show nothing of ours;
   `envctl tasks` lists train+val only; a test id is refused.
5. Harness, both modes, `--state never` on test. Check: a random-action stand-in on 3 test tasks; no response
   carries state; the recorder has it all.
6. A real run: prime-agent on train+val for a fixed budget, then the harness on test.

## Status (2026-11-13)

Built and verified against live emulators:

- **Step 1, splits.** `configs/splits/split_config.yaml` + `env_server/splits.py`. 1000 tasks over 20 game
  variants: 9 training games at ~25 train / 10 val / 15 test each, 11 held-out games wholly `test_env`
  (217 / 98 / 135 / 550). Start states kept whole; `verify` round-trips every task against the live benchmark and
  fails loudly on a reworded task. `python run_env_server.py generate_splits --show`.
- **Step 2, env server.** `env_server/{policy,sessions,codec,recorder,server,check}.py` + `run_env_server.py`.
  Live dev session on a train task: reset, valid steps, invalid step (step count unchanged, `valid: false`),
  save/load, close.
- **The firewall, asserted.** `python run_env_server.py check` passes: a test server lists nothing and offers no
  free play; a test task is refused on dev and a train task on test; `state` is refused on test; no test response
  carries a `state` key, a hidden name, or another task's goal; a dev session that asks for state gets it, without
  the grader's groups.
- **Step 3, client.** `user_template/{envclient.py,envctl}`, stdlib only. Driven end-to-end over HTTP from a clean
  workspace: `tasks` (315 listed), `scenes`, `open --state`, `actions`, steps, `save`/`load`, `info`, `close`, with
  `obs/<session>/<n>.png` and `.json` written per observation.
- **Sandbox image.** `storage_dir/containers/sandbox.sif`, 368M, from `docker://python:3.12`. Isolation confirmed
  from inside: `/work` holds only the workspace, `$HOME` is the separate agent home, and `/home1/ashokd`,
  `/home1/ashokd/projects/GameBoyRL`, `/project2/jessetho_1732` and `/scratch1/ashokd` are all absent. Outbound
  network works (HTTP 200 from pypi.org).

Two bugs found and fixed while testing, both worth remembering:

- Passing GameBoyRL's `parameters` into `gameboy_worlds` pointed `project_root` at the wrong repo and no ROM
  resolved. `gameboy_worlds` must load its own config — the rest of the repo calls it that way too.
- The grader's metric group is named per game (`pokemon_test`, `dejavu_test`, `harvest_moon_test`), so a
  hardcoded hidden-name list let `pokemon_test` through into dev state. The hidden set is now read off the live
  tracker (`TERMINATION_TRUNCATION_METRIC`, `SUBGOAL_METRIC`) plus a `*_test` suffix rule.

Not built yet: `env_server/sandbox.py`, `env_server/harness.py`, `run_user_eval.py`, and
`user_template/skills/envctl.md`.

**Blocked:** installing prime-agent into the sandbox. The installer was downloaded and reviewed (it verifies
sha256 against the channel manifest and writes only to `$PREFIX`, default `~/.local`; it touches no shell rc file),
but running it was refused by the harness's permission classifier. It needs to run once, inside the sandbox, where
`$HOME` is the agent home and the project is not mounted:

    apptainer exec --containall --cleanenv --bind <workspace>:/work \
      --home <storage_dir>/env_server/agent_home:/home/agent --pwd /work \
      <storage_dir>/containers/sandbox.sif sh /home/agent/install.sh

Until that runs, `live_agent` mode cannot be smoke-tested and the install question (prebuilt vs. built from
source) stays open — though the glibc finding above already argues for the prebuilt binary inside the container.
