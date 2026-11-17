# The CUSI port: end state (2026-11-14, network isolation added 2026-11-16)

Supersedes the Status and firewall sections of `plans/prime_agent_sandbox_plan.md`; the rest of that document
(why a server and a sandbox, tokens, the two evaluation modes, step budget) still holds.

## In one paragraph

The two boundaries CUSI relies on now exist for GameBoyRL and have been exercised together. An **env server** on
the host owns the splits and the emulators and decides, by a launch flag, what a client may see. A **sandbox image**
holds a pinned prime-agent 0.10.3 and nothing of this project, and runs with **no network**: its only ways out are
two Unix sockets, to the env server and to a model gateway that holds the API key. In the end-to-end test,
prime-agent ran headless inside that sandbox, drove a real Pokémon Red task with `envctl`, looked at the PNG frame,
read the screen signals, and closed the session, without installing anything at runtime and without the API key
ever being inside. What is **not** built is the
harness that runs this at scale (`sandbox.py`, `harness.py`, `run_user_eval.py`) and the repo skill that teaches the
agent `envctl`. Nothing of the port is committed yet.

## What was built

```
host (compute node)                                  sandbox: apptainer --containall --net --network none
-------------------                                  ----------------------------------------------------
run_env_server.py serve --unix_socket ...            no network: its own loopback only, no DNS
  policy / splits / sessions / codec / check         /work        the workspace, bind-mounted, writable
  └── envserver.sock ──────── bind-mounted ────────> sandbox-run → netbridge 127.0.0.1:8710 ← envctl
env_server/model_gateway.py  (holds the API key)     /home/agent  the agent home (node-local!)
  allowlisted models; strips web search             /opt/pa      prime-agent 0.10.3 (read-only)
  └── model.sock ─────────── bind-mounted ────────> netbridge 127.0.0.1:8720 ← prime-agent (models.json)
                                                     /opt/pa-kernel  pinned Python kernel, seeded on first use
```

| Piece | Where | State |
|---|---|---|
| Splits: 1000 tasks, 20 game variants, 217 / 98 / 135 / 550 train / val / test / test_env | `configs/splits/`, `env_server/splits.py` | built, verified |
| Env server and its policies | `env_server/`, `run_env_server.py` | built, verified (`run_env_server.py check` passes) |
| Client and CLI | `user_template/envclient.py`, `user_template/envctl` | built, verified over HTTP |
| Sandbox image | `containers/sandbox.def` → `storage_dir/containers/sandbox-pa.sif` (551 MB) | built, verified (31/31) |
| Network isolation: Unix-socket channels, in-sandbox bridge | `env_server/unix_http.py`, `containers/netbridge.py`, `containers/sandbox_run.sh` | built, verified |
| Model gateway (key on the host, model allowlist, web search stripped) | `env_server/model_gateway.py` | built, verified |
| Image checks | `containers/verify_sandbox.sh <sif>` | written, passing (35/35) |
| End-to-end test, isolated | `containers/smoke_e2e.sh` | written, passing |
| Sandbox launcher, harness, eval entry point | `env_server/sandbox.py`, `harness.py`, `run_user_eval.py` | **not built** |
| Repo skill teaching `envctl` | `user_template/skills/envctl.md` | **not built** |

## What a client sees, as it now stands

The allowlist in the original plan was replaced at your direction by two modes.

| | `dev` (train + val) | `test` (test + test_env) |
|---|---|---|
| Tasks listed | train + val | none; ids come from our harness |
| Free play, save/load | yes | no |
| `state` | the **whole** tracker report, every group, on request | **only** `termination_truncation` + `subgoals`, always |
| `obs` | frame (PNG), goal, actions, steps/max_steps, and `signals` | same |
| `obs.signals` | `agent_state`, `n_battles_lost` | same |

- "Whole" is literal. `codec.jsonable` is a total encoder: frames become lossless PNG, enums their name, classes
  their name, and it is depth-capped so the action class sitting in `previous_action_details` cannot send it
  walking the module graph (it did, once). Measured: 8 groups, 50.7 KB per step on dev; 1.9 KB on test.
- `signals` are the two RAM fields the playthrough leg reads. Both are on screen anyway (what kind of screen you
  are on, and the blackout animation), which is why you allowed them in obs. They are looked up by name across
  groups, because the group is named per game: Bomberman's tracker has neither, and its `signals` is `{}`.
- `check.py` asserts the test side as an allowlist on the way out: any group beyond the scored two fails the check.

## The sandbox image

Built by `apptainer build --fakeroot` from `containers/sandbox.def`. Everything that moves is pinned, and the pins
are labels on the image, so `apptainer inspect` answers "what exactly ran?" without the def:

| Pinned | How |
|---|---|
| Base: `python@sha256:cad37dd2…` | by digest (Apptainer rejects tag+digest, so `python:3.12` is only a label) |
| prime-agent 0.10.3, linux-x64, `sha256 811d9f0e…` | `PRIME_AGENT_VERSION` (skips the `stable` pointer) |
| The installer, `sha256 75522795…` | vendored, `containers/prime_agent_install.sh`: the build runs the bytes that were reviewed |
| uv 0.13.8 | pip, system-wide, so the installer does not `curl \| sh` an unpinned one |
| Kernel Python: CPython 3.11.18 | pre-installed where prime-agent's own `uv python install 3.11` then finds it |
| 54 kernel packages | `containers/kernel_constraints.txt` as `UV_CONSTRAINT`; the check confirms every pin is installed |
| 6 bundled skills | `containers/seed_kernel_skills.py`, recorded the way prime-agent records them |
| Telemetry | off: `PRIME_AGENT_TELEMETRY=0`, `DO_NOT_TRACK=1` |

The image needs the network to *build* (the R2 release host, PyPI, uv's Python downloads), and not to run, except
for the model API.

**The binary/state split.** The image holds runtimes; the agent home holds memory. `~/.prime` and `~/.config`,
including everything `/refine` writes, live only in the agent home the harness mounts, which is what makes
`isolated` and `carry` different. Checked: no `~/.prime`, no build HOME, no installer, no caches in the image. The
API key is not in the sandbox at all: the model gateway on the host adds it, and the sandbox's `models.json` says
`"apiKey": "none"`. (Before isolation it was passed in through the environment, and `auth.json` stayed an empty
`{}`.)

**The kernel had to be built three times; the reasons are the design.**

1. *No uv in the image → no Python tool at all.* The first image ran, and the agent reported `DONE` on a file it
   never wrote: `python_repl` failed with "uv is required", and the model ignored the error.
2. *The installer does pre-warm a kernel, but into `$HOME`* — the throwaway build HOME I then deleted.
3. *prime-agent strips `UV_PYTHON_INSTALL_DIR`* (and `UV_PYTHON`, …) before running uv, deliberately, so a repo
   cannot redirect the kernel's interpreter. The venv's interpreter therefore lands at uv's default, under
   `XDG_DATA_HOME`, which is not stripped — so that is set at build time only.

And it cannot simply be a read-only kernel in the image: with `PRIME_AGENT_KERNEL_PYTHON`, prime-agent installs
nothing and *disables* any Python skill missing from that interpreter, which is every skill `/refine` creates in
`carry` mode. So the kernel is built at `/home/agent/...` — the runtime path — stored in the image as a seed, and
`containers/prime_agent_wrapper.sh` copies it into each fresh agent home on first use (to a temp sibling, then an
atomic rename, so a session and its subagent cannot race). There it is writable.

## Verified

- `bash containers/verify_sandbox.sh …/sandbox-pa.sif` → **35 passed, 0 failed**: pins and labels; telemetry off;
  kernel interpreter in the image, 3.11.18, all pins installed, six skills recorded; fresh home seeded and
  writable; nothing of an agent home in the image; your home, GameBoyRL, project storage, ROM data and scratch all
  unreachable; `HOME=/home/agent`; a host-loopback service, the internet and DNS unreachable, no node sockets
  visible, `sandbox-run` bridging exactly the channel it is given; `/work` writable and holding only the workspace.
  Every check runs with the harness's real flags (`--containall --cleanenv --net --network none --home <agent
  home>:/home/agent`), and the network checks were confirmed to fail without `--network none`.
- File-write test: `hello.txt` written with the right contents, zero runtime installs.
- `bash containers/smoke_e2e.sh`, fully isolated → escape and gateway probes as in finding 3; the agent opened
  `gb.pokemon_red.003` through the env-server socket, stepped, attached `obs/<session>/0001.png`, and reported
  `GOAL: Pick up the item`, `AGENT_STATE: FREE_ROAM`, and a screen description I checked against the PNG (player in
  tall grass, item ball at the left edge: correct). The server logged the open and the close; the gateway
  forwarded 13 model calls; zero runtime installs; the key in neither the sandbox's environment nor its files.
- `python run_env_server.py check` passes after the codec change.

## Findings you need before running this for real

1. **Score from the server, never from the agent.** gemini-3.6-flash reported `DONE` on a task whose only tool call
   had failed. `success` comes from `TestEnvironmentMixin` via the env server, which is how it is designed;
   anything that reads the agent's final message as the outcome would be wrong.
2. **Agent homes must be node-local.** Seeding the kernel is 8,847 files: 41 s onto `/scratch1` (BeeGFS, metadata-
   bound), 3 s into `/dev/shm`, 0.4 s once seeded. `isolated` mode would pay that per task, and it is the small-file
   churn the cluster rules ask us not to put on shared filesystems. The harness should create agent homes in
   node-local memory and persist only what is worth keeping (sessions, `/refine` state — not `kernel-venv`).
3. **Network isolation — done (2026-11-16).** `--containall` alone gives no network namespace: the sandbox could
   connect to every service on the node, and did — measured, it reached the VS Code server and Claude Code's local
   ports. It now runs with `--net --network none` (works unprivileged here): no interface but its own loopback, no
   DNS. Its only two channels are Unix sockets bind-mounted in — the env server (`--unix_socket`) and the model
   gateway — exposed on its private loopback by `sandbox-run`, which starts `netbridge.py` for the channels named
   in `SANDBOX_BRIDGE`. prime-agent reaches the gateway through a custom provider in `~/.prime/agent/models.json`.
   The **gateway holds the API key**, so the key is never in the sandbox; it forwards only allowlisted model ids
   (no `:online`), strips `plugins` / `web_search_options` (OpenRouter's web search) and `models` / `route`
   (fallback routing), and logs every request without its content. Verified from inside: a host-loopback service,
   the internet and DNS unreachable; a disallowed model and an `:online` variant refused; the web fields stripped;
   the key in neither the environment nor any file. The network checks were confirmed to fail when `--network
   none` is dropped, so they are not vacuous. `env_server/egress_proxy.py` (an allowlisting CONNECT proxy) is kept
   but unused: it is how to grant a host such as PyPI deliberately, if that is ever wanted (see Not done).
4. **Models.** Your OpenRouter workspace guardrail blocks OpenAI models, and `OPENAI_API_KEY` in this shell is
   rejected (401). The tests use `google/gemini-3.6-flash` through OpenRouter, as the live runs do.
5. **Where to build.** There is no `debug` partition on this cluster, and this shell runs inside your 1-CPU
   `vscode-dev` allocation, so an `srun` asking for more CPUs queues forever. The image was built in place on one
   core (~10 min). The def's header now says so.

## Also done in this session

Committed (`f563f11`): `run_strategist.py` → `run_playthrough.py` and the other top-level renames (on-disk names
deliberately kept; `paths.PLAYTHROUGH_SEGMENT = "strategist"`), and all 17 slurm scripts routed through
`runs/playthrough/playthrough.sh`.

Uncommitted, beyond the port itself:

- `global_map.py`: a failure in the map now ends the run with a reproducible message, instead of being logged and
  swallowed; and `info["core"]` is read defensively like its neighbours.
- `guide.py`, `prompts.py`: a defeat with no recorded battle frames no longer kills the run
  (`images cannot be empty`, which ended `red_argon_resume` at episode 49 of 471); the prompt now says when there
  are no screenshots, and the call leaves `images` out.
- `plans/playthrough_splits_plan.md`: the plan for game-level splits on the playthrough leg. Its "vision senses"
  part is partly overtaken: the two RAM fields are now `obs.signals` on the env server — but the strategist does
  not go through the env server, so its consumers still read `get_info()` directly.

## Not done

- **The harness**: `env_server/sandbox.py` (the `apptainer exec` argv above, with node-local agent homes),
  `env_server/harness.py` (both modes, both memory policies, a commit per task), `run_user_eval.py`.
- **`user_template/skills/envctl.md`**, the repo skill that teaches the agent the env contract. The end-to-end test
  spelled the commands out in the prompt instead.
- **Untested:** that a skill `/refine` creates actually installs into the seeded kernel in `carry` mode (the design
  is built for it; no run has exercised it), and concurrent seeding by two processes. Under network isolation it
  will need a decision: installing a new skill needs PyPI, at least for its build backend, and the sandbox now has
  none. Either grant `pypi.org` + `files.pythonhosted.org` through `egress_proxy.py` (reopens package downloads)
  or bake the build backends in and install offline.
- **Still open from the threat review:** whether the `GameBoyWorlds` / `GameBoyRL` repos are public. If so, the task
  definitions and the success-check code are online — now reachable only through the model's own knowledge, since
  the sandbox has no web. Fixed on 2026-11-16, each with a check in `run_env_server.py check` that was confirmed to
  fail when its bug is put back: test allows exactly one reset per session, the one that starts the episode
  (`Policy.reset_again`; a second is a 403, and `describe()` reports `can_reset`); tokens are compared in constant
  time, as bytes, so a non-ASCII bearer is a 401 rather than a crash; and an unanticipated error returns only
  `{"error": "internal error", "incident": <id>}`, with the traceback in the server log under that id.
- **The playthrough leg's splits** (`plans/playthrough_splits_plan.md`), none of it started.
- `storage_dir/containers/sandbox.sif`, the first image, is now superseded by `sandbox-pa.sif` and can go when you
  are satisfied.

## Rebuilding and re-checking

From the repo root, on a compute node:

    module load apptainer
    export APPTAINER_CACHEDIR=/scratch1/$USER/apptainer_cache TMPDIR=/scratch1/$USER/tmp
    apptainer build --fakeroot <storage>/containers/sandbox-pa.sif containers/sandbox.def
    bash containers/verify_sandbox.sh <storage>/containers/sandbox-pa.sif
    bash containers/smoke_e2e.sh

To move to a newer prime-agent: change `PRIME_AGENT_VERSION` and the labels, re-vendor the installer, then
regenerate `kernel_constraints.txt` and `kernel_skills.json` from a runtime bootstrap. `verify_sandbox.sh` and the
end-to-end test's install-line count will tell you if any of those three drifted.
