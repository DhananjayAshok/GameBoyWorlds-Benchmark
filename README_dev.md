# Developer Reference

A guide to the scripts in `scripts/` and what each one does. Run a script with no arguments to see its usage; defaults live in `scripts/core/utils.sh`.

---

## RL Scripts (`scripts/rl/`)

| Script | What it does |
|--------|-------------|
| `create_traj.sh` | End-to-end trajectory creation for a single init_state: runs iterative RL to fill a replay buffer, then clusters observations into groups via `group_trajectories.sh`. Output feeds the VLM task-discovery pipeline. |
| `create_all_traj.sh` | Batch wrapper of `create_traj.sh` over every registered init_state for a game. Regenerates the state dictionary from GameBoyWorlds before iterating. |
| `iterative_training.sh` | Multi-round RL where each agent's curiosity buffer seeds the next round. Optionally retrains the world model between rounds. Called by `create_traj.sh`; can also be run directly for training-only workflows. |
| `sweep.sh` | Hyperparameter sweep over seeds, gammas, and algorithms. After the first pass it prunes to the best-k models by test reward, then re-runs only the winners to collect replay buffers. Called by `iterative_training.sh` when `--sweep true`. |
| `group_trajectories.sh` | Clusters a collected replay buffer by observation similarity (z-score gating) into grouped trajectory files used for downstream VLM task inference. Can be run standalone to re-cluster without retraining. |
| `collect_wm_buffers.sh` | Collects random-action and curiosity replay buffers for every train game/init_state in `--game`'s series, for the world-model baseline. No grouping. |

## Core RL Scripts (`scripts/core_rl/`)

| Script | What it does |
|--------|-------------|
| `train.sh` | Single RL training job with a given curiosity module (default: combinationbuffer PPO). Skips if the checkpoint exists unless `--overwrite_model true`. |
| `enjoy.sh` | Evaluates a trained model on a (possibly different) test game/init_state. |
| `default_rl.sh` | `train.sh` then `enjoy.sh`; supports `--train_only` / `--eval_only`. |
| `train_world_model.sh` | Trains a world model on a replay buffer (used by `curiosity_module=world_model` and the `world_model` executor). |
| `train_observation_embedder.sh` | Trains an observation encoder on replay buffer data for similarity-based curiosity. |

---

## VLM Scripts (`scripts/vlm/`)

| Script | What it does |
|--------|-------------|
| `propose_zeroshot.sh` | Calls the VLM to propose candidate tasks for the given init_states with no prior trajectory data. |
| `propose_all_zeroshot.sh` | Batch version of `propose_zeroshot.sh` across every registered init_state for a game. |
| `infer_tasks.sh` | Infers task labels from grouped trajectories produced by `group_trajectories.sh`. |
| `attempt_tasks.sh` | Runs the VLM agent on tasks from a task file under the two-stage judge. Writes `success_trajectories.{json,pkl}`. |
| `build_info.sh` | Distils (task, trajectory) pairs from a trajectory stem into `info_docs/` beside it. `--stage all` (default) is required before benchmarking. |
| `create_info_doc.sh` | Builds the info document for one `--source` (`curiosity` or `zeroshot`) of a game. |
| `create_all_info_docs.sh` | Builds info documents for every train game of `--game` and every source selected by `--mode`. |
| `infer_guidance.sh` | Annotates a trajectory with step-by-step guidance for practice. |
| `practice_tasks.sh` | Runs guided practice sessions from a guidance file, with a binary judge. |
| `clean_practice.sh` | Paraphrases tasks and VLM-filters every practice call in a practice dir (resumable). |
| `create_dataset.sh` | Builds `train_dataset.csv` / `validation_dataset.csv` from a cleaned practice dir. |
| `merge_practices.sh` | Merges the datasets of several practice dirs into one pair, tagged with a `source` column. |
| `train_vlm.sh` | LoRA fine-tunes a VLM via `llm-utils` (`train.py`); optionally pushes to the HuggingFace Hub. |

---

## Pipeline Scripts (`scripts/pipeline/`)

| Script | What it does |
|--------|-------------|
| `curiosity_all_tasks.sh` | `create_traj.sh` → `infer_tasks.sh` over all registered init_states. |
| `propose_and_attempt_all.sh` | Zero-shot proposal across all init_states, then a single combined attempt run. |
| `curiosity_and_zeroshot_all.sh` | Runs both data-collection verticals for a game. |
| `guidance_and_practice.sh` | `infer_guidance.sh` → `practice_tasks.sh` (→ `clean_practice.sh`) for one trajectory. |
| `collect_and_info_all.sh` | `curiosity_and_zeroshot_all.sh` for every train game of `--game`, then `create_all_info_docs.sh`. |

---

## Benchmark Scripts (`scripts/benchmark/`)

| Script | What it does |
|--------|-------------|
| `run_benchmark.sh` | Runs `run_benchmark.py` on a game with a given executor, supervisor and VLM. Records video and logs results to `results/benchmark/<game>/`. |
| `run_benchmark_info_retrieval.sh` | The `info_subgoal_retrieval` arm of `run_benchmark.sh`, with `--info_docs` resolved from disk via `--docs_*` flags. |

---

## Debug Scripts (`scripts/debug/`)

Read-only diagnostics over saved artifacts via `debug.py`; reports land under `<results_dir>/debug/<game>/`. `debug_all.sh` runs every stage in pipeline order (curiosity → infer → zeroshot → attempt → benchmark) and stops at the first failure.

---

## Core (`scripts/core/`)

| Script | What it does |
|--------|-------------|
| `utils.sh` | Loads config and venv; defines every script's default args and the arg-parsing helpers. |
| `all_train_states.sh` | Maps games to their train init_states. Regenerate with `python python_funcs.py task_dictionary`. |
| `serve_vllm.sh` / `stop_vllm.sh` | Start (blocking until healthy) / stop a vLLM server. |
| `llm-utils.sh` | Runs a command inside the `llm-utils` venv. |
| `clean_cache.sh` | Clears CleanRL artifacts and temporary emulator sessions. |
| `unsafe_clean_all.sh` | Wipes all storage; guarded by an exit you must remove first. |

---

## Playthrough (`run_playthrough.py`, `execution/strategist/`)

| File | What it does |
|------|-------------|
| `run_playthrough.py` | Builds one emulator session and runs `PokemonStrategist` on it. |
| `execution/strategist/pokemon/strategist.py` | The episode loop: perceive, stuck check, plan, dispatch, run a supervisor, review. |
| `execution/strategist/pokemon/goals.py` | The goal tree, subgoal planning and the completion check. |
| `execution/strategist/pokemon/knowledge.py` | The knowledge tree; `remember` files a fact, `recall` looks one up. |
| `execution/strategist/pokemon/guide.py` | Event-triggered rules and battle tips; defeat reflection. |
| `execution/strategist/pokemon/notepad.py` | Free-text working memory. |
| `execution/strategist/pokemon/npcs.py`, `npc_model.py`, `npc_map.py`, `npc_names.py` | The people registry: recognises characters on screen from their sprites, tracks where they stand and what they said, and adds them to the map text. Listens to every step through `MappingEnvironment`. |
| `execution/perception/pokemon/sprites.py` | Sprite matching used by the people registry. |
| `execution/strategist/pokemon/global_map/` | The map built from frames. `registration` aligns consecutive frames, `model` stores cells and areas and merges them, `env` wraps the environment so every step updates the map, `routes` plans paths, `text` and `render` draw it for prompts and reports, `judges` holds its model calls, `exploration` decides when to explore. |
| `execution/strategist/pokemon/asking.py` | One model question with retries that tell the model what was wrong with its reply. |
| `execution/strategist/pokemon/prompts.py` | Every strategist prompt. |
| `execution/strategist/report.py` | The per-episode archive (`report.pkl.gz`). |
| `execution/artifact.py` | Base class for the saved artifacts: one pickle per episode in which it changed. |
| `runs/playthrough/playthrough.sh` | One playthrough, then `debug.py playthrough view` and `diff`. |
| `runs/playthrough/all.sh` | Every Pokémon game in `runs/games.sh`, one after another, on one model. |

A run writes to three places:

1. **State**: `<storage_dir>/playthrough_artifacts/<game>/strategist/<name>/`, holding `episode_<n>/*.pkl`, `episode_<n>/report.pkl.gz` and `provenance_<stamp>.json`.
2. **Video**: under GameBoyWorlds' storage. The provenance file's `session_dir` is relative to `paths.playthrough_sessions_root(game=...)`; `paths.playthrough_video(session_dir, game=...)` resolves it.
3. **Emulator save states**: under GameBoyWorlds' `rom_data/.../states/`, one per episode, named `custom_strategist<name><n>.state`.

`python sync_data.py pull --set playthrough` (and `playthrough_videos`, `playthrough_states`) fetches published runs.

---

## Environment server and sandbox (`env_server/`, `containers/`, `agent_workspace/`)

| File | What it does |
|------|-------------|
| `run_env_server.py` | CLI: `serve`, `check`, `generate_splits`, `open_session`, `session_info`. |
| `env_server/server.py` | The HTTP handler and bearer-token checks. |
| `env_server/sessions.py` | One emulator per session; applies the policy to reset, step, save and load. Checkpoints and restores a playthrough session (`emulator.state` and `session.json` in `--checkpoint_dir`), keeps a train server's named saves in `--saves_dir`, and writes every session's video under `--obs_dir`. |
| `env_server/progress.py` | With `--progress_dir`, each session's progress read from game memory: badges, maps reached (carried across named saves), story events and engine flags raised, the party. One `<session>.jsonl` line per reset, save, load, close and 500 steps; a session that clears its game also goes in `clears.jsonl`. A session opened with `"run"` carries that name in its records. |
| `env_server/policy.py` | The `dev`, `test`, `playthrough` and `train` policies, including which GameBoyWorlds controller each serves (`low_level` buttons, or `state_wise` for playthrough and train). `train` opens only the `train_sandbox_games` of `split_config.yaml`, one session at a time (opening one closes the other), with full state on request, resets, and named saves that outlive the session. |
| `env_server/splits.py` | Generates, loads and verifies `configs/splits/benchmark_splits.json`. |
| `env_server/codec.py` | What a response may contain: the frame as PNG, and full or scored-only state. |
| `env_server/recorder.py` | Host-side record of every step when `--record_dir` is set. |
| `env_server/model_gateway.py` | The sandbox's model endpoint; holds the API key on the host. With `--max_cost` it adds up each call's cost in a ledger file and refuses calls once the cap is reached. |
| `env_server/check.py` | Asserts the policies and the HTTP layer against live sessions. |
| `containers/sandbox.def` | The image. Everything it pins is in `containers/versions.env`. |
| `containers/run_sandbox.sh` | Starts the server and gateway, runs a command in the sandbox, cleans up. Session videos go to `/work/obs`, read-only; `--max_cost` caps the model spend. |
| `containers/run_train_agent.sh`, `containers/train_agent/` | prime-agent developing a harness in a `--kind train` sandbox: its instructions (`AGENTS.md`, read-only at `/work/AGENTS.md`, the same whatever the harness), the loop that restarts it and checks a `/work/FINISHED` claim against `clears.jsonl` or accepts `/work/GIVE_UP`, `progress.py` for reading the progress records, and `harness_docs/<harness>.md`, each harness's `HARNESS.md`. The run exits 0 on a confirmed finish and 3 on giving up. |
| `containers/harness_repo.sh` | `sync <name> [--ref <ref>]` builds `harnesses/<name>` (gitignored here; its own repository with a fresh history) from a committed ref, `train_only` by default, refusing one that mentions a held-out game. `prepare` is how `run_sandbox.sh` clones it into `/work/harness` on branch `agent/<run>` the first time, and on later launches commits what the agent left uncommitted and merges the harness's `main` (kept as branch `upstream`), leaving a conflict for the agent. |
| `containers/train_watch.sh` | On the host during a train run: every 10 minutes fetches the agent's branches into `<run_dir>/harness_mirror.git` (a rewritten branch is kept beside the old one) and, with `--push_url`, pushes them under `--push_prefix/`; every hour, and at the end, snapshots its artifacts, logs and notes into `<run_dir>/archive/<time>/`, hard-linking unchanged files. |
| `containers/verify_sandbox.sh` | Checks a built image's contents and isolation. |
| `containers/sandbox_run.sh`, `netbridge.py` | Inside the image: expose the two Unix sockets as loopback ports, then run the command. |
| `containers/prime_agent_*.sh`, `kernel_*`, `seed_kernel_skills.py` | prime-agent specifics: the vendored installer, and the pinned Python environment it runs tools in. |
| `agent_workspace/envctl`, `envclient.py` | The client the agent is given. |
| `execution/remote_env.py` | `RemoteEnvironment`: the strategist harness's environment interface over an env-server session (`run_playthrough.py --remote`): the one in `$ENVCTL_SESSION`, or one it opens on `--game` itself. With named saves, `--resume_run` works remotely. |
| `execution/game_api.py` | The GameBoyWorlds names the harness uses: the real ones when GameBoyWorlds is installed, stand-ins inside the sandbox. |
| `containers/harness_requirements.txt` | The harness's pinned Python environment in the image (`/opt/harness`); `run_sandbox.sh --harness` copies the code to `/work/harness`. |

Routes, all JSON with `Authorization: Bearer <token>`. The server token reaches everything; a session's own token reaches only that session.

| Route | What it does |
|-------|-------------|
| `GET /tasks` | Tasks the policy lists. |
| `GET /scenes` | Free-play start states, `{game: [init_state]}`. |
| `POST /sessions` | Open a session: `{"task_id": ...}` or `{"game": ..., "init_state": ...}`, optional `"state": true` and `"run": <name>`. Returns the session and its token. |
| `GET /sessions/<id>` | Status: steps, `episode_over`, `success`, `subgoals`, `label`, `discarded_steps`. |
| `POST /sessions/<id>/reset`, `step`, `observe`, `save`, `load` | `step` takes `{"action": ...}`, `load` takes `{"handle": ...}`. On a train server `save` takes `{"name": ...}` and the name is the handle. |
| `POST /sessions/<id>/checkpoint` | Playthrough only: `{"label": ...}` checkpoints now; from then on the session resumes only from the latest label. |
| `DELETE /sessions/<id>` | Close it. |

To use a different agent, add its install to `sandbox.def` and pass its command after `--` to `run_sandbox.sh`. It needs to speak the OpenAI chat-completions API at `http://127.0.0.1:8720/v1`.
