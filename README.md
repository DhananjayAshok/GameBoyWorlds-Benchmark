<div align="center">
  <picture>
    <img alt="GameBoyWorlds" src="assets/logo_tilt.png" width="350px" style="max-width: 100%;">
  </picture>
  <br>

  **A Testbed for Self Improving Agents in Embodied Video Games**

  <br>
    <a href="https://github.com/DhananjayAshok/GameBoyRL/blob/main/LICENSE" target="_blank" rel="noopener noreferrer"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue"></a>
    <a href="https://example.com" target="_blank" rel="noopener noreferrer"><img alt="Paper" src="https://img.shields.io/badge/paper-arXiv-b31b1b"></a>
    <a href="https://example.com" target="_blank" rel="noopener noreferrer"><img alt="Website" src="https://img.shields.io/badge/website-online-brightgreen"></a>
    <a href="https://github.com/DhananjayAshok/GameBoyWorlds" target="_blank" rel="noopener noreferrer"><img alt="GameBoyWorlds" src="https://img.shields.io/badge/environments-GameBoyWorlds-f2c94c"></a>
</div>

<br>

## Overview

<img src="assets/logo.png" width="70px"> is the first testbed for self-improvement in complex, long-horizon video games. It challenges models to learn from experience alone, and learn continually during deployment in unfamiliar environments. 

![](assets/GameBoyWorldsTeaser.gif)

This repository establishes two things:

- **Execution.** When deployed without a harness, frontier VLMs fail more often than not on short-horizon execution tasks. Over 500 tasks, no frontier model achieves more than a 50% task success rate. Standard implementations of self-improvement methods **hurt** performance, suggesting that autonomous multimodal-grounding remains difficult. 

- **Playthrough.** Even when they are given a harness, VLMs struggle to clear games that are poorly represented in their training data. The result exposes that active knowledge acquisition and guided exploration are open challenges for embodied agents. 

---

## What brings you here?

**Just curious:** Don't miss our [project website](https://example.com) and, if you'd like more technical detail, [our paper](https://example.com). Questions are always welcome at [ashokd@usc.edu](mailto:ashokd@usc.edu).

**I want to use GameBoyWorlds:**  Read on.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/DhananjayAshok/GameBoyWorlds-Benchmark
cd GameBoyWorlds-Benchmark
git submodule update --init --recursive GameBoyWorlds cleanrl
```

`GameBoyWorlds` is the environment suite and `cleanrl` is the RL backend. The third submodule, `llm-utils`, is only needed to reproduce the PAE baseline and is left out here on purpose.

### 2. Create the environment

```bash
bash setup/create_env.sh --env_dir /path/to/env
```

This creates a Python 3.12 virtual environment at `env_dir` with `uv`, symlinks it to `setup/.venv`, installs everything from `setup/pyproject.toml`, and writes the `config.env` files for both this repo and GameBoyWorlds.

<details>
<summary>Manual alternative</summary>

```bash
uv venv /path/to/env --python 3.12
ln -s /path/to/env setup/.venv
cd setup && uv sync && cd ..
source setup/.venv/bin/activate
python configs/create_env_file.py
python GameBoyWorlds/configs/create_env_file.py
```
</details>

### 3. Configure

Edit `configs/private_vars.yaml`:

```yaml
storage_dir: "/path/to/your/storage"
...
```

`storage_dir` holds the heavy artifacts (trajectories, tasks, documents, checkpoints). Results — CSVs, plots, debug reports — go to `results_dir`, which defaults to `results/` inside the repo; change it in `configs/project_vars.yaml` if you want them elsewhere.

Set a storage directory in `GameBoyWorlds/configs/private_vars.yaml` too. It does **not** have to be the same path — GameBoyWorlds storage (ROMs, save states, recorded sessions) is independent of this repo's storage (tasks, trajectories, documents).

Model access is read from the environment, not from any config file. Export whichever of these you plan to use:

```bash
# Both must be exported and visible to the process that runs the commands below.
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
```

### 4. ROMs

Nothing will run until the ROMs are in place. Every ROM lives at

```
<GameBoyWorlds_storage_dir>/rom_data/<game_name>/ROM_NAME.extension
```

See [the ROM Setup section of the GameBoyWorlds README](https://github.com/DhananjayAshok/GameBoyWorlds#rom-setup) for the list of supported games, how to obtain each one legally, and the exact expected path.

### 5. (Optional) llm-utils, for the PAE baseline only

```bash
git submodule update --init --recursive llm-utils
bash setup/create_llm_utils_env.sh --env_dir /path/to/llm-env
```

This is a second, separate environment — fine-tuning pulls in a training stack that we keep out of the main one. Skip it unless you are reproducing PAE.

<details>
<summary>Manual alternative</summary>

```bash
uv venv /path/to/llm-env --python 3.12
ln -s /path/to/llm-env llm-utils/setup/.venv
cd llm-utils/setup && uv sync
```
</details>

---

## Verify your setup

**GameBoyWorlds.** All three demos should work:

```bash
python GameBoyWorlds/demos/emulator.py --game pokemon_red --play_mode human
python GameBoyWorlds/demos/environment.py --game pokemon_red
python GameBoyWorlds/demos/benchmark.py --game pokemon_red
```

The first opens a window you can play with the keyboard; the other two step the environment programmatically. If any of them cannot find a ROM, go back to step 4.

**Final Test:** One benchmark task against a cheap model:

```bash
bash scripts/benchmark/run_benchmark.sh \
    --game pokemon_red \
    --supervisor subgoal \
    --executor single_visual \
    --executor_vlm_model gpt-4o-mini \
    --executor_vlm_kind openai \
    --n_tasks 1
```

This needs `OPENAI_API_KEY`. If it finishes and writes a CSV, your install is good.

---

## Run a model on GameBoyWorlds-Execution

Execution is the short-task benchmark: a fixed list of tasks per game, each scored on its own, with success decided by the environment rather than by a judge. The arm we report is the **subgoal** supervisor over the **single_visual** executor.

```bash
bash scripts/benchmark/run_benchmark.sh \
    --game pokemon_red \
    --supervisor subgoal \
    --executor single_visual \
    --executor_vlm_model google/gemini-2.5-flash \
    --executor_vlm_kind openrouter
```

Any model OpenRouter serves works as `--executor_vlm_model`. To run a local model instead, swap `--executor_vlm_kind` for `vllm` (point it at a server started with `scripts/core/serve_vllm.sh`) or `huggingface` (loads the weights in-process). Nothing else about the command changes.

### Where the artifacts go

- **The scores.** `<results_dir>/benchmark/<game>/subgoal_single_visual_low_level_<model>.csv`, one row per task: whether it succeeded, how many steps it took, how many actions were invalid, which subgoals it reached, and the session directories it wrote.
- **The full record.** Each row's `session_dirs` points into GameBoyWorlds' `sessions/` directory, and each of those holds a `report.pkl.gz` — every supervisor call and every executor leg underneath it, with the images the model saw, the prompt, the raw response and the actions it produced. 
- **Video.** Recorded per session unless you pass `--save_video false`.

### Looking at what happened

`debug.py` reads those artifacts and writes markdown reports (with frames) under `<results_dir>/debug/<game>/`. 

```bash
python debug.py --game pokemon_red --executor single_visual benchmark --supervisor subgoal
```

You get one `episodes_<model>.md` per model — a frame-by-frame walk through each episode — and a `comparison.md` when two models are present.

### Results

![Execution leaderboard](assets/frontier_leaderboard.png)

![Model legend](assets/frontier_model_legend.png)

![Execution heatmap](assets/frontier_heatmap.png)

Frontier models fail to clear 50% performance on average, showing that short-horizon task execution in video game environments remains challenging. As you'll see from reproducing the self-improvement methods below, the standard approaches do not help at all. 

Solving GameBoyWorlds-Execution will require generalizable self-improvement methods that autonomously ground embodied agents to their environment. 

---

## Run a model on GameBoyWorlds-Playthrough

Unlike the original games, fan-made games are scarcely documented and so models trained on internet-scale corpora (e.g. Large Language Models) haven't already acquired as rich base of knowledge regarding the game areas or particular Fakémon. While good performance in PokémonRed may be a result of the backbone model's data being contaminated with walkthroughs and internet guides, the same concern is far less valid for more obscure fan-made games. You can find evidence of this in the HuggingFace dataset below:

<a href="https://huggingface.co/datasets/DJ-Research/GameBoyWorlds-Playthrough-Contamination-Check" target="_blank" rel="noopener noreferrer"><img alt="Contamination Check" src="https://img.shields.io/badge/dataset-Contamination%20Check-ffd21e?logo=huggingface"></a>

To run a playthrough: 

```bash
python run_playthrough.py \
    --game pokemon_red \
    --name gemini_run \
    --model google/gemini-3.6-flash \
    --vlm_kind openrouter \
    --max_episodes 400
```

`--name` is the identity of the playthrough. State is resumed by default, so re-running the same name continues where it left off rather than starting over. `--init_state starter` begins at the point where the first party member is chosen; `initial` is the true start of the game. You can override the model per layer with `--strategist_vlm_model`, `--supervisor_vlm_model` and `--executor_vlm_model`. `--resume_run <other name>` forks another playthrough's latest state into a new `--name`, which is how you replay from a point of interest without touching the original.

### How the harness works

The agent sees only the screen. Three layers sit between the model and the buttons:

- The **strategist** (`execution/strategist/`) owns the long game. Each episode it writes one short task.
- A **supervisor** (`execution/pokemon/supervisors.py`) carries that task out in at most `--supervisor_max_steps` steps, choosing an executor for each.
- **Executors** (`execution/pokemon/executors.py`) turn a step into button presses: walking, menus, dialogue, battles.

One episode runs like this:

1. **Perceive.** The tile recognizer labels the tiles on screen, and the frame is placed on the global map.
2. **Stuck check.** If recent episodes found nothing new or kept failing, the strategist is asked whether it is stuck. If so, exploration overrides the next task and the failing plan is abandoned.
3. **Plan.** The current goal is re-planned into subgoals every `--subgoal_every` episodes it stays open, using what the knowledge tree and the map say.
4. **Dispatch.** One call turns the goal, notepad, active guide rules and the map around the player into a task and guidance.
5. **Execute.** The supervisor runs the task. Every step it takes also updates the map.
6. **Review.** New areas are named, goals are checked off or revised, facts go into the knowledge tree, the notepad is updated, and guide rules are added or rewritten. A lost battle triggers a reflection that can add a "prepare first" subgoal.

### Where the artifacts go

The playthrough's memory lives at `<storage_dir>/playthrough_artifacts/<game>/strategist/<name>/episode_<n>/`. Six artifacts, each saved in the episodes where it changed:

| Artifact | What it holds |
|----------|---------------|
| `goals` | The goal tree — what the strategist is trying to do, and how it decomposed it |
| `globalmap` | The map it has built from screens alone: areas, doors between them, and where it has walked |
| `knowledge` | What it believes about the game's world, mechanics and its own situation, organised by topic |
| `guide` | Rules of thumb tied to events (entering a building, reaching a town, battles), seeded by `guide_seeds.yaml` and extended as it plays |
| `tiles` | The tile recognizer: the tile images it has seen and what it decided they are |
| `notepad` | Free-text thoughts it wrote to itself |

Alongside them, each episode directory holds a `report.pkl.gz` with every model call of that episode (`--report_detail` sets how deep it goes), and `provenance_<stamp>.json` in the run directory records the settings each process ran under and where its video is. Video goes to GameBoyWorlds' `sessions/<game>/strategist/<name>/<stamp>/`. The emulator save state for each episode is kept by GameBoyWorlds, which is what lets a run resume mid-game.

### Looking at what happened

```bash
# Every artifact as it stood at one moment
python debug.py --game pokemon_red playthrough view --name gemini_run --episode 120

# How each artifact CHANGED, save by save, from first version to latest
python debug.py --game pokemon_red playthrough diff --name gemini_run
```

`diff` shows you the moment the agent discovered a route, revised a belief, or spent forty episodes re-deriving something it already knew. Add `--artifact globalmap` (or `goals`, `knowledge`, `guide`, `tiles`, `notepad`) to narrow it to one. `bash runs/playthrough/playthrough.sh --name gemini_run --model google/gemini-3.6-flash` runs a playthrough and both reports in one go.

### Results

![Pokémon Red versus Pokémon Brown](assets/red_vs_brown.png)

The same system, the same number of episodes. In Pokémon Red — a game with two decades of walkthroughs in every pretraining corpus — it makes steady progress through the initial parts of the game. In Pokémon Brown, a fan-made game of comparable difficulty about which far less has been written, it never leaves the starter town. 

Solving GameBoyWorlds-Playthrough will require agents that can actively acquire information, conduct motivated exploration and self-improve continually, mid-deployment. 

---

## Bring your own agent

The harness above is ours. To evaluate a different agent — a coding agent, say, that writes its own harness — run it in the sandbox. The agent gets a workspace and a command-line client for the games, and nothing else: no network, no ROMs, no held-out tasks, no API key.

### The pieces

- **The environment server** (`run_env_server.py`, `env_server/`) runs on the host and owns the emulators. It serves sessions over HTTP: open a task, step it, get back the screen as a PNG, the goal, and whether the task is done.
- **The splits** (`configs/splits/`) divide the benchmark's tasks into `train`, `val`, `test` (held-out tasks in the training games) and `test_env` (every task of a held-out game). Tasks that share a start state always share a split.
- **The server's policy** is fixed when it starts and no request can change it:

  | | `--kind dev` | `--kind test` |
  |---|---|---|
  | Tasks it opens | `train`, `val` | `test`, `test_env` |
  | Lists tasks | yes | no |
  | Free play from a start state | yes | no |
  | Emulator state returned | full, on request | task success and subgoal progress only |
  | Save / load | yes | no |
  | Attempts per session | unlimited resets | one |

- **The model gateway** (`env_server/model_gateway.py`) is the sandbox's only route to a model. It forwards chat completions for the models you allow to OpenRouter and adds your API key on the host, so the key is never inside the sandbox. It strips web-search and fallback-routing fields.
- **The sandbox image** (`containers/sandbox.def`) is an [Apptainer](https://apptainer.org) image holding a Python runtime and the agent. It runs with no network at all; the server and gateway reach it as two Unix sockets.
- **The workspace** (`agent_workspace/`) is what the agent starts with: `envctl`, the command-line client, and `envclient.py`, the same thing as a Python module.

### Setup

You need Apptainer with unprivileged `--fakeroot` builds and `--net --network none` allowed. On an HPC system, build on a compute node.

```bash
# 1. Check the server's policies against live emulators
python run_env_server.py check

# 2. Build the image (about ten minutes, needs the network), then verify it
apptainer build --fakeroot <storage_dir>/containers/sandbox.sif containers/sandbox.def
bash containers/verify_sandbox.sh
```

`verify_sandbox.sh` checks that the image holds the pinned versions, and that from inside it your home directory, this repository, your storage, the ROMs and the network are all unreachable. It must pass before you trust a run.

The image is looked for at `<storage_dir>/containers/sandbox.sif`; set `SANDBOX_SIF` to keep it elsewhere. Sockets and the agent's home go in `/dev/shm`; set `SANDBOX_RUNTIME_DIR` to change that.

### Running an agent

```bash
export OPENROUTER_API_KEY=...

# Development: the agent can list and open train/val tasks itself
bash containers/run_sandbox.sh --kind dev --model google/gemini-3.6-flash \
    --workspace /path/to/workspace -- \
    prime-agent --provider gateway --model google/gemini-3.6-flash --cwd /work -p "<your prompt>"

# Evaluation: one held-out task, opened for the agent, one attempt
bash containers/run_sandbox.sh --kind test --task_id gb.pokemon_red.001 \
    --model google/gemini-3.6-flash --time_limit 1800 -- \
    prime-agent --provider gateway --model google/gemini-3.6-flash --cwd /work -p "<your prompt>"
```

Everything after `--` runs inside the sandbox with `/work` as its directory. The launcher starts the server and the gateway, runs the command, and shuts both down. Logs go to `<results_dir>/sandbox/<stamp>_<kind>/`; a test run also writes `result.json` there with `success` and the steps used. Add `--record_dir <dir>` to keep the full emulator state and every frame, host-side, whatever the agent was shown.

Inside the sandbox the agent drives the game with `envctl`:

```bash
python envctl tasks                     # dev only: the tasks it may practise on
python envctl open --task gb.pokemon_red.004
python envctl step "A"                  # writes obs/<session>/<n>.png and <n>.json
python envctl actions                   # the action strings this game accepts
python envctl close
```

In a test run the session is already open, so the agent starts at `envctl observe` or `envctl step`.

To regenerate the splits after changing `configs/splits/split_config.yaml` or the benchmark's task list, run `python run_env_server.py generate_splits --show`.

---

## Reproduce Self-Improvement Results

Each vertical below is independent. All three end in a benchmark run whose CSV you compare against the baseline from the Execution section above.

### World Model

Train a world model on curiosity-collected buffers, then let it drive the executor.

**1. Collect buffers.** Random-action and curiosity replay buffers for every train init_state of the game's series:

```bash
bash scripts/rl/collect_wm_buffers.sh --game pokemon_red
```

*Expected after this step:* one replay-buffer folder per (init_state, policy) pair under your storage directory, each holding the observations and actions of a full collection run. No grouping and no clustering happens here — these are raw buffers.

**2. Train the observation embedder, then the world model.**

```bash
bash scripts/core_rl/train_observation_embedder.sh --game pokemon_red --init_state initial
bash scripts/core_rl/train_world_model.sh --game pokemon_red --latest_replay_buffer_folder /path/from/step/1
```

*Expected after this step:* a world-model checkpoint and an observation-encoder checkpoint saved under the run name you used. 

**3. Benchmark it.**

```bash
bash scripts/benchmark/run_benchmark.sh \
    --game pokemon_red \
    --supervisor subgoal \
    --executor world_model \
    --world_model_run_name <run_name> \
    --executor_vlm_model gemma-4-31b-it \
    --executor_vlm_kind vllm
```

*Expected after this step:* a CSV named for the world-model executor (`..._world_model_<run_name>_...`), directly comparable to the baseline CSV. For test-only games, `--world_model_game` borrows the first installment game's checkpoint.

### PAE

PAE annotates a trajectory with step-by-step guidance, has the model practice against that guidance, filters what it produced, and fine-tune on what survives.

**1. Guidance and practice.**

```bash
bash scripts/pipeline/guidance_and_practice.sh \
    --game pokemon_red \
    --trajectory_path /path/to/trajectory/stem \
    --model_name gemma-4-31b-it \
    --vlm_kind vllm
```

*Expected after this step:* a guidance file beside the trajectory stem, and a practice directory holding one record per practice call — the frames, the guidance the model was given, what it did, and the binary judge's verdict. The script continues into cleaning by default; pass `--do_clean false` to stop before it.

**2. Clean.**

```bash
bash scripts/vlm/clean_practice.sh \
    --game pokemon_red \
    --practice_path /path/from/step/1 \
    --model_name gemma-4-31b-it \
    --vlm_kind vllm
```

*Expected after this step:* the same practice directory, with each call now carrying a paraphrased task and a filter verdict. This step is resumable — re-running it picks up where it left off rather than redoing calls.

**3. Build the dataset.**

```bash
bash scripts/vlm/create_dataset.sh --practice_path /path/from/step/2
```

*Expected after this step:* `train_dataset.csv` and `validation_dataset.csv` in the practice directory, split at `--val_frac` (0.2 by default). To train on more than one game or source at once, merge several practice directories first with `scripts/vlm/merge_practices.sh`, which tags each row with where it came from.

**4. Fine-tune.** This is the only step that needs the `llm-utils` environment.

```bash
bash scripts/vlm/train_vlm.sh \
    --train_file /path/to/train_dataset.csv \
    --validation_file /path/to/validation_dataset.csv \
    --model_name google/gemma-4-31b-it \
    --run_name pae_red
```

*Expected after this step:* LoRA adapter weights under the run name, optionally pushed to the Hub with `--push_to_hub true`.

**5. Benchmark it.** Serve the fine-tuned model and run the standard Execution command against it with `--executor_vlm_kind vllm`. You will want to provide an explicit `--supervisor_vlm_model` (and `--supervisor_vlm_kind`) pointing at the **base** model. 

### Curiosity-driven exploration

The curiosity vertical is the one that produces the **written guides** — per-game information documents distilled out of trajectories the agent collected itself, which a benchmark agent can then read at test time.

**You probably want to skip it.** We publish its output:

```bash
python sync_data.py pull --set curiosity               # dry run: reports what would be written
python sync_data.py pull --set curiosity --no_dry_run  # actually download
```

That gives you the trajectory annotations and the finished information documents, which is everything the benchmark reads. You can then jump to step 3. But if you want to run everything on your end:

**1. Collect and cluster.** RL agents explore under a curiosity reward; the resulting replay buffers are clustered by observation similarity into grouped trajectories.

```bash
bash scripts/rl/create_all_traj.sh --game pokemon_red --init_state initial --run_name my_run
```

*Expected after this step:* one grouped trajectory file per init_state. Each group is a cluster of observations the agent found distinctive; a healthy run gives you tens of groups per state, not two and not two thousand. If grouping collapses, tune `--z_min`.

**2. Infer tasks, propose, attempt, and distil.** One command runs the whole chain — task inference from the clusters, zero-shot proposal, attempting both under the two-stage judge, and building the documents:

```bash
bash scripts/pipeline/collect_and_info_all.sh \
    --game pokemon_red \
    --run_name my_run \
    --model_name google/gemini-2.5-flash \
    --vlm_kind openrouter
```

*Expected after this step:* under `<storage_dir>/proposed_tasks/<game>/<model>/`, a task file per source, a `success_trajectories.{json,pkl}` pair recording what the agent actually managed to do, and an `info_docs/` directory holding the written guides. The documents are the deliverable; the tasks and trajectories are how they were earned.

**3. Benchmark with retrieval.**

```bash
bash scripts/benchmark/run_benchmark_info_retrieval.sh \
    --game pokemon_red \
    --executor single_visual \
    --executor_vlm_model google/gemini-2.5-flash \
    --executor_vlm_kind openrouter \
    --docs_mode curiosity_only \
    --docs_run_name my_run
```

*Expected after this step:* a CSV named `info_subgoal_retrieval_..._curiosity_only.csv`. The control is `--supervisor info_subgoal_parametric` on the ordinary benchmark script: same planner, same everything after the document, but the document is one the model writes from its own priors given only the game's name. The gap between those two CSVs is what distillation bought.

### Results

![Self-improvement gain](assets/self_improvement_gain.png)

Nothing seems to work well. Self-improvement methods will need active development to work in the video game domain. 

---

## Development

Every script, what it does, and how the pieces fit together: [README_dev.md](README_dev.md).

---

## Citation

```bibtex

```
