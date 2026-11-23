# The strategist harness

A vision-language model plays the game through three layers. A strategist picks one short task per episode, a
supervisor plans it, and executors press buttons. Between episodes the strategist keeps memory: a global map built
from frames, a goal tree, a knowledge tree, a notepad, a strategy guide and a people registry.

Every episode a building survey (`execution/strategist/pokemon/buildings.py`) reads the map and the people registry
and, while a door nearby is not entered or someone nearby, indoors or out, is not talked to, sets the task ahead of
the goal. It gives up on a door or person after three tasks that got no nearer.

The tile recognizer and the global map are examples of a design change, as opposed to a patch: they solve whole
problems for the player (what is on screen, where things are, how places connect) by building real machinery
rather than by tweaking prompts.

Read `README.md` and `README_dev.md` here, then `run_playthrough.py` and `execution/strategist/`. The docs mention
`debug.py`; it is not in this repository, so write your own inspection scripts with the classes below.

# How it is run

The controller runs every playthrough, long or short, like this, and nothing else may open a game session:

```bash
cd /work/harness
/opt/harness/bin/python run_playthrough.py --remote --game <game> --name <run> --model "$SANDBOX_MODEL" \
    --vlm_kind gateway --max_episodes 100000 --record_dir <dir> [--start_checkpoint <run>:<id>]
```

Whatever you change, this command line must keep working, and the run must keep writing the two files below into
`--record_dir`: the controller, the checks and the short tests all depend on them.

- `checkpoints.jsonl`: one line each time the run can be forked from where it is: `id`, `t`, `episode`, `steps`,
  `server_save` (the name of the game state saved on the server) and `memory` (the files holding the harness's
  memory at that point). The strategist writes one after every episode. `--start_checkpoint <run>:<id>` starts a
  new run from that game state and that memory.
- `trajectory.jsonl`: one line per event: `t`, `episode`, `layer` (strategist, supervisor or executor), `kind`
  (intention, plan, attempt or outcome), `steps` and `text`. It is written by `execution/run_record.py`; when you
  add a layer or a decision that matters for understanding a run, record it there too.

# Its records

- `/work/storage/playthrough_artifacts/<game>/strategist/<run>/`: each run's memory, one `episode_<n>/` per
  episode, with `report.pkl.gz` (what happened, every model call, and its images). Load them with the classes in
  `execution/strategist/report.py` and `execution/strategist/pokemon/global_map/` (`GlobalMap.load`,
  `render_segment`). The memory files of checkpoints that were not kept are deleted after each cycle; the reports
  stay.
- `harness_commits.jsonl` in the same directory: the harness commit each launch of the run used, and whether it
  had uncommitted changes.

# Its options

- `--report_detail executor` keeps every frame the executors saw, which helps when debugging perception.
- `--supervisor_max_steps` (default 10) is the executor calls a supervisor may spend per episode.
- Each layer can use its own model (`--strategist_vlm_model`, `--supervisor_vlm_model`, `--executor_vlm_model`);
  in this sandbox only `$SANDBOX_MODEL` is available.

The controller does not pass these, so to change one, change its default in `run_playthrough.py`.
