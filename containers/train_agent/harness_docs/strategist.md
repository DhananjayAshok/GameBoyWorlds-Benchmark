# The strategist harness

A vision-language model plays the game through three layers. A strategist picks one short task per episode, a
supervisor plans it, and executors press buttons. Between episodes the strategist keeps memory: a global map built
from frames, a goal tree, a knowledge tree, a notepad, a strategy guide and a people registry.

Read `README.md` and `README_dev.md` here, then `run_playthrough.py` and `execution/strategist/`. The docs mention
`debug.py`; it is not in this repository, so write your own inspection scripts with the classes below.

# Its records

- `/work/storage/playthrough_artifacts/<game>/strategist/<name>/`: each run's memory, one `episode_<n>/` per
  episode, with `report.pkl.gz` (what happened, every model call, and its images). Load them with the classes in
  `execution/strategist/report.py` and `execution/strategist/pokemon/global_map/` (`GlobalMap.load`,
  `render_segment`).
- `harness_commits.jsonl` in the same directory: the harness commit each launch of the run used, and whether it
  had uncommitted changes.

# Its options

On top of the options every harness takes (see `/work/AGENTS.md`):

- `--max_episodes` defaults to 400: the episodes one launch may run.
- `--report_detail executor` keeps every frame the executors saw, which helps when debugging perception.
- `--supervisor_max_steps` (default 10) is the executor calls a supervisor may spend per episode.
- Each layer can use its own model (`--strategist_vlm_model`, `--supervisor_vlm_model`, `--executor_vlm_model`);
  in this sandbox only `$SANDBOX_MODEL` is available.

The strategist saves the game after every episode under a name built from `--name` with the underscores removed,
which is why two names that differ only by underscores collide.
