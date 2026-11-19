# Your job

You are developing a Pokémon-playing harness: the strategist in `/work/harness`. A vision-language model plays the
game through it. A strategist picks one short task per episode, a supervisor plans it, and executors press buttons;
between episodes the strategist keeps memory (a global map built from frames, a goal tree, a knowledge tree, a
notepad, a strategy guide, a people registry).

**The goal.** This harness will be deployed on a Pokémon game you will never see, one that is not on the internet:
a ROM hack in the style of Red (generation 1) or Crystal (generation 2). Every change you make must help there.
That means:

- **Game-agnostic.** No town, route, character, item, badge or map names; no coordinates; no walkthrough knowledge;
  nothing that only works because it is Red or Crystal. Prompts must not mention either game.
- **Vision-only at deployment.** When deployed, the harness gets the screen and nothing else: no RAM and no game
  state. The training server can return full state, but the harness must never depend on it.
- **Both generations.** Red- and Crystal-style menus, text boxes, battle screens and overworlds all differ. A fix
  that helps one and breaks the other is not a fix.

You practise on `pokemon_red` and `pokemon_crystal`, the only games this server opens. Treat them as stand-ins
for the unseen game, not as targets to memorise.

# How you work

Repeat this loop:

1. Read `/work/DEVLOG.md` and `/work/EXPERIMENTS.md` (create them if they do not exist).
2. Look at the latest playthroughs: their episode reports, maps, notes and videos (below).
3. Pick the most damaging failure you can see: the strategist stuck in a loop, the map splitting one place into
   copies, the executor misreading a menu, battles lost to bad choices, tasks too small to make progress.
4. Find its cause in the code. Fix it in the harness, generally.
5. Re-run on **both** games, from the start or from a saved point (`--resume_run`), and compare against the
   last runs.
6. Keep the change if it helps and breaks nothing; otherwise undo it. You have no git: keep a copy of a file
   (for example under `/work/backups/`) before a risky edit.
7. Write down what you changed, why, and the evidence in `DEVLOG.md`; add each run to `EXPERIMENTS.md`
   (run name, game, start, episodes, map squares and areas, badges, stuck episodes, notes).

Keep going. Nobody will check in on you; the logs are how your work is judged and how you remember it after
your session restarts. If `/work/STOP` exists, finish what you are doing, write the log, and stop.

# Your environment

- `/work/harness`: the code. Read `README.md` and `README_dev.md` there first, then `run_playthrough.py` and
  `execution/strategist/`. Everything under `/work/harness` is yours to change.
- `/work/storage/playthrough_artifacts/<game>/strategist/<name>/`: each run's memory, one `episode_<n>/` per
  episode, with `report.pkl.gz` (what happened, every model call, and its images). Load them with the classes in
  `execution/strategist/report.py` and `execution/strategist/pokemon/global_map/` (`GlobalMap.load`,
  `render_segment`). The docs mention `debug.py`; it is not available here, so write your own inspection
  scripts with those classes.
- `/work/obs/<session>/`: the video of every server session, read-only. A new file starts at each episode.
- Python for the harness is `/opt/harness/bin/python`. There is no network and no package installs.
- The only model is `$SANDBOX_MODEL`, through the gateway. Every call costs money and the whole run has a fixed
  budget; when it is spent, model calls fail with a spending-cap error. Do not waste calls: test on short runs.
- `git` is disabled. `/work/AGENTS.md` (this file) is read-only.

# Running a playthrough

One server session can be open at a time, and opening one closes any other, so run one playthrough at a time:
starting a second one kills the first.

```bash
cd /work/harness
/opt/harness/bin/python run_playthrough.py --remote --game pokemon_red --name red_baseline \
    --model "$SANDBOX_MODEL" --vlm_kind gateway --max_episodes 15 > /work/logs/red_baseline.log 2>&1
```

- `--game` is `pokemon_red` or `pokemon_crystal`. Each starts at the beginning of the game.
- `--name` is lowercase letters and underscores and must not end in a digit. Re-running the same name continues
  that run from its last episode. Saves ignore underscores, so never use two names that differ only by them.
- `--name new_run --resume_run old_run` forks `old_run`'s latest state, memory and game alike, into `new_run`:
  the way to replay a failure after a fix without replaying everything before it.
- A run of 15 episodes takes on the order of an hour. Run it in the background with its log in `/work/logs/` and
  check on it, rather than blocking on it.
- `--report_detail executor` keeps every frame the executors saw, which helps when debugging perception.
