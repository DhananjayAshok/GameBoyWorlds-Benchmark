# Your job

You are developing a Pokémon-playing harness: the code in `/work/harness` that lets a vision-language model play
a Game Boy Pokémon game from its screen. How this harness is designed is described in its own guide,
`/work/harness/HARNESS.md`, and its other docs; read them first. This file is about the job and the rules, which
are the same whatever the harness looks like.

**The goal.** This harness will be deployed on a Pokémon game you will never see, one that is not on the internet:
a ROM hack in the style of Red (generation 1) or Crystal (generation 2). Every change you make must help there.
That means:

- **Game-agnostic.** No town, route, character, item, badge or map names; no coordinates; no walkthrough knowledge;
  nothing that only works because it is Red or Crystal. Prompts and examples must not mention either game or
  anything in them.
- **Vision-only.** When deployed, the harness gets the screen and nothing else. See the next section: this rule
  has no exceptions.
- **Both generations.** Red- and Crystal-style menus, text boxes, battle screens and overworlds all differ. A fix
  that helps one and breaks the other is not a fix.

You practise on `pokemon_red` and `pokemon_crystal`, the only games this server opens. Treat them as stand-ins
for the unseen game, not as targets to memorise.

**What done looks like.** Your harness, unchanged between the two runs, plays `pokemon_red` and `pokemon_crystal`
from start to finish: through every badge, the Elite Four and the Champion. The server marks a playthrough as
cleared only when the Champion is beaten. Until you have watched your harness do that on both games, you have not
solved the problem, however good the short runs look. A harness that reaches the first route cleanly has not
shown it can get through a gym, a cave, a puzzle or a story event it has never seen.

# Game memory is for measuring, never for playing

The training server can see the game's memory (RAM): badges, the current map, story event flags, the party, and
more. You get some of it, for measuring how far a playthrough has got (below). **The harness itself must never
read game memory, game state, or anything derived from them, in any form: not to decide an action, not to check
whether a task is done, not to name a place, not as a fallback, not as a hint.** That includes the `state` the
server can return with an observation, the progress files under `/work/obs/progress`, the named-save metadata,
and anything you compute from them. The harness may use only the screen.

This is not a style rule. The game it will be deployed on has a different memory layout, no symbol tables and no
event names, so a harness that reads memory anywhere does not work there at all, however well it does here. Use
game memory only in your own inspection scripts, to judge runs and to find out what went wrong.

# How you work

Your main test is a **long playthrough**: one run that keeps going, episode after episode, towards the end of the
game. Start it from the beginning of the game. Starting from a point you saved earlier is fine when replaying the
start would teach you nothing new, but a fix is only proven once a run that includes it has carried on past the
place it fixed and kept making progress.

Repeat this loop:

1. Read `/work/DEVLOG.md` and `/work/EXPERIMENTS.md` (create them if they do not exist).
2. Look at the latest playthroughs: how far each got (below), their videos, and the harness's own records of them.
   Watch the videos; logs and summaries hide failures.
3. Find where the long playthroughs stall: the furthest point each game has reached, and what keeps the harness
   from getting past it. Fix that first. Smaller problems matter only when they stand between the harness and
   further progress.
4. Find its cause in the code. Fix it in the harness, generally.
5. Check the fix on a short run first: fork the run that showed the failure (`--resume_run`) and see whether it
   gets past that point. Then carry on with a long playthrough of **both** games.
6. Keep the change if it helps and breaks nothing; otherwise revert it (`git revert`).
7. Commit each fix on its own (below). Write down what you changed, why, and the evidence in `DEVLOG.md`; add each
   run to `EXPERIMENTS.md` (run name, game, start, harness commit, episodes, furthest point reached, badges, maps
   reached, story events, stuck episodes, notes).

**Judging progress.** Progress is how far into the game a run gets: badges won, new maps reached and new story
events. Fewer model calls, fewer episodes or more episodes ending `completed` mean nothing if the run gets no
further. Runs vary from one to the next, so do not call one change better than another from a single short run of
each.

The server measures this for every playthrough, from game memory:

```bash
/opt/harness/bin/python /opt/train/progress.py              # every run: badges, maps reached, events, last map
/opt/harness/bin/python /opt/train/progress.py --run <name> # one run's timeline: which maps, events and badges, when
```

Its records are in `/work/obs/progress/<session>.jsonl`, one line at every save, load and reset, every 500 steps,
and when a session closes or clears the game; `clears.jsonl` lists the games cleared. A run is matched by the
`--name` it was started with. Event and map names are the game's internal ones: use them to see what happened,
and never copy them into the harness.

Keep going. Nobody will check in on you; the logs and the commits are how your work is judged and how you
remember it after your session restarts. Ending a session while nothing is running and the games are not cleared
wastes the next one: before you stop, leave a playthrough running, or write in `DEVLOG.md` what the next session
should do first. If `/work/STOP` exists, finish what you are doing, write the log, and stop.

# Version control

`/work/harness` is a git repository and you are on your own branch. Commit every fix on its own, with a message
that says what was wrong, how you fixed it, and what evidence you have; commit before you start a run that tests
it, so the run is tied to the code that made it. Revert a change that did not help with `git revert`, rather than
rewriting history. Your commits are copied out of the sandbox and kept, so they are also your backup.

Changes may come from outside the sandbox on the branch `upstream`. They are merged into your branch for you when
they apply cleanly; when they do not, your session prompt says so, and you merge `upstream` yourself first,
keeping both sides' fixes. The repository has no remote: there is nothing to push to or pull from.

# Finishing or giving up

You can end the whole run in two ways. Either one stops it for good once your session ends, so do not do either
lightly.

- **Finished.** When your harness has cleared both games, write `/work/FINISHED`: which runs cleared which game,
  at which harness commit, and what you changed to get there. It is checked against the server's own record of
  cleared games (`/work/obs/progress/clears.jsonl`). A claim it does not confirm is rejected, and you carry on.
- **Give up.** If you have tried every approach you can think of and truly cannot see a way to make the harness
  clear the training games, write `/work/GIVE_UP`: how far the harness gets on each game, what stops it, every
  approach you tried and how each one failed, and why you believe nothing more you could do would get past it.
  Giving up is for having run out of ideas, not for a failure that is merely hard, slow or expensive to fix.

Do not end sessions early instead: a session that ends without new work, while neither file exists, only starts
the next one.

# Your environment

- `/work/harness`: the harness, yours to change. Read `HARNESS.md` there first.
- `/work/obs/<session>/`: the video of every server session, read-only. A new file starts at each save.
- `/work/obs/progress/`: the server's progress records (above), read-only.
- `/work/logs/`: put playthrough logs here. `/work/DEVLOG.md` and `/work/EXPERIMENTS.md`: your notes.
- Python for the harness is `/opt/harness/bin/python`. There is no network and no package installs.
- The only model is `$SANDBOX_MODEL`, through the gateway. Every call costs money and the whole run has a fixed
  budget; when it is spent, model calls fail with a spending-cap error. Spend it on long playthroughs and on
  short forks that check a fix; do not spend it on repeating the same short stretch of the game.
- `/work/AGENTS.md` (this file) and everything under `/opt/train` are read-only.

# Running a playthrough

Every harness is run the same way:

```bash
cd /work/harness
/opt/harness/bin/python run_playthrough.py --remote --game pokemon_red --name red_long \
    --model "$SANDBOX_MODEL" --vlm_kind gateway --max_episodes 400 > /work/logs/red_long.log 2>&1
```

- `--game` is `pokemon_red` or `pokemon_crystal`. A new name starts at the beginning of the game.
- `--name` is lowercase letters and underscores and must not end in a digit. Re-running the same name continues
  that run from where it stopped. Never use two names that differ only by underscores.
- `--name new_run --resume_run old_run` forks `old_run`'s latest state, harness memory and game alike, into
  `new_run`: the way to replay a failure after a fix without replaying everything before it.
- One server session can be open at a time, and opening one closes any other, so run one playthrough at a time:
  starting a second one kills the first.
- A whole game takes many hundreds of episodes. Run playthroughs in the background with their log in
  `/work/logs/` and check on them, rather than blocking on them.
- A running playthrough keeps the code it started with. To carry a fix into it, stop it and re-run the same
  `--name`; it continues from where it stopped.

`HARNESS.md` describes this harness's other options and where it keeps its own records.
