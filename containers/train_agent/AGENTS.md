# Your job

You are developing a Pokémon-playing harness: the code in `/work/harness` that lets a vision-language model play
a Game Boy Pokémon game from its screen. How this harness is designed is described in its own guide,
`/work/harness/HARNESS.md`, and its other docs. This file holds the rules that apply in every step of the work.
What to do in each step is in that step's own prompt.

**The goal.** This harness will be deployed on a Pokémon game you will never see, one that is not on the internet:
a ROM hack in the style of Red (generation 1) or Crystal (generation 2). Every change you make must help there.
That means:

- **Game-agnostic.** No town, route, character, item, badge or map names; no coordinates; no walkthrough knowledge;
  nothing that only works because it is Red or Crystal. Prompts and examples must not mention either game or
  anything in them. That includes directions: "the exit is to the north" is walkthrough knowledge of one map, not
  a rule for every game. `/opt/harness/bin/python /opt/train/agnostic_check.py` lists names from the training
  games found in the harness code; a fix is not accepted while it finds any.
- **Vision-only.** When deployed, the harness gets the screen and nothing else. See the next section: this rule
  has no exceptions.
- **Both generations.** Red- and Crystal-style menus, text boxes, battle screens and overworlds all differ. A fix
  that helps one and breaks the other is not a fix.

You practise on `pokemon_red` and `pokemon_crystal`. Treat them as stand-ins for the unseen game, not as targets
to memorise.

**What done looks like.** Your harness, unchanged between the two, plays both games from start to finish: through
every badge, the Elite Four and the Champion. The server marks a playthrough as cleared only when the Champion is
beaten, and the work ends when both games are cleared.

# Game memory is for measuring, never for playing

The training server can see the game's memory (RAM): the current map, where the player has stood, badges, story
event flags, the party, and more. You get some of it, for measuring how far a playthrough has got. **The harness
itself must never read game memory, game state, or anything derived from them, in any form: not to decide an
action, not to check whether a task is done, not to name a place, not as a fallback, not as a hint.** That
includes the progress files under `/work/obs/progress`, everything the controller writes under
`/work/playthroughs`, `/work/tests` and `/work/cycle`, and anything you compute from them. The harness may use
only the screen.

This is not a style rule. The game it will be deployed on has a different memory layout, no symbol tables and no
event names, so a harness that reads memory anywhere does not work there at all, however well it does here. Map
and event names in the records are the game's internal ones: use them to see what happened, and never copy them
into the harness.

# The cycle

The work goes round a fixed cycle, run by a controller. Each step starts you in a new conversation with that
step's prompt, so nothing carries over in your head: what you know is what is in the files. A step ends when the
record it asks for is accepted by `/opt/train/record.py`.

1. **Start.** You choose where the next long playthrough begins; the controller starts it. The games alternate.
2. **Long playthrough.** It runs for up to 3 hours. Every 5 minutes a check summarises the last 5 minutes, names
   the issues it shows, and judges whether the player is stuck. A `stuck` verdict ends the playthrough.
3. **Conclusion.** You decide which checkpoints to keep (at most 10%), write the issues table, ranked, and write
   one summary of the playthrough.
4. **Deep dive.** The top 5 issues, one at a time: confirm the failure in the records first, then investigate, fix
   in code, commit, and test with short tests. An issue the records do not show is closed with nothing changed.
5. **Larger issues.** You step back, name up to 3 larger problems in how the player behaves, and work on each.
6. **Iteration summary.** What changed, and what the next playthroughs should show because of it.

**You do not start, stop or continue playthroughs yourself.** The controller starts every long playthrough, and
short tests are started with `/opt/train/short_test.py` during an issue step. Running `run_playthrough.py`
yourself does not work: only the controller can open a game session.

# Where things are

- `/work/harness`: the harness, yours to change in the issue steps. Read `HARNESS.md` there first.
- `/work/playthroughs/<id>/`: one directory per long playthrough.
  - `playthrough.json`: how it started (harness commit, starting point) and how it ended.
  - `checks.md` (and `checks.jsonl`): every 5-minute check. `digests/`: what each check was written from.
  - `summary.md`, `issues.md`, `big_issues.md`, `save_states.jsonl`, `investigations/<issue>.md`: the conclusion
    and the work on the issues.
  - `harness/checkpoints.jsonl` and `harness/trajectory.jsonl`: the harness's own record of the run: every point
    it can be forked from, and its intentions, plans, attempts and outcomes.
  - `frames/`: frame sheets from its video.
- `/work/tests/<id>/`: one directory per short test, with `test.json` and its minute-by-minute `checks.jsonl`.
- `/work/iterations/<n>.md`: the iteration summaries.
- `/work/obs/<session>/`: the video of every server session, read-only. `/work/obs/progress/`: the server's
  progress records, read-only.
- `/work/logs/<run>.log`: each run's own log. `/work/agent_logs/cycle.log`: what the controller did.
- `/opt/train/taxonomy.md`: the issue taxonomy.

# Tools

All are run as `/opt/harness/bin/python /opt/train/<tool>.py`; `--help` describes each.

- `record.py <kind>`: hand in the record a step asks for. `--template` prints its shape.
- `digest.py --run <id> [--minutes N | --since T --until T]`: one stretch of a run in reading order: what the
  server measured, what the harness intended, planned, tried and concluded, and frames of the screen, described.
- `look.py --images <png ...> --question "..."`: show images to the model and ask about them. This is how you see
  the screen. What the harness says it did and what the screen shows can differ; when they do, the screen is
  right, and the difference is itself a finding.
- `short_test.py start|wait|verdict`: a short test of a fix, during an issue step.
- `progress.py [--run <id>]`: the server's progress records as a table or a timeline.
- `agnostic_check.py`: names from the training games found in the harness code.

# Version control

`/work/harness` is a git repository and you are on your own branch. Commit every fix on its own, with a message
that says what was wrong, how you fixed it, and what evidence you have. Commit before a short test, so the test
is tied to the code that made it. Undo a change with `git revert`, never by rewriting history. Your commits are
copied out of the sandbox and kept. The repository has no remote: there is nothing to push to or pull from.

Changes may come from outside the sandbox on the branch `upstream`; they are merged into your branch for you.

# Your environment

- Python for the harness and the tools is `/opt/harness/bin/python`. There is no network and no package installs.
- The only model is `$SANDBOX_MODEL`, through the gateway. Every call costs money and the whole run has a fixed
  budget; when it is spent, model calls fail with a spending-cap error.
- `/work/AGENTS.md` (this file) and everything under `/opt/train` are read-only.
- If `/work/STOP` exists, the controller stops at the next step.
