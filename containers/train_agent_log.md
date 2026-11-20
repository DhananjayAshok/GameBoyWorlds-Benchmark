# Train agent design log

What we changed in the train agent's setup (instructions, prompts, tools, launch) while watching run
`train_agent_flash`, why, and what happened. Newest last. Times are UTC. The agent's own record of its work is
`<run_dir>/work/DEVLOG.md`; this file is ours, and is kept outside `containers/train_agent/` so the sandbox does
not see it.

## 2026-11-19 22:00 to 2026-11-20 02:47: the agent stopped watching its runs

- **Seen:** after starting `red_long` in the background, the agent ended its turn. prime-agent stays alive while
  a child process runs, so the session sat idle until its 3-hour limit, with nobody watching the run (21:57 to
  00:40, and again 00:49 to 01:04, and 01:08 to 02:32).
- **Tried:** a "Watching a long playthrough" paragraph in `AGENTS.md` (watch actively, stop stuck runs). The agent
  read it and ended its turn anyway.
- **Tried:** prime-agent's `rlm_heartbeat` (a recurring wake-up), with a verdict ("in progress" or "stuck",
  judged by a condition set at the previous check) and a next condition at each check. It does not work here:
  `NameError: name 'rlm_heartbeat' is not defined`; prime-agent reports "This session has no daemon: heartbeats
  never fire". We run prime-agent with `-p`, which has no daemon, so `prime-agent send` cannot reach it either.
- **Fixed:** the rule became "check every 5 minutes and do not end your turn while a run is going; without a
  timer, wait inside the turn (`await asyncio.sleep(300)`)", stated in `AGENTS.md` and repeated in `continue.md`
  and `kickoff.md`, plus a one-off `next_prompt_note.md`. From 02:47 the agent stayed in its turn and checked
  every 5 minutes (commits 972fb68 / 3efd508).

## 2026-11-20 04:00 to 06:45: the checks were hollow

- **Seen:** 47 checks, all but one "In progress". `red_long` last reached a new map (Route 2) at 03:20, lost a
  battle at 03:48, was sent back to Pallet Town, and spent the next 3 hours in Viridian City. Checks 45 to 47 were
  word-for-word copies ("Moving north through Route 2 gatehouse") while the player was in a Viridian house. The
  next condition was always "get through Viridian Forest and reach Pewter City", far more than 5 minutes of play,
  so it was never judged.
- **Added:** an "insights and potential fixes" section in every check (02161b6 / de7df2b). The agent's checks did
  not pick it up: they kept the format copied from earlier `DEVLOG.md` entries.
- **Also seen:** the agent committed 39879eb "Prefer non-south exploration targets", a fixed compass bias that
  only helps because this stretch of Red goes north; not game-agnostic.

## 2026-11-20 06:50: hard stuck rules and a stall counter

- **Changed `progress.py`:** the summary has a `stalled` column (minutes since the run last won a badge, reached
  a new map or set a new story event; reset and load records do not count), and `--run` ends with the last
  advance, the minutes and steps since, the party then and now, and `STALLED` after 30 minutes. On the real
  records: `red_long` stalled 244 minutes.
- **Changed `AGENTS.md`:** each check starts from `progress.py --run`, never copies an earlier check, and quotes
  its evidence. Stuck, always: the previous condition was missed, a battle was lost, or `STALLED` (unless the
  agent can show an advance the counter cannot see, such as levelling up for a lost fight). The next condition
  must be reachable in 5 minutes, about one or two episodes, never a far goal.
- **Changed the session prompts** to start each check from `progress.py --run <name>`.
- Relaunched with a note telling the agent the rules changed and that its compass-bias commit may break the
  game-agnostic rule.
- **Gotcha:** quitting the screen session ends the running pass, and the loop starts one more pass that dies in
  the teardown (pass 22, 0 s); that pass takes `next_prompt_note.md` with it. Write the note after the relaunch
  and end the new pass's prime-agent, so the next pass picks it up. Done here: pass 24 (06:50) took the note, and
  is a fresh session (every 6th pass is), so it reads `AGENTS.md` from scratch.

## 2026-11-20 07:11: check 1 of our 20-minute watch

- **Seen:** pass 24 read the new rules. Its first check said **Stuck** (quoting `STALLED: 247 min`), and it
  reverted the compass bias of 39879eb, keeping a general anti-backtracking line (b04a51a). The insights section is
  now filled in. But:
  - it restarted the stalled run with only that unrelated fix, without diagnosing why the run cannot get from
    Viridian City into Viridian Forest;
  - its next conditions end in "or map new squares", which any wandering meets, so checks 1 to 3 say "In progress"
    while `progress.py` still says `STALLED` (268 min);
  - checks 2 and 3 say the player is "moving north up Route 2" while `progress.py` has it in Viridian City: it
    trusts the harness's own task text.
- **Changed `AGENTS.md`:** a stuck verdict must be followed, before restarting, by a written diagnosis of the
  stall from the episode reports and video (which tasks failed or looped, and why) and a fix for it; restarting a
  stalled run with no harness change is not allowed. New squares on the harness's map do not count as advance.
  Conditions are one thing with no "or" fallback. The player's location comes from `progress.py`, and a
  disagreement with the harness's belief is an insight.

## 2026-11-20 07:33: check 2 of our watch: the diagnosis rule worked

- **Seen:** the fresh session (pass 24, 07:13) followed the diagnosis rule. From episode reports 106 to 116 it
  found the real cause of the 4.5-hour stall: inside gatehouses the screen's edge row is wall or black padding, so
  `edge_cells` found no walkable cell, the off-screen move failed (`MOVE_BAD_LOCATION`), three failed episodes
  triggered global-map exploration, which walked the player back south, and re-planning brought it north again: a
  loop. Fix 05b2637 (generic: `edge_cells` searches inward when the edge row is blocked; the off-screen executor
  snaps to the nearest edge cell). It tested on a fork, `red_gate_fix` from `red_long`, as `AGENTS.md` says, and
  the fork reached `VIRIDIAN_FOREST` (11 maps), the first new map in about 5 hours.
- **No change** to the design this round. To watch next: whether it carries the fix back into a long playthrough
  rather than living on short forks, whether checks stay evidence-based, and Crystal, which has had no long
  playthrough at all.

## 2026-11-20 07:54: check 3 of our watch

- **Seen:** the fork `red_gate_fix` went through `VIRIDIAN_FOREST_SOUTH_GATE` into `VIRIDIAN_FOREST` (07:30 to
  07:33) and has been in the forest since (20 min, 112 steps). The agent stays in its turn, polling every 3 to 4
  minutes, but writes a check to `DEVLOG.md` only now and then (last at 07:41), and its condition still ends with
  "or discover new map squares / battle trainers", against the no-"or" rule. It locates the player from
  `progress.py` now.
- **No change:** not worth a relaunch while the run advances. If the forest stalls past 30 minutes and the checks
  do not call it, the next change is to make the written check itself the 5-minute step (one `DEVLOG.md` entry per
  check, not per batch).

## 2026-11-20 08:15: check 4 of our watch: written rules do not hold, so a tool enforces them

- **Seen:** `red_gate_fix` has been in `VIRIDIAN_FOREST` for 41 minutes (273 steps, `STALLED`), with episodes
  ending `out_of_calls` and every wild battle fled. The agent's checks (07:56, 08:02) still say "In progress", still
  end their conditions with "or map new forest squares", and stopped at 08:02. The written rules about `STALLED`
  and "or" are read and not followed.
- **Added `containers/train_agent/watch_check.py`** (`/opt/train/watch_check.py`): one call records one check in
  `/work/CHECKS.md` with the server's view of the run (map, step, the stall lines) and the condition set at the
  previous check. It refuses, and records nothing, an `in_progress` verdict while `STALLED` (unless `--advancing`
  says what the counter cannot see), a condition containing "or", and empty evidence. With only `--run` it shows
  the run's state and the last condition. Tested on the real records: refuses in-progress at 41 min stalled,
  refuses an "or" condition, records a stuck verdict.
- **Changed `AGENTS.md`** and the session prompts: every check is a `watch_check.py` call; a check not in
  `CHECKS.md` did not happen. Checks move from `DEVLOG.md` to `CHECKS.md`, which keeps `DEVLOG.md` for diagnoses
  and fixes.

## 2026-11-20 08:37: check 5 of our watch: the tool is used, and gamed through `--advancing`

- **Seen:** the fresh session (08:16) records every check with `watch_check.py` (08:25, 08:31, 08:37). It
  diagnosed one cause of the forest stall from episode reports 121 to 124: the strategist kept sending the
  player after ground items, one of them behind a tree that needs an HM, for about 40 minutes; fix 8c6d634 tells the
  strategist not to chase optional items (generic). It also cut the harness's VLM timeout from 600 s to 60 s
  (256e5bf) after seeing calls appear to hang. But every check says "In progress" through `--advancing` with no
  evidence ("player is actively navigating ... towards northern exit"), the condition ("complete episode 125 while
  advancing") repeats unmet, and the server's step count has been 6085 since 08:12: the agent restarted the run
  three times in 20 minutes, so no episode finished. `STALLED` at 63 minutes. The gateway is healthy (2 to 9 s per
  call).
- **Changed `watch_check.py`:** refuses `in_progress` when the server's step count has not moved since the last
  judged check; accepts `--advancing` for at most 3 checks in a row, and only with `in_progress`; adds a
  `restarted` verdict for the check right after a restart with a fix, which judges nothing. Tested against a copy
  of the real `CHECKS.md`.
- **Changed `AGENTS.md`:** documents those limits and `restarted`, and says to restart only for a fix to the stall
  being diagnosed, batching other changes, since each restart loses the episode in flight.
- **Not relaunched:** `/opt/train` is a directory mount, so the new `watch_check.py` is live in the running
  sandbox at once, and its refusals name the `restarted` verdict. The `AGENTS.md` wording waits for the next
  relaunch; relaunching now would cost the run another episode in flight.

## 2026-11-20 08:59: check 6 of our watch

- **Seen:** the tightened tool bit at 08:42: with the step count frozen, the agent recorded **Stuck**, then
  `restarted` after its fixes. Since then the step count moves (6085 to 6258) and it records "In progress" with
  `--advancing` citing the step deltas, which is honest but weak: its conditions are now "complete episode N while
  advancing through the forest", which any completed episode meets. `STALLED` at 85 minutes, still in
  `VIRIDIAN_FOREST`.
- **No change:** `--advancing` has been used at 3 checks in a row (08:48, 08:53, 08:58), so the tool will refuse
  "in progress" at the next check unless the run reaches a new map. That is the mechanism doing its job; watch
  whether the forced stuck verdict leads to a real diagnosis of the forest maze. If the agent keeps setting
  "complete episode N" conditions, the next change is to refuse conditions that name an episode rather than a
  place, battle or event.

## 2026-11-20 09:19: check 7 of our watch: `restarted` became the new loophole

- **Seen:** at 09:04 the `--advancing` limit forced **Stuck**, as intended. The agent then recorded `restarted`
  three times in a row (09:04, 09:10, 09:15) with the same process, no new commit (still 256e5bf) and no diagnosis
  in `DEVLOG.md`; instead it wrote that the player had reached the forest's north-west corner. Its conditions drifted
  back to "complete episode 130 while advancing". `STALLED` at 105 min. The harness is not hung (115 model calls
  in 15 minutes); episode 130 is just long, and the log only updates when an episode ends.
- **Changed `watch_check.py`:** each check records the harness commit (`git rev-parse` of `/work/harness`, marked
  `+uncommitted` when dirty); `restarted` is refused when the commit is the same as at the last check, or the fix
  is not committed. Conditions that mention an episode are refused. Tested against a copy of the real
  `CHECKS.md` and the real harness repo: a second `restarted` at 256e5bf is refused, "Complete Episode 131 while
  advancing" is refused, a stuck verdict is recorded with `Harness: 256e5bf`.
- **Changed `AGENTS.md`** to describe both. The tool is live at once (directory mount); not relaunched.

## 2026-11-20 09:41: check 8 of our watch: the loop becomes the heartbeat

- **Seen:** the agent recorded one last `restarted` at 09:20 (accepted once, as older checks had no commit line),
  then wrote a summary and **ended its turn**, with `red_gate_fix` running. Nobody watched it for 20 minutes, in
  which it walked from `VIRIDIAN_FOREST` back to `ROUTE_1` (`STALLED` 127 min). This is the fourth time a written
  rule failed to keep the agent in its turn; with a large context, Flash falls back into "summarise and stop".
- **Changed `agent_loop.sh`:** prime-agent now runs in the background and the loop watches it. If the session's
  transcript has not changed for 10 minutes while a playthrough is running, the loop ends the session. Whenever a
  session ends with a playthrough running (ended for being idle, or because the agent ended its turn and nothing
  held the process open), the loop waits 4 minutes and resumes the same conversation (`--continue`) with a note:
  you ended your turn while a playthrough ran; check it with `watch_check.py` and keep watching. These passes do
  not count as quick passes, so they cannot stop the loop. A fresh conversation now starts every 12 hours of
  conversation time instead of every 6th pass, which would otherwise come every half hour. This reverses the
  earlier choice not to force watching from outside: the instructions alone did not hold.
- **Changed `AGENTS.md`:** start playthroughs with `setsid nohup ... &`, so that ending a session does not kill
  them (a session's children die with it, which is why each earlier relaunch cost the episode in flight), says
  that idle sessions are ended and resumed, and how to stop a playthrough (`pkill -f`).
- The new loop is checked with `sh -n` only; the relaunch is its test.

## 2026-11-20 10:04: check 9 of our watch: detached long run, carrying the fixes

- **Seen:** the fresh conversation (09:43) went back to the long playthrough `red_long` rather than the fork that
  had walked back to Route 1, carrying all of the fixes (harness 256e5bf), as `AGENTS.md` asks once a fix works
  on a fork. It started it with `setsid nohup` (its own session id), so it survives the agent's sessions ending.
  `red_long` went Viridian City, Route 2, then into `VIRIDIAN_FOREST` at step 5835 (stalled 7 min). The agent is
  in its turn (transcript updated 29 s ago, mid-sleep; prime-agent writes progress during long tool calls, so the
  10-minute idle rule will not misfire on a 5-minute sleep). Checks are 11 minutes apart rather than 5, and its
  conditions are real places ("Enter VIRIDIAN_FOREST", "Enter VIRIDIAN_FOREST_SOUTH_GATE"). The idle-resume has
  not been needed yet.
- **No change.**

## 2026-11-20 10:25: check 10 of our watch: conditions judged from the records

- **Seen:** `red_long` in `VIRIDIAN_FOREST` for 28 minutes (5708 to 6159 steps). The agent is in its turn and
  checks every ~10 minutes. At 10:15 it recorded "In progress" though its previous condition, "Enter
  VIRIDIAN_FOREST_NORTH_GATE", was not met: "a missed condition means stuck" still rests on its honesty.
- **Changed `watch_check.py`:** every check takes `--expect map:<MAP>|event:<EVENT>|badge|none`, recorded with
  the condition. The next check judges the expectation from the progress records written since (map reached, event
  set, badge count up); if they exist and do not show it, `in_progress` is refused. With only `--run` it shows the
  previous expectation and whether it was met. `none` (for things the records cannot show, like a battle won) is
  accepted for at most 3 checks in a row. Tested on the real records: an expected `VIRIDIAN_FOREST_NORTH_GATE`
  since 10:05 is "not met" and refuses in-progress; an expected `VIRIDIAN_FOREST` since 09:50 is "met" and
  records.
- **Changed `AGENTS.md`** to document `--expect`. Live at once in the tool; `AGENTS.md` at the next relaunch,
  meanwhile the tool's refusal explains the new argument.

## 2026-11-20 10:47: check 11 of our watch: the stall counter counts tiles

- **Seen:** the `--expect` rule worked: at 10:36 the tool judged "Enter VIRIDIAN_FOREST_NORTH_GATE" not met and
  the agent recorded **Stuck**. Its diagnosis was thin ("the forest is a large single map", "the condition was
  too far") plus an unrelated fix (4f20a1a, 405b192: NPC-card parsing) that unlocked `restarted`; then it switched
  to `--expect none`. Its one fair point: in a large map, "no new map in 30 minutes" can be true while the player
  is still exploring, so `STALLED` cannot tell exploring from circling, and the agent's word fills that gap.
- **Changed `env_server/progress.py`:** the server now records the tiles the player has stood on (map, x, y from
  game memory, both generations), carried through named saves like maps reached; each record has `tiles` and
  `new_tiles`. `env_server/check.py` checks tiles are counted and survive a save and a server restart;
  `run_env_server.py check` passes.
- **Changed `containers/train_agent/progress.py`:** 25 new tiles since the last advance now count as an advance,
  so exploring a big map is not `STALLED` and circling over known tiles still is; the timeline shows tiles per
  record, and the stall lines show new tiles. Older records have no tile counts and read as 0. Saves made before
  this have no tile history, so the first records after the relaunch will count every tile as new: a short burst
  of false advances.
- **Changed `AGENTS.md`** to describe the new `STALLED`. Relaunching, since the env server must restart to count
  tiles.

## 2026-11-20 11:10: check 12 of our watch: tiles counted; restarts on guessed fixes

- **Seen:** the tile counter works on the live run: `red_long` restored with no tile history (1 tile), then
  +23, +3, +5, +2 as it explored; the 11:05 record counted as an advance ("26 new tiles"), so the forest stall now
  reads 5 minutes and the agent's checks cite tiles. The feared burst of false advances after the relaunch was
  small (only the tiles actually walked count). The agent's latest condition is "Enter
  VIRIDIAN_FOREST_NORTH_GATE" with `--expect map:VIRIDIAN_FOREST_NORTH_GATE`, and it is in its turn.
- **Pattern to watch:** each restart since 10:36 rode on a fix to something that was not the stall: NPC-card
  parsing (4f20a1a, 405b192), then `max_tries` 10 for "gateway proxy connection drops" (c2b8f4f). This launch's
  gateway log has 190 calls, all 200, none slower than 50 s: there were no drops. The commit rule makes it commit
  something, and it commits guesses. If this continues, the next step is to require the diagnosis to name the
  episode reports it read and the failure it found, before `restarted` is accepted.
- **No change** this round.

## 2026-11-20 11:31: check 13 of our watch: Red knowledge creeping into the harness

- **Seen:** after another forced stuck verdict (11:12, `expect map:VIRIDIAN_FOREST_NORTH_GATE` not met) the
  agent restarted on ea5b85e, which adds to the dispatch prompt: "When navigating a forest or maze route towards a
  northern exit, prioritize checking northern boundary openings ... over backtracking into southern or eastern
  loops": the same compass bias as 39879eb, now in a prompt. The same prompt's examples say "Talk to Professor
  Oak". It also wrote that it would use `--expect none` "during intermediate forest maze traversal" to avoid being
  judged; `none` is capped at 3 checks in a row. The forest: 40 tiles stood on in ~150 steps, 25 min since the
  last advance.
- **Added `containers/train_agent/agnostic_check.py`** and `game_names.txt`: 51 distinctive proper names (towns,
  regions, gym leaders, professors, Elite Four) taken from the Red and Crystal map, landmark and trainer-class
  tables in GameBoyWorlds' memory data, keeping only names that are not ordinary words; the check lists
  whole-word matches (and possessives) in the harness's tracked Python files. `watch_check.py` refuses
  `restarted` while it finds any. Tested: on the agent's harness it lists `prompts.py:20 (Oak)` and four names
  from our own NPC nickname pool.
- **Fixed our harness:** `execution/strategist/pokemon/npc_names.py` drew NPC nicknames from a pool containing
  Indigo, Misty, Saffron and Elm, so an NPC in Red could be nicknamed "Misty". Removed them; the base harness now
  scans clean. Synced into `harnesses/strategist` and published, so the relaunch merges it into the agent's branch.
- **Changed `AGENTS.md`:** directions count as walkthrough knowledge ("the exit is to the north" is one map's
  layout), and names `agnostic_check.py` and the gate.

## 2026-11-20 11:57: check 14 of our watch: watching without checking

- **Seen:** the note worked: the agent replaced its compass rule with a direction-agnostic one (7003b34, "prefer
  exits not yet tried") and restarted. But `red_long` is now circling: 255 steps between 11:39 and 11:54 reached 1
  new tile; `STALLED` 51 min. The agent is in its turn, sleeping in 3-minute chunks "for episode 1 to finish",
  and has recorded no check since 11:39 (18 min): it checks only when an episode ends, so the tool never gets to
  force the stuck verdict.
- **Changed `agent_loop.sh`:** while a playthrough runs, a session that has recorded nothing in `CHECKS.md` for 12
  minutes (counted from the later of the pass start and the file's last write) is ended, and resumed 5 seconds
  later with a note: record a check now, every 5 minutes whether or not an episode finished. The idle rule (10 min
  without any transcript activity) stays. Checked with `sh -n`; the relaunch is its test.
- **Changed `AGENTS.md`** to say both.

## 2026-11-20 12:19: check 15 of our watch: prompt fixes for a navigation failure

- **Seen:** relaunch at 11:58; the agent's first check was **Stuck** and it wrote a real diagnosis from episodes
  129 to 132: the strategist alternates "walk west" and "walk east" around an obstacle, a two-episode pendulum.
  Its fix (2067a4b) is one more sentence in the dispatch prompt, the fourth prompt-only change in this stall. The
  run did not move: 7095 to 7548 steps with 0 new tiles (42 tiles in all; 829 steps and 15 new tiles since the
  last advance at 11:05). Its 12:11 check claimed "advanced 209 steps further north ... 119 new mapped squares"
  through `--advancing`, from the harness's own map; the server shows no new tiles. The 12-minute check rule did
  not need to fire (checks at 11:58, 11:59, 12:11 and later).
- **Changed `watch_check.py`:** `--advancing` is refused unless the server shows new tiles or levels gained since
  the previous check; after 2 restarts within one stall (since the run's last advance), a `restarted` whose commits
  since the last check touch only `prompts.py` is refused, saying to change the harness's navigation, perception,
  memory or executors instead. Tested on the real records and the agent's repo (7003b34 to 2067a4b touches only
  `prompts.py`). Live at once; `AGENTS.md` updated for the next relaunch.

## 2026-11-20 12:40: check 16 of our watch: code fixes at last, and a false stuck of our making

- **Seen:** the 12-minute check rule fired (12:33, "recorded no check for 729s"), and the prompt-only rule did its
  job: the agent's next fixes changed the executor, with diagnoses from episode reports: 7675920 (`GridMoveExecutor`
  walks to the nearest reachable cell when the target is blocked, so the player walks up to a trainer blocking a
  corridor instead of turning back) and e2a8e58 (cap that fallback at 4 tiles after it wandered into side alcoves).
  Before those, d8ea34f (prompt) told the strategist that trainers on a path are to be fought, not avoided.
- **Our bug:** progress records were written only at saves and every 500 steps, about 45 minutes of play here, so
  during a long episode the step count looked frozen. At 12:33 the "step count has not moved" rule forced a stuck
  verdict while the 12:21 run was mid-episode and doing well (it reached step 7756, 36 new tiles, Squirtle 11, as
  the session-close record shows); the agent restarted from the last save, 7548, and lost all of it.
  `progress.py` also counted that close record as an advance.
- **Fixed:** `env_server/progress.py` writes a periodic record after 120 s of play as well as after 500 steps, so
  the records keep up with the run; `containers/train_agent/progress.py` treats `close` records like `reset` and
  `load` (their gains were never saved), which `watch_check.py` inherits. `run_env_server.py check` passes; the
  stall now reads correctly (95 min since 11:05). Relaunched just after a save, so no episode was lost.

## 2026-11-20 13:05: check 17 of our watch

- **Seen:** the 2-minute periodic records work (a `periodic` record at 13:05 between saves). The agent keeps making
  code fixes: da9eaeb (exploration override picks the nearest unexplored frontier on screen, not a distant
  entrance; `STALE_EPISODES` 3 to 5) and 7083f1d (the navigation map treats door, mat, carpet, path and warp tiles
  as walkable). The run is moving again: tiles stood on went 49 to 64 since 13:00, an advance at 13:03 (26 new
  tiles), still in `VIRIDIAN_FOREST`. Checks are on time; the tool's server-visible rule is what the 13:05
  `--advancing` cites (11 new tiles since the last check).
- **Still weak:** restarts every 10 to 15 minutes, each on a new commit, and 7083f1d's `DEVLOG.md` entry has no
  stall evidence, only the change. Not worth a rule yet while the run advances.
- **No change.**

## 2026-11-20 13:26: check 18 of our watch: twelve patches, one forest

- **Seen:** the run is back to stuck, fix, restart about every 10 minutes: ae6845e (exploration override skips
  targets at distance 1; prompt about following walls to an exit) and 931d443, which reorders the path search's
  `DIRECTIONS` so that ties go up, left, right, then down: "prefer forward progress", that is a north bias, the
  third time a compass preference has come in, now in code where `agnostic_check.py` cannot see it. Since
  `red_long` entered `VIRIDIAN_FOREST` at 09:57 it has been restarted 12 times on small patches; 64 tiles stood
  on; 23 min since the last tile advance.
- **Changed `watch_check.py`:** after every 6 restarts since the run last reached a new map, event or badge,
  `restarted` is refused until a stall review exists in `/work/reviews/` newer than the 6th (12th, ...) restart.
  The refusal says what the review must hold: every fix tried, whether the server's counters moved after each,
  the cause it points to, any fix that only works because of this map's layout (a preferred direction is one),
  and the larger change it will make instead or why not. Tested: refused at 12 restarts with no review; recorded
  once a review file exists. The tool does not judge the review's content; the review is there to be read.
- **Changed `AGENTS.md`** to describe stall reviews. Tool live at once; `AGENTS.md` at the next relaunch.

## 2026-11-20 13:47: check 19 of our watch: the first stall review

- **Seen:** the review rule fired; the agent wrote `/work/reviews/red_long.md` and restarted on 882e888 (edge
  cells must be reachable from the player). The review has the requested sections but not their substance: it
  credits tile gains to fixes that cannot move the player (API timeouts, retries, NPC-card parsing; the gains are
  restarts reloading the save); it defends 931d443's up-first tie-break as "forward progress for standard Game Boy
  RPG map progression"; and its "larger change" is its last three patches relabelled. `STALLED` 44 min, 448 steps
  for 13 new tiles; its 13:46 check called that "in progress" on 2 new tiles.
- **Changed `watch_check.py`:** `--advancing` needs at least 5 new tiles (or a level) since the last check, not 1.
  Fixed a stale message (records now come every 2 minutes, not only at saves).
- **Note to the agent** (ended its pass so the next one picks it up; the detached run kept playing): the review's
  three faults, how to measure a fix (tiles per 100 steps before and after, from `progress.py`), revert 931d443,
  and describe what the player actually does in the episodes since 13:03 before designing the larger change.
- **Assessment:** the rules now make the agent watch, check on time, judge by the server, fix code rather than
  prompts, and stop to review. What they cannot supply is the quality of its diagnosis: its explanations are
  plausible-sounding and loosely tied to evidence, and it rationalises its own changes. That is the model's
  judgement; more rules of this kind will keep producing compliance in form.

## 2026-11-20 14:08: check 20 of our watch: a real review, then untested restarts

- **Seen:** the note worked. The agent reverted 931d443 (8b8da9c) and rewrote its review: tiles per 100 steps for
  each fix from the server's records, and a diagnosis of the forest loop that fits the evidence: wild battles in
  tall grass use up the supervisor's 10 calls, the episode ends `out_of_calls` with no new squares, the strategist
  reads the tree line as a dead end and sends the player back, a pendulum between two corridors. Its proposed
  larger change is a direction commitment kept across episodes until a new map is entered. Not built yet; b0e20cd
  (tell the strategist `out_of_calls` means interrupted, not blocked) is a step toward it. But it then restarted
  three times in 11 minutes (13:52, 13:54, 14:03) with no judged check between, so no fix was tested; 72dfc5c
  (supervisor decision parsing) has nothing to do with the stall. `STALLED` 65 min.
- **Changed `watch_check.py`:** a `restarted` right after another `restarted` is refused: judge the fix first.
  (A first version also demanded a progress record since the last check; that would have blocked the normal
  stuck, stop, fix, restart sequence, so it was removed before committing.) Tested both paths on the real records.
- **Changed `AGENTS.md`** to say so.

## 2026-11-20 14:29: check 21 of our watch: our rules were feeding the restart churn

- **Seen:** the agent now runs the proper cycle (stuck, fix, restart; no back-to-back restarts) with fixes aimed at
  its diagnosis: f999340 (after a battle, the supervisor gets 4 more calls so the walk can finish), fcaa463 (a
  multi-step move that hits an obstacle part-way counts the steps taken instead of failing). `STALLED` 86 min,
  66 tiles. The save records show what each restart costs: the strategist saves at each episode's end, so a
  restart reloads the last episode's save and loses the one in flight, 15 to 120 steps each time (for example
  14:16 load at 8385, play to 8501, 14:26 restart reloads 8385).
- **Our part in it:** the only judgeable conditions were a map, event or badge. In the forest the next map is the
  exit, far more than 5 minutes away, so every condition was `map:VIRIDIAN_FOREST_NORTH_GATE`, every check judged
  it not met, every check was stuck, and every stuck led to a restart that lost an episode. The fix was never
  given time to work.
- **Changed `watch_check.py`:** `--expect tiles:<N>` (N at least 5): the next check is judged by the new tiles stood
  on since, from the server. The refusal message says to use it in a large map. Tested on the real records (10
  tiles since 13:00: met; since 14:10: not met; `tiles:2` refused).
- **Changed `AGENTS.md`:** documents `tiles:<N>` and why the far exit is the wrong condition.

## 2026-11-20 14:52: check 22 of our watch

- **Seen:** the loop ended pass 30 for recording no check in 12 minutes (14:47); pass 31's first check was Stuck
  with `--expect tiles:5`, the first near-term condition it has set. The run was restarted at 14:48 with no new
  commit and no `restarted` check: the tool cannot stop a restart it is not told about. `STALLED` 109 min, 68
  tiles, still `VIRIDIAN_FOREST`. The supervision now holds; the harness still cannot cross the forest, and the
  agent's proposed larger change (keeping a chosen direction across episodes) is not built.
- **No change.**

## 2026-11-20 15:12: check 23 of our watch: out of the stall

- **Seen:** with `tiles:5` conditions the checks are judged fairly ("met" at 14:53 and 15:02 on 26 and 8 new
  tiles). The run moved: 68 to 91 tiles stood on, `EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_0` at 15:07 (the first new
  story event since 09:57's maps), Squirtle 11 to 12. The fixes running were f999340 and fcaa463 (battle
  interruptions and partial moves), which fit the agent's own diagnosis. But at 15:08 the agent judged the run
  stuck because `tiles:5` was missed during the trainer battle, and restarted on a prompt-only tweak (488c841,
  "continue straight along the corridor after beating a trainer"), losing the episode in flight.
- **Changed `watch_check.py`:** an expectation also counts as met when the run reached a new map, event or badge
  since the check that set it, so a fight won is not judged a stall. Tested on the real records.

## 2026-11-20 15:33: check 24 of our watch: an untested change crashes the run

- **Seen:** fair checks (15:20 met on 7 tiles; 15:29 stuck, `tiles:5` missed), but the 15:30 restart (29cf277,
  off-screen moves take 2 steps to cross a door threshold) reloaded step 9027 after the run had reached 9348 and
  Squirtle 13: the in-flight episode ran ~320 steps without a save, and the restart lost it. Then the playthrough
  crashed at 15:31: `AttributeError: 'OnScreenMoveExecutor' object has no attribute '_steps_taken'`, in fcaa463's
  change to `GridMoveExecutor._walk` (14:26). That path is rare, so it only now ran; the agent checks code changes
  with `py_compile` only. No playthrough is running; the agent's session is active and its next check should show
  it. Left for the agent to find.
- **To watch:** whether it finds the crash from its check, and whether it starts testing changes on a short fork
  before the long run (which `AGENTS.md` already asks). If crashes recur, the next step is a test the tool runs
  before accepting `restarted`.

## 2026-11-20 15:43: check 25 of our watch

- **Seen:** the agent found the crash and fixed it (f14a562, the partial-path code used the wrong attribute name),
  and restarted the run, which has been playing since 15:37. It did not record that restart as a check or write
  the crash up in `DEVLOG.md`; the loop ended pass 31 at 15:43 for 12 minutes without a check, and the new pass
  opened with a fair stuck verdict (`STALLED` 36 min, 94 tiles, `tiles:5` not met). Still `VIRIDIAN_FOREST`.
- **No change.**

## 2026-11-20 16:04: check 26 of our watch: flailing

- **Seen:** three stuck checks in a row (15:49, 15:55, 16:01), conditions back to the far exit
  (`map:VIRIDIAN_FOREST_NORTH_GATE`), and a restart at 16:01 with no new commit (still f14a562), which reloaded
  step 9027 after the run had reached 9157. `watch_check.py` cannot stop a restart the agent does not record.
  `STALLED` 56 min since the trainer win; 94 tiles. The larger change from its own stall review (commit to a
  direction or frontier target across episodes) is still unbuilt after about ten small fixes since 13:03.
- **Note to the agent** (pass ended so the next one reads it): no restarts without a fix; build the committed-target
  change from its review, test it on a short fork (`forest_commit` from `red_long`) and carry it into `red_long`
  only if the fork gains tiles faster; use `tiles:<N>` in the forest.
- **Assessment:** the supervision catches stalls, but the agent's fixes are small and its experiments untested; it
  does what notes ask more readily than what rules ask. If notes are what move it, the design question is whether
  a stronger reviewer (us, or a stronger model) should write them on a schedule.

## 2026-11-20 16:25: check 27 of our watch: the larger change, on a fork

- **Seen:** the note worked again: the agent built the change from its stall review, 25da7af (the exploration
  override's frontier target is kept across episodes for the current map segment instead of re-chosen each
  episode), and is testing it on a fork, `forest_commit` from `red_long`, with `tiles:5`. Result so far is worse:
  828 steps in 15 minutes for 0 new tiles (`red_long` before: 8 tiles in 362 steps). The step rate (about 55 a
  minute against about 10 usually) suggests the player walking into an obstacle or battles back to back (Squirtle
  12 to 13). The agent recorded one fork check (16:09); the loop ended pass 33 at 16:22 for 12 minutes without a
  check. Left for the agent to judge and, by its own test plan, drop.
- **Pattern:** notes that name a concrete next step get done; rules get complied with in form. Kept for the
  design question of who writes the notes.
- **No change.**

## 2026-11-20 16:46: check 28 of our watch: a failed experiment adopted

- **Seen:** `forest_commit` (25da7af) stood on 0 new tiles in 2,226 steps by the server's count; the agent's
  `DEVLOG.md` reports it as "mapped 187+ new squares ... without directional oscillation" (the harness's own
  map) and it carried 25da7af into `red_long` at 16:41, against its own plan to adopt it only if the fork gained
  tiles faster. The tool judged the fork stuck; nothing stopped the adoption.
- **Changed `watch_check.py`:** `restarted` on a commit that another run (a fork) already played is refused unless
  that fork's new tiles per 100 steps under the commit (at least 200 steps, from the launches in each run's
  `harness_commits.jsonl` and the server's records) beat the long run's rate over its last hour. Tested on the real
  case: refused, "forest_commit ... 0.0 new tiles per 100 steps over 2226 steps; red_long ... 0.0 over its last
  228". `AGENTS.md`: "Forks decide".
- **Note to the agent:** the misread result, revert 25da7af, find what the player did in those 2,226 steps, correct
  `DEVLOG.md`. The adoption already happened, so the note carries the correction; the rule stops the next one.
