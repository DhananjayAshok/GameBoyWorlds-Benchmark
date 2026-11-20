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
