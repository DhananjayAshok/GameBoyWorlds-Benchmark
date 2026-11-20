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
