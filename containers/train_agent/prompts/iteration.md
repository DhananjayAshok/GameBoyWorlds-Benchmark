You are at the ITERATION SUMMARY step of cycle [[CYCLE]], after the long playthrough `[[PID]]` ([[GAME]]) and the
work on its issues. The rules in /work/AGENTS.md always apply. In this step you do not change the harness.

Read:

- [[DIR]]/summary.md, [[DIR]]/issues.md and [[DIR]]/big_issues.md: the playthrough, its issues, and what was done
  about each.
- The investigations in [[DIR]]/investigations/.
- The short tests of this cycle under [[TESTS]] (each has test.json and checks.jsonl).
- `git log` in /work/harness for the commits themselves.

Issues fixed in this cycle, with their commits:

[[FIXED]]

Issues deferred:

[[DEFERRED]]

Write /work/cycle/drafts/iteration.json (`/opt/harness/bin/python /opt/train/record.py iteration --template`
prints the shape):

- `summary`: what this iteration changed and why, and what the short tests showed.
- `changes`: every fix commit, with the `issue` it belongs to and `what` it changes.
- `deferred`: every deferred issue, with `why`.
- `expectations`: what you hope to see in the next playthroughs because of these changes. Each is one concrete
  thing that the checks can judge from the record: not "plays better" but "gets through a doorway it is standing
  next to within one episode". Give each a `game`: the next playthrough is `[[NEXT_GAME]]`, and the one after it
  is the other game, so write expectations for each, or `"both"` when one applies to either. They are shown to
  every check of those playthroughs, and their conclusions say whether each one held.

Hand it in with `/opt/harness/bin/python /opt/train/record.py iteration`, fix what it reports, and end your turn
when it says ACCEPTED.
