You are at the LARGER ISSUES step for the long playthrough `[[PID]]` ([[GAME]]). The rules in /work/AGENTS.md
always apply. In this step you do not change the harness. This step has [[BUDGET]]. With [[WARNING]] left, the
controller interrupts you once with a time check.

The small issues of this playthrough have been worked through one at a time. Now step back. Read:

- [[DIR]]/summary.md: the whole playthrough in one account.
- [[DIR]]/checks.md: every check.
- [[DIR]]/issues.md: the small issues and what was done about each.
- The previous playthrough: [[PREVIOUS]] (its summary.md, issues.md and big_issues.md), and the iteration
  summaries in /work/iterations/.

Look for the larger problems in how the player behaves: patterns no single small fix addresses. Small issues that
share one cause. A whole kind of situation the harness has no way to handle. A part of its design that keeps
producing new symptoms however often the symptoms are patched. Behaviour that is wasteful or aimless across the
entire run rather than at one moment.

List at most [[MAX]], and none that is only a restatement of one small issue. Write them as JSON to
/work/cycle/drafts/big_issues.json (`/opt/harness/bin/python /opt/train/record.py big_issues --template` prints the
shape): for each, `id` (B1, B2, ...), `description`, `where` (which playthroughs, checks and small issues show
it), `probable_cause` (what in the harness's design produces it), `proposed_change` (the design change that would
address it: real engineering, not a reworded prompt), `present_in_previous` (`"no"`, or the earlier issue it
matches as `<playthrough>:<id>`) and `rank` (1 first).

Hand it in with `/opt/harness/bin/python /opt/train/record.py big_issues`, fix what it reports, and end your turn
when it says ACCEPTED. Each one is then worked on in a step of its own.
