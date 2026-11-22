You are at the CONCLUSION step for the long playthrough `[[PID]]` ([[GAME]]), which has ended. The rules in
/work/AGENTS.md always apply. In this step you do not change the harness; you read the whole record and write
three things.

Read all of it first, starting with the screen review:

- [[DIR]]/screen_review.md: the whole playthrough judged from the screen, episode by episode. The reviewer was
  given only the tasks and subgoals the harness set, not its outcomes or the checks, so it is the account least
  shaped by what the harness believed. It flags each episode where the harness's reported outcome and the screen
  disagree. Its frame sheets are in [[DIR]]/review/. The checks below saw 5 minutes at a time; where they and the
  screen review differ, look at the frames before believing either.
- [[DIR]]/playthrough.json: how it started (harness commit, starting point) and how it ended.
- [[DIR]]/checks.md: every 5-minute check, with its summary, issues and verdict. The digests they were written
  from are in [[DIR]]/digests/.
- [[LOG]]: the run's own log (its end, if it crashed).
- The previous playthrough's issues: [[PREVIOUS]]

More of any stretch: `/opt/harness/bin/python /opt/train/digest.py --run [[PID]] --since <unix time> --until
<unix time>`. To look at frames: `/opt/harness/bin/python /opt/train/look.py --images <png ...> --question "..."`.

**1. Checkpoints to keep** (/work/cycle/drafts/saves.json, then `record.py saves`).
These checkpoints were recorded by the checks:

[[CANDIDATES]]

You may keep at most [[LIMIT]] of them (10%), and should keep fewer when in doubt: every other one is deleted.
The playthrough's final checkpoint is kept for you and does not count. Keep a checkpoint only if starting from it
would let you test something specific: just before an obstacle the player failed at, just before a battle it
lost. For each: `id`, `context` (where the player is and what just happened) and `useful_for`.

**2. Issues** (/work/cycle/drafts/issues.json, then `record.py issues`).
What the player agent struggles with, across the whole playthrough. Merge the same problem seen in several checks
into one issue. For each:

- `id`: I1, I2, ...
- `locus` and `code` from /opt/train/taxonomy.md.
- `description`: what it struggles with.
- `where`: which checks, times and episodes of this playthrough show it.
- `probable_cause`: the likely cause in the harness.
- `present_in_previous`: `"no"`, or the earlier issue it matches, as `<playthrough>:<issue id>`. Check the
  previous playthrough's issues file above, and earlier ones of this game under /work/playthroughs/.
- `likely_gain` and `solvability`: high, medium or low. Gain is how much further the playthroughs would get if
  this were fixed; solvability is how likely a fix is to work.
- `screen_review`: starts with `agrees`, `disagrees` or `not visible` (for a problem inside the harness that the
  screen cannot show), then says what the screen review shows about this issue. An issue the screen review
  contradicts needs strong evidence from the frames to stay.
- `rank`: 1 for the issue to address first, by likely gain and solvability together. Only the top [[TOP]] are
  addressed in this cycle, so rank with care.

**3. The summary** (/work/cycle/drafts/summary.json, then `record.py summary`).
`summary`: one account of everything about this playthrough: where it started, how far it got, what went well,
what stopped it; take how far it got from the screen review. `screen_review`: how your account stands with the
screen review, taking up by number ("episode 4") each episode it flags where the harness and the screen disagree,
and saying whether you filed it as an issue or why not. `expectations`: one entry, in order, for each of these expectations from the last iteration,
saying whether it held (`yes`, `no`, `unclear`) and the evidence:

[[EXPECTATIONS]]

`/opt/harness/bin/python /opt/train/record.py <kind> --template` prints each shape. Hand in all three with
`/opt/harness/bin/python /opt/train/record.py saves|issues|summary`, fix what each reports, and end your turn when
all three say ACCEPTED.
