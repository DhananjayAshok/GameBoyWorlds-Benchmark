You are at the DEEP DIVE step for one issue from the long playthrough `[[PID]]` ([[GAME]]). The rules in
/work/AGENTS.md always apply: the fix must be game-agnostic and vision-only. Work on this issue only.

```json
[[ISSUE]]
```

[[EARLIER]]
**1. Confirm the failure in the records, before anything else.** The issue was written by a check or the
conclusion, from a 5-minute window, and it can be wrong: in an earlier cycle a check filed "the harness froze"
while the server's records showed it taking steps the whole time, and the fix for that hang changed working code.
So go back to what was recorded while the game was played, not the checks' account of it:

- The server's records of the game (step count, map, position, grid spots), the harness's trajectory and its run
  log, and the frames, for the stretch the issue names:
  `/opt/harness/bin/python /opt/train/digest.py --run [[PID]] --since <unix time> --until <unix time>` prints all
  of these for any stretch. Frames: `/opt/harness/bin/python /opt/train/look.py --images <png ...> --question
  "..."`.
- The harness's own records of each episode: see /work/harness/HARNESS.md for where they are and how to load them.
- [[DIR]]/screen_review.md: each episode judged from its frames, with the frame sheets in [[DIR]]/review/.
- Look as hard for records that contradict the issue as for ones that support it. If the issue says the player
  did nothing, did the step count rise? If it says an action failed, does a frame show it working?

Write `## Confirmation` at the top of [[INVESTIGATION]]: the records you checked, what each shows, and whether the
failure happened as the issue describes. Then hand it in, before you change any code (it is only accepted while
the harness is exactly as it was at [[START]]): write /work/cycle/drafts/confirm.json and run
`/opt/harness/bin/python /opt/train/record.py confirm` (`--template` prints the shape).

If the records do not show the failure, `"confirmed": "no"`: change nothing, and hand in the fix record (step 5)
with `"status": "not_confirmed"` and a `note` saying what the records show happened instead. If they show a
different failure from the one filed, that is also `no`: say so in the note, and the next conclusion can file it.

**2. Investigate.** With the failure confirmed, find its cause in the code: the code in /work/harness, and the
checks that show it ([[DIR]]/checks.md, and their digests in [[DIR]]/digests/).

Add to [[INVESTIGATION]], after the confirmation: `## Diagnosis` (the cause, precisely: which code, which
decision, why it produces what was seen), `## Evidence` (the episodes, times and frames that show it) and `## Fix`
(what you will change and why that removes the cause, in any game).

**3. Fix it in the code and commit.** One commit per fix, with a message that says what was wrong, how you fixed
it and the evidence. The harness was at commit [[START]] when this step began.

**4. Test it, when a short test can show the difference.** A short test is a playthrough of at most [[MINUTES]]
minutes from a checkpoint, with one declared test and one stopping point. You have at most [[TESTS]] for this
issue. Commit first, then:

```bash
/opt/harness/bin/python /opt/train/short_test.py start --from <run>:<id> \
    --test "<the one thing that shows the fix works>" --stop "<the point at which the test is over>"
/opt/harness/bin/python /opt/train/short_test.py wait      # returns after a minute, with what happened
/opt/harness/bin/python /opt/train/short_test.py verdict --verdict not_yet|fixed|failed|stop_point_reached \
    --evidence "<what you saw>" [--passed yes|no]
```

After each `wait`, record a verdict: `not_yet` keeps the test running and you `wait` again; `fixed`, `failed` or
`stop_point_reached` (with `--passed`) ends it. Choose a checkpoint shortly before the place the issue shows up.
The most recent checkpoints of this playthrough (earlier ones are in [[DIR]]/harness/checkpoints.jsonl):

[[CHECKPOINTS]]

If a test fails, improve the fix, commit, and test again. If the last test you can run still fails, undo your
commits with `git revert` and record the issue as deferred.

This step has a time limit. If it runs out first, the controller stops any short test and tells you to wrap up:
you then either keep the fix unverified or revert it, in a few minutes.

**5. Hand in the result.** Write /work/cycle/drafts/fix.json and run
`/opt/harness/bin/python /opt/train/record.py fix`:

```json
{"issue": "[[ISSUE_ID]]", "status": "fixed", "diagnosis": "the cause, as you found it", "commits": ["<sha>"]}
```

Use `"status": "deferred"` with a `note` (why, and what you tried) when you cannot fix it in this step: for
example it needs a larger change than one issue. A deferred issue leaves the harness exactly as it was at
[[START]]. Fix what record.py reports, and end your turn when it says ACCEPTED.
