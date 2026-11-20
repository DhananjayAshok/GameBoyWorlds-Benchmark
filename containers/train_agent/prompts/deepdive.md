You are at the DEEP DIVE step for one issue from the long playthrough `[[PID]]` ([[GAME]]). The rules in
/work/AGENTS.md always apply: the fix must be game-agnostic and vision-only. Work on this issue only.

```json
[[ISSUE]]
```

[[EARLIER]]
**1. Investigate.** Find the actual cause, from the records and from the code, before you change anything.

- The checks that show it: [[DIR]]/checks.md, and their digests in [[DIR]]/digests/.
- More of any stretch: `/opt/harness/bin/python /opt/train/digest.py --run [[PID]] --since <unix time> --until
  <unix time>`. Frames: `/opt/harness/bin/python /opt/train/look.py --images <png ...> --question "..."`.
- The harness's own records of each episode: see /work/harness/HARNESS.md for where they are and how to load them.
- The code in /work/harness.

Write the investigation to [[INVESTIGATION]] with these sections: `## Diagnosis` (the cause, precisely: which
code, which decision, why it produces what was seen), `## Evidence` (the episodes, times and frames that show
it) and `## Fix` (what you will change and why that removes the cause, in any game).

**2. Fix it in the code and commit.** One commit per fix, with a message that says what was wrong, how you fixed
it and the evidence. The harness was at commit [[START]] when this step began.

**3. Test it, when a short test can show the difference.** A short test is a playthrough of at most [[MINUTES]]
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

**4. Hand in the result.** Write /work/cycle/drafts/fix.json and run
`/opt/harness/bin/python /opt/train/record.py fix`:

```json
{"issue": "[[ISSUE_ID]]", "status": "fixed", "diagnosis": "the cause, as you found it", "commits": ["<sha>"]}
```

Use `"status": "deferred"` with a `note` (why, and what you tried) when you cannot fix it in this step: for
example it needs a larger change than one issue. A deferred issue leaves the harness exactly as it was at
[[START]]. Fix what record.py reports, and end your turn when it says ACCEPTED.
