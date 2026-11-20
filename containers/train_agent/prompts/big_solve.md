You are at the LARGER ISSUE step for one larger issue from the long playthrough `[[PID]]` ([[GAME]]). The rules
in /work/AGENTS.md always apply: the change must be game-agnostic and vision-only. Work on this issue only.

```json
[[ISSUE]]
```

[[EARLIER]]
This is a design change, not a patch: build what the harness is missing, rather than rewording a prompt or
adding a special case. /work/harness/HARNESS.md points to examples of changes of this kind already in the harness.

**1. Investigate and design.** Read the records ([[DIR]]/summary.md, checks.md, issues.md, the digests in
[[DIR]]/digests/, and `/opt/harness/bin/python /opt/train/digest.py --run [[PID]] --since <t> --until <t>` for
more) and the code in /work/harness. Write [[INVESTIGATION]] with the sections `## Diagnosis` (what in the design
produces the behaviour), `## Evidence` (where it shows) and `## Fix` (the design you will build, and why it
handles this kind of situation in any game).

**2. Build it and commit.** Commit in coherent pieces, each with a message that says what it does and why. The
harness was at commit [[START]] when this step began.

**3. Test it.** You have at most [[TESTS]] short tests of at most [[MINUTES]] minutes each. Commit first, then:

```bash
/opt/harness/bin/python /opt/train/short_test.py start --from <run>:<id> \
    --test "<the one thing that shows the change works>" --stop "<the point at which the test is over>"
/opt/harness/bin/python /opt/train/short_test.py wait
/opt/harness/bin/python /opt/train/short_test.py verdict --verdict not_yet|fixed|failed|stop_point_reached \
    --evidence "<what you saw>" [--passed yes|no]
```

After each `wait`, record a verdict; `not_yet` keeps it running. The most recent checkpoints of this playthrough
(earlier ones are in [[DIR]]/harness/checkpoints.jsonl):

[[CHECKPOINTS]]

If the last test you can run fails, undo your commits with `git revert` and record the issue as deferred.

**4. Hand in the result.** Write /work/cycle/drafts/fix.json and run
`/opt/harness/bin/python /opt/train/record.py fix`:

```json
{"issue": "[[ISSUE_ID]]", "status": "fixed", "diagnosis": "the cause, as you found it", "commits": ["<sha>"]}
```

Use `"status": "deferred"` with a `note` when the change cannot be finished and shown to work in this step; a
deferred issue leaves the harness exactly as it was at [[START]]. Fix what record.py reports, and end your turn
when it says ACCEPTED.
