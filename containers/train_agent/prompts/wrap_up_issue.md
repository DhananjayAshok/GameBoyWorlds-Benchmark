TIME IS UP for this issue. Stop investigating and stop testing: any running short test has been stopped and no
new one can start. You have [[MINUTES]] minutes to settle the harness and hand in the fix record, then end your
turn.

Choose one, from what you know now:

- **Keep the fix, unverified.** Commit any finished work, then hand in `"status": "unverified"` with its
  `commits` and a `note` saying what the fix has not been shown to do. Only when you have good reason to think
  it works: not when the last short test failed.
- **Discard it.** `git revert` every commit you made in this step (and discard uncommitted changes), so the
  harness is exactly as it was when the step began, then hand in `"status": "deferred"` with a `note` saying
  what you tried and what is left.
- If it is already tested and passed, `"status": "fixed"`.

Write /work/cycle/drafts/fix.json and run `/opt/harness/bin/python /opt/train/record.py fix` (and hand in any
other record this step still needs: [[KINDS]]). Fix what it reports and end your turn when it says ACCEPTED.

What is still missing:
[[ERRORS]]
