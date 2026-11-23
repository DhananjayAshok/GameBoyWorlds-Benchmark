You are doing CHECK [[NUMBER]] of the long playthrough `[[PID]]` ([[GAME]]), [[MINUTES]] minutes in. The
playthrough keeps running while you check. You have [[BUDGET]]: read, judge, hand in the record, end your turn.
With [[WARNING]] left, the controller interrupts you once with a time check. Do not change the harness and do not
start or stop anything.

A check judges what happened; it does not explain it. Work only from the digest, its frames and the earlier
checks below. Do not open the harness's code, its pickles, its trajectory or its saved episode records: finding
the cause in the code is the deep dive's job, later, with the time for it. Hand in a first version of the record
within 2 minutes; you can improve it and hand it in again while time is left.

Read, in this order:

1. [[DIGEST]]: the last 5 minutes. What the server measured, what the harness intended, planned, tried and
   concluded, and what the screen showed (frames, described).
2. [[CHECKS]]: every earlier check of this playthrough, if there are any. Together they are the whole trajectory
   so far.

The digest's last section, "What the screen showed", lists the frame sheets of this window (each frame labelled
with its step and time) and a description of them. Those frames are your evidence for `on_screen`: read the
description and open the sheets. Only if that section says there are no frames does the window have none; then
the "Steps taken" line below says whether the game was being played. Few trajectory events are not by themselves
a sign that anything is wrong.

What the last iteration expects this playthrough to show:

[[EXPECTATIONS]]

What the controller measured, from game memory:

[[SIGNAL]]

Write the check as JSON to /work/cycle/drafts/check.json, in this shape
(`/opt/harness/bin/python /opt/train/record.py check --template` prints it):

- `summary`: four parts, each citing episode numbers and times.
  - `intentions`: what the strategist was trying to do.
  - `plans`: how the supervisor broke that down.
  - `attempts`: what the executors actually pressed and tried.
  - `on_screen`: what the player really did on screen. Take this from the frames and their description, not from
    the harness's own words: the harness can believe it did something it did not. If the frames do not answer
    something, ask with `/opt/harness/bin/python /opt/train/look.py --images <sheets> --question "..."`.
- `issues`: each problem you can see in this window, with `locus` and `code` from /opt/train/taxonomy.md,
  `observation` (how it can be seen, with episode and time), `cause` (one line: your best guess at the cause, from
  what you read here, without looking in the code) and
  `correct_behavior` (what the player should have done). An empty list if there are none.
- `stuck`: `verdict` is `"stuck"` or `"progressing"`, with `reasoning`. Judge from both the grid spots and maps the
  server counted and the earlier summaries. The player is stuck if, across the summaries, it is not meaningfully
  advancing through the game and reaching new ground: staying in the same area too long, losing to the same
  opponent again, repeating the same plan with the same result. A playthrough that is busy is not thereby
  progressing. The controller's recommendation above is advice, and your verdict decides: `stuck` ends this
  playthrough at once and starts the work of fixing what stopped it, which is the right call for a run that is
  going nowhere. If your verdict differs from the recommendation, `disagreement` must say what you see that the
  counters do not, with evidence.

Then hand it in:

```bash
/opt/harness/bin/python /opt/train/record.py check
```

Fix whatever it reports and run it again. When it says ACCEPTED, end your turn.
