You are at the START step of cycle [[CYCLE]]. The rules in /work/AGENTS.md always apply; read it if you have not.

[[NOTE]]The next long playthrough is `[[PID]]`, on `[[GAME]]`. The controller starts it; you only choose where it begins.

You can begin from the beginning of the game, or from any of these checkpoints (game state and harness memory
together):

[[CHECKPOINTS]]

The latest checkpoint is `[[LATEST]]`. Start there. The long playthrough is for parts of the game the harness has
not played yet; fixes are shown working at earlier points by the short tests in the issue steps, not by playing the
game again from an earlier point.

Start earlier only when a change can be seen nowhere after `[[LATEST]]`: for example, a fix for something that
happens only before that point, which a short test cannot show. Even then, take the latest checkpoint that comes
before it, not the beginning. Read what the last iteration expects of this playthrough, if the file exists:
[[LAST_ITERATION]]

Write your choice as JSON to /work/cycle/drafts/start.json:

```json
{"from": "[[LATEST]]", "why": "one or two sentences"}
```

If `from` is not `[[LATEST]]`, add `"earlier_because"`: the expectation or commit that needs the earlier start,
what happens before `[[LATEST]]` that it must see, and why a short test cannot show it. Without it the record is
not accepted.

`from` is `"beginning"` or one checkpoint exactly as listed, for example `"p001_red:57"`. Then hand it in:

```bash
/opt/harness/bin/python /opt/train/record.py start
```

When it says ACCEPTED, end your turn. Do nothing else in this step, apart from a launcher's note above if there
is one.
