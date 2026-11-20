You are at the START step of cycle [[CYCLE]]. The rules in /work/AGENTS.md always apply; read it if you have not.

[[NOTE]]The next long playthrough is `[[PID]]`, on `[[GAME]]`. The controller starts it; you only choose where it begins.

You can begin from the beginning of the game, or from any of these checkpoints (game state and harness memory
together):

[[CHECKPOINTS]]

To decide, read what the last iteration expects of this playthrough, if the file exists: [[LAST_ITERATION]]
Begin where the playthrough will teach you the most: usually the furthest point reached, so that it goes on into
new parts of the game; an earlier checkpoint or the beginning when a fix needs to be seen working from there.

Write your choice as JSON to /work/cycle/drafts/start.json:

```json
{"from": "beginning", "why": "one or two sentences"}
```

`from` is `"beginning"` or one checkpoint exactly as listed, for example `"p001_red:57"`. Then hand it in:

```bash
/opt/harness/bin/python /opt/train/record.py start
```

When it says ACCEPTED, end your turn. Do nothing else in this step, apart from a launcher's note above if there
is one.
