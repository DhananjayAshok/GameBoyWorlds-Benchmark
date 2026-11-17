CRITIQUE_SLICE_PROMPT = """You are analysing a segment of a failed attempt to complete a task in a game of [GAME].

Task: "[TASK]"

Actions taken in this segment (steps [START_IDX]-[END_IDX] of [TOTAL] total):
[ACTION_SEQUENCE]

The images show frames [START_IDX]-[END_IDX] of the trajectory, from left to right.

Describe what happened in this segment: what the player did, what went wrong (if anything), and any observations relevant to why the task was not completed.

Respond in exactly this format:
Segment summary: <one or two sentences describing what happened in this segment>
[STOP]"""

CRITIQUE_CONSOLIDATE_PROMPT = """You are analysing a failed attempt to complete a task in a game of [GAME].

Task: "[TASK]"

Below are summaries of each segment of the failed trajectory:
[SEGMENT_SUMMARIES]

[PRIOR_HINT_BLOCK]Based on the full trajectory above, provide a concise hint for how to better approach the task on the next attempt.

Respond in exactly this format:
Critique: <what went wrong overall>
Hint: <one or two sentence hint for a better approach>
[STOP]"""

PLAN_PROMPT = """You are planning how a player should complete a task in a game of [GAME].

Task: "[TASK]"

The image is the screen the player is looking at right now.

Here is what has been learned from past playthroughs of this game that may be relevant:
[INSIGHTS]

Break the task into an ordered sequence of steps, separated by the token [STEP].

Each step will be given to a player who CANNOT see the other steps and does not know how many remain. They see only the current screen and the step you wrote. So each step must stand entirely on its own.

Requirements for every step:

- Describe the step by what is VISIBLE on screen: objects, icons, cursors, doors, characters, menu entries, text. Refer to things the player can point at.
- Do NOT name buttons or directions. Write "move the cursor to the coat" rather than "press RIGHT twice to reach the coat", and "select the hand tool" rather than "press A". Which button achieves it is the player's problem, and the button that worked in a past playthrough may be wrong from this screen.
- Every step MUST carry a termination condition that is visually checkable — a state of the screen the player can look at and confirm. Write it as "... until <what the screen shows>". If you cannot name a visible condition that ends the step, the step is too vague: merge it into a neighbour or rewrite it.
- One step should be one coherent sub-goal, not a single input and not the whole task.
- Prefer few steps. Three or four good steps beat ten brittle ones.

Do not include a step for something the screen shows is already done.

Respond in exactly this format:
Plan: <step one, ending in a visible condition> [STEP] <step two, ending in a visible condition> [STEP] <...>
[STOP]"""

FILTER_INSIGHTS_PROMPT = """You are pruning recorded knowledge about [GAME] down to what could matter for one task.

The player's task is: "[TASK]"

The image is the screen they are looking at right now.

Here is everything recorded from past playthroughs that was retrieved for this situation:
[CANDIDATES]

Some of it will be about things that have nothing to do with this task or this place — advice about talking to a character when nobody is here, about a menu that this task never opens, about a room the player is not in and will not enter. That is what you are removing.

Keep an item if it could plausibly matter at ANY point while doing this task, not only on the screen as it looks this instant. The player will move, open menus and change rooms while working, and knowledge about where they are heading is exactly what is worth keeping. Something you drop is gone for the whole task.

So: drop only what is clearly about something absent and unrelated. **If you are unsure, KEEP it.** Removing one useful item costs more than leaving three useless ones.

Respond in exactly this format:
Keep: <comma-separated numbers, or ALL>
[STOP]"""

DISTILL_INSIGHTS_PROMPT = """You are consolidating what is known about [GAME] into a briefing for one task.

The player's task is: "[TASK]"

The image is the screen they are looking at right now.

Here is the knowledge kept from past playthroughs. It was recorded piecemeal, by different runs, so it repeats itself, contradicts itself in places, and states the same thing at several levels of detail:
[CANDIDATES]

Rewrite it as a short, ordered list of concrete statements.

- **Aggregate.** Where several items describe one thing, merge them into a single statement that carries every specific detail any of them had. Prefer the most specific version: if one says "an icon in the toolbar" and another says "the third icon from the left", the merged statement says the third icon from the left.
- **Cut redundancy.** Two items that say the same thing become one. An item that is a vaguer restatement of another is dropped entirely.
- **Be concrete.** Name the object, the place on the screen, the observable result. Drop anything that survives only as generic advice — "be careful", "explore thoroughly", "pay attention to the surroundings" — that is not knowledge, it is filler.
- **Stay faithful.** Every statement must be supported by the items above. Do not add knowledge, do not resolve a contradiction by inventing a third version, and do not promote a guess into a fact. If two items genuinely disagree, say so in one statement and keep both readings.
- Do not narrow a statement to only what is on this screen. The player will move and change rooms while doing this task, and knowledge about where they are going still belongs here.

Respond in exactly this format, one statement per line:
Insights:
- <statement>
- <statement>
[STOP]"""

JUDGE_SLICE_PROMPT = """You are examining a segment of a player's attempt at one step of a plan in a game of [GAME].

The step they were asked to complete: "[TASK]"

Actions taken in this segment (steps [START_IDX]-[END_IDX] of [TOTAL] total):
[ACTION_SEQUENCE]

The images show frames [START_IDX]-[END_IDX] of the attempt, from left to right.

Describe what visibly changed on screen across this segment, and whether anything in it shows the step's termination condition being met. Report what you can see, not what you assume the player intended.

Respond in exactly this format:
Segment summary: <one or two sentences describing what visibly happened>
[STOP]"""

JUDGE_CONSOLIDATE_PROMPT = """You are deciding whether a player completed one step of a plan in a game of [GAME].

The step they were asked to complete: "[TASK]"

Summaries of each segment of their attempt:
[SEGMENT_SUMMARIES]

The image is the screen as it stands NOW, at the end of the attempt. It is your primary evidence: the step is complete if and only if this screen shows its termination condition met.

The player stopped because: [STOP_REASON]. Note that a player who declared itself finished may be wrong — judge the screen, not the claim.

Answering "yes" when the step is not done sends the plan onward from a state it does not expect, and everything after it is built on a false premise. Answering "no" when it is done wastes the step budget repeating work. Judge honestly in both directions.

Respond in exactly this format:
Reasoning: <one or two sentences, referring to what is visible in the final screen>
Complete: <yes or no>
[STOP]"""

REGRESSION_CHECK_PROMPT = """You are checking whether a player of [GAME] has undone progress they had already made.

Earlier in this task they completed this step:
"[PREVIOUS_STEP]"

[EARLIER_STEPS_BLOCK]You are given two images. The FIRST is the screen at the moment that step was judged complete. The SECOND is the screen now, after a later step was attempted and failed.

What happened in between:
[SEGMENT_SUMMARIES]

Compare the two screens. Has the state that made the earlier step complete been lost? Examples of losing it: a tool that was selected is no longer selected, a menu that was open has closed, a door that was opened is shut again, an item that was held has been put back, the player has left the room they had reached.

Judge only what the two images show. Do not guess from the actions described — if the second screen still shows the earlier step's result, it was not undone, however erratic the play looks. If the images are too similar to tell, say no.

Respond in exactly this format:
Reasoning: <one or two sentences comparing the two screens>
Undone: <yes or no>
What was lost: <if yes, name the specific thing that is no longer true; otherwise write NONE>
[STOP]"""

RESUME_HINT_PROMPT = """You are advising a player of [GAME] who has just failed to complete one step of a plan and is about to try again.

The step: "[TASK]"

Summaries of what they just did:
[SEGMENT_SUMMARIES]

Why it is judged incomplete: [JUDGEMENT]

[REGRESSION_BLOCK]What past playthroughs of this game recorded that may bear on this:
[INSIGHTS]

[PRIOR_HINT_BLOCK]The image is the screen the player is looking at RIGHT NOW. They are NOT starting over — the game is exactly as this screen shows, including any progress or damage from the failed attempt.

Work in two parts.

**First, diagnose.** Say what is actually going wrong, using the reasons the player gave for each button beside what the frames show happened. Name the mechanism, not the symptom: not "they failed to select the tool" but why the presses that should have selected it did not.

**Then instruct.** Unlike the plan, which describes goals without mentioning controls, your hint names the actual controls: UP, DOWN, LEFT, RIGHT, A, B, START. Say **what each button does towards this goal** — which one moves the cursor, which one confirms, which one backs out of the menu they are stuck in. Use the recorded knowledge above wherever it names a control or what it does; that is what it is for.

**Do not give a count or a sequence.** Not "press DOWN four times, then A". The player acts one button at a time and looks at the screen again after each one, so a recipe written from this screen is wrong by its second step, and a player following it stops watching the screen. Give them the function of each control and the visible condition that tells them to stop: "DOWN moves the selection down the list — keep going until KEY1 is the circled entry, then A confirms it."

Requirements:

- The instruction must start from THIS screen. If the failed attempt left the player somewhere unexpected, say which button gets them out of it first.
- Correct a false belief explicitly before instructing: "you are two tiles left of the icon, not on it — RIGHT moves the cursor towards it, and A selects once it is highlighted" beats restating the goal.
- Do not repeat an instruction the summaries show already failed. If pressing A did nothing three times, do not say press A; say which button does the thing they were trying to do.
- Tie every button to an effect the player can see. A button named without saying what it changes on screen is no more useful than the plan step was.

Respond in exactly this format:
Diagnosis: <one or two sentences naming what is actually going wrong>
Hint: <which buttons do what towards this goal, and the visible condition to stop at; two sentences at most, no counts>
[STOP]"""

PLAN_FLAW_PROMPT = """You are reviewing whether a plan for a task in [GAME] is still worth following.

The overall task: "[OVERALL_TASK]"

The plan, with progress marked:
[PLAN_BLOCK]

The current step has just failed. What happened:
[FAILURE_HISTORY]

Why it is judged incomplete: [JUDGEMENT]
[REGRESSION_LINE]
What past playthroughs of this game recorded:
[INSIGHTS]

The image is the screen the player is looking at right now.

A plan can fail for two very different reasons, and you are deciding which:

**The plan is sound, the player is fumbling it.** The steps describe the right route; the player misread the screen, pressed the wrong control, or acted on the wrong object. A better hint fixes this. Answer **no**.

**The plan is wrong.** The screens show something the plan did not anticipate: the route it assumes does not exist, an object it names is not there, a step depends on a state that cannot be reached from here, the game works differently from what the plan assumed, or the player is somewhere the plan has no path from. No hint fixes this, because the player is being asked to do the wrong thing. Answer **yes**.

Be strict. Repeated failure alone is not evidence of a bad plan — a fumbled step fails repeatedly too. You need something visible on the screens that the plan is incompatible with. If you cannot name that thing, answer no.

If you answer yes, write a replacement for the current step and everything after it. Steps already marked DONE are finished and must not be re-planned; start from where the player is now. Keep the original plan's rules: describe what is VISIBLE, never name buttons or directions, and end every step with a visually checkable condition ("... until <what the screen shows>").

Respond in exactly this format:
Reasoning: <what on the screens does or does not contradict the plan>
Flawed: <yes or no>
Plan: <the replacement steps separated by [STEP], or NONE if not flawed>
[STOP]"""

DESCRIBE_SLICE_PROMPT = """You are watching frames [START_IDX]-[END_IDX] of [TOTAL] total frames from a game of [GAME].

Describe what the player does and what changes visually in this segment. Focus on actions taken and their outcomes. Do not assume any particular goal.

Respond in exactly this format:
Description: <concise description of the player's actions and visual changes in this segment>
[STOP]"""

DESCRIBE_CONSOLIDATE_PROMPT = """You are consolidating segment descriptions from a game of [GAME] into a single complete trajectory description.

Segment descriptions (in chronological order), each labelled with the frame range it covers:
[SEGMENT_DESCRIPTIONS]

Produce a single coherent description of the full trajectory from start to finish. Explicitly reference the frame ranges (e.g. "frames 1-10", "frames 11-20") as you describe what happens, so the reader can tell which part of the trajectory each event belongs to. Keep these frame-range references in the same form they appear in the segment labels above.

Respond in exactly this format:
Description: <complete description of the full trajectory, with frame ranges referenced inline>
[STOP]"""

JUDGE_BINARY_PROMPT = """Task: "[TASK]"

A player attempted to complete this task. Here is a description of what happened across the FULL trajectory:
"[DESCRIPTION]"

The images show only the FINAL frames of the trajectory. Task completion may have occurred earlier and may not be visible in these images.

Did the player successfully complete the task at any point during the trajectory? Use the description as your primary evidence — if it mentions something that closely matches task completion, count it as success even if it is not visible in the final frames shown.
[GOAL_CONDITION_NOTE]
The description references frame ranges (e.g. "frames 11-20"). Using these, identify the safe success point: the single frame number by which the task has SURELY been achieved. Pick the earliest frame you are confident the task is already complete. If the task was never completed, or you cannot tell from the description, respond with N/A.

Respond in exactly this format:
Reasoning: <your reasoning, referencing the description and any visual evidence>
Success: <yes or no>
Safe success point: <frame number, or N/A if never completed or unknown>
[STOP]"""

# Carries its own surrounding blank lines so the spacing is right when the slot is left empty.
JUDGE_GOAL_CONDITION_NOTE = """
The goal condition for this task is: "[GOAL_CONDITION]"

The goal condition is a strict guide, and only if the player has basically achieved the task with only minor, trivial differences from the goal condition should you consider it a success.
"""


# [FRAME_NOTE] and [EVIDENCE_NOTE] must match whether an entry frame is actually sent.
RELEVANCE_PROMPT = """You are deciding whether a piece of recorded knowledge about [GAME] is relevant to the situation a player is in right now.

The player's current task is: "[TASK]"

Here is the recorded entry:
[ENTRY]

[FRAME_NOTE]

Could this entry's knowledge be relevant to the player's current task on this current screen? Answer yes only if the entry genuinely fits the situation — the same or a very similar [KIND]. [EVIDENCE_NOTE]

Answering yes to something that does not fit produces a misleading hint, which is worse than no hint at all. Answering no to something that does fit wastes knowledge that was already paid for. Judge honestly in both directions.

Respond in exactly this format:
Reasoning: <one or two sentences, referring to the frames>
Relevant: <yes or no>
[STOP]"""
