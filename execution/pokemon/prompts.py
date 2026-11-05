PRIMITIVES = """1. MoveTo(description): walk to an object or entity that is visible on the screen, e.g. MoveTo(wooden signpost at (2, -4)).
2. MoveOff(direction description): walk off the edge of the screen in a direction, e.g. MoveOff(up, through the gap in the ledge) or MoveOff(left).
3. Interact(target): interact with the object or entity directly in front of the player, e.g. to talk to someone or read a sign. Name the target, e.g. Interact(wooden signpost).
4. Battle(optional instructions): fight the battle that is on screen, optionally with instructions, e.g. Battle() or Battle(use Thundershock and do not switch Pokemon).
5. Menu(specific instruction): use the menu system to do something specific, e.g. Menu(use a Potion on Pikachu). Only works in the main menu opened with Start, or in a menu that is already open.
6. Dialogue(): press through the dialogue that is on screen until it ends.
7. Free(instruction; stop when: condition): press single buttons (a, b, start, arrows) towards something specific, and say what to stop at. The condition must be something visible on screen, e.g. Free(walk up and down inside the patch of tall grass; stop when: a wild Pokemon appears). Use this when none of the others fit."""

PLAYTHROUGH_GUIDANCE_BLOCK = """
Guidance for this task:
[GUIDANCE]
"""

PLAYTHROUGH_LOG_BLOCK = """
What has happened so far, in order:
[LOG]
"""

PLAYTHROUGH_PLAN_PROMPT = """You are playing [GAME] and are planning how to complete a task.

Your task: "[TASK]"
[GUIDANCE_BLOCK][LOG_BLOCK]
The image is the current game screen. Here is what has been recognised on it:
[SCREEN_TILES]

These are the only things the player can be told to do:
[PRIMITIVES]

Break the task into subgoals, in order. Write each subgoal as the state of the game to reach, in plain words, not as one of the calls above: "the player is standing in front of the nurse", never "Interact(nurse)". Each subgoal should be a small step that one of the above can plausibly achieve from the screen it starts on, and should be worth checking off on its own. Write as few as the task needs. Start from where the player is now.

Respond in exactly this format:
Reasoning: <what the screen shows and how you intend to get the task done>
Subgoals:
- <first subgoal>
- <second subgoal>
...
[STOP]"""

PLAYTHROUGH_SUBGOAL_DECISION_PROMPT = """You are playing [GAME] and deciding what the player should do next.

Your task: "[TASK]"
[GUIDANCE_BLOCK]
Your plan:
[PLAN]

You are working on this subgoal: "[SUBGOAL]"
[LOG_BLOCK][ATTEMPTS_BLOCK]
The image is the current game screen. Here is what has been recognised on it:
[SCREEN_TILES]

You can make the player do exactly one of the following:
[PRIMITIVES]

Pick the single action that makes the most progress on the subgoal from this screen.

Respond in exactly this format:
Reasoning: <what you see, and why this action is the best next step>
Action: <exactly one action from the list, written as a call, e.g. MoveTo(door of the Poke Mart)>
[STOP]"""

PLAYTHROUGH_ATTEMPTS_BLOCK = """
What you already tried for this subgoal, and what came of it:
[ATTEMPTS]
"""

PLAYTHROUGH_JUDGE_PROMPT = """You are playing [GAME] and are checking what the player just did.

Your task: "[TASK]"
[GUIDANCE_BLOCK]
You are working on this subgoal: "[SUBGOAL]"

You asked the player to: [REQUEST]

The first image is the screen before, and the second image is the screen after. The player is always drawn in the centre of the screen, so when the player walks, the rest of the screen shifts the other way.

Before, this had been recognised on the screen, with the player at (0, 0):
[SCREEN_TILES]
[AFTER_BLOCK][TEXT_BLOCK]
The game reported: [EXECUTOR_REPORT]
That report can be wrong about where the player ended up, so judge from the images.

Say whether the subgoal is now finished, and write a short note for the log of this playthrough. The note is the only record that survives, so say what actually happened and anything learned about the game or the map.

Respond in exactly this format:
Reasoning: <compare the two screens and the text>
Subgoal complete: <yes or no>
Summary: <one or two sentences for the log>
[STOP]"""

PLAYTHROUGH_TEXT_BLOCK = """
Text that appeared on screen during it:
[TEXT]
"""

PLAYTHROUGH_AFTER_BLOCK = """
After it, this is recognised on the screen, with the player at (0, 0):
[AFTER_TILES]
"""

PLAYTHROUGH_ACHIEVABLE_PROMPT = """You are playing [GAME] and are stuck.

Your task: "[TASK]"
[GUIDANCE_BLOCK]
Your plan:
[PLAN]

You have been trying this subgoal and it is not working: "[SUBGOAL]"
[LOG_BLOCK]
What you tried for this subgoal:
[ATTEMPTS]

The image is the current game screen. Here is what has been recognised on it:
[SCREEN_TILES]

Decide whether to carry on with a new plan or to give up. Give up only if the task cannot be done from here at all, for example because it needs something the player does not have and cannot get.

If you carry on, write a fresh plan for what is left, starting from the current screen. It may drop the subgoal that failed, or reach it another way.

Respond in exactly this format:
Reasoning: <why it is not working, and what to do about it>
Decision: <replan or quit>
Reason: <if quitting, why the task cannot be done; otherwise None>
Subgoals:
- <first subgoal of the new plan, if replanning>
- <second subgoal>
...
[STOP]"""

PLAYTHROUGH_TASK_CHECK_PROMPT = """You are playing [GAME] and have worked through your plan.

Your task: "[TASK]"
[GUIDANCE_BLOCK][LOG_BLOCK]
The image is the current game screen. Here is what has been recognised on it:
[SCREEN_TILES]

Decide whether the task is now complete.

Respond in exactly this format:
Reasoning: <what was achieved, and whether it adds up to the task>
Task complete: <yes or no>
Summary: <one or two sentences describing how the playthrough went>
[STOP]"""

PLAYTHROUGH_CLARIFY_TARGET_PROMPT = """You are playing [GAME] and deciding what the player should do next.

Your task: "[TASK]"
[GUIDANCE_BLOCK]
The image is the current game screen. Here is what has been recognised on it:
[SCREEN_TILES]

You asked the player to move to "[TARGET]", but that description matches more than one thing on the screen:
[CANDIDATES]

Rewrite the description so that it matches exactly one thing on the screen: the one you meant. Include its coordinates and whatever sets it apart from the others.

Respond in exactly this format:
Reasoning: <which of the candidates you meant, and why>
Target: <the new description>
[STOP]"""

PLAYTHROUGH_MENU_OBJECTIVE_PROMPT = """You are playing [GAME] and are about to use the game's menus.

Your task: "[TASK]"
[GUIDANCE_BLOCK]
The image is the current game screen. Here is what has been recognised on it:
[SCREEN_TILES]

The player is currently [AGENT_STATE].

You decided to use the menus for this: "[INSTRUCTION]"

Write one clear, specific objective for whoever operates the menus. They see only the screen and your objective, not the task, so name exactly what should end up selected, used, or read.

Then say whether this needs the main menu, the one opened with the Start button from the overworld, which holds the Pokedex, your Pokemon, the bag and the trainer card. Answer no if it is about a menu that is already open on the screen, such as a PC, a shop, a yes/no choice or a battle menu.

Respond in exactly this format:
Reasoning: <what the task needs from the menus>
Objective: <one sentence, specific>
Start menu: <yes or no>
[STOP]"""

PLAYTHROUGH_BATTLE_TIPS_BLOCK = """
Your battle tips, learned from earlier battles; follow them where they apply:
[TIPS]
"""

PLAYTHROUGH_BATTLE_INSTRUCTIONS_PROMPT = """You are playing [GAME] and a battle has just started.

Your task: "[TASK]"
[GUIDANCE_BLOCK][BATTLE_TIPS_BLOCK]
The image is the current battle screen.

Say how this battle should be fought, for whoever gives the orders in it. Keep it short: which Pokemon to use, which kind of move to favour, whether to try to run, or whether to just win normally. Answer None if there is nothing special to say and the battle should simply be won.

Respond in exactly this format:
Reasoning: <what the battle screen shows and what it means for the task>
Instructions: <one short sentence, or None>
[STOP]"""
