STRATEGIST_RECENT_BLOCK = """
The last few episodes, oldest first:
[RECENT]
"""

DISPATCH_PROMPT = """You are playing [GAME] and are the strategist: you decide what is worth doing next, and hand one short task at a time to a player who does the actual playing.

[RECENT_BLOCK]
The goal you are working towards, from the broadest goal down to it:
[GOAL]

The player is at: [CURRENT]

Your notepad:
[NOTEPAD]
[GUIDE]
The image is the current game screen. Here is what has been recognised on it:
[SCREEN_TILES][GLOBAL_MAP]

Write one task for the player that moves them towards the goal. The player is competent but has no memory of anything before this moment, gets roughly [MAX_STEPS] actions, and can only walk, talk to people, use menus and fight battles. So the task must be something reachable from this screen in that budget: "walk into the Pokemon Center and heal your team", not "beat the next gym".

Then write the guidance: everything the player needs and could not work out from the screen. Where the thing they want is, what has already been tried and failed, what to avoid. Do not repeat the task in it.

Respond in exactly this format:
Reasoning: <where the player is and what the next concrete step is>
Task: <one sentence, concrete, achievable from this screen>
Guidance: <one to four sentences for the player, or None>
[STOP]"""

REVIEW_PROMPT = """You are playing [GAME] and are the strategist. One episode has just finished.

Here is the player's report of the episode:
[DIGEST]

Write one line summarising the episode for your own log.[TOWN_QUESTION]

Respond in exactly this format:
Reasoning: <what actually happened and what it means>
Summary: <one sentence>[TOWN_FORMAT]
[STOP]"""

REVIEW_TOWN_QUESTION = """

New doors appeared on your map this episode. Say whether the player reached a town or city this episode that it had never been to before; a route, a cave or a building is not a town. Towns and cities the player has already reached: [TOWNS]."""

REVIEW_TOWN_FORMAT = """
Town: <the name of the town or city reached for the first time this episode, or None>"""

KNOWLEDGE_EXTRACT_PROMPT = """You are playing [GAME] and are the strategist. One episode has just finished, and you keep a database of what you know about the game.

Here is the player's report of the episode:
[DIGEST]

The first image is the screen when the episode started, the second is the screen when it ended.

List the facts from this episode that are worth remembering for the rest of the game: what people said, where places are and how they connect, what items and Pokemon were found or used, what worked, what failed and why, and what the story now asks of the player. Leave out the play-by-play of what the player pressed, and anything that will not matter later.

Each fact is stored on its own, away from this report, so it must make sense without it: name the people, places and things it is about rather than saying "he", "there" or "it".

Write at most [MAX_FACTS] facts, one per line. If nothing is worth remembering, write None.

Respond in exactly this format:
Reasoning: <what this episode taught you>
Facts:
- <fact>
- <fact>
[STOP]"""

KNOWLEDGE_HEADER = """You are playing [GAME] and keep a database of what you know, organised as a tree of topics. Here is some knowledge you wish to remember:
[KNOWLEDGE]
"""

KNOWLEDGE_NAVIGATE_PROMPT = KNOWLEDGE_HEADER + """
You may already have this in your database. First, navigate to the place where it is best stored.

You are at: [PATH]
[DESCRIPTION]

The entries under it:
[CHILDREN]

Choose the entry this knowledge belongs in, or NEW if none of them fit and it needs a new entry here.

Respond in exactly this format:
Reasoning: <which entry fits and why>
Choice: <the number of an entry, or NEW>
[STOP]"""

KNOWLEDGE_LEAF_PROMPT = KNOWLEDGE_HEADER + """
The closest entry in your database is:
[PATH]
[DESCRIPTION]

Decide what to do with it:
KNOWN: the entry already says this. Nothing changes.
UPDATE: the entry is about the same thing but is missing this knowledge or gets it wrong. Rewrite its description so it includes the new knowledge and keeps everything still true.
NEW: the knowledge is about something else. It gets its own entry next to this one.

Respond in exactly this format:
Reasoning: <how the knowledge compares to the entry>
Choice: <KNOWN, UPDATE or NEW>
New Description: <the full rewritten description if UPDATE, otherwise None>
[STOP]"""

KNOWLEDGE_CREATE_PROMPT = KNOWLEDGE_HEADER + """
It will be stored under: [PATH]
[DESCRIPTION]

The entries already there:
[SIBLINGS]

Write the new entry. If it belongs directly under [LAST], write None for both parent lines. If it is the first of a group of related entries that deserves its own heading, give that heading as the parent and the entry will go under it.

No title may repeat one of the entries already there.

Respond in exactly this format:
Reasoning: <what the knowledge is and where it fits>
Parent Title: <a short heading, or None>
Parent Description: <one sentence on what the heading covers, or None>
Entry Title: <a short title>
Entry Description: <the knowledge itself, distilled to what is worth remembering>
[STOP]"""

KNOWLEDGE_RECALL_PROMPT = """You are playing [GAME] and keep a database of what you know, organised as a tree of topics. You want to recall what you know about:
[QUERY]

You are at: [PATH]
[DESCRIPTION]

The entries under it:
[CHILDREN]

Choose up to [MAX_BRANCHES] entries that are most likely to hold something relevant, or NOT FOUND if none of them could.

Respond in exactly this format:
Reasoning: <which entries could be relevant and why>
Choice: <the numbers of at most [MAX_BRANCHES] relevant entries, separated by commas, or NOT FOUND>
[STOP]"""

RETRY_BLOCK = """

Your previous reply was:
[PREVIOUS]

That reply could not be used: [PROBLEM] Respond again in the required format."""

GOAL_CHILDREN_BLOCK = """
It already had these subgoals, which are all finished with:
[CHILDREN]
"""

GOAL_UNSUCCESSFUL_BLOCK = """
You have been working towards this goal for [EPISODES] episodes and have not achieved it yet. Your notepad may say what went wrong.
"""

SUBGOAL_QUERY_PROMPT = """You are playing [GAME] and are the strategist: you break long-term goals down into smaller subgoals that a player can work through one at a time.

The goal you are planning for, from the broadest goal down to it:
[GOAL]
[EXTRAS]
Your notepad:
[NOTEPAD]

Before planning, you can look things up in your database of what you know about the game. Write the questions whose answers would help you decide how this goal should be broken down.

Each question is looked up on its own, in a database that does not know where you are, what you are doing or what has just happened. So every question must make sense to someone who knows the game but knows nothing about your situation: name the places, people, items and Pokemon it is about, and never write "here", "this town", "the next gym" or "right now". Keep each question specific and about one thing.
[ROUTE_OPTION]
If the plan is already clear, or looking things up would not help, write None instead.

Respond in exactly this format:
Reasoning: <what you would need to know to plan this goal>
Query: <one self-contained question>
Query: <one self-contained question>[ROUTE_FORMAT]
([LINE_KINDS], or instead the single line `Queries: None`)
[STOP]"""

SUBGOAL_ROUTE_OPTION = """
You can also look up routes on your map of the places you have been. The player is at: [CURRENT]. For each place you want to know how to reach, write a `Route:` line saying where you want to go: a particular place you know by name, or a kind of place, such as "somewhere to heal your Pokemon". Unlike the questions, routes are always worked out from where the player is now.
"""

SUBGOAL_ROUTE_FORMAT = """
Route: <where you want to go from here>"""

SUBGOAL_CREATE_PROMPT = """You are playing [GAME] and are the strategist: you break long-term goals down into smaller subgoals that a player can work through one at a time.

The goal you are planning for, from the broadest goal down to it:
[GOAL]
[EXTRAS]
Your notepad:
[NOTEPAD]

What you looked up in your database of what you know about the game:
[KNOWLEDGE]

The player is at: [CURRENT]

Routes you looked up on your map:
[ROUTES]

Places on your map that may lead somewhere you have not been yet:
[EXPLORE]

Routes are only true from where the player is now, and the player will move while working through the subgoals. So describe each subgoal by where to get to and what to do there, not by copying the steps of a route.

Decide whether this goal needs to be broken down. If it is one clear objective the player can go straight for, write None. Otherwise, list the subgoals that together achieve it, in the order they should be done. Each subgoal is something to achieve in the game, not a button to press.

For each subgoal, give a short description, any details that would help achieve it (or None), and why it is needed for the goal. Separate the three with `|`, and do not use `|` anywhere else. No two subgoals may have the same description, and none may repeat the goal itself or one of its existing subgoals.

Respond in exactly this format:
Reasoning: <what the goal needs, and whether it must be broken down>
Subgoal: <description> | <details, or None> | <why it is needed>
Subgoal: <description> | <details, or None> | <why it is needed>
(one `Subgoal:` line per subgoal, or instead the single line `Subgoals: None`)
[STOP]"""

GOAL_CHECK_PROMPT = """You are playing [GAME] and are the strategist. One episode has just finished.

The goal to check, from the broadest goal down to it:
[GOAL]

Your notepad:
[NOTEPAD]

Here is the player's report of the episode:
[DIGEST]

Has this goal been achieved, in this episode or before it? Answer Yes only if the report or your notepad clearly shows it has happened.

Respond in exactly this format:
Reasoning: <the evidence for and against>
Achieved: <Yes or No>
[STOP]"""

GOAL_ABANDON_OPTION = """
ABANDON: the goal cannot be achieved, or is no longer worth pursuing. Drop it."""

GOAL_DECISION_PROMPT = """You are playing [GAME] and are the strategist. One episode has just finished, and this goal has not been achieved yet:
[GOAL]

Your notepad:
[NOTEPAD]

Here is the player's report of the episode:
[DIGEST]

Decide what to do with the goal:
CONTINUE: the goal is still right and is on its way; it just needs longer. Nothing changes.
UPDATE: the goal is roughly right but aimed at the wrong thing or described wrongly. Rewrite it.[ABANDON_OPTION]

Respond in exactly this format:
Reasoning: <how the goal is going, and what should happen to it>
Choice: <one of [CHOICES]>
New Description: <the rewritten goal if UPDATE, otherwise None>
New Details: <the rewritten details if UPDATE, or None>
Reason: <why you chose to update or abandon it, or None if CONTINUE>
[STOP]"""

NAVIGATE_SELECT_PROMPT = """You are playing [GAME] and keep a map of the places you have been. You want to go to:
[REQUEST]

You are at: [CURRENT]

The places on your map, nearest first:
[PLACES]

Choose every place that fits where you want to go. The nearest one you know a route to will be used. If none of them fit, write None.

Respond in exactly this format:
Reasoning: <which places fit and why>
Choice: <the numbers of the places that fit, separated by commas, or None>
[STOP]"""

NOTEPAD_UPDATE_PROMPT = """You are playing [GAME] and are the strategist. You keep a notepad of working thoughts about the goal you are pursuing: what has been tried, what worked, what failed, and what to try next. One episode has just finished.

The goal you are working towards, from the broadest goal down to it:
[GOAL]

The player is at: [CURRENT]

Your notepad:
[NOTEPAD]

Here is the player's report of the episode:
[DIGEST]

Your one-line summary of the episode: [SUMMARY]

Update the notepad with what this episode taught you about reaching the goal. Choose one:
APPEND: add new notes to the end of the notepad, keeping everything already there.
REWRITE: replace the whole notepad, when parts of it are now wrong, out of date or repeated. Keep everything that is still true and useful.

Keep the notes short and useful for the episodes to come. Do not narrate the episode step by step.

Respond in exactly this format:
Reasoning: <what this episode changes about your notes>
Action: <APPEND or REWRITE>
Notes:
<the notes to append, or the whole new notepad>
[STOP]"""

SEGMENT_NAME_PROMPT = """You are playing [GAME] and are the strategist. You keep a map of everything the player has seen, made of separate areas: an area is ground the player walked around in, and the player leaves it through a door, stairs, a cave entrance or anything else that makes the screen go black. [SITUATION]

The image is the whole of that area as seen so far, stitched together from the screens: the blue line is the path the player walked, [MARK_IMAGE]numbered boxes are doors and stairs. Here it is as text too, north up, one character per step[MARK_TEXT]:
[MAP]
Legend: [LEGEND]

The tasks you gave the player recently, oldest first:
[TASKS]

[STEPS_TITLE], oldest first:
[STEPS]

Other areas you have already named:
[NAMED]

[ASK] Use the name the game gives it if a sign, the town map or dialogue said so; otherwise a short descriptive name, such as "Pewter City Pokemon Center" or "the cave north of Route 3". Name the whole area, not one building in it. If this area is part of a place you have already named, such as another floor of the same building or another part of the same town, make that clear in its name. The description is for finding your way back later: what kind of place it is, how it is laid out, its notable buildings, people and objects, and its exits.

Also say what kind of area it is: outdoors (a town, city or route under the open sky), building (any floor inside a building, such as a house, a Pokemon Center, a shop, a gym, a lab or a gatehouse), or cave (a cave, forest, tunnel or other enclosed place that is not a building).

Respond in exactly this format:
Reasoning: <what the map and the player's steps show about this area>
Name: [NAME_FORMAT]
Description: [DESCRIPTION_FORMAT]
Kind: <outdoors, building or cave>
[STOP]"""

MERGE_JUDGE_PROMPT = """You are playing [GAME] and are the strategist. You keep a map of everything the player has seen, made of separate areas: an area is ground the player walked around in, and the player leaves it through a door, stairs, a cave entrance or anything else that makes the screen go black. Sometimes the player comes back to a place and the map does not recognise it, so the same place ends up as two areas. Other times two different places simply look alike: the floors of one house, or the same kind of building in two towns.

Two areas on your map look alike:
Area A, "[NAME_A]": [DESCRIPTION_A]
Its doors and stairs: [DOORS_A]
Area B, "[NAME_B]": [DESCRIPTION_B]
Its doors and stairs: [DOORS_B]

The names and descriptions were written when each area was first seen and can be wrong, so trust the images over them.

The first image is area A and the second is area B, each as seen so far; numbered boxes are doors and stairs. The third image lays B over A where they line up best: squares outlined green look the same in both and squares outlined red look different. Squares tinted blue were seen only in B, and squares with neither an outline nor a tint were seen only in A: what one area shows where the other has never looked is not a difference, since that part of the other area is simply unexplored. Lined up this way they share [SHARED] squares and [MATCHES] of them look the same. People move around, so a person standing in a different place is not a difference; furniture, walls, buildings and the shape of the ground are.

Are A and B the same place seen twice, or two different places that look alike? If they are the same place they will be joined into one area named "[NAME_A]".

Respond in exactly this format:
Reasoning: <what in the images and descriptions shows whether they are one place or two>
Decision: <SAME or DIFFERENT>
[STOP]"""

STUCK_PROMPT = """You are playing [GAME] and are the strategist: each episode you set the player one task. Lately [SIGNS].

That can mean you are stuck: going round in circles, trying the same thing that keeps failing, or looking for the way on in the wrong place. But it can also be normal progress that is simply slow: a long conversation, a cutscene, choosing from a menu, a battle, or a task that needs several episodes in one place. Only being stuck calls for a change of approach.

You are in [AREA]. Your map has [AREAS] separate area(s) so far. The image is the screen now.

Your recent episodes, oldest first, with their tasks, how they ended and what happened:
[EPISODES]

Are you stuck? If you answer YES, your next few tasks will be to explore the parts of this area you have not seen yet, instead of what you have been trying.

Respond in exactly this format:
Reasoning: <what the recent episodes show: steady progress, or the same failure again and again>
Stuck: <YES or NO>
[STOP]"""

DEFEAT_REFLECTION_PROMPT = """You are playing [GAME] and are the strategist. The player has just LOST a battle: all of their Pokemon fainted and they blacked out, which sends them back to the last Pokemon Center they used with their Pokemon healed and some of their money gone. The episode ended there.

The task you had set for the episode: [TASK]
Your current goal: [GOAL]
Where the battle happened: [WHERE]

The player's report of the episode:
[DIGEST]

The battle, as the player fought it:
[BATTLE]

The first image is the battle screen when the player started fighting it, the second is the last battle screen before the defeat.

You keep battle tips: short rules shown to the player at the start of every battle and on every turn of it. Your battle tips now:
[TIPS]

Think about why this battle was lost: the opponent was too strong for the player's team, the player's Pokemon were already hurt when it started, the wrong moves were chosen, a Pokemon should have been switched or healed, or something else. Then decide:
1. What went wrong, in one or two sentences.
2. Whether to change your battle tips, so the next battle goes better. A good tip is short and worth following in every battle, not just against this opponent. You may keep at most [MAX_TIPS] tips.
3. Whether the current goal can simply be tried again as it is (CONTINUE), or the player must first do something else to get ready for it (PREPARE), such as training their Pokemon in nearby grass, buying Potions, healing before setting out, or catching another Pokemon. A PREPARE step is added to your goals before the current one.
4. Whether the player should go back to where the battle happened (YES), for example because a trainer or gym leader there still blocks the way, or not (NO), for example because it was a wild Pokemon on a route.

Respond in exactly this format:
Reasoning: <why the battle was lost, and what that means for what to do next>
Wrong: <what went wrong, in one or two sentences>
Change: <ADD | <tip>, or EDIT | <number> | <new tip>, or REMOVE | <number>>
(at most [MAX_CHANGES] `Change:` lines, or instead the single line `Change: NONE`; only tips marked yours can be edited or removed)
Plan: <CONTINUE, or PREPARE | <the step to add before the current goal>>
Return: <YES or NO>
[STOP]"""

DEFEAT_RETURN_LINE = "You lost a battle in episode [EPISODE] at [WHERE] You decided the player should go back there once ready."

SEGMENT_NAME_NEW = "Name the area and describe it."
SEGMENT_NAME_REVIEW = """You named this area "[NAME]" and described it as: [DESCRIPTION]
Its map has grown from about [BEFORE] to [AFTER] squares since then, so look again. Keep the name and description if they still fit, by answering UNCHANGED. Change them only if the larger map shows the area better: for example if the name was for one building or corner and the area turns out to be a whole town, or the name is wrong."""

GUIDE_UPDATE_PROMPT = """You are playing [GAME] and are the strategist. You keep a strategy guide: short rules you follow every time something particular happens in the game. Each rule is shown to you again whenever its trigger happens, so a good rule is one that will be worth following every time, in any town, not just this once.

The triggers are: [TRIGGERS].

Your strategy guide now:
[RULES]

In the episode that just finished:
[EVENTS]

Here is the player's report of the episode:
[DIGEST]

Decide whether this episode taught you something that belongs in the guide: a habit worth having every time the player enters or leaves a building, or one of your own rules that turned out wrong or could be said better. Most episodes teach nothing new, and then the answer is NONE. Never add a rule that repeats one already there.

Respond in exactly this format:
Reasoning: <what this episode shows about entering or leaving buildings>
Change: ADD | <trigger> | <the rule, one sentence>
Change: EDIT | <number of one of your rules> | <the rewritten rule>
Change: REMOVE | <number of one of your rules>
(at most [MAX_CHANGES] `Change:` lines, or instead the single line `Change: NONE`)
[STOP]"""

NOTEPAD_EXTRACT_PROMPT = """You are playing [GAME] and are the strategist. You have just finished with these goals:
[CLOSED]

This is your notepad of working thoughts from pursuing them. It is about to be cleared:
[NOTEPAD]

List the facts in the notepad that are worth remembering for the rest of the game, long after these goals: where places are and how they connect, what people said, what items and Pokemon were found or used, what worked, what failed and why. Leave out what only mattered for finishing these goals.

Each fact is stored on its own, away from this notepad, so it must make sense without it: name the people, places and things it is about rather than saying "he", "there" or "it".

Write at most [MAX_FACTS] facts, one per line. If nothing is worth remembering, write None.

Respond in exactly this format:
Reasoning: <what in the notepad will still matter later>
Facts:
- <fact>
- <fact>
[STOP]"""
