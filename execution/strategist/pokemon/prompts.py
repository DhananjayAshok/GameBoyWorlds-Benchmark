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

The image is the current game screen. Here is what has been recognised on it:
[SCREEN_TILES]

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

Write one line summarising the episode for your own log.

Respond in exactly this format:
Reasoning: <what actually happened and what it means>
Summary: <one sentence>
[STOP]"""

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

PLACE_KINDS_BLOCK = """Places come in three kinds:
Major: a town, city or route, the places the game's own town map names.
Bridge: a place whose purpose is to connect major locations, such as a gatehouse between a route and a town, a cave or a tunnel.
Internal: a room inside a major location, such as a Pokemon Center, a shop, a gym or a house. Each floor of a building is its own room. Give the major location it is inside as its parent.

The places you already know:
[KNOWN]

If the place is one of these, use its name and kind exactly as written. Otherwise give it the name the game uses for it, or a short descriptive name if the game does not show one."""

LOCATE_PROMPT = """You are playing [GAME] and are the strategist. You keep a map of the places the player has been, and you do not currently know where the player is.

The image is the current game screen. Here is what has been recognised on it:
[SCREEN_TILES]

""" + PLACE_KINDS_BLOCK + """ If the screen does not show a place at all (a battle, a menu, dialogue covering the view, a black screen), write Unknown.

Respond in exactly this format:
Reasoning: <what the screen shows and which place it is>
Kind: <Major, Bridge, Internal or Unknown>
Name: <the place's name, or None if Unknown>
Parent: <for Internal, the major location it is inside; otherwise None>
[STOP]"""

TRANSITION_CHECK_PROMPT = """You are playing [GAME] and are the strategist. While the player was playing, the screen went black for a moment. The first [N_BEFORE] images are the screen before it went black, oldest first, and the last [N_AFTER] images are the screen after, oldest first.

Decide whether the player moved to a different place: through a door into or out of a building, up or down stairs, into or out of a cave or gatehouse, or to another town or route.

Only a move to a different place counts. These do not: a battle starting or ending, a menu, the Pokemon or item screens opening or closing, dialogue, a screen flash, or a cutscene that ends back where it started. If the screen after the black shows the same place as before it, answer No.

Respond in exactly this format:
Reasoning: <what the screens before and after show>
Transition: <Yes or No>
[STOP]"""

TRANSITION_NAME_PROMPT = """You are playing [GAME] and are the strategist. You keep a map of the places the player has been. The player has just moved from one place to another, and the screen went black in between. The first [N_BEFORE] images are the screen before, oldest first, and the last [N_AFTER] images are the screen after, oldest first.

The player was in: [PREVIOUS]
Places already known to connect to it: [CONNECTED]

""" + PLACE_KINDS_BLOCK + """ If the screen after does not show enough to tell where the player is, write Unknown.

Then say how the player got from the previous place to this one, and whether the same way leads back.

Respond in exactly this format:
Reasoning: <what the screens show and which place the player is in now>
Kind: <Major, Bridge, Internal or Unknown>
Name: <the place's name, or None if Unknown>
Parent: <for Internal, the major location it is inside; otherwise None>
How: <how the player got here from the previous place, e.g. "through the door at the top of the building", or None>
Two-way: <Yes if the same way leads back, No if it does not or you cannot tell>
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
