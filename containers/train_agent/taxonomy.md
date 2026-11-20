# Issue taxonomy

Every issue you record names one **locus** (which part of the harness carries the problem) and one **code** (what
kind of problem it is). Use the codes below exactly as written. If nothing fits, write `new:<group>/<name>` and
define it in the issue's description; it will be reviewed for adding here.

Pick the code whose fix would most plausibly have saved the time lost. When two compete, pick the one earlier in
the chain of cause and effect.

## Locus

| locus | meaning |
|---|---|
| `strategist/goal` | the goal the strategist is pursuing is wrong, stale, or not the next thing the game needs |
| `strategist/task` | the goal is right but the task written for the episode is wrong, vague, too large or a repeat |
| `strategist/memory` | what the harness remembers (map, notes, knowledge, people, guide) is wrong, missing or unused |
| `supervisor/plan` | the first or re-issued plan is wrong for the situation: wrong target, missing step, no strategy |
| `supervisor/replan` | a replan or verdict undoes or abandons progress, or fails to react to a failed leg |
| `supervisor/decomposition` | step size, end condition or leg length makes a correct plan unexecutable |
| `supervisor/perception` | the supervisor misreads the screen it is judging (position, state, which thing is which) |
| `executor/reasoning` | the hint was right and achievable; the executor's choice of action was wrong |
| `executor/perception` | the executor misread the current frame (object, position, cursor) |
| `executor/action-format` | unparseable or invalid output is the main cost |
| `perception/tiles` | the tile recognizer or screen description is wrong, and the layers above reasoned over it |
| `perception/history-misuse` | the history or a stuck warning was accurate and was ignored |
| `environment` | cost imposed by the game or by the harness's plumbing, not by a decision |

## Codes

### Progress over a whole playthrough

**`progress/area-loop`**: the player stays in one area, revisiting the same places, across many episodes, with
no new ground reached. Exclude a deliberate search that keeps covering new ground.

**`progress/repeat-loss`**: the player loses to the same opponent again without having changed anything that
would change the result (levels, party, items, tactics).

**`progress/aimless-grind`**: battles are fought for experience with no opponent they are meant to beat, or long
after the party is strong enough, or without the levels actually rising.

**`progress/no-retry`**: an obstacle was the reason for a detour (training, buying, healing) and the player never
goes back to try it again.

**`progress/backtrack`**: the player leaves the frontier and returns to places already finished, with no need
there.

**`progress/optional-detour`**: time goes to optional content (items, side rooms, optional conversations) while
the way forward is known.

### Memory and goals

**`memory/stale-belief`**: the harness believes something about the world that is no longer true or never was
(where the player is, what is done, what a place connects to) and keeps acting on it.

**`memory/goal-drift`**: the goal being pursued changes without being finished or deliberately dropped, or the
goal tree no longer matches what the game is asking for.

**`memory/lost-knowledge`**: something the harness saw or was told earlier (a hint, a requirement, a location) is
not there when it is needed.

**`memory/map-error`**: the harness's own map is wrong: places merged that are different, one place split in
two, a connection missing or invented, blocked ground marked walkable or the reverse.

### Navigation and movement

**`nav/leg-oscillation`**: the direction is chosen afresh every leg, so runs cancel each other and the net
movement is small. Exclude one committed wrong direction (`plan/goal-misread`).

**`nav/alignment-overshoot`**: cannot stop on a one-tile target (a doorway, a gap, a stair, the tile next to a
person): legs pass it and the turn or the interaction comes from the wrong tile.

**`nav/no-search-strategy`**: the target's place is unknown and the search has no order: exits left untried,
buildings passed without entering, dead ends circled.

**`nav/path-blocked`**: the route chosen is blocked (a ledge, water, a person, an obstacle needing a tool) and
the same route is tried again instead of another route or the thing that unblocks it.

### Plans

**`plan/missing-precondition`**: the task needs a state or an item that is not there yet, and the plan goes on as
if it were, often although the game said so.

**`plan/mechanic-unused`**: the task needs a particular action (interact, use an item, a menu path) and the plan
only gives directions, so it is never used.

**`plan/goal-misread`**: the task text is resolved to the wrong thing (wrong building, wrong person, wrong
direction), or what counts as done keeps being redefined.

### Supervisor control

**`replan/undoes-progress`**: a replan or verdict reverses real progress: a working approach abandoned, a
finished step marked undone, an unfinished step passed over.

**`decomp/granularity`**: the plan is right but its steps cannot be executed: a leg shorter than the interaction
it must finish, a multi-step menu action split across legs, an end condition that does not match the action.

### Perception

**`percept/target-misbound`**: the intended thing is bound to the wrong thing on screen: the wrong person of
two, the wrong door, a menu entry taken for another.

**`percept/state-misread`**: the screen's state is misread: a prompt needing a choice treated as text to dismiss,
a frozen screen described as changing, a battle or menu not recognised, moving taken for stuck or the reverse.

**`percept/sub-tile`**: the target or the position is below what can be seen: a few pixels, a dark scene.

**`percept/landmark-ignored`**: a label, sign, name or line of dialogue on screen that answers the question is
not used.

### Mechanics

**`mech/unknown-control`**: what to do is understood, but not which inputs do it: how to close a box, how to
reach an option, how to use a move outside battle.

**`mech/misunderstood`**: the harness's model of a game rule is wrong: what an item does, what blocks movement,
what a type or a status does, how a menu behaves.

### Executor behaviour

**`exec/no-commit`**: the cursor or the player reaches the target and the confirming press never comes, or only
the first half of a two-part instruction is done.

**`exec/overshoot`**: the goal is a state to stop in (a page, a cursor position) and the executor presses past
it.

**`exec/list-bias`**: the first or highlighted entry is chosen again and again while an untried one is visible,
or only part of a list is ever looked at.

**`exec/repeat-completed`**: a conversation or a check that has finished, or already said no, is started again
because nothing records it as done.

**`exec/no-adaptation`**: in a battle or another contested state, the same action continues although it is
failing and a better one is available.

### Recovery and degeneracy

**`recover/ignored-feedback`**: the game says what is wrong and the relevant part of the action does not change.

**`degen/loop`**: the same action on the same target with the same result, four times or more.

**`degen/no-op`**: long runs of inputs with no effect: walking into walls, pressing under a text box.

**`degen/wander`**: plausible but aimless movement with no pattern of revisits.

### Budget and environment

**`budget/structural`**: the right thing was being done with little waste, and the episode's budget ran out.

**`env/attrition`**: unskippable content (scenes, forced encounters) used the time.

**`env/chance`**: the action was right and the roll was unlucky.

**`env/harness`**: an episode was cut short, crashed, or lost to a failed model call or a parsing error.
