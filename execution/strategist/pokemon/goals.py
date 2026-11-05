"""
What the strategist is trying to achieve: a tree of goals, and the standing goals it can
fall back on.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from execution.artifact import StrategistArtifact, update
from execution.strategist.pokemon import prompts as P
from execution.strategist.pokemon.asking import Parser, ask, field_values, single_value
from execution.strategist.pokemon.knowledge import KnowledgeTree, recall
from utils import log_error, log_warn
from utils.lm_inference import clean_value

STATUSES = ("incomplete", "abandoned", "complete")
OPEN_STATUSES = ("incomplete",)

BADGE_COUNT = 8

GOAL_ATTEMPTS = 3
MAX_SUBGOAL_STEPS = 10
MIN_ABANDON_DEPTH = 2

GoalPath = Tuple[str, ...]
VLMCaller = Callable[[str], Callable[..., str]]


@dataclass
class GoalNode:
    description: str
    details: Optional[str] = None
    status: str = "incomplete"
    source: str = "initial"
    source_description: Optional[str] = None
    achievable_without_subgoals: Optional[int] = None
    frontier_since: Optional[int] = None
    status_reason: Optional[str] = None


GOAL_FIELDS = tuple(f.name for f in fields(GoalNode))


class GoalSubtree:
    def __init__(self, root: GoalNode, subtrees: Optional[List["GoalSubtree"]] = None, depth: int = 0) -> None:
        self.root = root
        self.depth = depth
        self.subtrees: List["GoalSubtree"] = []
        for subtree in subtrees or []:
            self.add(subtree)

    @classmethod
    def create_tree(cls, subgoals: List[GoalNode], title: str) -> "GoalSubtree":
        tree = cls(GoalNode(description=title))
        for subgoal in subgoals:
            tree.add(subgoal)
        return tree

    def add(self, child) -> "GoalSubtree":
        subtree = child if isinstance(child, GoalSubtree) else GoalSubtree(child)
        subtree._set_depth(self.depth + 1)
        self.subtrees.append(subtree)
        return subtree

    def _set_depth(self, depth: int) -> None:
        self.depth = depth
        for subtree in self.subtrees:
            subtree._set_depth(depth + 1)

    def walk(self) -> Iterator["GoalSubtree"]:
        yield self
        for subtree in self.subtrees:
            yield from subtree.walk()

    def nodes(self) -> List[GoalNode]:
        return [tree.root for tree in self.walk()]

    def leaves(self) -> List["GoalSubtree"]:
        return [tree for tree in self.walk() if not tree.subtrees]

    def find(self, description: str) -> Optional["GoalSubtree"]:
        wanted = description.strip().lower()
        return next((tree for tree in self.walk()
                     if tree.root.description.strip().lower() == wanted), None)

    def open_leaves(self) -> List["GoalSubtree"]:
        return [tree for tree in self.leaves() if tree.root.status in OPEN_STATUSES]

    def frontier(self) -> Optional["GoalSubtree"]:
        if self.root.status not in OPEN_STATUSES:
            return None
        for subtree in self.subtrees:
            if subtree.root.status in OPEN_STATUSES:
                return subtree.frontier()
        return self

    def path_to(self, target: "GoalSubtree", prefix: GoalPath = ()) -> Optional[GoalPath]:
        path = prefix + (self.root.description,)
        if self is target:
            return path
        return next((found for subtree in self.subtrees
                     if (found := subtree.path_to(target, path)) is not None), None)

    def paths(self, prefix: GoalPath = ()) -> Dict[GoalPath, GoalNode]:
        path = prefix + (self.root.description,)
        out = {path: self.root}
        for subtree in self.subtrees:
            out.update(subtree.paths(path))
        return out

    def at(self, path: GoalPath) -> Optional["GoalSubtree"]:
        if not path or path[0] != self.root.description:
            return None
        if len(path) == 1:
            return self
        return next((found for subtree in self.subtrees if (found := subtree.at(path[1:])) is not None), None)

    def render(self, indent: int = 0) -> str:
        details = f" -- {self.root.details}" if self.root.details else ""
        lines = [f"{'  ' * indent}- {self.root.description} [{self.root.status}]{details}"]
        for subtree in self.subtrees:
            lines.append(subtree.render(indent + 1))
        return "\n".join(lines)


class GoalTree(StrategistArtifact):
    frontier_path: Optional[GoalPath] = None

    def __init__(self, tree: GoalSubtree) -> None:
        self.tree = tree
        self.frontier_path = None

    def _at(self, path: GoalPath) -> GoalSubtree:
        subtree = self.tree.at(path)
        if subtree is None:
            log_error(f"No goal at {' > '.join(path)!r}.")
        return subtree

    def frontier(self) -> Optional[GoalSubtree]:
        return self.tree.frontier()

    def path_to(self, subtree: GoalSubtree) -> Optional[GoalPath]:
        return self.tree.path_to(subtree)

    @update
    def add_subtree(self, subtree: GoalSubtree) -> GoalSubtree:
        frontier = self.frontier()
        if frontier is None:
            log_error(f"No incomplete goal to attach {subtree.root.description!r} under.")
        return frontier.add(subtree)

    @update
    def add_under(self, path: GoalPath, child) -> GoalSubtree:
        return self._at(path).add(child)

    @update
    def set_status(self, path: GoalPath, status: str, reason: Optional[str] = None) -> None:
        if status not in STATUSES:
            log_error(f"Goal status must be one of {STATUSES}, got {status!r}.")
        node = self._at(path).root
        node.status = status
        node.status_reason = reason

    @update
    def edit(self, path: GoalPath, **changes: Any) -> None:
        unknown = [name for name in changes if name not in GOAL_FIELDS]
        if unknown:
            log_error(f"GoalNode has no fields {unknown}; it has {GOAL_FIELDS}.")
        node = self._at(path).root
        for name, value in changes.items():
            setattr(node, name, value)

    @update
    def set_frontier(self, path: GoalPath, since: int) -> None:
        self._at(path).root.frontier_since = since
        self.frontier_path = path

    def render(self) -> str:
        return self.tree.render()

    def diff(self, old: "GoalTree") -> Dict[str, Any]:
        new_nodes = self.tree.paths()
        old_nodes = old.tree.paths()
        status_changed = {
            path: (old_nodes[path].status, node.status)
            for path, node in new_nodes.items()
            if path in old_nodes and old_nodes[path].status != node.status
        }
        fields_changed = {}
        for path, node in new_nodes.items():
            if path not in old_nodes:
                continue
            changed = {name: (getattr(old_nodes[path], name), getattr(node, name))
                       for name in GOAL_FIELDS
                       if name not in ("description", "status")
                       and getattr(old_nodes[path], name) != getattr(node, name)}
            if changed:
                fields_changed[path] = changed
        return {
            "status_changed": status_changed,
            "fields_changed": fields_changed,
            "added": [p for p in new_nodes if p not in old_nodes],
            "removed": [p for p in old_nodes if p not in new_nodes],
            "frontier": (old.frontier_path, self.frontier_path),
        }


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


class PokemonGoalTree(GoalTree):
    def __init__(self) -> None:
        tree = GoalSubtree(GoalNode(description="Clear the game"))
        starter = tree.add(GoalNode(description="Obtain your first Pokemon"))
        starter.add(GoalNode(description="Progress through the game until you can choose to obtain a Pokemon"))
        for n in range(1, BADGE_COUNT + 1):
            badge = tree.add(GoalNode(description=f"Obtain Badge {n}"))
            badge.add(GoalNode(description=f"Visit the city with the {ordinal(n)} Gym"))
            badge.add(GoalNode(description="Defeat the Gym Leader"))
        champion = tree.add(GoalNode(description="Become the Champion"))
        champion.add(GoalNode(description="Cross Victory Road"))
        champion.add(GoalNode(description="Defeat the Elite 4 and Champion"))
        super().__init__(tree)


def default_goals() -> List[GoalNode]:
    return [
        GoalNode(description="Explore", details="Speak to everyone in the area.",
                 source="default"),
        GoalNode(description="Train", details="Level up your Pokemon",
                 source="default"),
        GoalNode(description="Progress", details="Journey to the next unvisited city",
                 source="default"),
    ]


# ----------------------------------------------------------------------------
# Prompt pieces
# ----------------------------------------------------------------------------


def _goal_text(path: GoalPath, subtree: GoalSubtree) -> str:
    text = " > ".join(path[1:] or path)
    if subtree.root.details:
        text += f"\nDetails: {subtree.root.details}"
    return text


def frontier_text(goals: GoalTree) -> str:
    subtree = goals.frontier()
    if subtree is None:
        return "(every goal is finished)"
    return _goal_text(goals.path_to(subtree), subtree)


def goal_path_text(path: GoalPath) -> str:
    return " > ".join(path[1:] or path)


def _children_block(subtree: GoalSubtree) -> str:
    if not subtree.subtrees:
        return ""
    lines = []
    for child in subtree.subtrees:
        reason = f": {child.root.status_reason}" if child.root.status_reason else ""
        lines.append(f"- {child.root.description} [{child.root.status}{reason}]")
    return P.GOAL_CHILDREN_BLOCK.replace("[CHILDREN]", "\n".join(lines))


def _unsuccessful_block(episodes: int) -> str:
    if episodes <= 0:
        return ""
    return P.GOAL_UNSUCCESSFUL_BLOCK.replace("[EPISODES]", str(episodes))


def _notepad_text(notepad: str) -> str:
    return notepad.strip() or "(empty)"


def _knowledge_text(queries: List[str], found: Dict[Any, Dict[str, Any]]) -> str:
    if not queries:
        return "(you did not look anything up)"
    if not found:
        return "(nothing relevant was found)"
    return "\n".join(f"- {entry['title']}: {entry['description']}" for entry in found.values())


def _routes_text(routes: List[Tuple[str, Optional[str]]]) -> str:
    if not routes:
        return "(you did not look any up)"
    return "\n\n".join(f"To \"{request}\":\n{found or '(it could not be worked out)'}"
                       for request, found in routes)


def _goal_prompt(template: str, game: str, path: GoalPath, subtree: GoalSubtree, notepad: str) -> str:
    return (template.replace("[GAME]", game)
            .replace("[GOAL]", _goal_text(path, subtree))
            .replace("[NOTEPAD]", _notepad_text(notepad)))


# ----------------------------------------------------------------------------
# Parsers
# ----------------------------------------------------------------------------


def _distinct(values: List[str], key: str) -> Tuple[Optional[List[str]], Optional[str]]:
    cleaned: List[str] = []
    for value in values:
        value = clean_value(value)
        if value is None:
            return None, f"A `{key}:` line was empty. Every `{key}:` line must hold one request."
        if value.lower() not in (seen.lower() for seen in cleaned):
            cleaned.append(value)
    return cleaned, None


def _parse_queries(allow_routes: bool) -> Parser:
    kinds = "one `Query:` line per question" + (" and one `Route:` line per place" if allow_routes else "")

    def parse(output: str) -> Tuple[Any, Optional[str]]:
        queries = field_values(output, "Query")
        routes = field_values(output, "Route")
        nones = field_values(output, "Queries")
        if routes and not allow_routes:
            return None, "`Route:` lines cannot be used here. Write only `Query:` lines, or `Queries: None`."
        if not queries and not routes and len(nones) == 1 and clean_value(nones[0]) is None:
            return ([], []), None
        if not (queries or routes) or nones:
            return None, f"Write {kinds}, or the single line `Queries: None`, and not both."
        queries, problem = _distinct(queries, "Query")
        if problem is not None:
            return None, problem
        routes, problem = _distinct(routes, "Route")
        if problem is not None:
            return None, problem
        return (queries, routes), None
    return parse


def _parse_subgoals(taken: List[str]) -> Parser:
    fmt = ("Write one `Subgoal: <description> | <details, or None> | <why it is needed>` line per "
           "subgoal, or the single line `Subgoals: None`, and not both.")

    def parse(output: str) -> Tuple[Any, Optional[str]]:
        lines = field_values(output, "Subgoal")
        nones = field_values(output, "Subgoals")
        if not lines and len(nones) == 1 and clean_value(nones[0]) is None:
            return [], None
        if not lines or nones:
            return None, fmt
        seen = {description.strip().lower() for description in taken}
        nodes: List[GoalNode] = []
        for line in lines:
            parts = line.split("|")
            if len(parts) != 3:
                return None, f"`Subgoal: {line}` does not split into exactly three parts on `|`. {fmt}"
            description, details, why = (clean_value(part) for part in parts)
            if description is None or why is None:
                return None, f"`Subgoal: {line}` must have both a description and why it is needed."
            if description.lower() in seen:
                return None, (f"The subgoal {description!r} repeats the goal, one of its existing "
                              "subgoals, or another subgoal. Every description must be different.")
            seen.add(description.lower())
            nodes.append(GoalNode(description=description, details=details, source="subgoal_creation",
                                  source_description=why))
        return nodes, None
    return parse


def _parse_achieved(output: str) -> Tuple[Any, Optional[str]]:
    value, problem = single_value(output, "Achieved")
    if problem is not None:
        return None, problem + " It must say Yes or No."
    answer = (value or "").rstrip(".").strip().lower()
    if answer not in ("yes", "no"):
        return None, f"The `Achieved:` line must say Yes or No, got {value!r}."
    return answer == "yes", None


def _parse_decision(choices: Tuple[str, ...], siblings: List[str]) -> Parser:
    taken = {description.strip().lower() for description in siblings}

    def parse(output: str) -> Tuple[Any, Optional[str]]:
        choice, problem = single_value(output, "Choice")
        if problem is not None:
            return None, problem + f" It must be one of {', '.join(choices)}."
        choice = (choice or "").rstrip(".").strip().upper()
        if choice not in choices:
            return None, f"The `Choice:` line must be one of {', '.join(choices)}, got {choice!r}."
        if choice == "CONTINUE":
            return (choice, None, None, None), None
        reason, problem = single_value(output, "Reason")
        if problem is not None or reason is None:
            return None, f"{choice} needs a `Reason:` line saying why."
        if choice == "ABANDON":
            return (choice, None, None, reason), None
        description, problem = single_value(output, "New Description")
        if problem is not None or description is None:
            return None, "UPDATE needs the rewritten goal on a `New Description:` line."
        details, problem = single_value(output, "New Details")
        if problem is not None:
            return None, "UPDATE needs a `New Details:` line, holding None if there are no details."
        if description.lower() in taken:
            return None, f"The new description {description!r} is already used by another goal next to it."
        return (choice, description, details, reason), None
    return parse


# ----------------------------------------------------------------------------
# Subgoal creation
# ----------------------------------------------------------------------------


def create_subgoals(goals: GoalTree, knowledge: KnowledgeTree, current: str, explore: str, notepad: str,
                    navigator: Callable[[str], Optional[str]], vlm_caller: VLMCaller, game: str,
                    episode_number: int, unsuccessful_for: int = 0, parameters: Optional[dict] = None) -> None:
    routes: List[Tuple[str, Optional[str]]] = []
    for step in range(MAX_SUBGOAL_STEPS):
        subtree = goals.frontier()
        if subtree is None:
            return
        path = goals.path_to(subtree)
        extras = (_children_block(subtree) + _unsuccessful_block(unsuccessful_for if step == 0 else 0))
        allow_routes = step == 0

        prompt = (_goal_prompt(P.SUBGOAL_QUERY_PROMPT, game, path, subtree, notepad)
                  .replace("[EXTRAS]", extras)
                  .replace("[ROUTE_OPTION]", P.SUBGOAL_ROUTE_OPTION if allow_routes else "")
                  .replace("[ROUTE_FORMAT]", P.SUBGOAL_ROUTE_FORMAT if allow_routes else "")
                  .replace("[LINE_KINDS]", "one `Query:` line per question and one `Route:` line per place"
                           if allow_routes else "one `Query:` line per question")
                  .replace("[CURRENT]", current))
        lookups = ask(vlm_caller("subgoal_query"), prompt, _parse_queries(allow_routes), "Subgoal query",
                      GOAL_ATTEMPTS, parameters)
        if lookups is None:
            log_warn(f"No usable queries for {path[-1]!r}; planning it without looking anything up.",
                     parameters)
            lookups = ([], [])
        queries, requests = lookups
        found: Dict[Any, Dict[str, Any]] = {}
        for query in queries:
            for entry in recall(knowledge, query, vlm_caller("subgoal_recall"), game, parameters):
                found.setdefault(entry["path"], entry)
        routes += [(request, navigator(request)) for request in requests]

        prompt = (_goal_prompt(P.SUBGOAL_CREATE_PROMPT, game, path, subtree, notepad)
                  .replace("[EXTRAS]", extras)
                  .replace("[KNOWLEDGE]", _knowledge_text(queries, found))
                  .replace("[CURRENT]", current)
                  .replace("[ROUTES]", _routes_text(routes))
                  .replace("[EXPLORE]", explore))
        taken = [path[-1]] + [child.root.description for child in subtree.subtrees]
        subgoals = ask(vlm_caller("subgoal_create"), prompt, _parse_subgoals(taken), "Subgoal creation",
                       GOAL_ATTEMPTS, parameters)
        if subgoals is None:
            log_warn(f"No usable subgoals for {path[-1]!r}; treating it as achievable without subgoals.",
                     parameters)
            subgoals = []
        if not subgoals:
            goals.edit(path, achievable_without_subgoals=episode_number, episode_number=episode_number)
            return
        for subgoal in subgoals:
            goals.add_under(path, subgoal, episode_number=episode_number)
        goals.edit(path, achievable_without_subgoals=None, episode_number=episode_number)
        goals.set_frontier(goals.path_to(goals.frontier()), episode_number, episode_number=episode_number)


# ----------------------------------------------------------------------------
# Completion check
# ----------------------------------------------------------------------------


def check_goals(goals: GoalTree, digest: str, notepad: str, vlm_caller: VLMCaller, game: str,
                episode_number: int, parameters: Optional[dict] = None) -> List[Tuple[GoalPath, str]]:
    closed: List[Tuple[GoalPath, str]] = []
    subtree = goals.frontier()
    checked_active = False
    while subtree is not None:
        path = goals.path_to(subtree)
        prompt = _goal_prompt(P.GOAL_CHECK_PROMPT, game, path, subtree, notepad).replace("[DIGEST]", digest)
        achieved = ask(vlm_caller("goal_check"), prompt, _parse_achieved, "Goal check", GOAL_ATTEMPTS,
                       parameters)
        if achieved is None:
            log_warn(f"No usable completion check for {path[-1]!r}; treating it as not achieved.", parameters)
            achieved = False
        if not achieved:
            if not checked_active and subtree.root.source != "override" and _decide(
                    goals, path, subtree, digest, notepad, vlm_caller, game, episode_number, parameters):
                closed.append((path, "abandoned"))
            return closed
        goals.set_status(path, "complete", episode_number=episode_number)
        closed.append((path, "complete"))
        checked_active = True
        subtree = goals.frontier()
    return closed


def _decide(goals: GoalTree, path: GoalPath, subtree: GoalSubtree, digest: str, notepad: str,
            vlm_caller: VLMCaller, game: str, episode_number: int, parameters: Optional[dict]) -> bool:
    can_abandon = len(path) - 1 >= MIN_ABANDON_DEPTH
    choices = ("CONTINUE", "UPDATE", "ABANDON") if can_abandon else ("CONTINUE", "UPDATE")
    parent = goals.tree.at(path[:-1]) if len(path) > 1 else None
    siblings = [child.root.description for child in parent.subtrees if child is not subtree] if parent else []
    prompt = (_goal_prompt(P.GOAL_DECISION_PROMPT, game, path, subtree, notepad)
              .replace("[DIGEST]", digest)
              .replace("[ABANDON_OPTION]", P.GOAL_ABANDON_OPTION if can_abandon else "")
              .replace("[CHOICES]", ", ".join(choices)))
    decision = ask(vlm_caller("goal_decision"), prompt, _parse_decision(choices, siblings), "Goal decision",
                   GOAL_ATTEMPTS, parameters)
    if decision is None:
        log_warn(f"No usable decision for {path[-1]!r}; continuing with it.", parameters)
        return False
    choice, description, details, reason = decision
    if choice == "ABANDON":
        goals.set_status(path, "abandoned", reason, episode_number=episode_number)
        return True
    if choice == "UPDATE":
        goals.edit(path, description=description, details=details, source_description=reason,
                   frontier_since=None, achievable_without_subgoals=None, episode_number=episode_number)
    return False
