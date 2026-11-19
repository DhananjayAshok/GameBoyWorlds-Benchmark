from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from execution.artifact import StrategistArtifact, update
from execution.strategist.pokemon import prompts as P
from execution.strategist.pokemon.asking import Parser, ask
from utils import log_error
from utils.lm_inference import clean_value, parse_key_value

KnowledgePath = Tuple[str, ...]

KNOWLEDGE_ATTEMPTS = 3
RECALL_MAX_BRANCHES = 3


@dataclass
class KnowledgeNode:
    title: str
    description: str


class KnowledgeSubtree:
    # `leaves` holds every child, not only childless ones; the name is part of the saved format.
    def __init__(self, root: KnowledgeNode, leaves: Optional[List["KnowledgeSubtree"]] = None,
                 depth: int = 0) -> None:
        self.root = root
        self.depth = depth
        self.leaves: List["KnowledgeSubtree"] = []
        for leaf in leaves or []:
            self.add(leaf)

    def add(self, child) -> "KnowledgeSubtree":
        node = child if isinstance(child, KnowledgeSubtree) else KnowledgeSubtree(child)
        node._set_depth(self.depth + 1)
        self.leaves.append(node)
        return node

    def _set_depth(self, depth: int) -> None:
        self.depth = depth
        for leaf in self.leaves:
            leaf._set_depth(depth + 1)

    def walk(self) -> Iterator["KnowledgeSubtree"]:
        yield self
        for leaf in self.leaves:
            yield from leaf.walk()

    def nodes(self) -> List[KnowledgeNode]:
        return [tree.root for tree in self.walk()]

    def paths(self, prefix: KnowledgePath = ()) -> Dict[KnowledgePath, KnowledgeNode]:
        path = prefix + (self.root.title,)
        out = {path: self.root}
        for leaf in self.leaves:
            out.update(leaf.paths(path))
        return out

    def at(self, path: KnowledgePath) -> Optional["KnowledgeSubtree"]:
        if not path or path[0] != self.root.title:
            return None
        if len(path) == 1:
            return self
        return next((found for leaf in self.leaves if (found := leaf.at(path[1:])) is not None), None)


class KnowledgeTree(StrategistArtifact):
    def __init__(self, tree: KnowledgeSubtree) -> None:
        self.tree = tree

    def _at(self, path: KnowledgePath) -> KnowledgeSubtree:
        subtree = self.tree.at(path)
        if subtree is None:
            log_error(f"No knowledge at {' > '.join(path)!r}.")
        return subtree

    @update
    def add_under(self, path: KnowledgePath, child) -> KnowledgeSubtree:
        return self._at(path).add(child)

    @update
    def set_description(self, path: KnowledgePath, description: str) -> None:
        self._at(path).root.description = description

    def diff(self, old: "KnowledgeTree") -> Dict[str, Any]:
        new_nodes = self.tree.paths()
        old_nodes = old.tree.paths()
        description_changed = {
            path: (old_nodes[path].description, node.description)
            for path, node in new_nodes.items()
            if path in old_nodes and old_nodes[path].description != node.description
        }
        return {
            "description_changed": description_changed,
            "added": [p for p in new_nodes if p not in old_nodes],
            "removed": [p for p in old_nodes if p not in new_nodes],
        }


class PokemonKnowledgeTree(KnowledgeTree):
    def __init__(self) -> None:
        tree = KnowledgeSubtree(KnowledgeNode(title="All", description="All"))
        tree.add(KnowledgeNode(title="Active Story",
                               description="What is currently happening and what the player is in the middle of."))
        tree.add(KnowledgeNode(title="Game Lore",
                               description="Background world knowledge that does not change as the game is played."))
        tree.add(KnowledgeNode(title="NPCs",
                               description="Who has been met, what they said, and what they want."))
        tree.add(KnowledgeNode(title="Locations",
                               description="Where places are, how they connect, and what is in them."))
        tree.add(KnowledgeNode(title="Items",
                               description="What has been found, what is available, and where."))
        tree.add(KnowledgeNode(title="Pokemon",
                               description="What has been caught, trained, or fought, and how it performed."))
        tree.add(KnowledgeNode(title="Game Control Meta",
                               description="How to issue tasks and guidance to the player effectively: which kinds "
                                           "of instructions worked or failed, and why, so later tasks are "
                                           "phrased and scoped better."))
        super().__init__(tree)


def _path_text(path: KnowledgePath) -> str:
    return " > ".join(path)


def _node_prompt(template: str, game: str, text: str, path: KnowledgePath, node: KnowledgeSubtree) -> str:
    children = "\n".join(f"{i}. {child.root.title}: {child.root.description}"
                         for i, child in enumerate(node.leaves))
    siblings = "\n".join(f"- {child.root.title}" for child in node.leaves)
    return (template.replace("[GAME]", game)
            .replace("[KNOWLEDGE]", text)
            .replace("[QUERY]", text)
            .replace("[PATH]", _path_text(path))
            .replace("[DESCRIPTION]", node.root.description)
            .replace("[CHILDREN]", children)
            .replace("[SIBLINGS]", siblings or "(none)")
            .replace("[LAST]", node.root.title))


def _child_index(choice: str, children: List[KnowledgeSubtree]) -> Optional[int]:
    choice = choice.strip().rstrip(".").strip()
    numbered = re.match(r"(\d+)\s*[.):\-]?\s+(.+)$", choice)
    if numbered and int(numbered.group(1)) < len(children):
        title = children[int(numbered.group(1))].root.title.strip().lower()
        if numbered.group(2).strip().lower() == title:
            choice = numbered.group(1)
    if choice.isdigit():
        return int(choice) if int(choice) < len(children) else None
    matches = [i for i, child in enumerate(children) if child.root.title.strip().lower() == choice.lower()]
    return matches[0] if len(matches) == 1 else None


def _parse_navigation(children: List[KnowledgeSubtree]) -> Parser:
    valid = f"one of the entry numbers 0 to {len(children) - 1}, or NEW"

    def parse(output: str) -> Tuple[Any, Optional[str]]:
        choice = clean_value(parse_key_value(output, "Choice"))
        if choice is None:
            return None, f"The `Choice:` line was missing or empty. It must be {valid}."
        if choice.upper() == "NEW":
            return "NEW", None
        index = _child_index(choice, children)
        if index is None:
            return None, f"{choice!r} is not a valid choice. It must be {valid}."
        return index, None
    return parse


def _parse_recall(children: List[KnowledgeSubtree]) -> Parser:
    valid = (f"at most {RECALL_MAX_BRANCHES} entry numbers from 0 to {len(children) - 1} separated by commas, "
             "or NOT FOUND")

    def parse(output: str) -> Tuple[Any, Optional[str]]:
        choice = clean_value(parse_key_value(output, "Choice"))
        if choice is None:
            return None, f"The `Choice:` line was missing or empty. It must be {valid}."
        if choice.upper().replace("_", " ") == "NOT FOUND":
            return [], None
        indices: List[int] = []
        for part in choice.split(","):
            index = _child_index(part, children)
            if index is None:
                return None, f"{part.strip()!r} is not a valid entry. The choice must be {valid}."
            if index not in indices:
                indices.append(index)
        if len(indices) > RECALL_MAX_BRANCHES:
            return None, f"{len(indices)} entries were chosen. The choice must be {valid}."
        return indices, None
    return parse


def _parse_leaf(output: str) -> Tuple[Any, Optional[str]]:
    choice = clean_value(parse_key_value(output, "Choice"))
    if choice is None or choice.upper() not in ("KNOWN", "UPDATE", "NEW"):
        return None, f"The `Choice:` line must be KNOWN, UPDATE or NEW, got {choice!r}."
    choice = choice.upper()
    description = clean_value(parse_key_value(output, "New Description"))
    if choice == "UPDATE" and description is None:
        return None, "UPDATE needs the full rewritten description on the `New Description:` line."
    return (choice, description), None


def _parse_create(sibling_titles: List[str]) -> Parser:
    taken = {title.strip().lower() for title in sibling_titles}

    def parse(output: str) -> Tuple[Any, Optional[str]]:
        parent_title = clean_value(parse_key_value(output, "Parent Title"))
        parent_description = clean_value(parse_key_value(output, "Parent Description"))
        entry_title = clean_value(parse_key_value(output, "Entry Title"))
        entry_description = clean_value(parse_key_value(output, "Entry Description"))
        if entry_title is None or entry_description is None:
            return None, "Both `Entry Title:` and `Entry Description:` must be filled in."
        if (parent_title is None) != (parent_description is None):
            return None, "`Parent Title:` and `Parent Description:` must both be filled in, or both be None."
        top_title = parent_title or entry_title
        if top_title.strip().lower() in taken:
            return None, f"The title {top_title!r} is already used by an entry there. Choose a different title."
        if parent_title is not None and parent_title.strip().lower() == entry_title.strip().lower():
            return None, "The entry's title must differ from its parent's."
        entry = KnowledgeNode(title=entry_title, description=entry_description)
        if parent_title is None:
            return KnowledgeSubtree(entry), None
        return KnowledgeSubtree(KnowledgeNode(title=parent_title, description=parent_description),
                                [KnowledgeSubtree(entry)]), None
    return parse


def _create(knowledge: KnowledgeTree, knowledge_text: str, path: KnowledgePath, vlm_call: Callable[..., str],
            game: str, episode_number: int, parameters: Optional[dict]) -> Dict[str, Any]:
    at = knowledge.tree.at(path)
    prompt = _node_prompt(P.KNOWLEDGE_CREATE_PROMPT, game, knowledge_text, path, at)
    subtree = ask(vlm_call, prompt, _parse_create([child.root.title for child in at.leaves]),
                  "Knowledge create", KNOWLEDGE_ATTEMPTS, parameters)
    if subtree is None:
        return {"outcome": "failed", "stage": "create", "path": path}
    knowledge.add_under(path, subtree, episode_number=episode_number)
    added = path + tuple(node.title for node in [subtree.root] + [leaf.root for leaf in subtree.leaves])
    return {"outcome": "added", "path": added}


def remember(knowledge: KnowledgeTree, knowledge_text: str, vlm_call: Callable[..., str], game: str,
             episode_number: int, parameters: Optional[dict] = None) -> Dict[str, Any]:
    node = knowledge.tree
    path: KnowledgePath = (node.root.title,)
    while node.leaves:
        prompt = _node_prompt(P.KNOWLEDGE_NAVIGATE_PROMPT, game, knowledge_text, path, node)
        choice = ask(vlm_call, prompt, _parse_navigation(node.leaves), "Knowledge navigation",
                     KNOWLEDGE_ATTEMPTS, parameters)
        if choice is None:
            return {"outcome": "failed", "stage": "navigation", "path": path}
        if choice == "NEW":
            return _create(knowledge, knowledge_text, path, vlm_call, game, episode_number, parameters)
        node = node.leaves[choice]
        path = path + (node.root.title,)
    if len(path) == 1:
        return _create(knowledge, knowledge_text, path, vlm_call, game, episode_number, parameters)
    prompt = _node_prompt(P.KNOWLEDGE_LEAF_PROMPT, game, knowledge_text, path, node)
    decision = ask(vlm_call, prompt, _parse_leaf, "Knowledge leaf", KNOWLEDGE_ATTEMPTS, parameters)
    if decision is None:
        return {"outcome": "failed", "stage": "leaf", "path": path}
    choice, description = decision
    if choice == "KNOWN":
        return {"outcome": "known", "path": path}
    if choice == "UPDATE":
        knowledge.set_description(path, description, episode_number=episode_number)
        return {"outcome": "updated", "path": path}
    return _create(knowledge, knowledge_text, path[:-1], vlm_call, game, episode_number, parameters)


def _recall_at(node: KnowledgeSubtree, path: KnowledgePath, query: str, vlm_call: Callable[..., str],
               game: str, parameters: Optional[dict]) -> List[Dict[str, Any]]:
    if not node.leaves:
        if len(path) == 1:
            return []
        return [{"path": path, "title": node.root.title, "description": node.root.description}]
    prompt = (_node_prompt(P.KNOWLEDGE_RECALL_PROMPT, game, query, path, node)
              .replace("[MAX_BRANCHES]", str(RECALL_MAX_BRANCHES)))
    indices = ask(vlm_call, prompt, _parse_recall(node.leaves), "Knowledge recall", KNOWLEDGE_ATTEMPTS,
                  parameters)
    found: List[Dict[str, Any]] = []
    for index in indices or []:
        child = node.leaves[index]
        found += _recall_at(child, path + (child.root.title,), query, vlm_call, game, parameters)
    return found


def recall(knowledge: KnowledgeTree, query: str, vlm_call: Callable[..., str], game: str,
           parameters: Optional[dict] = None) -> List[Dict[str, Any]]:
    return _recall_at(knowledge.tree, (knowledge.tree.root.title,), query, vlm_call, game, parameters)
