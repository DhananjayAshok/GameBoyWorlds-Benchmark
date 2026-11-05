"""
Rules of thumb the strategist follows when something happens in the game: seeded by hand, and added
to by the strategist as it plays.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from execution.artifact import StrategistArtifact, update
from execution.strategist.pokemon import prompts as P
from execution.strategist.pokemon.asking import ask, field_values
from execution.strategist.pokemon.global_map import BUILDING, OUTDOORS, GlobalMap
from utils import log_info

ENTERED_BUILDING, EXITED_BUILDING, NEW_TOWN, BATTLE = ("entered building", "exited building", "entered new town",
                                                        "battle")
EVENT_TRIGGERS = (ENTERED_BUILDING, EXITED_BUILDING, NEW_TOWN)
TRIGGERS = EVENT_TRIGGERS + (BATTLE,)
MAX_BATTLE_TIPS = 8
SEEDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guide_seeds.yaml")
ACTIVE_EPISODES = 5
MAX_CHANGES = 3
GUIDE_ATTEMPTS = 2

@dataclass
class GuideRule:
    id: int
    trigger: str
    text: str
    source: str
    created: int
    updated: int


@dataclass
class GuideEvent:
    episode: int
    trigger: str
    previous: str
    place: str
    rules: Tuple[int, ...]
    segment: Optional[int] = None

    @property
    def text(self) -> str:
        if self.trigger == NEW_TOWN:
            return f"the player {self.trigger}: {self.place}"
        return f"the player {self.trigger}: {self.previous} → {self.place}"


def load_seeds(path: str = SEEDS_PATH) -> List[Dict[str, str]]:
    with open(path) as handle:
        seeds = yaml.safe_load(handle) or []
    for seed in seeds:
        if seed.get("trigger") not in TRIGGERS or not str(seed.get("rule", "")).strip():
            raise ValueError(f"Bad strategy guide seed in {path}: {seed!r}")
    return seeds


def door_trigger(gmap: GlobalMap, source: int, destination: int) -> Optional[str]:
    kinds = (gmap.info(source).kind, gmap.info(destination).kind)
    return {(OUTDOORS, BUILDING): ENTERED_BUILDING, (BUILDING, OUTDOORS): EXITED_BUILDING}.get(kinds)


class StrategyGuide(StrategistArtifact):
    def __init__(self) -> None:
        self.rules: List[GuideRule] = []
        self.removed: List[GuideRule] = []
        self.events: List[GuideEvent] = []
        self.towns: List[str] = []
        self.next_id = 0

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        self.__dict__.setdefault("towns", [])

    def is_new_town(self, town: str) -> bool:
        return town.strip().lower() not in {t.lower() for t in self.towns}

    def rule(self, rule_id: int) -> Optional[GuideRule]:
        return next((r for r in self.rules if r.id == rule_id), None)

    def rules_for(self, trigger: str) -> List[GuideRule]:
        return [r for r in self.rules if r.trigger == trigger]

    def times_fired(self, rule_id: int) -> int:
        return sum(rule_id in event.rules for event in self.events)

    def missing_seeds(self, seeds: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
        have = {(r.trigger, r.text) for r in self.rules + self.removed if r.source == "seed"}
        return [s for s in seeds if (s["trigger"], s["rule"].strip()) not in have]

    def active_event(self, gmap: GlobalMap, episode: int) -> Optional[GuideEvent]:
        if not self.events:
            return None
        event = self.events[-1]
        here = gmap.resolve(gmap.segment, (0, 0))[0]
        if episode - event.episode >= ACTIVE_EPISODES or (
                event.segment is not None and gmap.resolve(event.segment, (0, 0))[0] != here):
            return None
        return event

    @update
    def add_rule(self, trigger: str, text: str, source: str, episode: int) -> GuideRule:
        rule = GuideRule(self.next_id, trigger, text.strip(), source, episode, episode)
        self.next_id += 1
        self.rules.append(rule)
        return rule

    @update
    def edit_rule(self, rule_id: int, text: str, episode: int) -> None:
        rule = self.rule(rule_id)
        rule.text, rule.updated = text.strip(), episode

    @update
    def remove_rule(self, rule_id: int) -> None:
        rule = self.rule(rule_id)
        self.rules.remove(rule)
        self.removed.append(rule)

    @update
    def record_event(self, event: GuideEvent) -> None:
        self.events.append(event)
        if event.trigger == NEW_TOWN:
            self.towns.append(event.place)

    def diff(self, old: "StrategyGuide") -> Dict[str, Any]:
        before = {r.id: r for r in old.rules}
        return {
            "added": [r for r in self.rules if r.id not in before],
            "edited": [(before[r.id], r) for r in self.rules if r.id in before and before[r.id].text != r.text],
            "removed": [r for r in old.rules if self.rule(r.id) is None],
            "events": self.events[len(old.events):],
            "towns": self.towns[len(old.towns):],
        }


def guide_block(guide: StrategyGuide, gmap: GlobalMap, episode: int) -> str:
    event = guide.active_event(gmap, episode)
    if event is None or not guide.rules_for(event.trigger):
        return ""
    lines = [f"\nYour strategy guide: the player {event.trigger} ({event.place}) in episode "
             f"{event.episode}, so these rules of yours apply. Follow them unless the goal clearly needs "
             f"something else first:"]
    lines += [f"- {r.text}" for r in guide.rules_for(event.trigger)]
    return "\n".join(lines) + "\n"


def record_events(guide: StrategyGuide, gmap: GlobalMap, warps: Sequence[Tuple], episode: int,
                  parameters: Optional[dict] = None) -> List[GuideEvent]:
    events = []
    for _, source, _, destination, _ in warps:
        source, destination = gmap.resolve(source, (0, 0))[0], gmap.resolve(destination, (0, 0))[0]
        trigger = door_trigger(gmap, source, destination)
        if trigger is None:
            continue
        event = GuideEvent(episode, trigger, gmap.place_label(source), gmap.place_label(destination),
                           tuple(r.id for r in guide.rules_for(trigger)), destination)
        guide.record_event(event, episode_number=episode)
        events.append(event)
        log_info(f"Strategy guide: the player {trigger} ({event.place}); {len(event.rules)} rule(s) apply.",
                 parameters)
    return events


def record_new_town(guide: StrategyGuide, town: Optional[str], previous: str, episode: int,
                    parameters: Optional[dict] = None) -> Optional[GuideEvent]:
    if not town or not guide.is_new_town(town):
        return None
    event = GuideEvent(episode, NEW_TOWN, previous, town.strip(), tuple(r.id for r in guide.rules_for(NEW_TOWN)))
    guide.record_event(event, episode_number=episode)
    log_info(f"Strategy guide: the player {NEW_TOWN} ({town.strip()}); {len(event.rules)} rule(s) apply.", parameters)
    return event


def town_question(guide: StrategyGuide) -> Tuple[str, str]:
    reached = ", ".join(guide.towns) if guide.towns else "none yet"
    return (P.REVIEW_TOWN_QUESTION.replace("[TOWNS]", reached), P.REVIEW_TOWN_FORMAT)


def battle_tips(guide: StrategyGuide) -> List[str]:
    return [r.text for r in guide.rules_for(BATTLE)]


def _rules_text(guide: StrategyGuide, triggers: Tuple[str, ...]) -> str:
    rules = [r for r in guide.rules if r.trigger in triggers]
    if not rules:
        return "(none yet)"
    return "\n".join(f"[{r.id}] " + (f"when the player {r.trigger}: " if len(triggers) > 1 else "") + f"{r.text} "
                     f"({'yours' if r.source == 'strategist' else 'given to you, cannot be changed'})"
                     for r in rules)


def _parse_reflection(guide: StrategyGuide):
    def parse(output: str) -> Tuple[Any, Optional[str]]:
        wrong = field_values(output, "Wrong")
        if len(wrong) != 1 or not wrong[0].strip():
            return None, "There must be exactly one `Wrong:` line saying what went wrong."
        plan = field_values(output, "Plan")
        if len(plan) != 1:
            return None, "There must be exactly one `Plan:` line: CONTINUE, or PREPARE | <subgoal>."
        parts = [p.strip() for p in plan[0].split("|", 1)]
        if parts[0].upper().rstrip(".") == "CONTINUE":
            prepare = None
        elif parts[0].upper() == "PREPARE" and len(parts) == 2 and parts[1]:
            prepare = parts[1]
        else:
            return None, f"`Plan:` must be CONTINUE, or PREPARE | <subgoal>. Got: {plan[0]!r}"
        back = field_values(output, "Return")
        if len(back) != 1 or back[0].strip().upper().rstrip(".") not in ("YES", "NO"):
            return None, "There must be exactly one `Return:` line, YES or NO."
        changes, problem = _change_lines(guide, output, (BATTLE,))
        if problem is not None:
            return None, problem
        adds = sum(c[0] == "ADD" for c in changes) - sum(c[0] == "REMOVE" for c in changes)
        if len(guide.rules_for(BATTLE)) + adds > MAX_BATTLE_TIPS:
            return None, (f"You may keep at most {MAX_BATTLE_TIPS} battle tips; remove or rewrite one of yours "
                          f"instead of adding.")
        return {"wrong": wrong[0].strip(), "changes": changes, "prepare": prepare,
                "return": back[0].strip().upper().startswith("YES")}, None
    return parse


def reflect_on_defeat(guide: StrategyGuide, task: str, goal: str, where: str, digest: str, battle: str,
                      images: List[Any], vlm_call, game: str, episode: int,
                      parameters: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    prompt = (P.DEFEAT_REFLECTION_PROMPT.replace("[GAME]", game).replace("[TASK]", task).replace("[GOAL]", goal)
              .replace("[WHERE]", where).replace("[DIGEST]", digest).replace("[BATTLE]", battle)
              .replace("[TIPS]", _rules_text(guide, (BATTLE,))).replace("[MAX_CHANGES]", str(MAX_CHANGES))
              .replace("[MAX_TIPS]", str(MAX_BATTLE_TIPS)))
    answer = ask(vlm_call, prompt, _parse_reflection(guide), "Defeat reflection", GUIDE_ATTEMPTS, parameters,
                 images=images)
    if answer is not None:
        _apply_changes(guide, answer["changes"], episode, parameters)
    return answer


def _change_lines(guide: StrategyGuide, output: str, triggers: Tuple[str, ...]) -> Tuple[Any, Optional[str]]:
    values = field_values(output, "Change")
    if not values:
        return None, "There must be at least one `Change:` line, or the single line `Change: NONE`."
    if len(values) == 1 and values[0].strip().rstrip(".").upper() == "NONE":
        return [], None
    if len(values) > MAX_CHANGES:
        return None, f"Write at most {MAX_CHANGES} `Change:` lines."
    fixed = triggers[0] if len(triggers) == 1 else None
    changes = []
    for value in values:
        parts = [p.strip() for p in value.split("|")]
        action = parts[0].upper()
        if action == "ADD":
            if fixed is not None:
                if len(parts) != 2 or not parts[1]:
                    return None, f"`Change: ADD | <rule>` needs the rule. Got: {value!r}"
                changes.append(("ADD", fixed, parts[1]))
            elif len(parts) != 3 or parts[1].lower() not in triggers or not parts[2]:
                return None, (f"`Change: ADD | <trigger> | <rule>` needs a trigger from {', '.join(triggers)} "
                              f"and the rule. Got: {value!r}")
            else:
                changes.append(("ADD", parts[1].lower(), parts[2]))
            continue
        if action not in ("EDIT", "REMOVE"):
            return None, f"Each change must start with ADD, EDIT or REMOVE, or be NONE. Got: {value!r}"
        try:
            rule = guide.rule(int(parts[1].strip("[]")))
        except (IndexError, ValueError):
            rule = None
        if rule is None or rule.source != "strategist" or rule.trigger not in triggers:
            return None, (f"{action} needs the number of one of your own rules listed here, and only rules marked "
                          f"'yours' can be changed. Got: {value!r}")
        if any(change[1] == rule.id for change in changes if change[0] != "ADD"):
            return None, f"Change each rule at most once. Rule [{rule.id}] appears twice."
        if action == "EDIT":
            if len(parts) != 3 or not parts[2]:
                return None, f"`Change: EDIT | <number> | <new rule>` needs the new rule. Got: {value!r}"
            changes.append(("EDIT", rule.id, parts[2]))
        else:
            changes.append(("REMOVE", rule.id, None))
    return changes, None


def _apply_changes(guide: StrategyGuide, changes: Sequence[Tuple], episode: int,
                   parameters: Optional[dict] = None) -> None:
    for action, key, text in changes:
        if action == "ADD":
            rule = guide.add_rule(key, text, "strategist", episode, episode_number=episode)
            log_info(f"Strategy guide: added rule [{rule.id}] for '{key}': {text}", parameters)
        elif action == "EDIT":
            guide.edit_rule(key, text, episode, episode_number=episode)
            log_info(f"Strategy guide: rewrote rule [{key}]: {text}", parameters)
        else:
            guide.remove_rule(key, episode_number=episode)
            log_info(f"Strategy guide: removed rule [{key}].", parameters)


def update_guide(guide: StrategyGuide, events: Sequence[GuideEvent], digest: str, vlm_call, game: str,
                 episode: int, parameters: Optional[dict] = None) -> int:
    prompt = (P.GUIDE_UPDATE_PROMPT.replace("[GAME]", game)
              .replace("[EVENTS]", "\n".join(f"- {e.text}" for e in events))
              .replace("[RULES]", _rules_text(guide, EVENT_TRIGGERS))
              .replace("[TRIGGERS]", ", ".join(EVENT_TRIGGERS))
              .replace("[DIGEST]", digest)
              .replace("[MAX_CHANGES]", str(MAX_CHANGES)))
    changes = ask(vlm_call, prompt, lambda output: _change_lines(guide, output, EVENT_TRIGGERS),
                  "Strategy guide update", GUIDE_ATTEMPTS, parameters)
    _apply_changes(guide, changes or [], episode, parameters)
    return len(changes or [])
