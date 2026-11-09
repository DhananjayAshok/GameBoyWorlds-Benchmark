"""
The model steps of the NPC registry: who is this person (a card), is this look-alike somebody already met (a judge),
what was said (a summary) and what is the person called (a real name).

Code does everything it can first (see ``npcs.py``); the model is asked only what code cannot tell:

* a **card** for a person nobody has described yet: kind, which way the shown picture faces, a one-line description,
  and whether it is somebody already met seen from another side;
* a **judge** for a person who looks exactly like one met elsewhere: usually another person, sometimes a rival or a
  professor who travels;
* a **summary** of each conversation with its speaker's own name when the text gives one.

Every call is batched, every reply is parsed strictly, and an unusable reply is retried at the next episode end; after
two unusable replies the person (or conversation) is left as it is. Nothing here can break a run: the caller wraps it.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from execution.perception.pokemon.grid import cell_box, draw_box
from execution.strategist.pokemon.asking import field_values
from utils import log_info, log_warn
from utils.lm_inference import clean_value

#: People (or conversations) sent to the model in one call; the rest wait for the next episode end.
MAX_BATCH = 24
#: A look-alike is compared with at most this many people.
MAX_CANDIDATES = 8
#: Replies that cannot be used before the person (or conversation) is left as it is.
MAX_TRIES = 2

NPC_CARD_PROMPT = """You are looking at [GAME]. Image 1 is the game screen with one character outlined in red. Image 2 is that character enlarged. [CANDIDATES]Answer about the outlined character only.

Respond in exactly this format:
Kind: <person | trainer | pokemon | object>
Facing: <down | up | left | right>
Description: <one sentence on appearance and likely role, for example "bald man in a white coat, a scientist">
Same as: <the number of the matching earlier character, or None>
[STOP]

Kind: object means a statue, doll, machine or anything that is not a character. Facing is which way the character faces, as seen on the screen."""

NPC_CANDIDATES_BLOCK = """The images after those are characters already met; each is "[k] name: description, last seen in an area". The outlined character may be one of them seen from another side, or a different person who looks alike. Say "Same as: k" only if it is the same individual.
[LIST]
"""

NPC_JUDGE_PROMPT = """You are looking at [GAME]. Image 1 is the screen with one character outlined in red, now in [AREA]. The images after it are characters you met before who look exactly like it:
[LIST]
In this game many people share a look, so a look-alike in another place is usually a different person. Say it is the same person only if the story makes that likely (a rival, a professor, a character who travels).

Respond in exactly this format:
Same as: <number, or None>
Reason: <one short sentence>
[STOP]"""

NPC_SUMMARY_PROMPT = """In [GAME] the player just had this exchange [WITH] in [AREA]:
<<<[TEXT]>>>
[HINT]
Respond in exactly this format:
Summary: <one sentence: what was said and anything the player got or must do>
Speaker name: <the speaker's own name if the text gives it (for example "I'm PROF. OAK" or "OAK: ..."), else None>
Name type: <personal | class | None>
Outcome: <info | gift | battle | choice | other>
[STOP]

Name type: class means a trainer class such as YOUNGSTER or LASS rather than a personal name."""

KINDS = ("person", "trainer", "pokemon", "object")
OUTCOMES = ("info", "gift", "battle", "choice", "other")
NAME_TYPES = ("personal", "class")
TAG = re.compile(r"^\s*([A-Z][A-Z.' ]{1,12}):")


# -- pictures ---------------------------------------------------------------------------------------------------

def _grey(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    return frame[:, :, 0] if frame.ndim == 3 and frame.shape[2] == 1 else frame


def screen_with_box(frame: np.ndarray, cell: Tuple[int, int]):
    """The screen with a red box on the square a person stands on."""
    return draw_box(_grey(frame), *cell_box(*cell))


def sprite_x4(view) -> np.ndarray:
    return cv2.resize(np.asarray(view.pixels, dtype=np.uint8), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)


def look_sheet(look, max_views: int = 4) -> np.ndarray:
    """Up to four views of a look side by side, enlarged."""
    return np.concatenate([sprite_x4(v) for v in look.views[:max_views]], axis=1)


# -- naming and listing -----------------------------------------------------------------------------------------

def display_name(npc) -> str:
    return npc.real_name or npc.name


def _area(reg, gmap, segment: int) -> str:
    try:
        return gmap.info(segment).name or "an area you have not named"
    except Exception:
        return "an unknown area"


def _last_summary(reg, npc) -> str:
    for cid in reversed(npc.conversations):
        c = reg.conversations.get(cid)
        if c is not None and (c.summary or c.text):
            return c.summary or c.text[:120].replace("\n", " ")
    return ""


def _describe(reg, gmap, npc, number: int, met: bool = False) -> str:
    seen = reg.last_seen(gmap, npc)
    where = _area(reg, gmap, seen[0]) if seen else "an unknown area"
    text = f"[{number}] {display_name(npc)}: {npc.description or 'no description yet'}, "
    text += f"met in {where}" if met else f"last seen in {where}"
    said = _last_summary(reg, npc) if met else ""
    return text + (f', said: "{said[:120]}"' if said else "")


def _candidates(reg, npc) -> List[Any]:
    """The people a new person might really be: live, not on screen with it, no duplicates, at most a handful."""
    out, seen = [], {npc.id}
    for cid in npc.candidates:
        if cid not in reg.npcs:
            continue
        other = reg.get(cid)
        if other.id in seen or other.merged_into is not None:
            continue
        seen.add(other.id)
        out.append(other)
    return out[:MAX_CANDIDATES]


# -- parsing ----------------------------------------------------------------------------------------------------

def _word(output: str, key: str, allowed: Tuple[str, ...]) -> Tuple[Optional[str], Optional[str]]:
    values = field_values(output, key)
    if len(values) != 1:
        return None, f"exactly one {key} line is needed, found {len(values)}"
    value = values[0].lower().strip(" .\"'*")
    for word in allowed:
        if value == word or value.startswith(word + " ") or value.startswith(word + ","):
            return word, None
    return None, f"{key} must be one of {', '.join(allowed)}"


def _same_as(output: str, limit: int) -> Tuple[Optional[int], Optional[str]]:
    values = field_values(output, "Same as")
    if len(values) != 1:
        return None, f"exactly one Same as line is needed, found {len(values)}"
    value = clean_value(values[0])
    if value is None:
        return 0, None
    match = re.match(r"^\[?(\d+)\]?\b", value.strip())
    if not match or not 1 <= int(match.group(1)) <= limit:
        return None, f"Same as must be None or a number from 1 to {limit}"
    return int(match.group(1)), None


def parse_card(output: str, n_candidates: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    kind, problem = _word(output, "Kind", KINDS)
    if problem:
        return None, problem
    facing, problem = _word(output, "Facing", ("down", "up", "left", "right"))
    if problem:
        facing = None                                            # a missing facing is not worth a retry
    same, problem = _same_as(output, n_candidates) if n_candidates else (0, None)
    if problem:
        return None, problem
    description = clean_value((field_values(output, "Description") or [""])[0])
    return {"kind": kind, "facing": facing, "description": (description or "")[:200], "same": same}, None


def parse_judge(output: str, n_candidates: int) -> Tuple[Optional[int], Optional[str]]:
    return _same_as(output, n_candidates)


def parse_summary(output: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    summary = clean_value((field_values(output, "Summary") or [""])[0])
    if not summary:
        return None, "a Summary line is needed"
    name = clean_value((field_values(output, "Speaker name") or [""])[0])
    kind = (field_values(output, "Name type") or [""])[0].lower().strip(" .\"'*")
    outcome = (field_values(output, "Outcome") or [""])[0].lower().strip(" .\"'*")
    return {"summary": summary[:300], "name": name[:40] if name else None,
            "name_type": kind if kind in NAME_TYPES else None,
            "outcome": outcome if outcome in OUTCOMES else "other"}, None


def _replies(out: Any, n: int) -> List[str]:
    """One reply per prompt. A thinking model answers ``Reasoning: ...  Response:  Kind: person``: only what follows
    the last ``Response:`` is the answer (the parsers look for lines that start with their key)."""
    out = out if isinstance(out, list) else [out]
    out = [str(x).rsplit("Response:", 1)[1] if "Response:" in str(x) else str(x) for x in out]
    return [x.strip() for x in out] + [""] * (n - len(out))


# -- merging ----------------------------------------------------------------------------------------------------

def try_merge(reg, keep, drop) -> bool:
    """Fold ``drop`` into ``keep`` if they were never on screen together and their pictures can be joined."""
    if keep.id == drop.id or frozenset((keep.id, drop.id)) in reg.covisible:
        return False
    keep_look, drop_look = reg.look_of(keep), reg.look_of(drop)
    if keep_look.id != drop_look.id:
        others = [n for n in reg.live_npcs() if n.id != drop.id and reg.look_of(n).id == drop_look.id]
        if others:                                               # the drop look is shared: sightings would point wrong
            return False
        reg.merge_looks(keep_look.id, drop_look.id)
    if keep.kind == "unknown" and drop.kind != "unknown":
        keep.kind = drop.kind
    if not keep.description and drop.description:
        keep.description = drop.description
    reg.merge_npcs(keep, drop)
    keep.needs = None
    return True


# -- the steps --------------------------------------------------------------------------------------------------

def run_model(reg, vlm_card: Callable[..., Any], vlm_judge: Callable[..., Any], game: str, gmap) -> None:
    """Cards for people nobody described, judges for look-alikes met in another area (two batched calls)."""
    cards = [n for n in reg.live_npcs() if n.needs == "card"][:MAX_BATCH]
    judges = [n for n in reg.live_npcs() if n.needs == "judge"][:MAX_BATCH]
    if cards:
        _cards(reg, cards, vlm_card, game, gmap)
    if judges:
        _judges(reg, judges, vlm_judge, game, gmap)


def _failed(reg, npc) -> None:
    npc.tries += 1
    if npc.tries >= MAX_TRIES:
        npc.needs = None
        reg.stats["model_gave_up"] += 1


def _cards(reg, people, vlm_call, game: str, gmap) -> None:
    prompts, images, cands = [], [], []
    for npc in people:
        others = _candidates(reg, npc)
        cands.append(others)
        block = ""
        if others:
            lines = "\n".join(_describe(reg, gmap, o, k + 1) for k, o in enumerate(others))
            block = NPC_CANDIDATES_BLOCK.replace("[LIST]", lines)
        prompts.append(NPC_CARD_PROMPT.replace("[GAME]", game).replace("[CANDIDATES]", block))
        look = reg.look_of(npc)
        first = npc.sightings[0].view if npc.sightings else 0
        view = look.views[min(first, len(look.views) - 1)]
        pics = ([screen_with_box(npc.context, npc.context_cell)] if npc.context is not None and npc.context_cell
                else [])
        pics += [sprite_x4(view)] + [look_sheet(reg.look_of(o)) for o in others]
        images.append(pics)
    out = _replies(vlm_call(texts=prompts, images=images, max_new_tokens=2500), len(people))
    for npc, others, reply in zip(people, cands, out):
        card, problem = parse_card(reply, len(others))
        if card is None:
            log_warn(f"NPC card for {npc.name} unusable: {problem}")
            _failed(reg, npc)
            continue
        reg.stats["cards"] += 1
        if card["same"] and try_merge(reg, others[card["same"] - 1], npc):
            reg.stats["merged_by_model"] += 1
            continue
        _apply_card(reg, npc, card)


def _apply_card(reg, npc, card: Dict[str, Any]) -> None:
    look = reg.look_of(npc)
    if card["kind"] == "object":
        look.kind_hint = "object"
    if npc.kind in ("unknown", "person") or card["kind"] == "pokemon":
        npc.kind = card["kind"]
    if card["description"]:
        npc.description = card["description"]
    if card["facing"] and npc.sightings:
        view = look.views[min(npc.sightings[0].view, len(look.views) - 1)]
        view.facing = card["facing"]
    npc.needs = None


def _judges(reg, people, vlm_call, game: str, gmap) -> None:
    prompts, images, cands = [], [], []
    for npc in people:
        others = _candidates(reg, npc)
        cands.append(others)
        seen = reg.last_seen(gmap, npc)
        lines = "\n".join(_describe(reg, gmap, o, k + 1, met=True) for k, o in enumerate(others))
        prompts.append(NPC_JUDGE_PROMPT.replace("[GAME]", game).replace("[LIST]", lines)
                       .replace("[AREA]", _area(reg, gmap, seen[0]) if seen else "an unknown area"))
        pics = ([screen_with_box(npc.context, npc.context_cell)] if npc.context is not None and npc.context_cell
                else [sprite_x4(reg.look_of(npc).views[0])])
        images.append(pics + [look_sheet(reg.look_of(o)) for o in others[:1]])
    out = _replies(vlm_call(texts=prompts, images=images, max_new_tokens=1500), len(people))
    for npc, others, reply in zip(people, cands, out):
        if not others:
            npc.needs = None
            continue
        same, problem = parse_judge(reply, len(others))
        if same is None:
            log_warn(f"NPC judge for {npc.name} unusable: {problem}")
            _failed(reg, npc)
            continue
        reg.stats["judged"] += 1
        if same and try_merge(reg, others[same - 1], npc):
            reg.stats["merged_by_model"] += 1
            continue
        peers = [n for n in reg.live_npcs() if n.id != npc.id and reg.look_of(n).id == reg.look_of(npc).id
                 and n.description]
        if peers and not npc.description:                       # a look already described is not described again
            best = Counter((p.kind, p.description) for p in peers).most_common(1)[0][0]
            npc.kind, npc.description = best
        npc.needs = None


def _same_text(text: str) -> str:
    return re.sub(r"[\s▼▶]+", " ", text).strip().casefold()


def run_summaries(reg, vlm_call: Callable[..., Any], game: str, gmap, only=None) -> None:
    """One batched, text-only call for every conversation with text that has no summary yet (or just ``only``)."""
    todo = [c for c in (only if only is not None else reg.conversations.values())
            if c.text.strip() and not c.summarised][:MAX_BATCH]
    done = {}
    for c in reg.conversations.values():
        if c.summary:
            done.setdefault(_same_text(c.text), c)
    for c in list(todo):                                         # the same words again: the earlier summary stands
        twin = done.get(_same_text(c.text))
        if twin is not None and twin is not c:
            c.summary, c.speaker_name, c.name_type, c.summarised = (twin.summary, twin.speaker_name,
                                                                    twin.name_type, True)
            c.outcome = "battle" if c.led_to == "battle" else twin.outcome
            todo.remove(c)
    if not todo:
        return
    prompts = []
    for c in todo:
        with_whom = (f"with {display_name(reg.get(c.npc))}" if c.npc in reg.npcs
                     else "with a sign" if c.sign else "with someone")
        tag = TAG.match(c.text)
        hint = f'The text shows the tag "{tag.group(1)}:" before the speech.\n' if tag else ""
        prompts.append(NPC_SUMMARY_PROMPT.replace("[GAME]", game).replace("[WITH]", with_whom)
                       .replace("[AREA]", _area(reg, gmap, gmap.resolve(c.segment, c.square)[0]))
                       .replace("[TEXT]", c.text[:1500]).replace("[HINT]", hint))
    out = _replies(vlm_call(texts=prompts, max_new_tokens=1500), len(todo))
    for c, reply in zip(todo, out):
        parsed, problem = parse_summary(reply)
        if parsed is None:
            log_warn(f"NPC summary of {c.id} unusable: {problem}")
            c.tries += 1
            if c.tries >= MAX_TRIES:
                c.summarised = True
            continue
        c.summary, c.speaker_name, c.name_type, c.summarised = (parsed["summary"], parsed["name"],
                                                                parsed["name_type"], True)
        c.outcome = "battle" if c.led_to == "battle" else parsed["outcome"]
        reg.stats["summaries"] += 1


def apply_real_names(reg) -> None:
    """A person who gave their own name in most of their conversations is called that from now on, unless the name
    is already somebody else's (two people are never merged because they share a name)."""
    taken = {n.real_name.casefold() for n in reg.live_npcs() if n.real_name}
    for npc in reg.live_npcs():
        if npc.real_name:
            continue
        names = Counter((c.speaker_name, c.name_type) for c in
                        (reg.conversations[i] for i in npc.conversations if i in reg.conversations)
                        if c.speaker_name)
        for (name, kind), _ in names.most_common():
            if kind != "personal" or name.casefold() in taken:
                continue
            npc.real_name, npc.name_type = name, kind
            taken.add(name.casefold())
            reg.stats["real_names"] += 1
            break
