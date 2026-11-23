from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from execution.perception.pokemon.grid import GRID_SKIP
from execution.strategist.pokemon import prompts as P
from execution.strategist.pokemon.asking import ask, single_value

from .model import AREA_KINDS
from .render import render_overlay, render_segment
from .routes import plan_route, render_plan
from .text import LEGEND, _door_text, entrance_text, segment_ascii

if TYPE_CHECKING:
    from .model import Cell, GlobalMap

MERGE_JUDGE_ATTEMPTS = 2
STUCK_ATTEMPTS = 2
NAME_ATTEMPTS = 2
MAX_NAMED_LINES = 30

UNCHANGED = "UNCHANGED"


def _parse_segment_name(review: bool):
    def parse(output: str) -> Tuple[Any, Optional[str]]:
        answer = []
        for key in ("Name", "Description"):
            value, problem = single_value(output, key)
            if problem is not None or not value:
                return None, problem or f"The `{key}:` line was empty."
            if value.strip().rstrip(".").upper() == UNCHANGED:
                if not review:
                    return None, f"This area has no {key.lower()} yet, so `{key}: {UNCHANGED}` is not possible."
                value = None
            answer.append(value)
        kind, problem = single_value(output, "Kind")
        kind = (kind or "").strip().rstrip(".").lower()
        if problem is not None or kind not in AREA_KINDS:
            return None, problem or f"`Kind:` must be one of {', '.join(AREA_KINDS)}, not {kind!r}."
        return (*answer, kind), None
    return parse


def name_segment(gmap: GlobalMap, segment: int, mark: Optional[Cell], left: bool, tasks: str, steps: str, vlm_call,
                 game: str, episode: int, parameters: Optional[dict] = None) -> Optional[Tuple[bool, str, str]]:
    import cv2
    from PIL import Image

    info = gmap.info(segment)
    review = info.name is not None
    named = [f"- {other.name}: {other.description}" for si, other in sorted(gmap.segment_info.items())
             if other.name is not None and si != segment and si not in gmap.merged_into]
    if left:
        situation, steps_title = "The player has just left an area", "What the player did before leaving the area"
        mark_image, mark_text = "where the player left the area", "where the player left"
    else:
        situation, steps_title = "The player has been exploring an area", "What the player did in the last episode"
        mark_image, mark_text = "where the player is now", "where the player is now"
    situation += " you have already named." if review else " that has no name yet."
    ask_text = (P.SEGMENT_NAME_REVIEW.replace("[NAME]", info.name).replace("[DESCRIPTION]", info.description or "")
                .replace("[BEFORE]", str(info.named_pixels // GRID_SKIP ** 2))
                .replace("[AFTER]", str(gmap.filled_pixels(segment) // GRID_SKIP ** 2))
                if review else P.SEGMENT_NAME_NEW)
    came = entrance_text(gmap, segment)
    prompt = (P.SEGMENT_NAME_PROMPT.replace("[GAME]", game)
              .replace("[SITUATION]", situation)
              .replace("[ENTRANCE]", P.SEGMENT_NAME_ENTRANCE.replace("[ENTRANCE]", came) if came else "")
              .replace("[MARK_IMAGE]", f"the magenta box is {mark_image}, and " if mark is not None else "and ")
              .replace("[MARK_TEXT]", f", with X {mark_text}" if mark is not None else "")
              .replace("[MAP]", segment_ascii(gmap, segment, mark))
              .replace("[LEGEND]", LEGEND)
              .replace("[TASKS]", tasks or "(none yet)")
              .replace("[STEPS_TITLE]", steps_title)
              .replace("[STEPS]", steps or "(nothing was recorded)")
              .replace("[NAMED]", "\n".join(named[-MAX_NAMED_LINES:]) or "(none yet)")
              .replace("[ASK]", ask_text)
              .replace("[NAME_FORMAT]", f'<a short name, or {UNCHANGED} to keep "{info.name}">' if review
                       else "<a short name>")
              .replace("[DESCRIPTION_FORMAT]", f"<one to three sentences, or {UNCHANGED} to keep the current one>"
                       if review else "<one to three sentences>"))
    canvas = render_segment(gmap, segment, highlight=[mark] if mark is not None else [], doors=True)
    images = [Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))] if canvas is not None else []
    answer = ask(vlm_call, prompt, _parse_segment_name(review), "Segment naming", NAME_ATTEMPTS, parameters,
                 images=images)
    if answer is None:
        return None
    changed = gmap.set_name(segment, answer[0], answer[1], answer[2], episode)
    gmap.stats["segments_renamed" if review and changed else "segments_named" if changed else "names_kept"] += 1
    return changed, info.name, info.description


def _parse_merge_decision(output: str) -> Tuple[Any, Optional[str]]:
    decision, problem = single_value(output, "Decision")
    if problem is not None:
        return None, f"{problem} It must be SAME or DIFFERENT."
    decision = (decision or "").strip().strip(".").upper()
    if decision not in ("SAME", "DIFFERENT"):
        return None, f"The decision must be SAME or DIFFERENT, not {decision!r}."
    return decision == "SAME", None


def judge_merge(gmap: GlobalMap, source: int, target: int, overlap: Dict[str, Any], vlm_call, game: str,
                parameters: Optional[dict] = None) -> Optional[bool]:
    import cv2
    from PIL import Image

    a, b = gmap.info(target), gmap.info(source)
    prompt = (P.MERGE_JUDGE_PROMPT.replace("[GAME]", game)
              .replace("[NAME_A]", a.name or "").replace("[DESCRIPTION_A]", a.description or "")
              .replace("[DOORS_A]", _door_text(gmap, target))
              .replace("[NAME_B]", b.name or "").replace("[DESCRIPTION_B]", b.description or "")
              .replace("[DOORS_B]", _door_text(gmap, source))
              .replace("[SHARED]", str(overlap["shared"])).replace("[MATCHES]", str(overlap["matches"])))
    canvases = [render_segment(gmap, target, trail=False, doors=True),
                render_segment(gmap, source, trail=False, doors=True),
                render_overlay(gmap, source, target, overlap["offset"])]
    images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in canvases if c is not None]
    return ask(vlm_call, prompt, _parse_merge_decision, "Merge judgement", MERGE_JUDGE_ATTEMPTS, parameters,
               images=images)


# At most this many looked-up routes are kept and shown at every dispatch.
MAX_ACTIVE_ROUTES = 2
NAVIGATE_ATTEMPTS = 2


def _parse_choice(count: int):
    valid = f"numbers from 0 to {count - 1} separated by commas, or None"

    def parse(output: str) -> Tuple[Any, Optional[str]]:
        choice, problem = single_value(output, "Choice")
        if problem is not None:
            return None, f"{problem} It must hold {valid}."
        if choice is None:
            return [], None
        indices: List[int] = []
        for part in choice.split(","):
            part = part.strip().rstrip(".").strip()
            if not part.isdigit() or int(part) >= count:
                return None, f"{part!r} is not an area number. The choice must be {valid}."
            if int(part) not in indices:
                indices.append(int(part))
        return indices, None
    return parse


def navigate_map(gmap: GlobalMap, request: str, vlm_caller, game: str,
                 parameters: Optional[dict] = None) -> Optional[str]:
    live =[si for si, seg in enumerate(gmap.segments) if seg and si not in gmap.merged_into]
    plans = {si: plan_route(gmap, [si]) for si in live}
    entries = sorted(live, key=lambda si: (plans[si] is None, plans[si]["cost"] if plans[si] else 0, si))
    lines = []
    for index, si in enumerate(entries):
        plan, info = plans[si], gmap.info(si)
        note = ("no known route" if plan is None else "you are here" if not plan["hops"]
                else f"{len(plan['hops'])} door(s) away")
        lines.append(f"{index}. {gmap.area_label(si)}" + (f": {info.description}" if info.description else "")
                     + f" ({note})")
    prompt = (P.NAVIGATE_SELECT_PROMPT.replace("[GAME]", game)
              .replace("[REQUEST]", request)
              .replace("[CURRENT]", gmap.place_label())
              .replace("[PLACES]", "\n".join(lines)))
    indices = ask(vlm_caller("navigate_select"), prompt, _parse_choice(len(entries)), "Navigate selection",
                  NAVIGATE_ATTEMPTS, parameters)
    if indices is None:
        return None
    if not indices:
        return f"No area on your map fits \"{request}\"."
    chosen = [entries[i] for i in indices]
    plan = plan_route(gmap, chosen)
    if plan is not None and plan["hops"]:
        gmap.routes = (gmap.routes + [{"request": request, "targets": chosen, "step": gmap.step}])[-MAX_ACTIVE_ROUTES:]
        gmap.stats["routes_planned"] += 1
    return render_plan(gmap, chosen, plan)


def _parse_stuck(output: str) -> Tuple[Any, Optional[str]]:
    decision, problem = single_value(output, "Stuck")
    if problem is not None:
        return None, f"{problem} It must be YES or NO."
    decision = (decision or "").strip().strip(".").upper()
    if decision not in ("YES", "NO"):
        return None, f"The answer must be YES or NO, not {decision!r}."
    return decision == "YES", None


def judge_stuck(gmap: GlobalMap, signs: str, episodes: str, frame: np.ndarray, vlm_call, game: str,
                parameters: Optional[dict] = None) -> Optional[bool]:
    here = gmap.info(gmap.segment)
    area = f'the area you named "{here.name}": {here.description}' if here.name else "an area you have not named yet"
    prompt = (P.STUCK_PROMPT.replace("[GAME]", game).replace("[SIGNS]", signs).replace("[EPISODES]", episodes)
              .replace("[AREA]", area).replace("[AREAS]", str(gmap.n_areas)))
    return ask(vlm_call, prompt, _parse_stuck, "Stuck check", STUCK_ATTEMPTS, parameters, images=[frame])
