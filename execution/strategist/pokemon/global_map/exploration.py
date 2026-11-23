from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from .routes import plan_route
from .text import LEGEND, ascii_window, door_lines, elsewhere_lines, frontier_targets, route_lines

if TYPE_CHECKING:
    from .model import Cell, GlobalMap

# Episodes without a new cell, or failed in a row, before the strategist is asked whether it is stuck.
STALE_EPISODES = 3
FAILURE_EPISODES = 3
# Episodes exploration stays in charge once the strategist judges itself stuck.
EXPLORE_EPISODES = 3


def explore_elsewhere(gmap: GlobalMap) -> List[Dict[str, Any]]:
    groups: Dict[Any, Dict[str, Any]] = {}
    unreachable = []
    for si, seg in enumerate(gmap.segments):
        if not seg or si in gmap.merged_into or si == gmap.segment or not gmap.frontier(si):
            continue
        plan = plan_route(gmap, [si])
        if plan is None or not plan["hops"]:
            unreachable.append(si)
            continue
        first = plan["hops"][0]
        group = groups.setdefault((first["depart"], first["to"]), {"plan": plan, "target": si, "areas": []})
        group["areas"].append((si, plan))
        if plan["cost"] < group["plan"]["cost"]:
            group["plan"], group["target"] = plan, si
    out = sorted(groups.values(), key=lambda g: g["plan"]["cost"])
    for g in out:
        g["areas"].sort(key=lambda a: a[1]["cost"])
    if unreachable:
        out.append({"plan": None, "target": None, "areas": [(si, None) for si in unreachable]})
    return out


def explore_text(gmap: GlobalMap) -> str:
    here, lines = gmap.area_label(gmap.segment), []
    edge = len(gmap.frontier())
    if edge:
        lines.append(f"- {here}, where you are: {edge} unexplored edge square{'s' if edge != 1 else ''}.")
    unentered = sum(door["leads_to"] is None for door in gmap.door_list())
    if unentered:
        lines.append(f"- {here}, where you are: {unentered} door(s) or stairs not entered yet.")
    groups = explore_elsewhere(gmap)
    if groups:
        lines.append("Other areas with unexplored edges, grouped by the way out of here:")
        lines += elsewhere_lines(gmap, groups)
    return "\n".join(lines) or "(none)"


def exploring(gmap: GlobalMap, episode: Optional[int]) -> bool:
    return episode is not None and gmap.exploring_until is not None and episode <= gmap.exploring_until


def keep_exploring(gmap: GlobalMap, until: int) -> None:
    gmap.exploring_until = until


def stuck_signs(gmap: GlobalMap, recent_episodes: Iterable[int], failure_streak: int) -> Optional[str]:
    stale = gmap.episodes_without_new_cells(list(recent_episodes))
    signs = []
    if stale >= STALE_EPISODES:
        signs.append(f"the last {stale} episodes found nothing new for your map")
    if failure_streak >= FAILURE_EPISODES:
        signs.append(f"the last {failure_streak} episodes did not complete their task")
    return " and ".join(signs) if signs else None


def prompt_block(gmap: GlobalMap, recent_episodes: Iterable[int],
                 local_kinds: Optional[Dict[Cell, str]] = None,
                 explore: bool = False, episode: Optional[int] = None,
                 failed_tasks: Iterable[str] = (), busy: bool = False) -> Tuple[str, Optional[Dict[str, Any]]]:
    """The map section of the dispatch prompt and, when exploring, the target that overrides the task.
    Does not change the map: the caller applies override["until"] with `keep_exploring`.
    `busy`: the building survey already sets this episode's task, so exploring does not override it."""
    recent = list(recent_episodes)
    stale = gmap.episodes_without_new_cells(recent)
    targets, mass = frontier_targets(gmap, local_kinds)
    here = gmap.info(gmap.segment)
    lines = [f"You are in the area you named {gmap.place_label()}: {here.description}"] if here.name else []
    if here.explored is not None:
        lines.append("You have fully explored this area: no edge of its map is left unexplored.")
    lines += [
        "MAP OF EVERYTHING YOU HAVE SEEN IN THIS AREA (north is up, one character per step):",
        ascii_window(gmap),
        "Legend: " + LEGEND,
        f"Squares mapped here: {len(gmap.current)} (in {gmap.n_areas} separate area(s) overall). "
        f"New squares found in the last episode: "
        f"{gmap.new_cells_by_episode.get(max(recent), 0) if recent else 0}; "
        f"episodes in a row with nothing new: {stale}.",
    ]
    lines += door_lines(gmap)
    lines += route_lines(gmap)
    if targets:
        lines.append("The nearest edge of the map you have NOT explored yet in each direction "
                     "(x = steps to the right, y = steps up, measured from you):")
        for t in targets:
            where = "on screen, reachable" if t["on_screen"] else "off screen"
            lines.append(f"- ({t['dx']}, {t['dy']}): {t['steps']} steps {t['sector']} ({where})")
        lines.append("Most unexplored edge lies: " +
                     ", ".join(f"{s} ({n} squares)" for s, n in mass.most_common(3)) + ".")
    else:
        lines.append("Every reachable edge of this area has been explored; the way on is a door, "
                     "stairs or a path you have not tried.")
    override = None
    triggered = explore and episode is not None and not busy
    # Exploration stays in charge for a few episodes: one is not enough to rule a direction out.
    active = exploring(gmap, episode) and not busy
    elsewhere = explore_elsewhere(gmap) if (triggered or active) and not targets else []
    if (triggered or active) and (targets or elsewhere):
        until = gmap.exploring_until if active else episode + EXPLORE_EPISODES - 1
        left = until - episode + 1
        stuck = (f"You judged that you are stuck, so what you have been trying is not working, and you are exploring "
                 f"({left} episode(s) of exploring left).")
        if targets:
            heading = mass.most_common(1)[0][0]
            override = next(t for t in targets if t["sector"] == heading)   # every direction with mass has one
            override = dict(override, kind="edge", area=gmap.segment, until=until)
            lines.append(
                f"{stuck} This episode's task MUST be to walk to the unexplored edge of {gmap.area_label(gmap.segment)} "
                f"at ({override['dx']}, {override['dy']}), {override['steps']} steps {override['sector']}, "
                f"and keep going into the area beyond it.")
        else:
            override = {"kind": "elsewhere", "areas": [si for g in elsewhere for si, _ in g["areas"]],
                        "until": until}
            lines.append(
                f"{stuck} Nothing is left to explore in {gmap.area_label(gmap.segment)}. This episode's task MUST "
                f"be to head for one of the areas below that still has unexplored edges, and explore them. "
                f"Areas that share the way out of here are listed together:")
            lines += elsewhere_lines(gmap, elsewhere)
        failed = [t for t in dict.fromkeys(failed_tasks) if t][:3]
        if failed:
            lines.append("Do NOT set any of these recent tasks again — each of them failed: " +
                         "; ".join(f'"{t}"' for t in failed) + ".")
    return "\n".join(lines), override
