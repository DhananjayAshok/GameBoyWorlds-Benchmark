"""
The building survey: wherever the player is, go inside every building once and talk to everyone in it.

Read off the global map (doors, where they lead, what is unexplored) and the people registry (who has been
talked to). It only writes rules into the strategist's prompts; the strategist still writes every task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from execution.strategist.pokemon.global_map import BUILDING, CAVE, GlobalMap, door_uses, entrance, sector
from execution.strategist.pokemon.global_map.routes import plan_route, render_plan

Cell = Tuple[int, int]

# A door, person or edge is given up on after this many tasks that neither reached it nor got nearer.
MAX_TRIES = 3
# Doors and unexplored edges farther than this many steps from the player belong to somewhere else.
TOWN_REACH = 24
MAX_LINES = 6

INSIDE, OUTSIDE, MAYBE_INSIDE = "inside", "outside", "maybe inside"

HEADER = ("BUILDINGS AND PEOPLE. Wherever you are, go inside every building once and talk to everyone in it before "
          "you leave: that is how you find out what a place has. Until the place you are in is finished, this comes "
          "before your goal. Only two things come before it: a rule of your strategy guide that applies now, and "
          "healing your team when it is too hurt to go on. Talk to each person once: do not talk again to someone you "
          "have already talked to unless your goal or the task in hand needs it.")
COORDINATES = "x = steps to the right, y = steps up, measured from you"


@dataclass
class Door:
    cells: List[Cell]
    leads_to: Optional[int]
    dx: int
    dy: int
    uses: int
    tries: int
    waiting: int = 0

    @property
    def steps(self) -> int:
        return abs(self.dx) + abs(self.dy)

    @property
    def where(self) -> str:
        return _where(self.dx, self.dy)

    @property
    def line(self) -> str:
        return f"- {self.where}" + ("" if self.steps else ": step off it and walk back onto it to go through")


@dataclass
class Survey:
    lines: List[str] = field(default_factory=list)
    aim: Optional[Dict[str, Any]] = None
    log: str = ""

    @property
    def text(self) -> str:
        return "\n".join([HEADER] + self.lines) if self.lines else ""


def _where(dx: int, dy: int) -> str:
    steps = abs(dx) + abs(dy)
    return f"({dx}, {dy}), {steps} steps {sector(dx, dy)}" if steps else "(0, 0), where you stand"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def setting(gmap: GlobalMap, segment: int) -> Optional[str]:
    """Whether an area is inside a building; None for a cave, which is passed through, not surveyed.
    An area with no name yet is outside if nothing led into it and maybe inside if a door did."""
    kind = gmap.info(segment).kind
    if kind == CAVE:
        return None
    if kind is not None:
        return INSIDE if kind == BUILDING else OUTSIDE
    return OUTSIDE if entrance(gmap, segment) is None else MAYBE_INSIDE


def _door_tries(gmap: GlobalMap, segment: int, cells: Iterable[Cell]) -> int:
    squares = {(segment, cell) for cell in cells}
    return sum(n for key, n in gmap.survey_tries.items()
               if key[0] == "door" and gmap.resolve(key[1], key[2]) in squares)


def doors_of(gmap: GlobalMap, segment: int, origin: Optional[Cell] = None) -> List[Door]:
    """The doors of an area, nearest to `origin` (the player by default) first, without links back into itself."""
    segment = gmap.resolve(segment, (0, 0))[0]
    ox, oy = gmap.pos if origin is None else origin
    doors = []
    for door in gmap.door_list(segment):
        leads_to = None if door["leads_to"] is None else gmap.resolve(door["leads_to"], (0, 0))[0]
        if leads_to == segment:
            continue
        dx, dy = min(((g[0] - ox, g[1] - oy) for g in door["cells"]), key=lambda d: abs(d[0]) + abs(d[1]))
        doors.append(Door(door["cells"], leads_to, dx, dy, door_uses(gmap, segment, door["cells"]),
                          _door_tries(gmap, segment, door["cells"])))
    return sorted(doors, key=lambda d: (d.steps, d.dx, d.dy))


def _waiting(gmap: GlobalMap, people: Iterable[Any]) -> List[Any]:
    """People not talked to yet that are still worth a try, nearest first."""
    px, py = gmap.pos
    out = [m for m in people if m.talked == "no" and gmap.survey_tries[("person", m.npc_id)] < MAX_TRIES]
    return sorted(out, key=lambda m: (abs(m.square[0] - px) + abs(m.square[1] - py), m.number))


def _given_up(gmap: GlobalMap, people: Iterable[Any]) -> int:
    return sum(m.talked == "no" and gmap.survey_tries[("person", m.npc_id)] >= MAX_TRIES for m in people)


def _way_out(gmap: GlobalMap, segment: int, doors: List[Door]) -> str:
    outside = [d for d in doors if d.leads_to is not None and setting(gmap, d.leads_to) == OUTSIDE]
    if outside:
        return f"through the door at {outside[0].where}, to {gmap.area_label(outside[0].leads_to)}"
    parent = gmap.parent(segment)
    if parent is not None:
        plan = plan_route(gmap, [parent])
        if plan is not None and plan["hops"]:
            return "towards " + gmap.area_label(parent) + ". " + render_plan(gmap, [parent], plan).splitlines()[-1]
    came = entrance(gmap, segment)
    back = [d for d in doors if came is not None and d.leads_to == came[0]]
    if back:
        return f"through the door you came in by, at {back[0].where}, to {gmap.area_label(back[0].leads_to)}"
    return "the way you came in"


def settle(gmap: GlobalMap, people_in: Optional[Callable[[int], List[Any]]] = None) -> Optional[str]:
    """Judge the last task's aim: a try is counted when it was neither reached nor got nearer.
    Returns what was counted, for the log."""
    aim = gmap.survey_aim
    gmap.aim_survey(None)
    if aim is None:
        return None
    here = gmap.resolve(gmap.segment, gmap.pos)[0]
    segment = gmap.resolve(aim["segment"], (0, 0))[0]
    px, py = gmap.pos
    if aim["kind"] == "edge":
        if gmap.new_cells_by_episode.get(aim["episode"], 0) > 0:
            return None
        key, what = ("edge", aim["segment"]), "an unexplored edge showed nothing new"
    elif aim["kind"] == "person":
        if people_in is None:
            return None
        mark = next((m for m in people_in(segment) if m.npc_id == aim["npc"]), None)
        if mark is not None and (mark.talked != "no" or (
                here == segment and abs(mark.square[0] - px) + abs(mark.square[1] - py) < aim["steps"])):
            return None
        key, what = ("person", aim["npc"]), "a person was not talked to"
    else:
        cell = gmap.resolve(aim["segment"], aim["cell"])[1]
        door = next((d for d in doors_of(gmap, segment) if cell in d.cells), None)
        if door is None or (aim["kind"] == "door" and door.leads_to is not None):
            return None
        if aim["kind"] == "return" and here != segment:
            return None
        if here == segment and door.steps < aim["steps"]:
            return None
        key, what = ("door", aim["segment"], aim["cell"]), "a door was not entered"
    gmap.count_survey_try(key)
    left = MAX_TRIES - gmap.survey_tries[key]
    return f"{what} (aimed at in episode {aim['episode']}); " + (f"{left} more tries" if left > 0 else "given up on")


def survey(gmap: GlobalMap, people: Optional[List[Any]] = None,
           people_in: Optional[Callable[[int], List[Any]]] = None,
           edges: Iterable[Dict[str, Any]] = (), episode: int = 0) -> Survey:
    """
    The survey rules for the dispatch prompt and what this episode's task is sent for (`aim`), if anything.

    people: the people registry's marks for the area the player is in, or None when the registry is off.
    people_in: the marks for any other area, to tell which buildings still have someone not talked to.
    edges: the nearest unexplored map edges from the player (text.frontier_targets).
    """
    if not gmap.current:
        return Survey()
    here = gmap.resolve(gmap.segment, gmap.pos)[0]
    where = setting(gmap, here)
    if where is None:
        return Survey(log="in a cave: no survey")
    name = gmap.area_label(here)
    doors = doors_of(gmap, here)
    new = [d for d in doors if d.leads_to is None and d.tries < MAX_TRIES]
    shut = [d for d in doors if d.leads_to is None and d.tries >= MAX_TRIES]
    edges = sorted(edges, key=lambda t: t["steps"])
    edge_ok = gmap.survey_tries[("edge", here)] < MAX_TRIES
    out = Survey()
    base = {"segment": here, "episode": episode}

    if where in (INSIDE, MAYBE_INSIDE):
        place = f"inside {name}" if where == INSIDE else f"in {name}, which you came into through a door"
        waiting = _waiting(gmap, people or [])
        if waiting:
            out.lines.append(f"You are {place}. People here you have NOT talked to yet ({COORDINATES}), "
                             f"nearest first:")
            px, py = gmap.pos
            for m in waiting[:MAX_LINES]:
                out.lines.append(f"- {_where(m.square[0] - px, m.square[1] - py)}"
                                 + (": may have moved, look for them" if m.stale else ""))
            first = waiting[0]
            out.lines.append("This episode's task MUST be to walk up to the first one listed and talk to them until "
                             "they have nothing more to say. Say where they stand: the player does not know any "
                             "label from your list of people. Do not leave before everyone here is talked to.")
            out.aim = dict(base, kind="person", npc=first.npc_id,
                           steps=abs(first.square[0] - px) + abs(first.square[1] - py))
            out.log = f"{place}; {len(waiting)} person(s) not talked to yet"
            return out
        everyone = ("Everyone you have found here has been talked to" if people
                    else "You have found nobody here to talk to" if people is not None
                    else "Talk to everyone here you have not spoken to yet first")
        if _given_up(gmap, people or []):
            everyone += f" ({_plural(_given_up(gmap, people), 'person')} could not be talked to: leave them)"
        if new:
            out.lines.append(f"You are {place}. {everyone}. Doors or stairs here you have NOT been through yet "
                             f"({COORDINATES}), nearest first:")
            out.lines += [d.line for d in new[:MAX_LINES]]
            out.lines.append("This episode's task MUST be to go through the first one listed.")
            out.aim = dict(base, kind="door", cell=new[0].cells[0], steps=new[0].steps)
            out.log = f"{place}; {len(new)} door(s) not entered yet"
            return out
        if edges and edge_ok and people is not None:
            edge = edges[0]
            out.lines.append(f"You are {place}. {everyone}, but part of this place has not been seen yet, and more "
                             f"people may be there. This episode's task MUST be to walk to the unexplored map edge "
                             f"at {_where(edge['dx'], edge['dy'])} and look around.")
            out.aim = dict(base, kind="edge")
            out.log = f"{place}; looking at the part not seen yet"
            return out
        if where == INSIDE:
            done = (f"{everyone}, and you have been through every door found here: this building is finished"
                    if people is not None else
                    "You have been through every door found here. Once you have talked to everyone here you have "
                    "not spoken to yet, this building is finished")
            out.lines.append(f"You are {place}. {done}. Do not stay to look around or talk to the same people "
                             f"again. Unless your goal needs something else in this building, this episode's task "
                             f"MUST then be to leave {_way_out(gmap, here, doors)}.")
        out.log = f"{place}; finished"
        return out

    px, py = gmap.pos

    def away(m: Any) -> int:
        return abs(m.square[0] - px) + abs(m.square[1] - py)

    near = [d for d in doors if d.steps <= TOWN_REACH]
    waiting = [m for m in _waiting(gmap, people or []) if away(m) <= TOWN_REACH]
    if not near and not waiting:
        return Survey(log=f"outside in {name}; no door or person left within {TOWN_REACH} steps")
    indoors = [d for d in near if d.leads_to is not None and setting(gmap, d.leads_to) in (INSIDE, MAYBE_INSIDE)]
    if people_in is not None:
        for d in indoors:
            if d.tries < MAX_TRIES:
                d.waiting = len(_waiting(gmap, people_in(d.leads_to)))
    new = [d for d in new if d.steps <= TOWN_REACH]
    back = [d for d in near if d.waiting]
    if new or back or waiting:
        out.lines.append(f"You are outside in {name}. What is left to do near you ({COORDINATES}), nearest first "
                         f"in each list:")
        if waiting:
            out.lines.append("People out here you have NOT talked to yet:")
            out.lines += [f"- {_where(m.square[0] - px, m.square[1] - py)}"
                          + (": may have moved, look for them" if m.stale else "") for m in waiting[:MAX_LINES]]
        if new:
            out.lines.append("Doors you have NOT been through yet:")
            out.lines += [d.line for d in new[:MAX_LINES]]
        if back:
            out.lines.append("Buildings you left before talking to everyone inside:")
            out.lines += [f"- the door at {d.where}, to {gmap.area_label(d.leads_to)}: "
                          f"{_plural(d.waiting, 'person')} not talked to yet" for d in back[:MAX_LINES]]
        door = min(new + back, key=lambda d: d.steps, default=None)
        if waiting and (door is None or away(waiting[0]) < door.steps):
            first = waiting[0]
            out.lines.append(f"The nearest of these is the person at {_where(first.square[0] - px, first.square[1] - py)}"
                             f": this episode's task MUST be to walk up to them and talk to them until they have "
                             f"nothing more to say. Say where they stand: the player does not know any label from "
                             f"your list of people.")
            out.aim = dict(base, kind="person", npc=first.npc_id, steps=away(first))
        else:
            out.lines.append(f"The nearest of these is the door at {door.where}: this episode's task MUST be to walk "
                             f"to it and go inside. Once inside, talk to everyone before leaving.")
            out.aim = dict(base, kind="door" if door.leads_to is None else "return", cell=door.cells[0],
                           steps=door.steps)
        out.log = (f"outside in {name}; {len(waiting)} person(s) out here not talked to, {len(new)} door(s) not "
                   f"entered yet, {len(back)} building(s) with people not talked to")
    else:
        edge = next((t for t in edges if t["steps"] <= TOWN_REACH), None)
        done = (f"You are outside in {name}. You have been inside every building whose door you have found within "
                f"{TOWN_REACH} steps of you" + (", and talked to everyone found in them and out here"
                                                if people_in is not None else "") + ".")
        if edge is not None and edge_ok:
            out.lines.append(f"{done} More buildings and people may be just past the edge of your map: this "
                             f"episode's task MUST be to walk to the unexplored map edge at "
                             f"{_where(edge['dx'], edge['dy'])} and look for more.")
            out.aim = dict(base, kind="edge")
            out.log = f"outside in {name}; looking past the map edge for more doors"
        else:
            out.lines.append(f"{done} This place is finished: carry on with your goal.")
            out.log = f"outside in {name}; finished"
    entered = [d for d in indoors if not d.waiting]
    if entered:
        out.lines.append("Buildings here you have already been inside and finished. Do not go in again to look "
                         "around; go in only when your goal needs that place:")
        out.lines += [f"- the door at {d.where}, to {gmap.area_label(d.leads_to)}"
                      + (f" (been through it {d.uses} times)" if d.uses > 1 else "") for d in entered[:MAX_LINES]]
    shut = [d for d in shut if d.steps <= TOWN_REACH]
    if shut:
        out.lines.append(f"{_plural(len(shut), 'door')} here could not be entered in {MAX_TRIES} tries (locked, "
                         f"blocked or not a door); do not try again unless something in the game changes that: "
                         + ", ".join(d.where for d in shut[:MAX_LINES]) + ".")
    return out


def summary(gmap: GlobalMap, people_in: Optional[Callable[[int], List[Any]]] = None) -> Optional[str]:
    """What the map and the people registry say about the buildings near the player, for the goal check and
    the subgoal planner. None when the player is not in or just outside a building."""
    if not gmap.current:
        return None
    here = gmap.resolve(gmap.segment, gmap.pos)[0]
    where = setting(gmap, here)
    if where is None:
        return None
    town, origin = here, gmap.pos
    if where != OUTSIDE:
        came = entrance(gmap, here)
        if came is None or setting(gmap, came[0]) != OUTSIDE:
            return None
        town, origin = came
    doors = [d for d in doors_of(gmap, town, origin) if d.steps <= TOWN_REACH]
    if not doors:
        return None
    parts = []
    for d in doors:
        if d.leads_to is None or setting(gmap, d.leads_to) not in (INSIDE, MAYBE_INSIDE):
            continue
        left = len(_waiting(gmap, people_in(d.leads_to))) if people_in is not None else 0
        state = (f"{_plural(left, 'person')} not talked to yet" if left
                 else "everyone found there talked to" if people_in is not None else "been inside")
        parts.append(f"{gmap.area_label(d.leads_to)} ({state})")
    new = sum(d.leads_to is None and d.tries < MAX_TRIES for d in doors)
    shut = sum(d.leads_to is None and d.tries >= MAX_TRIES for d in doors)
    text = (f"From your map and your list of people, for the doors found within {TOWN_REACH} steps of "
            f"{'you' if where == OUTSIDE else 'the door you came in by'} in {gmap.area_label(town)}: "
            + (f"you have been inside {_plural(len(parts), 'building')}: " + "; ".join(parts) if parts
               else "you have not been inside any building yet") + ". ")
    text += (f"{_plural(new, 'door')} there you have not been through yet." if new
             else "No door found there is left that you have not been through.")
    if shut:
        text += f" {_plural(shut, 'door')} could not be entered."
    if people_in is not None:
        ox, oy = origin
        outside = sum(abs(m.square[0] - ox) + abs(m.square[1] - oy) <= TOWN_REACH
                      for m in _waiting(gmap, people_in(town)))
        text += (f" {_plural(outside, 'person')} outside there not talked to yet." if outside
                 else " Everyone found outside there has been talked to.")
    return text
