import functools
import math
import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import click
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from debug_scripts import markdown as md
from debug_scripts.benchmark import _storage_link
from debug_scripts.frames import _FONT, to_pil
from execution.perception.pokemon.tiles import TileRecognizer
from execution.strategist.pokemon.global_map import GlobalMap, render_segment, segment_bounds
from execution.strategist.pokemon.goals import PokemonGoalTree
from execution.strategist.pokemon.guide import StrategyGuide
from execution.strategist.pokemon.knowledge import PokemonKnowledgeTree
from execution.strategist.pokemon.notepad import ThoughtNotepad
from execution.strategist.report import episode_report_path, load_episode_report
from python_scripts.paths import debug_dir, debug_frames_dir, strategist_dir
from utils import log_error, log_info
from utils.fundamental import depathify

ARTIFACTS = {
    "goals": PokemonGoalTree,
    "knowledge": PokemonKnowledgeTree,
    "tiles": TileRecognizer,
    "notepad": ThoughtNotepad,
    "globalmap": GlobalMap,
    "guide": StrategyGuide,
}
TITLES = {"goals": "Goals", "knowledge": "Knowledge", "tiles": "Tiles",
          "notepad": "Thought notepad", "globalmap": "Global map", "guide": "Strategy guide"}
OPTIONAL = {"globalmap", "guide"}

TILE_SCALE = 6
IMAGE_SCALE = 2


class Report:
    def __init__(self, report_dir: str, images_dir: str, overwrite: bool, run_dir: str) -> None:
        self.report_dir = report_dir
        self.images_dir = images_dir
        self.overwrite = overwrite
        self.run_dir = run_dir

    def image_path(self, *parts: str) -> str:
        path = os.path.join(self.images_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def fresh(self, path: str) -> bool:
        return self.overwrite or not os.path.exists(path)

    def img(self, alt: str, path: Optional[str]) -> str:
        return md.img(alt, path, self.report_dir) if path else ""


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------


def contact_sheet(report: Report, items: Sequence[Tuple[str, np.ndarray]], path: str,
                  columns: int = 12) -> Optional[str]:
    if not items:
        return None
    if not report.fresh(path):
        return path
    images = [to_pil(array) for _, array in items]
    cell_w = max(image.width for image in images) * TILE_SCALE
    cell_h = max(image.height for image in images) * TILE_SCALE
    label_h, pad = 16, 4
    cols = min(columns, len(images))
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (pad + cols * (cell_w + pad), pad + rows * (cell_h + label_h + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, ((label, _), image) in enumerate(zip(items, images)):
        row, col = divmod(i, cols)
        x, y = pad + col * (cell_w + pad), pad + row * (cell_h + label_h + pad)
        sheet.paste(image.resize((image.width * TILE_SCALE, image.height * TILE_SCALE), Image.NEAREST), (x, y))
        draw.text((x, y + cell_h + 1), label, fill="black", font=_FONT)
    sheet.save(path)
    return path


# ----------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------


def path_text(path: Tuple[str, ...]) -> str:
    *parents, last = path
    return " › ".join([*parents, f"**{last}**"])


def count_phrase(counts: List[Tuple[int, str]]) -> str:
    parts = [f"{n} {label}" for n, label in counts if n]
    return ", ".join(parts) if parts else "no change"


# ----------------------------------------------------------------------------
# Views: (artifact, report, episode) -> (markdown blocks, one-line summary)
# ----------------------------------------------------------------------------


def goal_lines(tree, level: int, focus) -> List[str]:
    node = tree.root
    line = f"{'  ' * level}- `[{node.status}]` **{node.description}**"
    if node.details:
        line += f" — {node.details}"
    extras = []
    if node.source != "initial":
        extras.append(f"source: {node.source}" + (f" ({node.source_description})" if node.source_description else ""))
    if node.achievable_without_subgoals is not None:
        extras.append(f"achievable without subgoals (ep {node.achievable_without_subgoals})")
    if node.frontier_since is not None:
        extras.append(f"frontier since ep {node.frontier_since}")
    if node.status_reason:
        extras.append(f"reason: {node.status_reason}")
    if extras:
        line += f" _({'; '.join(extras)})_"
    if tree is focus:
        line += " **← current focus**"
    lines = [line]
    for subtree in tree.subtrees:
        lines += goal_lines(subtree, level + 1, focus)
    return lines


def view_goals(goals: PokemonGoalTree, report: Report, episode: int) -> Tuple[List[str], str]:
    tree = goals.tree
    statuses = pd.Series([node.status for node in tree.nodes()]).value_counts()
    focus = tree.frontier()
    focus_text = focus.root.description if focus else "nothing (every goal is closed)"
    complete = int(statuses.get("complete", 0))
    blocks = [
        md.para(f"Current focus: **{focus_text}**"),
        md.table(statuses.rename_axis("status").reset_index(name="goals")),
        md.h2("Tree"),
        "\n".join(goal_lines(tree, 0, focus)) + "\n",
    ]
    return blocks, f"{complete}/{len(tree.nodes())} goals complete; focus: {focus_text}"


def knowledge_lines(tree, level: int) -> List[str]:
    lines = [f"{'  ' * level}- **{tree.root.title}**: {tree.root.description}"]
    for leaf in tree.leaves:
        lines += knowledge_lines(leaf, level + 1)
    return lines


def view_knowledge(knowledge: PokemonKnowledgeTree, report: Report, episode: int) -> Tuple[List[str], str]:
    tree = knowledge.tree
    branches = pd.DataFrame([
        {"branch": leaf.root.title, "entries": len(leaf.nodes()) - 1} for leaf in tree.leaves
    ])
    total = len(tree.nodes()) - 1 - len(tree.leaves)
    blocks = [
        md.table(branches),
        md.h2("Tree"),
        "\n".join(knowledge_lines(tree, 0)) + "\n",
    ]
    return blocks, f"{total} entries across {len(tree.leaves)} branches"


def tile_sections(report: Report, tiles, prefix: str) -> List[str]:
    blocks = []
    by_tag: Dict[str, list] = {}
    for tile in tiles:
        by_tag.setdefault(tile.tag, []).append(tile)
    for tag, members in sorted(by_tag.items(), key=lambda item: -len(item[1])):
        items, rows = [], []
        for tile in members:
            label = str(len(items)) if tile.image is not None else "-"
            if tile.image is not None:
                items.append((label, tile.image))
            rows.append({"#": label, "hash": (tile.hash or "")[:10], "description": tile.description or ""})
        sheet = contact_sheet(report, items, report.image_path("tiles", f"{prefix}_{depathify(tag)}.png"))
        blocks += [md.h3(f"{tag} ({len(members)})"), report.img(tag, sheet)]
        if any(row["description"] for row in rows):
            blocks.append(md.table(pd.DataFrame(rows)))
    return blocks


def view_tiles(recognizer: TileRecognizer, report: Report, episode: int) -> Tuple[List[str], str]:
    counts = pd.Series([tile.tag for tile in recognizer.tiles], dtype=object).value_counts()
    blocks = [
        md.para(f"Recognizer name: `{recognizer.name}`, game `{recognizer.game}`."),
        md.table(counts.rename_axis("tag").reset_index(name="tiles")),
        md.h2("Tiles by tag"),
        *tile_sections(report, recognizer.tiles, f"ep{episode}"),
    ]
    return blocks, f"{len(recognizer.tiles)} tiles across {len(counts)} tags"


def view_notepad(notepad: ThoughtNotepad, report: Report, episode: int) -> Tuple[List[str], str]:
    n_lines = len(notepad.text.splitlines())
    blocks = [md.code(notepad.text) if notepad.text else md.note("The notepad is empty.")]
    return blocks, f"{notepad.length} words over {n_lines} lines"


MAP_DIRECTION = re.compile(r"MUST be to walk to the unexplored edge at \((-?\d+), (-?\d+)\), (\d+) steps ([a-z-]+)")
MAP_LEGEND = "Green: unexplored edge. Blue: trail. Red: the player."
DOOR_LEGEND = "Numbered boxes are doors and stairs: cyan once the area behind is known, orange if not entered yet."


def doors_table(gmap, doors) -> str:
    return md.table(pd.DataFrame([
        {"door": i, "squares": ", ".join(map(str, door["cells"])),
         "leads to": (f"segment {door['leads_to']}: {gmap.area_label(door['leads_to'])}"
                      if door["leads_to"] is not None else "not entered yet")}
        for i, door in enumerate(doors)]))


@functools.lru_cache(maxsize=None)
def episode_brief(run_dir: str, episode: int) -> Tuple[str, str]:
    if not os.path.exists(episode_report_path(run_dir, episode)):
        return "", "no archived report"
    archive = load_episode_report(run_dir, episode)
    dispatch = [call for call in archive.strategist_calls if call.stage == "dispatch"]
    match = MAP_DIRECTION.search(str(dispatch[-1].prompt)) if dispatch else None
    direction = (f"map override: {match.group(3)} steps {match.group(4)} to the unexplored edge at "
                 f"({match.group(1)}, {match.group(2)})" if match else "no map override")
    return archive.episode.task or "", direction


def map_image(report: Report, gmap: GlobalMap, segment: int, path: str, **kwargs) -> Optional[str]:
    if report.fresh(path):
        canvas = render_segment(gmap, segment, **kwargs)
        if canvas is None:
            return None
        cv2.imwrite(path, canvas)
    return path


def map_replay(report: Report, gmap: GlobalMap, episode: int, n_frames: int = 120) -> Optional[str]:
    si = gmap.segment
    path = report.image_path("globalmap", f"ep{episode}_seg{si}_build.gif")
    if not report.fresh(path):
        return path
    steps = [t for t, s, _ in gmap.trail if s == si]
    bounds = segment_bounds(gmap, si)
    if not steps or bounds is None:
        return None
    ends = sorted((GlobalMap.load(report.run_dir, n).step, n)
                  for n in GlobalMap.saved_episodes(report.run_dir) if n <= episode)
    seg = gmap.segments[si]
    frames = []
    for t in sorted(set(range(steps[0], steps[-1] + 1, max(1, (steps[-1] - steps[0]) // n_frames))) | {steps[-1]}):
        canvas = render_segment(gmap, si, upto_step=t, bounds=bounds)
        if canvas is None:
            continue
        ep = next((n for end, n in ends if t <= end), episode)
        task, direction = episode_brief(report.run_dir, ep)
        width = canvas.shape[1]
        band = np.zeros((70, width, 3), np.uint8)
        seen = sum(1 for c in seg.values() if c.first_seen <= t)
        cv2.putText(band, f"episode {ep}   step {t}   squares mapped {seen}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(band, direction[:width // 7], (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255) if direction.startswith("map override") else (180, 180, 180), 1)
        cv2.putText(band, ("task: " + task)[:width // 7], (6, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1)
        frames.append(Image.fromarray(cv2.cvtColor(np.vstack([band, canvas]), cv2.COLOR_BGR2RGB)))
    if not frames:
        return None
    frames += [frames[-1]] * 15
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=True)
    return path


def view_globalmap(gmap: GlobalMap, report: Report, episode: int) -> Tuple[List[str], str]:
    exploring = gmap.exploring_until
    replay = map_replay(report, gmap, episode)
    blocks = [
        md.para(f"{gmap.n_cells} squares mapped in {gmap.n_areas} area(s) ({len(gmap.segments)} segments opened, "
                f"{len(gmap.merged_into)} merged into an older one), over {gmap.step} steps."),
        md.para(f"Player at {gmap.pos} in segment {gmap.segment}. "
                + (f"Forced exploration until episode {exploring}." if exploring is not None and exploring >= episode
                   else "Not in forced exploration.")),
        md.h2("Build replay of the current segment"),
        report.img("build replay", replay) if replay else md.note("The current segment is empty."),
        md.para(MAP_LEGEND),
        md.h2("Episodes"),
        md.table(pd.DataFrame([
            {"episode": n, "new squares": added, "task": episode_brief(report.run_dir, n)[0],
             "map": episode_brief(report.run_dir, n)[1]}
            for n, added in sorted(gmap.new_cells_by_episode.items())
        ])),
        md.h2("Segments"),
    ]
    live = [si for si, seg in enumerate(gmap.segments) if seg]
    blocks.append(md.table(pd.DataFrame([
        {"segment": si, "squares": len(gmap.segments[si]), "pixels": gmap.filled_pixels(si),
         "pixels at last naming": gmap.info(si).named_pixels, "name": gmap.info(si).name or "(unnamed)",
         "named in episode": gmap.info(si).episode, "fully explored since": gmap.info(si).explored,
         "description": gmap.info(si).description or ""}
        for si in live])))
    blocks.append(md.para(MAP_LEGEND + " " + DOOR_LEGEND))
    for si in live:
        seg, info = gmap.segments[si], gmap.info(si)
        png = map_image(report, gmap, si, report.image_path("globalmap", f"ep{episode}_seg{si}.png"), doors=True)
        blocks += [md.h3(f"Segment {si}{' (current)' if si == gmap.segment else ''}: {gmap.place_label(si)}, "
                         f"{info.kind or 'kind not known'}, {len(seg)} squares")]
        if info.description:
            blocks.append(md.para(info.description))
        if info.aliases:
            blocks.append(md.para("Also named, before merges: " + "; ".join(name for name, _ in info.aliases)))
        blocks.append(report.img(f"segment {si}", png))
        doors = gmap.door_list(si)
        if doors:
            blocks.append(doors_table(gmap, doors))
    verdicts = gmap.merge_verdicts
    if verdicts:
        blocks += [md.h2("Merges judged"), verdicts_table(verdicts)]
    blocks.append(md.h2("Counters"))
    blocks.append(md.table(pd.Series(dict(gmap.stats), dtype=int).sort_index().rename_axis("counter")
                           .reset_index(name="count")) if gmap.stats else md.note("No counters yet."))
    return blocks, f"{gmap.n_cells} squares in {gmap.n_areas} area(s); at {gmap.pos} in segment {gmap.segment}"


def verdicts_table(verdicts) -> str:
    return md.table(pd.DataFrame([
        {"segments": f"{a} and {b}", "names": " / ".join(v["names"]), "same place": v["same"],
         "matching squares": v["matches"], "at step": v["step"]} for (a, b), v in sorted(verdicts.items())]))


def rules_table(rules, guide: StrategyGuide) -> str:
    return md.table(pd.DataFrame([
        {"id": r.id, "trigger": r.trigger, "rule": r.text, "source": r.source, "created": r.created,
         "updated": r.updated, "fired": guide.times_fired(r.id)} for r in rules]))


def events_table(events) -> str:
    return md.table(pd.DataFrame([
        {"episode": e.episode, "trigger": e.trigger, "from": e.previous, "to": e.place,
         "rules shown": ", ".join(map(str, e.rules)) or "none"} for e in events]))


def stage_replies(run_dir: str, episodes, stage: str, label: str) -> List[str]:
    blocks = []
    for n in sorted(set(episodes)):
        if not os.path.exists(episode_report_path(run_dir, n)):
            continue
        for call in load_episode_report(run_dir, n).strategist_calls:
            if call.stage == stage:
                blocks.append(md.details(f"Episode {n}: {label}", md.code(str(call.response))))
    return blocks


def guide_updates(run_dir: str, episodes) -> List[str]:
    return stage_replies(run_dir, episodes, "guide_update", "guide update reply")


def view_guide(guide: StrategyGuide, report: Report, episode: int) -> Tuple[List[str], str]:
    blocks = [md.h2("Rules"), rules_table(guide.rules, guide) if guide.rules else md.note("No rules.")]
    if guide.removed:
        blocks += [md.h2("Removed rules"), rules_table(guide.removed, guide)]
    blocks.append(md.h2("Triggers"))
    blocks.append(events_table(guide.events) if guide.events else md.note("No trigger has fired yet."))
    blocks += [md.h2("Towns reached"), md.para(", ".join(guide.towns) if guide.towns else "None yet.")]
    updates = guide_updates(report.run_dir, (e.episode for e in guide.events))
    if updates:
        blocks += [md.h2("The strategist's guide updates"), *updates]
    own = sum(r.source == "strategist" for r in guide.rules)
    return blocks, f"{len(guide.rules)} rules ({own} the strategist's own), {len(guide.events)} triggers fired"


VIEWS: Dict[str, Callable[..., Tuple[List[str], str]]] = {
    "goals": view_goals, "knowledge": view_knowledge, "tiles": view_tiles,
    "notepad": view_notepad, "globalmap": view_globalmap, "guide": view_guide,
}


# ----------------------------------------------------------------------------
# Diffs: (new, old, diff, report, old_ep, new_ep) -> (markdown blocks, one-line summary)
# ----------------------------------------------------------------------------


def diff_goals(new, old, d, report: Report, a: int, b: int) -> Tuple[List[str], str]:
    new_nodes, old_nodes = new.tree.paths(), old.tree.paths()
    blocks = []
    if d["status_changed"]:
        blocks += [md.h3("Status changes"), md.table(pd.DataFrame([
            {"goal": path_text(path), "from": before, "to": after}
            for path, (before, after) in d["status_changed"].items()
        ]))]
    if d["added"]:
        blocks += [md.h3("Added"), md.bullets(
            f"+ {path_text(p)} `[{new_nodes[p].status}]`" + (f" — {new_nodes[p].details}" if new_nodes[p].details else "")
            + (f" _(source: {new_nodes[p].source})_" if new_nodes[p].source != "initial" else "")
            for p in d["added"])]
    if d["removed"]:
        blocks += [md.h3("Removed"), md.bullets(f"- {path_text(p)} `[{old_nodes[p].status}]`" for p in d["removed"])]
    return blocks, count_phrase([(len(d["status_changed"]), "status changes"), (len(d["added"]), "added"),
                                 (len(d["removed"]), "removed")])


def diff_knowledge(new, old, d, report: Report, a: int, b: int) -> Tuple[List[str], str]:
    new_nodes, old_nodes = new.tree.paths(), old.tree.paths()
    blocks = []
    if d["description_changed"]:
        blocks += [md.h3("Rewritten"), md.bullets(
            f"{path_text(path)}: ~~{before}~~ → {after}" for path, (before, after) in d["description_changed"].items())]
    if d["added"]:
        blocks += [md.h3("Added"), md.bullets(f"+ {path_text(p)}: {new_nodes[p].description}" for p in d["added"])]
    if d["removed"]:
        blocks += [md.h3("Removed"), md.bullets(f"- {path_text(p)}: {old_nodes[p].description}" for p in d["removed"])]
    return blocks, count_phrase([(len(d["description_changed"]), "rewritten"), (len(d["added"]), "added"),
                                 (len(d["removed"]), "removed")])


def diff_tiles(new, old, d, report: Report, a: int, b: int) -> Tuple[List[str], str]:
    blocks = []
    if d["new_tiles"]:
        counts = pd.Series(d["new_tile_count_by_tag"]).sort_values(ascending=False)
        blocks += [md.table(counts.rename_axis("tag").reset_index(name="new tiles")),
                   *tile_sections(report, d["new_tiles"], f"ep{a}_to_ep{b}")]
    return blocks, count_phrase([(len(d["new_tiles"]), "new tiles")])


def diff_notepad(new, old, d, report: Report, a: int, b: int) -> Tuple[List[str], str]:
    change = f"{d['length_change']:+d} words"
    if d["kind"] == "unchanged":
        return [], "no change"
    if d["kind"] == "cleared":
        return [md.h3("Cleared"), md.details("Text before clearing", md.code(old.text))], f"cleared ({change})"
    if d["kind"] == "appended":
        return [md.h3("Appended"), md.code("\n".join(d["added_lines"]))], \
            f"appended {len(d['added_lines'])} lines ({change})"
    blocks = [md.h3("Rewritten"), md.code("\n".join(d["unified"]), "diff"),
              md.details("Full new text", md.code(new.text))]
    return blocks, (f"rewritten: {len(d['added_lines'])} lines added, {len(d['removed_lines'])} removed "
                    f"({change})")


def diff_globalmap(new, old, d, report: Report, a: int, b: int) -> Tuple[List[str], str]:
    task, direction = episode_brief(report.run_dir, b)
    (s0, p0), (s1, p1) = d["position"]
    blocks = [
        md.para(f"Task in episode {b}: {task.rstrip('.') or '(none)'}. {direction[0].upper() + direction[1:]}."),
        md.para(f"{d['steps']} steps; the player went from {p0} in segment {s0} to {p1} in segment {s1}."),
    ]
    if d["opened"]:
        blocks += [md.h3("Segments opened"), md.bullets(f"+ segment {si}" for si in d["opened"])]
    if d["merged"]:
        blocks += [md.h3("Segments merged"), md.bullets(
            f"segment {s} folded into segment {target}, shifted by {offset}" for s, (target, offset) in d["merged"].items())]
    before, after = d["exploring"]
    if before != after:
        blocks.append(md.para(f"Forced exploration: until episode {before} → until episode {after}."))
    for si in sorted(set(d["new_cells"]) | set(d["reclassified"])):
        if not new.segments[si]:
            continue
        added, relabelled = d["new_cells"].get(si, []), d["reclassified"].get(si, [])
        png = map_image(report, new, si, report.image_path("globalmap", f"ep{a}_to_ep{b}_seg{si}.png"),
                        highlight=added, trail_from=old.step, doors=True)
        blocks += [md.h3(f"Segment {si}: {len(added)} new squares, {len(relabelled)} recognised since"),
                   report.img(f"segment {si}", png)]
    if d["new_cells"] or d["reclassified"]:
        blocks.append(md.para(f"Magenta: squares new since episode {a}. Blue: this episode's trail. "
                              "Green: unexplored edge. Red: the player."))
    before = old.segment_info
    named = {si: info for si, info in new.segment_info.items()
             if info.name is not None and si not in new.merged_into
             and (si not in before or before[si].name != info.name)}
    if named:
        blocks += [md.h3("Segments named"), md.bullets(f"segment {si}: **{info.name}**: {info.description}"
                                                       for si, info in sorted(named.items()))]
        blocks += stage_replies(report.run_dir, range(a + 1, b + 1), "segment_name", "segment naming reply")
    judged = {pair: v for pair, v in new.merge_verdicts.items() if old.merge_verdicts.get(pair) != v}
    if judged:
        blocks += [md.h3("Merges judged"), verdicts_table(judged)]
        blocks += stage_replies(report.run_dir, range(a + 1, b + 1), "merge_judge", "merge judgement reply")
    doors = d.get("doors", {})
    if doors:
        blocks += [md.h3("Doors labelled"), md.bullets(f"segment {si}, square {p}: leads to segment {dest}, "
                                                       f"{new.area_label(dest)}"
                                                       for (si, p), dest in sorted(doors.items()))]
    if d["stats"]:
        blocks += [md.h3("Counter changes"), md.table(pd.Series(d["stats"], dtype=int).sort_index()
                                                     .rename_axis("counter").reset_index(name="change"))]
    return blocks, count_phrase([(sum(len(v) for v in d["new_cells"].values()), "new squares"),
                                 (len(named), "segments named"), (len(doors), "doors labelled"),
                                 (sum(len(v) for v in d["reclassified"].values()), "squares recognised"),
                                 (len(d["opened"]), "segments opened"), (len(d["merged"]), "merged"),
                                 (d["steps"], "steps")])


def diff_guide(new, old, d, report: Report, a: int, b: int) -> Tuple[List[str], str]:
    blocks = []
    if d["events"]:
        blocks += [md.h3("Triggers fired"), events_table(d["events"])]
    if d["towns"]:
        blocks += [md.h3("New towns"), md.bullets(d["towns"])]
    if d["added"]:
        blocks += [md.h3("Added"), md.bullets(f"+ [{r.id}] when the player {r.trigger}: {r.text} _({r.source})_"
                                              for r in d["added"])]
    if d["edited"]:
        blocks += [md.h3("Rewritten"), md.bullets(f"[{after.id}]: ~~{before.text}~~ → {after.text}"
                                                  for before, after in d["edited"])]
    if d["removed"]:
        blocks += [md.h3("Removed"), md.bullets(f"- [{r.id}] {r.text}" for r in d["removed"])]
    blocks += guide_updates(report.run_dir, range(a + 1, b + 1))
    return blocks, count_phrase([(len(d["events"]), "triggers fired"), (len(d["added"]), "added"),
                                 (len(d["edited"]), "rewritten"), (len(d["removed"]), "removed")])


DIFFS: Dict[str, Callable[..., Tuple[List[str], str]]] = {
    "goals": diff_goals, "knowledge": diff_knowledge, "tiles": diff_tiles,
    "notepad": diff_notepad, "globalmap": diff_globalmap, "guide": diff_guide,
}


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------


def setup(obj: dict, name: str, mode: str) -> Tuple[str, Report]:
    parameters, game = obj["parameters"], obj["game"]
    run_dir = strategist_dir(parameters, game=game, name=name)
    if not os.path.isdir(run_dir):
        log_error(f"[strategist] no strategist run at {run_dir}. run_strategist.py --name {name!r} "
                  f"--game {game} writes it.", parameters)
    segment = depathify(name)
    report_dir = debug_dir(parameters, game=game, stage="strategist", sub=(segment, mode),
                           output_dir=obj["output_dir"])
    images_dir = _storage_link(report_dir, "images",
                               debug_frames_dir(parameters, game=game, stage="strategist", sub=(segment, mode)),
                               parameters)
    report = Report(report_dir=report_dir, images_dir=images_dir, overwrite=obj["overwrite"], run_dir=run_dir)
    return run_dir, report


def selected(artifact: Optional[str], run_dir: str) -> List[str]:
    if artifact:
        return [artifact]
    return [key for key, cls in ARTIFACTS.items() if key not in OPTIONAL or cls.exists(run_dir)]


@click.group(name="strategist")
def debug_strategist():
    """Inspect a strategist run's saved artifacts (goals, knowledge, tiles, notepad, global map,
    strategy guide)."""


ARTIFACT_OPTION = click.option("--artifact", type=click.Choice(list(ARTIFACTS)), default=None,
                               help="One artifact to report on. Omit for all of them.")
NAME_OPTION = click.option("--name", required=True, help="The strategist run's --name.")


@debug_strategist.command(name="view")
@NAME_OPTION
@ARTIFACT_OPTION
@click.option("--episode", type=int, default=None,
              help="Show each artifact as of this episode: its latest save at or before it. "
                   "Defaults to the latest save of each.")
@click.pass_obj
def debug_strategist_view(obj, name, artifact, episode):
    """Each artifact as it stood at one episode."""
    parameters = obj["parameters"]
    run_dir, report = setup(obj, name, "view")
    rows = []
    for key in selected(artifact, run_dir):
        cls = ARTIFACTS[key]
        eligible = [n for n in cls.saved_episodes(run_dir) if episode is None or n <= episode]
        if not eligible:
            where = f" at or before episode {episode}" if episode is not None else ""
            log_error(f"[strategist] no saved {cls.__name__} under {run_dir}{where}.", parameters)
        loaded = eligible[-1]
        blocks, summary = VIEWS[key](cls.load(run_dir, loaded, parameters=parameters), report, loaded)
        path = md.write_report(os.path.join(report.report_dir, f"{key}.md"), [
            md.h1(f"{TITLES[key]}: {name}, episode {loaded}"),
            md.para(f"_{summary}_"),
            *blocks,
        ])
        rows.append({"artifact": md.link(TITLES[key], path, report.report_dir), "from episode": loaded,
                     "summary": summary})
    requested = f"episode {episode}" if episode is not None else "the latest episode"
    index = md.write_report(os.path.join(report.report_dir, "index.md"), [
        md.h1(f"Strategist artifacts: {name}"),
        md.para(f"Run directory: `{run_dir}`"),
        md.para(f"As of {requested}. Each artifact is shown from its own latest save at or before that point, "
                "so the episodes below can differ."),
        md.table(pd.DataFrame(rows)),
    ])
    log_info(f"[strategist] view report: {index}", parameters)


@debug_strategist.command(name="diff")
@NAME_OPTION
@ARTIFACT_OPTION
@click.pass_obj
def debug_strategist_diff(obj, name, artifact):
    """How each artifact changed, save by save, from its first version to its latest."""
    parameters = obj["parameters"]
    run_dir, report = setup(obj, name, "diff")
    rows = []
    for key in selected(artifact, run_dir):
        cls = ARTIFACTS[key]
        episodes = cls.saved_episodes(run_dir)
        if not episodes:
            log_error(f"[strategist] no saved {cls.__name__} under {run_dir}.", parameters)
        previous = cls.load(run_dir, episodes[0], parameters=parameters)
        first_blocks, first_summary = VIEWS[key](previous, report, episodes[0])
        blocks = [
            md.h1(f"{TITLES[key]} history: {name}"),
            md.para(f"{len(episodes)} saved versions, at episodes {', '.join(map(str, episodes))}."),
            md.h2(f"Episode {episodes[0]}: first saved version"),
            md.para(f"_{first_summary}_"),
            md.details("Full first version", "\n".join(first_blocks)),
        ]
        steps = []
        for a, b in zip(episodes, episodes[1:]):
            current = cls.load(run_dir, b, parameters=parameters)
            diff_blocks, summary = DIFFS[key](current, previous, current.diff(previous), report, a, b)
            blocks += [md.h2(f"Episode {a} → {b}"), md.para(f"_{summary}_"), *diff_blocks]
            steps.append(f"{a}→{b}: {summary}")
            previous = current
        path = md.write_report(os.path.join(report.report_dir, f"{key}.md"), blocks)
        rows.append({"artifact": md.link(TITLES[key], path, report.report_dir), "versions": len(episodes),
                     "first": episodes[0], "latest": episodes[-1],
                     "last change": steps[-1] if steps else "(only one version)"})
    index = md.write_report(os.path.join(report.report_dir, "index.md"), [
        md.h1(f"Strategist artifact history: {name}"),
        md.para(f"Run directory: `{run_dir}`"),
        md.para("Each artifact is diffed between consecutive episodes where it was saved, oldest first."),
        md.table(pd.DataFrame(rows)),
    ])
    log_info(f"[strategist] diff report: {index}", parameters)
