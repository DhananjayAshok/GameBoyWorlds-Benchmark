"""
The layer above :class:`~execution.pokemon.supervisors.PokemonPlayThroughSupervisor`.

It owns the long game -- what to aim at, what is known about the world, and where the
player is -- and dispatches one supervisor per short-horizon task, folding the result back
into its own state.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from gameboy_worlds.emulation.pokemon import AgentState
from tqdm import tqdm

from execution.artifact import episode_dirs
from execution.perception.pokemon.tiles import TileRecognizer, verbalize_tiles
from execution.pokemon.executors import CombatExecutor
from execution.pokemon.supervisors import PokemonPlayThroughSupervisor
from execution.report import (EnvironmentStepRecord, ExecutorVLMCallRecord, SupervisorReport, SupervisorVLMCallRecord,
                              per_prompt_token_counts)
from execution.strategist.pokemon import prompts as P
from execution.strategist.pokemon.asking import ask, single_value
from execution.strategist.pokemon.goals import (GoalNode, GoalSubtree, PokemonGoalTree, check_goals,
                                                create_subgoals, frontier_text, goal_path_text)
from execution.strategist.pokemon.guide import (StrategyGuide, battle_tips, guide_block, load_seeds, record_events,
                                                record_new_town, reflect_on_defeat, town_question, update_guide)
from execution.strategist.pokemon.knowledge import PokemonKnowledgeTree, remember
from execution.strategist.pokemon.notepad import ThoughtNotepad
from execution.strategist.pokemon.global_map import (FAILURE_EPISODES, GlobalMap, MappingEnvironment, explore_text,
                                                     exploring, judge_merge, judge_stuck, name_segment, navigate_map,
                                                     prompt_block, stuck_signs)
from execution.perception.pokemon.navigation import cell_kinds
from execution.strategist.report import (REPORT_DETAILS, EpisodeArchive, EpisodeRecord, StrategistReport,
                                         StrategistVLMCallRecord, save_episode_report, trim_episode)
from python_scripts.paths import strategist_dir, strategist_session_rel
from utils import VLM, load_parameters, log_error, log_info, log_warn
from utils.lm_inference import clean_value, parse_key_value
from utils.parsing import parse_list

#: One file per process, stamped like the run it records, so a resumed run writes its own.
PROVENANCE_PREFIX = "provenance"
MAX_BATTLE_REASON = 200
#: A frame is black when this fraction of its pixels are at or below BLACK_PIXEL_LEVEL (a door warp
#: leaves grey lines, so "every pixel" is too strict).
BLACK_PIXEL_LEVEL = 10
BLACK_FRACTION = 0.95
MAX_SETTLE_STEPS = 8


def settle_screen(env) -> None:
    """While the screen is black, let the game run on (no button) so an episode never ends mid door-warp."""
    for _ in range(MAX_SETTLE_STEPS):
        frame = np.asarray(env.get_info()["core"]["current_frame"])
        if float(np.mean(frame <= BLACK_PIXEL_LEVEL)) < BLACK_FRACTION:
            return
        captured, _ = env._emulator.step(None)
        if captured is None:
            return

# One run writes to three trees, and only the first is addressable from (game, name) alone.
# Recovering the other two, given a provenance_<stamp>.json payload:
#
#   1. State, under this project's storage:
#        paths.strategist_dir(game=payload["game"], name=payload["init_kwargs"]["name"])
#      holds episode_<n>/*.pkl, episode_<n>/report.pkl.gz and the provenance files. Also
#      recorded verbatim as payload["init_kwargs"]["storage"].
#
#   2. Video, under GameBoyWorlds' storage. payload["session_dir"] is relative to
#      paths.strategist_sessions_root(game=...) and resolves with:
#        paths.strategist_session_abs(payload["session_dir"], game=payload["game"])
#        paths.strategist_video(payload["session_dir"], game=payload["game"])
#      The second returns None when the run saved no video. Do NOT rebuild this path from
#      the stamp in the provenance filename; see _resolve_session_dir for why it differs.
#      A payload["session_dir"] of None means the run predates this field, or the emulator
#      exposed no session path -- glob sessions/<game>/strategist/<name>/*/*/videos/0.mp4.
#
#   3. Emulator save states, under GameBoyWorlds' rom_data, NOT under either of the above:
#        <gbw storage>/rom_data/<series>/<game>/states/custom_<treat_name(name)><n>.state
#      one per saved episode, named by _emulator_state_name. There is no accessor for the
#      states directory on this side; GameBoyWorlds builds it as
#      gbw_parameters[f"{game}_rom_data_path"] + "/states/".


def treat_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name)


class PokemonStrategist:
    RECENT_EPISODES = 5
    MAX_DISPATCH_ATTEMPTS = 2
    MAX_EXTRACT_ATTEMPTS = 2
    MAX_NOTEPAD_ATTEMPTS = 3
    MAX_FACTS = 15
    MIN_NAMING_STEPS = 3
    RETURN_EPISODES = 3

    def __init__(
        self,
        env,
        game: str = "pokemon_red",
        name: str = "default",
        max_episodes: int = 20,
        supervisor_max_steps: int = 10,
        strategist_vlm_model: Optional[str] = None,
        strategist_vlm_kind: Optional[str] = None,
        max_new_tokens: int = 4800,
        supervisor_max_new_tokens: int = 4800,
        subgoal_every: int = 5,
        report_detail: str = "strategist",
        resume: bool = True,
        resume_run: Optional[str] = None,
        resume_episode: Union[str, int] = "latest",
        on_episode_complete: Optional[Callable[["StrategistReport"], None]] = None,
        override_goal: Optional[str] = None,
        parameters: Optional[dict] = None,
        **supervisor_kwargs: Any,
    ) -> None:
        self._env = env
        self._game = game
        self._name = name
        self._max_episodes = max_episodes
        self._supervisor_max_steps = supervisor_max_steps
        self._strategist_vlm_model = strategist_vlm_model
        self._strategist_vlm_kind = strategist_vlm_kind
        self._max_new_tokens = max_new_tokens
        #: Named, not left to ``supervisor_kwargs``, where it would collide with this class's
        #: own ``max_new_tokens``.
        self._supervisor_max_new_tokens = supervisor_max_new_tokens
        self._subgoal_every = subgoal_every
        self._report_detail = report_detail
        self._resume = resume
        self._on_episode_complete = on_episode_complete
        self._parameters = load_parameters(parameters)
        if isinstance(subgoal_every, bool) or not isinstance(subgoal_every, int) or subgoal_every <= 0:
            log_error(f"subgoal_every must be a positive integer, got {subgoal_every!r}.", self._parameters)
        if report_detail not in REPORT_DETAILS:
            log_error(f"report_detail must be one of {REPORT_DETAILS}, got {report_detail!r}.",
                      self._parameters)
        if treat_name(name)[-1:].isdigit():
            log_error(f"Strategist name {name!r} must not end in a digit.", self._parameters)
        self._supervisor_kwargs = supervisor_kwargs
        self._vlm_instance: Optional[VLM] = None

        self._dir = strategist_dir(self._parameters, game=game, name=name)
        self._resume_run = resume_run
        self._resume_episode = resume_episode
        self._resume_dir = self._dir

        if resume_episode != "latest":
            try:
                self._resume_episode = int(resume_episode)
            except (TypeError, ValueError):
                log_error(f"resume_episode must be 'latest' or an integer, got {resume_episode!r}.",
                          self._parameters)
        if resume_run is not None and resume_run != name:
            if not resume:
                log_error(f"resume_run={resume_run!r} was given with resume=False, which asks "
                          f"to start from that run's state and from nothing at the same time.",
                          self._parameters)
            if os.path.exists(self._dir):
                log_error(f"resume_run={resume_run!r} would fork that run's state into {name!r}, "
                          f"but {self._dir} already exists.", self._parameters)
            self._resume_dir = strategist_dir(self._parameters, game=game, name=resume_run)

        #: --fresh starts every artifact over too.
        load_episode = None if self._resume_episode == "latest" else self._resume_episode
        self.tiles = self._load_or_default(
            TileRecognizer, lambda: TileRecognizer(name=name, game=game, parameters=self._parameters),
            load_episode)
        self.goals = self._load_or_default(PokemonGoalTree, PokemonGoalTree, load_episode)
        self.knowledge = self._load_or_default(PokemonKnowledgeTree, PokemonKnowledgeTree, load_episode)
        self.notepad = self._load_or_default(ThoughtNotepad, ThoughtNotepad, load_episode)
        self.global_map = self._load_or_default(GlobalMap, GlobalMap, load_episode)
        self.guide = self._load_or_default(StrategyGuide, StrategyGuide, load_episode)

        if not resume:
            self.start_episode = 0
        elif self._resume_episode != "latest":
            self.start_episode = self._resume_episode + 1
        else:
            resumed_from = episode_dirs(self._resume_dir)
            self.start_episode = (max(resumed_from) + 1) if resumed_from else 0
        self._override_goal = override_goal
        if override_goal is not None and self.start_episode == 0:
            self.goals.tree = GoalSubtree.create_tree(
                [GoalNode(description=override_goal, source="override")], self.goals.tree.root.description)
        if self.start_episode in episode_dirs(self._dir):
            log_error(f"episode_{self.start_episode} already exists under {self._dir!r}; "
                      "refusing to overwrite it.", self._parameters)
        if self.start_episode > 0:
            self._env.load_custom_state(
                self._emulator_state_name(self._resume_run or name, self.start_episode - 1))
        for seed in self.guide.missing_seeds(load_seeds()):
            self.guide.add_rule(seed["trigger"], seed["rule"], "seed", self.start_episode,
                                episode_number=self.start_episode)

        self._identified: Optional[Dict[Any, Any]] = None
        self._screen_tiles: Optional[str] = None
        #: Episodes run by *this* instance, which is what ``max_episodes`` bounds.
        self._episodes_run = 0
        self._ran = False
        self._explore_now = False

        self._started_at = datetime.now()
        self._session_dir = self._resolve_session_dir()
        os.makedirs(self._dir, exist_ok=True)

        self.report = StrategistReport(
            game=game,
            strategist_name=self.__class__.__name__,
            init_kwargs=self._run_config(),
        )
        self._write_provenance()
        self.save_all(self.start_episode)

    def _run_config(self) -> dict:
        return {
            "name": self._name,
            "max_episodes": self._max_episodes,
            "supervisor_max_steps": self._supervisor_max_steps,
            "strategist_vlm_model": self._strategist_vlm_model,
            "strategist_vlm_kind": self._strategist_vlm_kind,
            "max_new_tokens": self._max_new_tokens,
            "supervisor_max_new_tokens": self._supervisor_max_new_tokens,
            "subgoal_every": self._subgoal_every,
            "report_detail": self._report_detail,
            "storage": self._dir,
            "resume": self._resume,
            "resume_run": self._resume_run,
            "resume_episode": self._resume_episode,
            "resume_dir": self._resume_dir,
            "supervisor_kwargs": dict(self._supervisor_kwargs),
            "override_goal": self._override_goal,
        }

    def _resolve_session_dir(self) -> Optional[str]:
        """The emulator's session directory, cut to ``<name>/<stamp>/<n>_<hash>``.

        Read off the emulator rather than rebuilt from the naming convention: the stamp in
        the session path is minted by ``run_strategist.py`` before the emulator boots, while
        this class's own ``_started_at`` is minted after, so the two disagree whenever
        construction crosses a second boundary. Recording what the emulator actually used is
        the only reliable link between the two halves of a run.

        :return: The relative session directory, or ``None`` if the emulator does not expose
            one (``Environment._emulator`` having moved, or a session path outside the
            strategist root).
        :rtype: Optional[str]
        """
        emulator = getattr(self._env, "_emulator", None)
        session_path = getattr(emulator, "session_path", None) if emulator is not None else None
        if not session_path:
            log_warn("The environment exposed no session_path, so this run's provenance "
                     "cannot record where its video was written.", self._parameters)
            return None
        return strategist_session_rel(session_path, game=self._game, parameters=self._parameters)

    def _write_provenance(self) -> str:
        payload = {
            "started_at": self._started_at.isoformat(timespec="seconds"),
            "game": self._game,
            "strategist": self.__class__.__name__,
            "session_dir": self._session_dir,
            "init_kwargs": self._run_config(),
        }
        stamp = self._started_at.strftime("%Y%m%d_%H%M%S")
        path = self._path(f"{PROVENANCE_PREFIX}_{stamp}.json")
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
        log_info(f"Strategist provenance written to {path}", self._parameters)
        return path

    def _path(self, filename: str) -> str:
        return os.path.join(self._dir, filename)

    def _load_or_default(self, cls, factory: Callable[[], Any], episode_number: Optional[int]) -> Any:
        if self._resume and cls.exists(self._resume_dir):
            artifact = cls.load(self._resume_dir, episode_number, parameters=self._parameters)
            if self._resume_dir != self._dir:
                artifact.last_written = None
            return artifact
        return factory()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    @property
    def _vlm(self) -> VLM:
        if self._vlm_instance is None:
            if not self._strategist_vlm_model:
                log_error(f"{self.__class__.__name__} tried to make a VLM call but no "
                          f"strategist_vlm_model was given.", self._parameters)
            self._vlm_instance = VLM(self._strategist_vlm_model, self._strategist_vlm_kind)
        return self._vlm_instance

    def _vlm_call(self, stage: str, **kwargs: Any) -> Any:
        result, records = self._vlm_infer(stage, **kwargs)
        self.report.event_log.extend(records)
        return result

    def _vlm_infer(self, stage: str, **kwargs: Any) -> tuple:
        kwargs.setdefault("max_new_tokens", self._max_new_tokens)
        inferred = self._vlm.infer(**kwargs)
        result, meta = inferred["output"], inferred["meta"]
        texts = kwargs.get("texts")
        images = kwargs.get("images") or []
        records: List[StrategistVLMCallRecord] = []
        if isinstance(texts, list):
            responses = result if isinstance(result, list) else [result] * len(texts)
            token_counts = per_prompt_token_counts(meta, len(texts))
            for index, (prompt, response) in enumerate(zip(texts, responses)):
                call_images = images[index] if index < len(images) and isinstance(images[index], list) else images
                records.append(StrategistVLMCallRecord(
                    stage=stage, images=call_images, prompt=prompt, response=response,
                    input_tokens=token_counts[index][0], output_tokens=token_counts[index][1]))
        else:
            records.append(StrategistVLMCallRecord(
                stage=stage, images=images, prompt=texts, response=result,
                input_tokens=meta["input_tokens"], output_tokens=meta["output_tokens"]))
        return result, records

    def _vlm_caller(self, stage: str):
        def call(**kwargs: Any) -> Any:
            return self._vlm_call(stage, **kwargs)
        return call

    # ------------------------------------------------------------------
    # Screen and prompt blocks
    # ------------------------------------------------------------------

    def _current_place(self) -> str:
        return self.global_map.place_label() if self.global_map.current else "unknown"

    def _current_frame(self) -> np.ndarray:
        return self._env.get_info()["core"]["current_frame"]

    def _env_done(self) -> bool:
        return bool(self._env._emulator.check_if_done())

    def _perceive(self, frame: np.ndarray, episode_number: int) -> str:
        self.tiles.set_vlm_call(self._vlm_caller("perception"))
        self.tiles.record_tiles(frame, episode_number)
        self._identified = self.tiles.identify_tiles(frame)
        self._screen_tiles = verbalize_tiles(self._identified)
        return self._screen_tiles

    def _recent_block(self) -> str:
        recent = self.report.episodes[-self.RECENT_EPISODES:]
        if not recent:
            return ""
        lines = [f"- episode {e.n}, task \"{e.task}\" "
                 f"[{e.status}]: {e.summary or 'no summary'}" for e in recent]
        return P.STRATEGIST_RECENT_BLOCK.replace("[RECENT]", "\n".join(lines))

    def _fill(self, prompt: str) -> str:
        return (prompt.replace("[GAME]", self._game)
                .replace("[RECENT_BLOCK]", self._recent_block())
                .replace("[MAX_STEPS]", str(self._supervisor_max_steps))
                .replace("[GOAL]", frontier_text(self.goals))
                .replace("[CURRENT]", self._current_place())
                .replace("[NOTEPAD]", self.notepad.text.strip() or "(empty)")
                .replace("[GUIDE]", guide_block(self.guide, self.global_map, self.start_episode + self._episodes_run)))

    # ------------------------------------------------------------------
    # The strategist's own calls
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Global map
    # ------------------------------------------------------------------

    def _map_opening_screen(self, frame: np.ndarray) -> None:
        free = self._env.get_info()["pokemon_core"]["agent_state"] == AgentState.FREE_ROAM
        self.global_map.on_step(frame, self._identified if free else {}, free)

    def _failure_streak(self) -> Tuple[int, List[str]]:
        """Episodes in a row, newest first, that did not complete, and their tasks."""
        streak, failed = 0, []
        for e in reversed(self.report.episodes):
            if e.status == "completed":
                break
            streak += 1
            failed.append(e.task)
        return streak, failed

    def _abandon_failing_plan(self, n: int) -> None:
        """
        When exploration takes over because the plan kept failing, drop the plan as well.

        Exploring alone gets the player somewhere new, but the goal tree still points at what was
        failing, so the strategist plans its way straight back. Abandon the generated part of the
        current goal path, down from the deepest goal the game started with (those are never touched),
        record why in the knowledge tree, and let the goal planner re-plan from there this episode.
        Once per trigger.
        """
        if not self._explore_now or exploring(self.global_map, n):
            return
        streak, failed = self._failure_streak()
        if streak < FAILURE_EPISODES:
            return
        frontier = self.goals.frontier()
        if frontier is None:
            return
        path = self.goals.path_to(frontier)
        generated = next((path[:i] for i in range(1, len(path) + 1)
                          if self.goals.tree.at(path[:i]).root.source not in ("initial", "default", "override")),
                         None)
        if generated is None:
            return
        top, leaf = self.goals.tree.at(generated).root.description, frontier.root.description
        tasks = "; ".join(f'"{t}"' for t in list(dict.fromkeys(t for t in failed if t))[:3])
        self.goals.set_status(generated, "abandoned",
                              f"abandoned after {streak} failed episodes in a row ({tasks}); this approach "
                              f"does not work from here", episode_number=n)
        self._remember_all([f"The plan '{top}' (which came down to '{leaf}') failed {streak} episodes in a row "
                            f"and was abandoned: that approach does not work from here, so do not plan it again."], n)
        self.global_map.stats["plans_abandoned"] += 1
        log_info(f"Global map: the plan {top!r} kept failing; abandoned it and re-planning.", self._parameters)

    def _supervisor_steps(self, report: SupervisorReport, marker: Optional[Tuple[int, int]],
                          left: bool = True) -> str:
        n_notes, n_events = marker if marker is not None else (0, 0)
        lines = []
        if n_notes < self.MIN_NAMING_STEPS and len(self.report.episodes) > 1:
            earlier = self.report.episodes[-2].supervisor_report
            if earlier is not None:
                lines += [f"- (previous episode) {note}" for note in earlier.narrative[-self.MIN_NAMING_STEPS:]]
        lines += [f"- {note}" for note in report.narrative[:n_notes]]
        decisions = [e for e in report.event_log[:n_events]
                     if isinstance(e, SupervisorVLMCallRecord) and e.stage == "decision"]
        if decisions and left:
            action = clean_value(parse_key_value(str(decisions[-1].response), "Action"))
            if action:
                lines.append(f"- (under way when the screen went black) {action}")
        return "\n".join(lines)

    def _judge_merge(self, source: int, target: int, overlap: Dict[str, Any]) -> Optional[bool]:
        gmap = self.global_map
        names = (gmap.info(source).name, gmap.info(target).name)
        same = judge_merge(gmap, source, target, overlap, self._vlm_caller("merge_judge"), self._game,
                           self._parameters)
        verdict = "unanswered" if same is None else "the same place" if same else "different places"
        log_info(f"Global map: judged segments {source} '{names[0]}' and {target} '{names[1]}' {verdict}.",
                 self._parameters)
        return same

    def _consolidate(self) -> None:
        for m in self.global_map.consolidate(judge=self._judge_merge):
            log_info(f"Global map: merged segment {m['source']} into {m['target']} at {m['offset']} "
                     f"({m['matches']} of {m['shared']} squares match).", self._parameters)

    def _name_segments(self, report: SupervisorReport, n: int) -> None:
        gmap = self.global_map
        self._consolidate()
        gmap.queue_grown()
        queue, gmap.naming_queue = gmap.naming_queue, []
        tasks = "\n".join(f"- episode {e.n}: {e.task} [{e.status}]" + (f": {e.summary}" if e.summary else "")
                          for e in self.report.episodes[-self.RECENT_EPISODES:])
        done = set()
        for entry in queue:
            left = entry.get("reason", "door") == "door"
            if left:
                segment, mark = gmap.resolve(entry["segment"], entry["exit"])
            else:
                segment = gmap.resolve(entry["segment"], (0, 0))[0]
                mark = gmap.pos if gmap.segment == segment else None
            if segment in done or not gmap.segments[segment] or (left and gmap.info(segment).name is not None):
                continue
            done.add(segment)
            marker = entry["marker"] if left else (len(report.narrative), len(report.event_log))
            before = gmap.info(segment).name
            answer = name_segment(gmap, segment, mark, left, tasks, self._supervisor_steps(report, marker, left),
                                  self._vlm_caller("segment_name"), self._game, n, self._parameters)
            if answer is None:
                continue
            changed, name, description = answer
            if before is None:
                log_info(f"Global map: named segment {segment} '{name}': {description}", self._parameters)
            elif changed:
                log_info(f"Global map: renamed segment {segment} '{before}' -> '{name}': {description}",
                         self._parameters)
            else:
                log_info(f"Global map: kept the name of segment {segment} '{name}'.", self._parameters)

    def _global_map_block(self) -> str:
        if not self.global_map.current:
            return ""
        local = cell_kinds(self._identified, self.global_map.ledge_patterns) if self._identified else None
        self.global_map.prune_routes()
        _, failed = self._failure_streak()
        block, override = prompt_block(self.global_map, list(self.global_map.new_cells_by_episode), local,
                                       explore=self._explore_now,
                                       episode=self.start_episode + self._episodes_run,
                                       failed_tasks=failed)
        if override is not None and override["kind"] == "edge":
            log_info(f"Global map: exploration overrides the task: head to ({override['dx']}, {override['dy']}) "
                     f"{override['sector']} in {self.global_map.area_label(override['area'])}.", self._parameters)
        elif override is not None:
            names = ", ".join(self.global_map.area_label(si) for si in override["areas"])
            log_info(f"Global map: exploration overrides the task: nothing left here, head for one of: {names}.",
                     self._parameters)
        back = self.global_map.defeat_return
        if back is not None and self.start_episode + self._episodes_run - back["episode"] <= self.RETURN_EPISODES:
            block += "\n" + (P.DEFEAT_RETURN_LINE.replace("[EPISODE]", str(back["episode"]))
                             .replace("[WHERE]", self.global_map.defeat_route(back["defeat"])))
        return "\n\n" + block

    def _check_stuck(self, frame: np.ndarray, n: int) -> bool:
        gmap = self.global_map
        if exploring(gmap, n):
            return False
        streak, _ = self._failure_streak()
        signs = stuck_signs(gmap, list(gmap.new_cells_by_episode), streak)
        if signs is None:
            return False
        episodes = "\n".join(f"- episode {e.n}: {e.task} [{e.status}]" + (f": {e.summary}" if e.summary else "")
                             for e in self.report.episodes[-self.RECENT_EPISODES:])
        stuck = judge_stuck(gmap, signs, episodes, frame, self._vlm_caller("stuck_check"), self._game,
                            self._parameters)
        gmap.stats["stuck_checks"] += 1
        gmap.stats["stuck_yes"] += bool(stuck)
        verdict = "unanswered" if stuck is None else "stuck: exploring" if stuck else "not stuck"
        log_info(f"Global map: {signs}; the strategist judged itself {verdict}.", self._parameters)
        return bool(stuck)

    def _dispatch(self, frame: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
        prompt = (self._fill(P.DISPATCH_PROMPT)
                  .replace("[SCREEN_TILES]", self._screen_tiles or "(nothing was recognised)")
                  .replace("[GLOBAL_MAP]", self._global_map_block()))
        dispatched = ask(self._vlm_caller("dispatch"), prompt, _parse_dispatch, "Dispatch",
                         self.MAX_DISPATCH_ATTEMPTS, self._parameters, images=[frame])
        return dispatched if dispatched is not None else (None, None)

    def _review(self, digest: str) -> Tuple[str, Optional[str]]:
        new_doors = self.global_map.new_entrances()
        question, answer = town_question(self.guide) if new_doors else ("", "")
        prompt = (self._fill(P.REVIEW_PROMPT).replace("[DIGEST]", digest)
                  .replace("[TOWN_QUESTION]", question).replace("[TOWN_FORMAT]", answer))
        output = self._vlm_call("review", texts=prompt)
        self.global_map.mark_entrances_seen()
        summary = clean_value(parse_key_value(output, "Summary")) or ""
        town = clean_value(parse_key_value(output, "Town")) if new_doors else None
        if new_doors:
            log_info(f"Strategy guide: {new_doors} new door square(s) on the map; asked about a new town, "
                     f"answer: {town}.", self._parameters)
        return summary, None if town is None or town.strip().rstrip(".").lower() == "none" else town

    def _extract_knowledge(self, digest: str, before: np.ndarray, after: np.ndarray, n: int) -> None:
        prompt = (self._fill(P.KNOWLEDGE_EXTRACT_PROMPT)
                  .replace("[DIGEST]", digest)
                  .replace("[MAX_FACTS]", str(self.MAX_FACTS)))
        facts = ask(self._vlm_caller("extract"), prompt, _parse_facts, "Knowledge extraction",
                    self.MAX_EXTRACT_ATTEMPTS, self._parameters, images=[before, after])
        self._remember_all(facts or [], n)

    @staticmethod
    def _battle_log(report: SupervisorReport) -> Tuple[str, List[np.ndarray]]:
        lines, frames = [], []
        for executor in report.executor_reports:
            if executor.executor_name != CombatExecutor.__name__:
                continue
            lines.append(f"Orders for the fight: {executor.task}")
            for call in executor.vlm_call_log:
                if not isinstance(call, ExecutorVLMCallRecord) or call.tag != "action":
                    continue
                response = str(call.response)
                action = clean_value(parse_key_value(response, "Action")) or "?"
                move = clean_value(parse_key_value(response, "Move"))
                reason = " ".join((clean_value(parse_key_value(response, "Reasoning")) or "").split())
                lines.append(f"- {action}" + (f" move {move}" if move and move.lower() != "none" else "")
                             + (f": {reason[:MAX_BATTLE_REASON]}" if reason else ""))
            if executor.notes and executor.notes.strip():
                lines.append(f"Battle text seen: {executor.notes.strip()}")
            steps = [s for s in executor.steps if isinstance(s, EnvironmentStepRecord)]
            if steps:
                frames = [steps[0].frame_before, steps[-1].frame_before]
        return ("\n".join(lines) or "(no battle was recorded)"), frames

    def _reflect_on_defeat(self, report: SupervisorReport, task: str, n: int) -> None:
        gmap = self.global_map
        defeat = gmap.defeats[-1] if gmap.defeats else None
        where = gmap.defeat_route(defeat) if defeat is not None else "not known"
        battle, images = self._battle_log(report)
        answer = reflect_on_defeat(self.guide, task, frontier_text(self.goals), where, report.digest, battle,
                                   images, self._vlm_caller("defeat_reflection"), self._game, n, self._parameters)
        if answer is None:
            log_warn("Defeat reflection gave no usable reply; nothing was changed.", self._parameters)
            return
        place = f" ({gmap.area_label(gmap.resolve(defeat['segment'], defeat['pos'])[0])})" if defeat else ""
        self._remember_all([f"The player lost a battle in episode {n}{place}: {answer['wrong']}"], n)
        if answer["prepare"]:
            frontier = self.goals.frontier()
            if frontier is not None:
                path = self.goals.path_to(frontier)
                details = f"Added after losing a battle in episode {n}: {answer['wrong']}"
                self.goals.add_under(path, GoalNode(description=answer["prepare"], details=details, source="defeat"),
                                     episode_number=n)
                self.goals.edit(path, achievable_without_subgoals=None, episode_number=n)
                self.goals.set_frontier(self.goals.path_to(self.goals.frontier()), n, episode_number=n)
        if answer["return"] and defeat is not None:
            gmap.defeat_return = {"defeat": defeat, "episode": n}
        log_info(f"Defeat reflection: {answer['wrong']} Plan: "
                 f"{'PREPARE ' + answer['prepare'] if answer['prepare'] else 'CONTINUE'}; "
                 f"return: {'yes' if answer['return'] else 'no'}; {len(answer['changes'])} tip change(s).",
                 self._parameters)

    def _remember_all(self, facts: List[str], n: int) -> None:
        for fact in facts:
            remember(self.knowledge, fact, self._vlm_caller("knowledge"), self._game, n, self._parameters)

    def _update_notepad(self, digest: str, summary: str, n: int) -> None:
        prompt = (self._fill(P.NOTEPAD_UPDATE_PROMPT)
                  .replace("[DIGEST]", digest)
                  .replace("[SUMMARY]", summary or "(none)"))
        update = ask(self._vlm_caller("notepad_update"), prompt, _parse_notepad_update, "Notepad update",
                     self.MAX_NOTEPAD_ATTEMPTS, self._parameters)
        if update is None:
            log_warn("No usable notepad update; leaving the notepad as it was.", self._parameters)
            return
        action, notes = update
        if action == "APPEND":
            self.notepad.add(notes, episode_number=n)
        else:
            self.notepad.update(notes, episode_number=n)

    def _absorb_notepad(self, closed: List[Tuple[Tuple[str, ...], str]], n: int) -> None:
        if self.notepad.text.strip():
            prompt = (self._fill(P.NOTEPAD_EXTRACT_PROMPT)
                      .replace("[CLOSED]", "\n".join(f"- {goal_path_text(path)} [{status}]" for path, status in closed))
                      .replace("[MAX_FACTS]", str(self.MAX_FACTS)))
            facts = ask(self._vlm_caller("notepad_extract"), prompt, _parse_facts, "Notepad extraction",
                        self.MAX_EXTRACT_ATTEMPTS, self._parameters)
            self._remember_all(facts or [], n)
        self.notepad.clear(episode_number=n)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_all(self, episode_number: int) -> None:
        artifacts = [self.tiles, self.goals, self.knowledge, self.notepad, self.guide, self.global_map]
        for artifact in artifacts:
            artifact.save(self._dir, episode_number, parameters=self._parameters)
        self._env.save_custom_state(self._emulator_state_name(self._name, episode_number))

    @staticmethod
    def _emulator_state_name(run_name: str, episode_number: int) -> str:
        """``strategist<name><n>``, saved by the emulator as ``custom_strategist<name><n>.state``.

        The ``strategist`` prefix is what makes these separable from the states
        ``Environment._simulate`` writes, which are named for a uuid4 hex and deleted in a
        ``finally`` that a killed job never reaches. A uuid hex cannot begin with
        ``strategist``, so the sync set's ``custom_strategist*`` pattern cannot pick one up.
        Must stay alphanumeric: ``save_custom_state`` rejects anything else.
        """
        return f"strategist{treat_name(run_name)}{episode_number}"

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def _stop_reason(self) -> Optional[str]:
        if self._episodes_run >= self._max_episodes:
            return "max_episodes"
        if self.goals.tree.root.status == "complete":
            return "goals_complete"
        if self._env_done():
            return "env_done"
        return None

    def run(self) -> StrategistReport:
        if self._ran:
            log_error(f"{self.__class__.__name__}.run() called twice on one instance. The "
                      "report accumulates, so this would merge two runs into one record.",
                      self._parameters)
        self._ran = True
        # Bounded by max_episodes, which is the only stop reason that can be counted
        # ahead of time; the environment ending the run early just leaves the bar short
        # of its total.
        progress = tqdm(total=self._max_episodes, desc="episodes", unit="ep")
        while True:
            stop = self._stop_reason()
            if stop is not None:
                self.report.stop_reason = stop
                progress.set_postfix_str(f"stopped: {stop}")
                break
            record = self._run_episode(self.start_episode + self._episodes_run)
            self._episodes_run += 1
            progress.update(1)
            progress.set_postfix_str(record.status)
        progress.close()
        return self.report

    def _plan_goals(self, n: int) -> None:
        subtree = self.goals.frontier()
        if subtree is None:
            return
        path = self.goals.path_to(subtree)
        since = subtree.root.frontier_since
        if path != self.goals.frontier_path or since is None:
            self.goals.set_frontier(path, n, episode_number=n)
            unsuccessful_for = 0
        else:
            unsuccessful_for = n - since
            if unsuccessful_for <= 0 or unsuccessful_for % self._subgoal_every:
                return
        self.global_map.routes = []
        create_subgoals(self.goals, self.knowledge, self._current_place(), explore_text(self.global_map),
                        self.notepad.text,
                        lambda request: navigate_map(self.global_map, request, self._vlm_caller, self._game,
                                                     self._parameters),
                        self._vlm_caller, self._game, n, unsuccessful_for, self._parameters)

    def _run_episode(self, n: int) -> EpisodeRecord:
        #: Where this episode's own calls start, so the archive holds one episode rather
        #: than every call the run has made so far.
        calls_from = len(self.report.event_log)
        frame = self._current_frame()
        self._perceive(frame, n)
        self._map_opening_screen(frame)
        self._explore_now = self._check_stuck(frame, n)
        self._abandon_failing_plan(n)
        self._plan_goals(n)
        task, guidance = self._dispatch(frame)

        record = EpisodeRecord(n=n, task=task or "", guidance=guidance)
        self.report.episodes.append(record)

        if task is None:
            record.status = "no_task"
            record.summary = "The strategist could not write a task from this screen."
            self._finish_episode(record, calls_from)
            return record

        env = MappingEnvironment(self._env, self.global_map, self.tiles, self._parameters)
        warps_from = len(self.global_map.warps)
        supervisor = PokemonPlayThroughSupervisor(
            task=task, env=env, game=self._game, max_steps=self._supervisor_max_steps,
            tile_recognizer=self.tiles, episode_number=n, guidance=guidance, parameters=self._parameters,
            max_new_tokens=self._supervisor_max_new_tokens, battle_tips=battle_tips(self.guide),
            **self._supervisor_kwargs,
        )
        env.marker = lambda: (len(supervisor.report.narrative), len(supervisor.report.event_log))
        result = supervisor.evaluate()
        report = result["report"]
        record.supervisor_report = report
        record.status = str(result.get("status", "unknown"))
        self.report.event_log.append(report)
        # Let a mid-door-warp black screen clear, so the next task starts from the new place.
        settle_screen(self._env)
        self._name_segments(report, n)
        events = record_events(self.guide, self.global_map, self.global_map.warps[warps_from:], n, self._parameters)

        before, frame = frame, self._current_frame()
        self._perceive(frame, n)

        closed = check_goals(self.goals, report.digest, self.notepad.text, self._vlm_caller, self._game, n,
                             self._parameters)
        summary, town = self._review(report.digest)
        record.summary = summary or record.summary
        town_event = record_new_town(self.guide, town, self._current_place(), n, self._parameters)
        if town_event is not None:
            events.append(town_event)
        if report.narrative or any(r.notes and r.notes.strip() for r in report.executor_reports):
            self._extract_knowledge(report.digest, before, frame, n)
        self._update_notepad(report.digest, summary, n)
        if closed:
            self._absorb_notepad(closed, n)
        if events:
            update_guide(self.guide, events, report.digest, self._vlm_caller("guide_update"), self._game, n,
                         self._parameters)
        if record.status == "lost_battle":
            self._reflect_on_defeat(report, task, n)

        self._finish_episode(record, calls_from)
        return record

    def _archive_episode(self, record: EpisodeRecord, calls_from: int) -> None:
        calls = [event for event in self.report.event_log[calls_from:]
                 if isinstance(event, StrategistVLMCallRecord)]
        archive = EpisodeArchive(
            game=self._game,
            strategist_name=self.__class__.__name__,
            init_kwargs=self._run_config(),
            detail=self._report_detail,
            episode=trim_episode(record, self._report_detail),
            strategist_calls=calls,
        )
        save_episode_report(self._dir, record.n, archive, self._parameters)

    def _finish_episode(self, record: EpisodeRecord, calls_from: int = 0) -> None:
        self.global_map.refresh(self.tiles._by_hash)
        added = self.global_map.end_episode(record.n, judge=self._judge_merge, episode_number=record.n)
        log_info(f"Global map: {added} new squares this episode, {self.global_map.n_cells} in all, "
                 f"{self.global_map.n_areas} area(s).", self._parameters)
        self.save_all(record.n)
        self._archive_episode(record, calls_from)
        if self._on_episode_complete is not None:
            # A run of this length ends by wall clock, not by finishing, so the record has
            # to be on disk before the next episode starts. A checkpoint that cannot be
            # written must not take the run down with it.
            try:
                self._on_episode_complete(self.report)
            except Exception as error:  # noqa: BLE001
                log_warn(f"Could not checkpoint the strategist report: {error}", self._parameters)


def _parse_dispatch(output: str) -> Tuple[Any, Optional[str]]:
    task = clean_value(parse_key_value(output, "Task"))
    if task is None:
        return None, "The `Task:` line was missing or empty. It must be one concrete sentence."
    return (task, clean_value(parse_key_value(output, "Guidance"))), None


def _parse_notepad_update(output: str) -> Tuple[Any, Optional[str]]:
    action, problem = single_value(output, "Action")
    action = (action or "").rstrip(".").strip().upper()
    if problem is not None or action not in ("APPEND", "REWRITE"):
        return None, "There must be exactly one `Action:` line, saying APPEND or REWRITE."
    lines = output.splitlines()
    starts = [i for i, line in enumerate(lines) if line.replace("**", "").strip().lower().startswith("notes:")]
    if len(starts) != 1:
        return None, f"There must be exactly one `Notes:` line, found {len(starts)}."
    first = lines[starts[0]].replace("**", "").strip()[len("notes:"):].strip()
    body = [first] + lines[starts[0] + 1:]
    notes = "\n".join(line for line in body if line.strip() != "[STOP]").strip()
    if not notes:
        return None, "The notes under `Notes:` were empty."
    return (action, notes), None


def _parse_facts(output: str) -> Tuple[Any, Optional[str]]:
    if "facts:" not in output.lower():
        return None, "There was no `Facts:` line. Give the facts as a list under it, or write `Facts: None`."
    facts = [fact for fact in (clean_value(item) for item in parse_list(output, "Facts")) if fact]
    if len(facts) > PokemonStrategist.MAX_FACTS:
        return None, f"{len(facts)} facts were given. Write at most {PokemonStrategist.MAX_FACTS}."
    return facts, None
