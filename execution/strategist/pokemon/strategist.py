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
from tqdm import tqdm

from execution.artifact import episode_dirs
from execution.perception.pokemon.tiles import TileRecognizer, verbalize_tiles
from execution.pokemon.supervisors import PokemonPlayThroughSupervisor
from execution.report import per_prompt_token_counts
from execution.strategist.pokemon import prompts as P
from execution.strategist.pokemon.asking import ask, single_value
from execution.strategist.pokemon.goals import (PokemonGoalTree, check_goals, create_subgoals, frontier_text,
                                                goal_path_text)
from execution.strategist.pokemon.knowledge import PokemonKnowledgeTree, remember
from execution.strategist.pokemon.location import (FrameTail, LocationStore, Place, episode_frames, follow_transitions,
                                                   is_black, locate, place_text, settle_screen, verify_location)
from execution.strategist.pokemon.notepad import ThoughtNotepad
from execution.strategist.report import (REPORT_DETAILS, EpisodeArchive, EpisodeRecord, StrategistReport,
                                         StrategistVLMCallRecord, save_episode_report, trim_episode)
from python_scripts.paths import strategist_dir, strategist_session_rel
from utils import VLM, load_parameters, log_error, log_info, log_warn
from utils.lm_inference import clean_value, parse_key_value
from utils.parsing import parse_list

#: One file per process, stamped like the run it records, so a resumed run writes its own.
PROVENANCE_PREFIX = "provenance"

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
        self.locations = self._load_or_default(LocationStore, LocationStore, load_episode)
        self.knowledge = self._load_or_default(PokemonKnowledgeTree, PokemonKnowledgeTree, load_episode)
        self.notepad = self._load_or_default(ThoughtNotepad, ThoughtNotepad, load_episode)

        if not resume:
            self.start_episode = 0
        elif self._resume_episode != "latest":
            self.start_episode = self._resume_episode + 1
        else:
            resumed_from = episode_dirs(self._resume_dir)
            self.start_episode = (max(resumed_from) + 1) if resumed_from else 0
        if self.start_episode in episode_dirs(self._dir):
            log_error(f"episode_{self.start_episode} already exists under {self._dir!r}; "
                      "refusing to overwrite it.", self._parameters)
        if self.start_episode > 0:
            self._env.load_custom_state(
                self._emulator_state_name(self._resume_run or name, self.start_episode - 1))

        #: The end of the frame stream so far, so a door warp that straddles two episodes is still seen whole.
        self._frame_tail = FrameTail()
        #: A different place named by the last location check and not yet accepted (needs a second check).
        self._relocation: Optional[Place] = None
        self._identified: Optional[Dict[Any, Any]] = None
        self._screen_tiles: Optional[str] = None
        #: Episodes run by *this* instance, which is what ``max_episodes`` bounds.
        self._episodes_run = 0
        self._ran = False

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

    @property
    def current_location(self) -> Optional[Place]:
        return self.locations.current

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
                .replace("[CURRENT]", place_text(self.current_location))
                .replace("[NOTEPAD]", self.notepad.text.strip() or "(empty)"))

    # ------------------------------------------------------------------
    # The strategist's own calls
    # ------------------------------------------------------------------

    def _dispatch(self, frame: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
        prompt = (self._fill(P.DISPATCH_PROMPT)
                  .replace("[SCREEN_TILES]", self._screen_tiles or "(nothing was recognised)"))
        dispatched = ask(self._vlm_caller("dispatch"), prompt, _parse_dispatch, "Dispatch",
                         self.MAX_DISPATCH_ATTEMPTS, self._parameters, images=[frame])
        return dispatched if dispatched is not None else (None, None)

    def _review(self, digest: str) -> str:
        prompt = self._fill(P.REVIEW_PROMPT).replace("[DIGEST]", digest)
        output = self._vlm_call("review", texts=prompt)
        return clean_value(parse_key_value(output, "Summary")) or ""

    def _extract_knowledge(self, digest: str, before: np.ndarray, after: np.ndarray, n: int) -> None:
        prompt = (self._fill(P.KNOWLEDGE_EXTRACT_PROMPT)
                  .replace("[DIGEST]", digest)
                  .replace("[MAX_FACTS]", str(self.MAX_FACTS)))
        facts = ask(self._vlm_caller("extract"), prompt, _parse_facts, "Knowledge extraction",
                    self.MAX_EXTRACT_ATTEMPTS, self._parameters, images=[before, after])
        self._remember_all(facts or [], n)

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
        for artifact in (self.tiles, self.goals, self.locations, self.knowledge, self.notepad):
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
        create_subgoals(self.goals, self.knowledge, self.locations, self.notepad.text, self._vlm_caller,
                        self._game, n, unsuccessful_for, self._parameters)

    def _run_episode(self, n: int) -> EpisodeRecord:
        #: Where this episode's own calls start, so the archive holds one episode rather
        #: than every call the run has made so far.
        calls_from = len(self.report.event_log)
        frame = self._current_frame()
        self._perceive(frame, n)
        if self.current_location is None:
            locate(self.locations, frame, self._screen_tiles, self._identified, self._vlm_caller, self._game, n,
                   self._parameters)
        elif not is_black(frame):
            self._relocation = verify_location(self.locations, frame, self._screen_tiles, self._identified,
                                               self._vlm_caller, self._game, n, self._relocation, self._parameters)
        self._plan_goals(n)
        task, guidance = self._dispatch(frame)

        record = EpisodeRecord(n=n, task=task or "", guidance=guidance)
        self.report.episodes.append(record)

        if task is None:
            record.status = "no_task"
            record.summary = "The strategist could not write a task from this screen."
            self._finish_episode(record, calls_from)
            return record

        supervisor = PokemonPlayThroughSupervisor(
            task=task, env=self._env, game=self._game, max_steps=self._supervisor_max_steps,
            tile_recognizer=self.tiles, episode_number=n, guidance=guidance, parameters=self._parameters,
            max_new_tokens=self._supervisor_max_new_tokens,
            **self._supervisor_kwargs,
        )
        result = supervisor.evaluate()
        report = result["report"]
        record.supervisor_report = report
        record.status = str(result.get("status", "unknown"))
        self.report.event_log.append(report)
        # An episode can end mid-door-warp. Let the screen come back before reading the frames, so the
        # transition is judged in this episode and the next task is dispatched from the new place.
        settled = settle_screen(self._env)
        follow_transitions(self.locations, episode_frames(report) + settled, self._frame_tail, self.tiles,
                           self._vlm_caller, self._game, n, self._parameters)

        before, frame = frame, self._current_frame()
        self._perceive(frame, n)

        closed = check_goals(self.goals, report.digest, self.notepad.text, self._vlm_caller, self._game, n,
                             self._parameters)
        summary = self._review(report.digest)
        record.summary = summary or record.summary
        if report.narrative or any(r.notes and r.notes.strip() for r in report.executor_reports):
            self._extract_knowledge(report.digest, before, frame, n)
        self._update_notepad(report.digest, summary, n)
        if closed:
            self._absorb_notepad(closed, n)

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
