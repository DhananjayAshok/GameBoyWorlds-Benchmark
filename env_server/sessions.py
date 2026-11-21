import json
import os
import re
import secrets
import threading
import time
import uuid
from typing import Optional

import numpy as np
from gameboy_worlds import get_environment, get_test_environment

from env_server import codec, splits
from env_server.policy import Policy
from env_server.progress import Progress
from env_server.recorder import Recorder
from utils.log_handling import log_info, log_warn


class SessionError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


CHECKPOINT_STATE = "emulator.state"
CHECKPOINT_META = "session.json"
SAVE_NAME = re.compile(r"^[A-Za-z0-9_]{1,128}$")
VIDEO_ROLL_SECONDS = 60
FRAME_INDEX = "frame_index.jsonl"


def read_checkpoint(*, directory: Optional[str]) -> Optional[dict]:
    if not directory:
        return None
    meta = os.path.join(directory, CHECKPOINT_META)
    if not os.path.exists(meta) or not os.path.exists(os.path.join(directory, CHECKPOINT_STATE)):
        return None
    with open(meta) as f:
        return json.load(f)


class Session:
    def __init__(self, *, policy: Policy, task_id: Optional[str], game: Optional[str],
                 init_state: Optional[str], want_state: bool, max_steps: int,
                 record_dir: Optional[str], session_name: Optional[str], parameters: dict,
                 checkpoint_dir: Optional[str] = None, checkpoint_every: int = 0,
                 saved: Optional[dict] = None, obs_dir: Optional[str] = None,
                 saves_dir: Optional[str] = None, progress_dir: Optional[str] = None,
                 run: Optional[str] = None) -> None:
        self._parameters = parameters
        self.saves_dir = saves_dir
        self.run = run
        self.policy = policy
        self.task_id = task_id
        self.id = saved["session"] if saved else uuid.uuid4().hex[:16]
        self.token = saved["token"] if saved else secrets.token_urlsafe(24)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self._checkpointed_at = 0
        self.label: Optional[str] = saved.get("label") if saved else None
        self.discarded_steps = int(saved.get("discarded_steps", 0)) if saved else 0
        self._video_started = time.time()
        self._indexed: tuple = (None, 0)
        self._meta: Optional[dict] = saved
        self.want_state = True if not policy.state_optional else bool(want_state)
        self.success: Optional[float] = None
        self.episode_over = False
        self.started = False
        self.closed = False
        self.last_used = time.time()
        self.lock = threading.Lock()
        self._handles: set = set()

        # gameboy_worlds loads its own config (ROM paths); it must not be handed this repo's parameters.
        emulator_kwargs = {"headless": True, "session_name": session_name, "max_steps": max_steps}
        if policy.long_running:
            emulator_kwargs["save_video"] = True
            emulator_kwargs["session_name"] = session_name or f"env_server/{self.id}"
        if obs_dir:
            emulator_kwargs["save_video"] = True
            emulator_kwargs["session_name"] = os.path.join(os.path.abspath(obs_dir), self.id)
        if task_id is not None:
            row = splits.row_of(task_id=task_id, parameters=parameters)
            self.goal = str(row["task"])
            self.game = str(row["game"])
            self.env = get_test_environment(row=row, controller_variant=policy.controller, **emulator_kwargs)
        else:
            self.goal = ""
            self.game = game
            self.env = get_environment(game=game, environment_variant="default",
                                       controller_variant=policy.controller, init_state=init_state,
                                       **emulator_kwargs)
        self.recorder = Recorder(directory=record_dir, session=self.id, task_id=task_id, game=self.game,
                                 policy=policy.kind, frames=not policy.long_running,
                                 parameters=parameters) if record_dir else None
        if self.env._emulator.save_video:
            log_info(f"session {self.id} video: {self.env._emulator.session_path}", parameters=parameters)
        memory = getattr(self.env._emulator.state_parser, "memory", None)
        self.progress = Progress(directory=progress_dir, session=self.id, game=self.game, run=run,
                                 memory=memory, parameters=parameters) \
            if progress_dir and memory is not None else None
        if saved:
            self._restore(saved=saved)

    @property
    def is_task(self) -> bool:
        return self.task_id is not None

    def describe(self) -> dict:
        return {"session": self.id, "run": self.run, "task_id": self.task_id, "game": self.game, "goal": self.goal,
                "policy": self.policy.kind, "controller": self.policy.controller, "state_enabled": self.want_state,
                "state_mode": self.policy.state_mode,
                "can_reset": self.policy.reset_again or not self.started,
                "save_load": self.policy.save_load, "named_saves": self.policy.named_saves,
                "free_play": not self.is_task,
                "episode_over": self.episode_over, "success": self.success,
                "steps": self._steps(), "max_steps": self._max_steps(), "subgoals": self._subgoals(),
                "label": self.label, "discarded_steps": self.discarded_steps}

    def _steps(self) -> int:
        return int(getattr(self.env._emulator, "step_count", 0))

    def _max_steps(self) -> Optional[int]:
        return None if not self.is_task else int(getattr(self.env._emulator, "max_steps", 0))

    def _subgoals(self) -> Optional[dict]:
        metric = self.env._emulator.state_tracker.metrics.get("subgoals")
        if metric is None:
            return None
        return {"all": [s.NAME for s in metric._subgoals],
                "completed": [s.NAME for s in metric._subgoals if s.completed]}

    def checkpoint(self) -> None:
        if not self.checkpoint_dir:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._checkpointed_at = self._steps()
        self._meta = {**self.describe(), "token": self.token, "started": self.started}
        # State before meta, each replaced atomically: a crash between the two leaves a loadable pair
        # whose step count is at most one interval behind the emulator's.
        self._write(CHECKPOINT_STATE, lambda f: self.env._emulator._pyboy.save_state(f))
        self._write_meta()

    def _write(self, name: str, write) -> None:
        path = os.path.join(self.checkpoint_dir, name)
        with open(path + ".tmp", "wb") as f:
            write(f)
        os.replace(path + ".tmp", path)

    def _write_meta(self) -> None:
        self._write(CHECKPOINT_META, lambda f: f.write(json.dumps(self._meta, indent=1).encode()))

    def mark(self, *, label: str) -> dict:
        if not self.checkpoint_dir:
            raise SessionError("this server keeps no checkpoints", 403)
        self.last_used = time.time()
        self.label = label
        self.checkpoint()
        return {"label": label, "steps": self._steps()}

    def _discard_since_checkpoint(self) -> None:
        lost = self._steps() - self._checkpointed_at
        if lost <= 0 or self._meta is None:
            return
        self.discarded_steps += lost
        self._meta = {**self._meta, "discarded_steps": self.discarded_steps}
        self._write_meta()

    def _apply_state(self, *, state_path: str, saved: dict) -> None:
        emulator = self.env._emulator
        with open(state_path, "rb") as f:
            emulator._pyboy.load_state(f)
        # Zero ticks redraws the screen from the loaded state without advancing the game.
        emulator._pyboy.tick(0, True)
        emulator.step_count = int(saved["steps"])
        # Screen-matched subgoals are seen only at the moment they happen, so they are restored by name.
        done = set((saved.get("subgoals") or {}).get("completed", []))
        metric = emulator.state_tracker.metrics.get("subgoals")
        for subgoal in (metric._subgoals if metric is not None else []):
            subgoal.completed = subgoal.NAME in done
        emulator._update_listeners_after_actions(np.array([emulator.get_current_frame()]))
        if emulator.save_video:
            emulator.video_writer.start_video()
            self._video_started = time.time()

    def _restore(self, *, saved: dict) -> None:
        self._apply_state(state_path=os.path.join(self.checkpoint_dir, CHECKPOINT_STATE), saved=saved)
        self._checkpointed_at = self._steps()
        self.started = bool(saved["started"])
        self.episode_over = bool(saved["episode_over"])
        self.success = saved["success"]
        log_info(f"restored session {self.id} at step {self._steps()}", parameters=self._parameters)

    def _payload(self, *, info: Optional[dict], action_success=None, valid: bool = True,
                 error: Optional[str] = None) -> dict:
        report = info if isinstance(info, dict) else self.env.get_info()
        out = codec.public_obs(frame=self.env.get_observation(), goal=self.goal,
                               actions=self._actions(), steps=self._steps(), max_steps=self._max_steps(),
                               report=report, action_success=action_success)
        if self.policy.transitions and isinstance(info, dict) \
                and "previous_action_details" in info.get("core", {}):
            out["transition"] = codec.transition(info=info)
        payload = {"obs": out, "valid": valid, "error": error, "done": self.episode_over,
                   "success": self.success, "session": self.id}
        if self.want_state:
            payload["state"] = codec.state_for(report=report, mode=self.policy.state_mode)
        return payload

    def _actions(self) -> dict:
        try:
            strings = self.env.get_action_strings()
        except Exception as exc:
            log_warn(f"get_action_strings failed: {exc}", parameters=self._parameters)
            return {}
        return {(k.__name__ if isinstance(k, type) else str(k)): str(v) for k, v in strings.items()}

    def reset(self) -> dict:
        if self.started and not self.policy.reset_again:
            raise SessionError("this episode has started and cannot be reset on this server", 403)
        self.started = True
        self.last_used = time.time()
        _, info = self.env.reset()
        self._index_frames(kind="reset")
        self.episode_over = False
        self.success = None
        if self.progress:
            self.progress.start_over()
            self._write_progress(kind="reset")
        if self.recorder:
            self.recorder.write(kind="reset", action=None, info=info, success=None,
                                frame=self.env.get_observation())
        return self._payload(info=info)

    def step(self, *, action: str) -> dict:
        self.last_used = time.time()
        if self.episode_over:
            raise SessionError("the episode is over; reset first" if self.policy.reset_again
                               else "the episode is over", 409)
        # The emulator is live from construction, so a step starts the episode as much as a reset does.
        self.started = True
        obs, reward, terminated, truncated, info = self.env.step_str(action)
        if obs is None and info is None:
            return self._payload(info=None, valid=False,
                                 error="not a valid action; see obs.actions for the format")
        if isinstance(info, dict) and info.get("invalid_action"):
            return self._payload(info=None, valid=False,
                                 error="that action is not available on this screen; see obs.actions")
        self.episode_over = bool(terminated or truncated)
        if self.episode_over and self.is_task:
            self.success = 1.0 if bool(terminated) else 0.0
        action_success = info.get("core", {}).get("previous_action_details", (None,) * 4)[3] \
            if isinstance(info, dict) else None
        if self.recorder:
            self.recorder.write(kind="step", action=action, info=info, success=self.success,
                                frame=self.env.get_observation())
        if self.progress:
            self.progress.observe(steps=self._steps())
            if self.success == 1.0:
                self._write_progress(kind="cleared")
            elif self.progress.due(steps=self._steps()):
                self._write_progress(kind="periodic")
        # One high-level action can take many presses, so this counts presses since the last checkpoint.
        if self.episode_over or (self.checkpoint_every and self.label is None
                                 and self._steps() - self._checkpointed_at >= self.checkpoint_every):
            self.checkpoint()
        self._index_frames(kind="step")
        if self.policy.long_running and time.time() - self._video_started >= VIDEO_ROLL_SECONDS:
            self._new_video()
        return self._payload(info=info, action_success=action_success)

    def _index_frames(self, *, kind: str) -> None:
        emulator = self.env._emulator
        if not (self.policy.long_running and emulator.save_video
                and getattr(emulator.video_writer, "video_path", None)):
            return
        writer = emulator.video_writer
        video = os.path.relpath(writer.video_path, emulator.session_path)
        first = self._indexed[1] if self._indexed[0] == video else 0
        self._indexed = (video, writer.frames_written)
        row = {"t": round(time.time(), 2), "kind": kind, "step": self._steps(), "video": video,
               "frames": [first, writer.frames_written]}
        try:
            with open(os.path.join(emulator.session_path, FRAME_INDEX), "a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError as exc:
            log_warn(f"frame index for session {self.id}: {exc}", parameters=self._parameters)

    def _new_video(self) -> None:
        emulator = self.env._emulator
        if emulator.save_video and emulator.video_writer.video_running:
            emulator.video_writer.start_video()
        self._video_started = time.time()

    def observe(self) -> dict:
        self.last_used = time.time()
        return self._payload(info=None)

    def _write_progress(self, *, kind: str, save: Optional[str] = None) -> None:
        try:
            self.progress.write(kind=kind, steps=self._steps(), subgoals=self._subgoals(),
                                success=self.success, save=save)
        except Exception as exc:
            log_warn(f"progress for session {self.id}: {exc}", parameters=self._parameters)

    def _named_paths(self, *, name) -> tuple:
        if not isinstance(name, str) or not SAVE_NAME.match(name):
            raise SessionError("a save name is 1-128 letters, digits or underscores")
        base = os.path.join(self.saves_dir, self.game, name)
        return base + ".state", base + ".json"

    def save(self, *, name: Optional[str] = None) -> dict:
        if not self.policy.save_load:
            raise SessionError("save/load is not available on this server", 403)
        self.last_used = time.time()
        if self.policy.named_saves:
            state_path, meta_path = self._named_paths(name=name)
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            with open(state_path + ".tmp", "wb") as f:
                self.env._emulator._pyboy.save_state(f)
            os.replace(state_path + ".tmp", state_path)
            meta = {"steps": self._steps(), "subgoals": self._subgoals()}
            if self.progress:
                meta["progress"] = self.progress.lineage()
            with open(meta_path + ".tmp", "w") as f:
                json.dump(meta, f)
            os.replace(meta_path + ".tmp", meta_path)
            if self.progress:
                self._write_progress(kind="save", save=name)
            self._new_video()
            return {"handle": name, "steps": self._steps()}
        handle = uuid.uuid4().hex[:16]
        self.env.save_custom_state(handle)
        self._handles.add(handle)
        return {"handle": handle}

    def load(self, *, handle: str) -> dict:
        if not self.policy.save_load:
            raise SessionError("save/load is not available on this server", 403)
        self.last_used = time.time()
        if self.policy.named_saves:
            state_path, meta_path = self._named_paths(name=handle)
            if not os.path.exists(state_path) or not os.path.exists(meta_path):
                raise SessionError("unknown handle", 404)
            with open(meta_path) as f:
                saved = json.load(f)
            self.started = True
            self._apply_state(state_path=state_path, saved=saved)
            self._index_frames(kind="load")
            if self.progress:
                self.progress.restore(lineage=saved.get("progress"))
                self._write_progress(kind="load", save=handle)
        else:
            if handle not in self._handles:
                raise SessionError("unknown handle", 404)
            self.env.load_custom_state(handle)
        self.episode_over = False
        self.success = None
        return self._payload(info=None)

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for handle in self._handles:
                try:
                    self.env.delete_custom_state(handle)
                except Exception as exc:
                    log_warn(f"deleting save state {handle}: {exc}", parameters=self._parameters)
            self._handles.clear()
            try:
                if self.label is None or self.episode_over:
                    self.checkpoint()
                elif self.checkpoint_dir:
                    self._discard_since_checkpoint()
            except Exception as exc:
                log_warn(f"checkpointing session {self.id}: {exc}", parameters=self._parameters)
            if self.progress and self.started:
                self._write_progress(kind="close")
            try:
                self.env.close()
            except Exception as exc:
                log_warn(f"closing session {self.id}: {exc}", parameters=self._parameters)
            if self.recorder:
                self.recorder.finish(success=self.success)


class Sessions:
    def __init__(self, *, policy: Policy, max_sessions: int, idle_seconds: float,
                 record_dir: Optional[str], max_steps: Optional[int], parameters: dict,
                 session_name: Optional[str] = None, checkpoint_dir: Optional[str] = None,
                 checkpoint_every: int = 500, obs_dir: Optional[str] = None,
                 saves_dir: Optional[str] = None, progress_dir: Optional[str] = None) -> None:
        self._parameters = parameters
        self.policy = policy
        self.progress_dir = progress_dir
        if checkpoint_dir and policy.kind != "playthrough":
            raise ValueError(f"a {policy.kind} server does not checkpoint; only playthrough does")
        if policy.named_saves != bool(saves_dir):
            raise ValueError(f"a {policy.kind} server {'needs' if policy.named_saves else 'takes no'} saves_dir")
        self.obs_dir = obs_dir
        self.saves_dir = saves_dir
        # A checkpoint directory holds one session, so a checkpointing server serves one.
        self.max_sessions = 1 if checkpoint_dir else max_sessions
        self.idle_seconds = idle_seconds
        self.record_dir = record_dir
        self.max_steps = policy.max_steps if max_steps is None else max_steps
        self.session_name = session_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self._sessions: dict = {}
        self._opening = 0
        self._lock = threading.Lock()

    def _build(self, *, task_id, game, init_state, want_state, saved=None, run=None) -> Session:
        return Session(policy=self.policy, task_id=task_id, game=game, init_state=init_state,
                       want_state=want_state, max_steps=self.max_steps, record_dir=self.record_dir,
                       session_name=self.session_name, parameters=self._parameters,
                       checkpoint_dir=self.checkpoint_dir, checkpoint_every=self.checkpoint_every,
                       saved=saved, obs_dir=self.obs_dir, saves_dir=self.saves_dir,
                       progress_dir=self.progress_dir, run=run)

    def listed(self) -> list:
        tasks = splits.listed(splits_wanted=self.policy.splits_listed, parameters=self._parameters)
        if not self.policy.train_games_only:
            return tasks
        allowed = set(splits.train_sandbox_games(parameters=self._parameters))
        return [t for t in tasks if splits.game_of(task_id=t["task_id"]) in allowed]

    def free_play(self) -> dict:
        if not self.policy.free_play:
            return {}
        return splits.free_play_states(parameters=self._parameters)

    def open(self, *, task_id: Optional[str] = None, game: Optional[str] = None,
             init_state: Optional[str] = None, want_state: bool = False,
             run: Optional[str] = None) -> Session:
        if (task_id is None) == (game is None):
            raise SessionError("pass exactly one of task_id or game (with init_state)")
        if run is not None and (not isinstance(run, str) or not SAVE_NAME.match(run)):
            raise SessionError("a run name is 1-128 letters, digits or underscores")
        if task_id is not None:
            if not self.policy.visible(task_id=task_id, parameters=self._parameters):
                raise SessionError("no such task", 404)
        else:
            if not self.policy.free_play:
                raise SessionError("free play is not available on this server", 403)
            allowed = splits.free_play_states(parameters=self._parameters)
            if game not in allowed or init_state not in allowed[game]:
                raise SessionError("no such start point", 404)
        self.reap()
        # A checkpointed run is one session for good: opening its task again hands back that session,
        # restored from disk if it was closed, and never a fresh start.
        saved = read_checkpoint(directory=self.checkpoint_dir)
        if saved and saved["task_id"] != task_id:
            raise SessionError(f"this run is {saved['task_id']}; it cannot open another task", 409)
        replaced = []
        with self._lock:
            for existing in (self._sessions.values() if self.checkpoint_dir else ()):
                return existing
            if self.policy.opening_replaces:
                while self._sessions and len(self._sessions) + self._opening >= self.max_sessions:
                    oldest = min(self._sessions.values(), key=lambda s: s.last_used)
                    replaced.append(self._sessions.pop(oldest.id))
            if len(self._sessions) + self._opening >= self.max_sessions:
                raise SessionError("too many open sessions; close one first", 429)
            self._opening += 1
        for session in replaced:
            log_info(f"closing session {session.id} to open a new one", parameters=self._parameters)
            session.close()
        session = None
        try:
            session = self._build(task_id=task_id, game=game, init_state=init_state, want_state=want_state,
                                  saved=saved, run=run)
        finally:
            with self._lock:
                self._opening -= 1
                if session is not None:
                    self._sessions[session.id] = session
        log_info(f"opened session {session.id} ({task_id or f'{game}/{init_state}'})",
                 parameters=self._parameters)
        return session

    def get(self, *, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionError("no such session", 404)
        return session

    def token_of(self, *, session_id: str) -> Optional[str]:
        with self._lock:
            session = self._sessions.get(session_id)
        return None if session is None else session.token

    def close(self, *, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise SessionError("no such session", 404)
        session.close()

    def saves(self) -> list:
        if not self.saves_dir:
            raise SessionError("this server keeps no named saves", 403)
        found = []
        for game in sorted(os.listdir(self.saves_dir)) if os.path.isdir(self.saves_dir) else []:
            for entry in sorted(os.listdir(os.path.join(self.saves_dir, game))):
                if entry.endswith(".state"):
                    path = os.path.join(self.saves_dir, game, entry)
                    found.append({"game": game, "name": entry[:-len(".state")],
                                  "t": round(os.path.getmtime(path), 1), "bytes": os.path.getsize(path)})
        return found

    def delete_save(self, *, game: str, name: str) -> None:
        if not self.saves_dir:
            raise SessionError("this server keeps no named saves", 403)
        if not SAVE_NAME.match(game) or not SAVE_NAME.match(name):
            raise SessionError("a save is named by its game and 1-128 letters, digits or underscores")
        base = os.path.join(self.saves_dir, game, name)
        if not os.path.exists(base + ".state"):
            raise SessionError("no such save", 404)
        for path in (base + ".state", base + ".json"):
            if os.path.exists(path):
                os.remove(path)

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()

    def reap(self) -> None:
        if self.policy.long_running:
            return
        cutoff = time.time() - self.idle_seconds
        with self._lock:
            stale = [s for s in self._sessions.values() if s.last_used < cutoff]
            for session in stale:
                self._sessions.pop(session.id, None)
        for session in stale:
            log_info(f"reaping idle session {session.id}", parameters=self._parameters)
            session.close()
