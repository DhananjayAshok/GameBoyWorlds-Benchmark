"""
Shared persistence for the strategist's state: goals, knowledge, tiles, the global map and the guide are
each one artifact, checkpointed per episode under the same strategist directory.
"""

from __future__ import annotations

import functools
import os
import pickle
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from utils import file_makedir, log_error, log_info

EPISODE_DIR_RE = re.compile(r"episode_(\d+)")


def episode_dirs(strategist_dir: str) -> Dict[int, str]:
    """Every ``episode_<n>`` directory directly under ``strategist_dir``, keyed by ``n``."""
    if not os.path.isdir(strategist_dir):
        return {}
    found = {}
    for entry in os.listdir(strategist_dir):
        match = EPISODE_DIR_RE.fullmatch(entry)
        if match:
            found[int(match.group(1))] = os.path.join(strategist_dir, entry)
    return found


def update(method):
    """Marks the artifact as changed in ``episode_number``, which the caller must pass by keyword."""
    @functools.wraps(method)
    def wrapper(self, *args, episode_number: int, **kwargs):
        result = method(self, *args, **kwargs)
        self.last_written = episode_number
        return result
    return wrapper


class StrategistArtifact(ABC):
    #: Names of instance attributes that should not be pickled (e.g. closures, open
    #: handles). Subclasses override this to exclude what they can't or shouldn't persist.
    _TRANSIENT: Tuple[str, ...] = ()

    last_written: Optional[int] = None

    @abstractmethod
    def diff(self, old: "StrategistArtifact") -> Dict[str, Any]:
        """What changed going from ``old`` (an earlier instance of the same subclass) to self."""
        raise NotImplementedError

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for name in self._TRANSIENT:
            state.pop(name, None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        for name in self._TRANSIENT:
            self.__dict__.setdefault(name, None)

    @classmethod
    def _filename(cls) -> str:
        return f"{cls.__name__.lower()}.pkl"

    @classmethod
    def _episode_path(cls, strategist_dir: str, episode_number: int) -> str:
        return os.path.join(strategist_dir, f"episode_{episode_number}", cls._filename())

    @classmethod
    def saved_episodes(cls, strategist_dir: str) -> List[int]:
        return sorted(n for n, path in episode_dirs(strategist_dir).items()
                      if os.path.exists(os.path.join(path, cls._filename())))

    @classmethod
    def exists(cls, strategist_dir: str) -> bool:
        """Whether there is anything for :meth:`load` to find, with no episode_number given."""
        return bool(cls.saved_episodes(strategist_dir))

    def save(self, strategist_dir: str, episode_number: int, parameters: Optional[dict] = None) -> Optional[str]:
        if self.last_written is not None and self.last_written != episode_number:
            return None
        if not os.path.isdir(strategist_dir):
            log_error(f"Cannot save {self.__class__.__name__} under {strategist_dir!r}, which "
                      "does not exist.", parameters)
        if self.last_written is None:
            self.last_written = episode_number
        path = self._episode_path(strategist_dir, episode_number)
        file_makedir(path)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, strategist_dir: str, episode_number: Optional[int] = None,
             parameters: Optional[dict] = None) -> Any:
        if episode_number is None:
            episodes = cls.saved_episodes(strategist_dir)
            if not episodes:
                log_error(f"No saved {cls.__name__} found under {strategist_dir!r}.", parameters)
            episode_number = max(episodes)
            path = cls._episode_path(strategist_dir, episode_number)
            with open(path, "rb") as f:
                obj = pickle.load(f)
            log_info(f"Loaded {cls.__name__} from episode {episode_number} at {path}", parameters)
        else:
            path = cls._episode_path(strategist_dir, episode_number)
            if not os.path.exists(path):
                log_error(f"No saved {cls.__name__} for episode {episode_number} at {path}.", parameters)
            with open(path, "rb") as f:
                obj = pickle.load(f)
        obj.last_written = episode_number
        return obj
