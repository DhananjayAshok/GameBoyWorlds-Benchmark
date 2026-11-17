from __future__ import annotations

import gzip
import os
import pickle
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Union

import numpy as np

from execution.artifact import episode_dirs
from execution.report import ExecutorReport, SupervisorReport
from utils import file_makedir, log_warn

# How much of an episode is archived; executor legs carry every frame, so keeping them is opt-in.
REPORT_DETAILS = ("strategist", "supervisor", "executor")

EPISODE_REPORT_FILENAME = "report.pkl.gz"


@dataclass
class StrategistVLMCallRecord:
    stage: str
    images: List[np.ndarray]
    prompt: str
    response: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class EpisodeRecord:
    n: int
    task: str
    guidance: Optional[str] = None
    supervisor_report: Optional[SupervisorReport] = None
    status: str = "incomplete"
    summary: str = ""


@dataclass
class StrategistReport:
    game: str
    strategist_name: str
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    episodes: List[EpisodeRecord] = field(default_factory=list)
    event_log: List[Union[StrategistVLMCallRecord, SupervisorReport]] = field(default_factory=list)
    stop_reason: str = "incomplete"

    def __str__(self) -> str:
        lines = [f"Strategist: {self.strategist_name} on {self.game}",
                 f"Stopped: {self.stop_reason} after {len(self.episodes)} episodes",
                 "",
                 f"{'#':>3}  {'status':<24} task"]
        lines += [f"{episode.n:>3}  {episode.status:<24} {episode.task}" for episode in self.episodes]
        return "\n".join(lines)


@dataclass
class EpisodeArchive:
    """One episode of a run, saved beside that episode's artifacts."""

    game: str
    strategist_name: str
    init_kwargs: Dict[str, Any]
    detail: str
    episode: EpisodeRecord
    strategist_calls: List[StrategistVLMCallRecord] = field(default_factory=list)


def trim_episode(record: EpisodeRecord, detail: str) -> EpisodeRecord:
    # Returns a copy: the live record is still being read by the running strategist.
    if detail == "executor" or record.supervisor_report is None:
        return record
    kept = ([e for e in record.supervisor_report.event_log if not isinstance(e, ExecutorReport)]
            if detail == "supervisor" else [])
    return replace(record, supervisor_report=replace(record.supervisor_report, event_log=kept))


def episode_report_path(run_dir: str, episode_number: int) -> str:
    return os.path.join(run_dir, f"episode_{episode_number}", EPISODE_REPORT_FILENAME)


def save_episode_report(run_dir: str, episode_number: int, archive: EpisodeArchive,
                        parameters: Optional[dict] = None) -> Optional[str]:
    path = episode_report_path(run_dir, episode_number)
    try:
        file_makedir(path)
        with gzip.open(path, "wb", compresslevel=6) as handle:
            pickle.dump(archive, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path
    # A failed write must not end a run whose episode is already paid for in VLM calls.
    except Exception as error:  # noqa: BLE001
        log_warn(f"Could not write the episode report to {path}: {error}", parameters)
        return None


def load_episode_report(run_dir: str, episode_number: int) -> EpisodeArchive:
    with gzip.open(episode_report_path(run_dir, episode_number), "rb") as handle:
        return pickle.load(handle)


def saved_episode_reports(run_dir: str) -> List[int]:
    return sorted(n for n, path in episode_dirs(run_dir).items()
                  if os.path.exists(os.path.join(path, EPISODE_REPORT_FILENAME)))
