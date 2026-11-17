from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from utils.fundamental import file_makedir
from utils.log_handling import log_error
from utils.parameter_handling import load_parameters

TASK_SECTION = "Task Understanding"
IMAGE_SECTION = "Image Understanding"


@dataclass
class Provenance:
    source: str = ""
    executor: Optional[str] = None
    controller_variant: Optional[str] = None
    model: Optional[str] = None
    trajectory_stem: Optional[str] = None
    built_at: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    @property
    def label(self) -> str:
        return self.source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "executor": self.executor,
            "controller_variant": self.controller_variant,
            "model": self.model,
            "trajectory_stem": self.trajectory_stem,
            "built_at": self.built_at,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Provenance":
        data = data or {}
        return cls(
            source=data.get("source", ""),
            executor=data.get("executor"),
            controller_variant=data.get("controller_variant"),
            model=data.get("model"),
            trajectory_stem=data.get("trajectory_stem"),
            built_at=data.get("built_at"),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
        )


@dataclass
class Entry:
    category: str
    description: str = ""
    examples: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    frame: Optional[str] = None          # relative to the document's frames_root
    example_frames: List[str] = field(default_factory=list)
    init_states: List[str] = field(default_factory=list)
    source: Optional[str] = None

    # Absolute path to ``frame``, filled in by ``load_document``; not persisted.
    resolved_frame: Optional[str] = None

    def evidence_block(self, include_frame_note: bool = False) -> str:
        lines = [f"Category: {self.category}", f"Description: {self.description}"]
        if self.examples:
            lines.append("Examples:")
            lines.extend(f"- {e}" for e in self.examples)
        if include_frame_note and self.frame:
            lines.append("(a representative frame for this entry is shown as an image)")
        return "\n".join(lines)

    def insights_block(self) -> str:
        return "\n".join(f"- {i}" for i in self.insights)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "description": self.description,
            "examples": list(self.examples),
            "insights": list(self.insights),
            "frame": self.frame,
            "example_frames": list(self.example_frames),
            "init_states": list(self.init_states),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entry":
        return cls(
            category=data.get("category", ""),
            description=data.get("description", ""),
            examples=list(data.get("examples") or []),
            insights=list(data.get("insights") or []),
            frame=data.get("frame"),
            example_frames=list(data.get("example_frames") or []),
            init_states=list(data.get("init_states") or []),
        )


@dataclass
class InfoDocument:
    game: str = ""
    task_entries: List[Entry] = field(default_factory=list)
    image_entries: List[Entry] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)

    # Relative to ``storage_dir``; entry frame paths are relative to it.
    frames_root: str = ""

    @staticmethod
    def _check_section(section: str) -> None:
        if section not in (TASK_SECTION, IMAGE_SECTION):
            raise ValueError(
                f"Unknown info-document section {section!r}. "
                f"Expected {TASK_SECTION!r} or {IMAGE_SECTION!r}."
            )

    def entries(self, section: str) -> List[Entry]:
        self._check_section(section)
        return self.task_entries if section == TASK_SECTION else self.image_entries

    def set_entries(self, section: str, entries: List[Entry]) -> None:
        self._check_section(section)
        if section == TASK_SECTION:
            self.task_entries = entries
        else:
            self.image_entries = entries

    @property
    def n_entries(self) -> int:
        return len(self.task_entries) + len(self.image_entries)

    def copy(self) -> "InfoDocument":
        def copied(entries: List[Entry]) -> List[Entry]:
            return [replace(e, examples=list(e.examples), insights=list(e.insights),
                            example_frames=list(e.example_frames),
                            init_states=list(e.init_states))
                    for e in entries]

        return InfoDocument(
            game=self.game,
            task_entries=copied(self.task_entries),
            image_entries=copied(self.image_entries),
            provenance=replace(self.provenance),
            frames_root=self.frames_root,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game": self.game,
            "provenance": self.provenance.to_dict(),
            "frames_root": self.frames_root,
            "task_entries": [e.to_dict() for e in self.task_entries],
            "image_entries": [e.to_dict() for e in self.image_entries],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InfoDocument":
        return cls(
            game=data.get("game", ""),
            task_entries=[Entry.from_dict(e) for e in data.get("task_entries") or []],
            image_entries=[Entry.from_dict(e) for e in data.get("image_entries") or []],
            provenance=Provenance.from_dict(data.get("provenance")),
            frames_root=data.get("frames_root", ""),
        )


_TASK_HEADING = "Task category:"
_IMAGE_HEADING = "Image category:"


def _render_entry(entry: Entry, heading: str) -> str:
    lines = [f"### {heading} {entry.category}", f"**Description:** {entry.description}"]
    if entry.frame:
        lines.append(f"**Frame:** {entry.frame}")
    if entry.init_states:
        lines.append(f"**Init states:** {', '.join(entry.init_states)}")
    lines.append("**Examples:**")
    lines.extend(f"- {e}" for e in entry.examples or ["(none recorded)"])
    if entry.example_frames:
        lines.append(f"**Example frames:** {', '.join(entry.example_frames)}")
    lines.append("**Insights:**")
    lines.extend(f"- {i}" for i in entry.insights or ["(none recorded)"])
    return "\n".join(lines)


def render_document(doc: InfoDocument) -> str:
    out = [f"# Info: {doc.game}", ""]
    if doc.provenance.source:
        out += [f"*Source: {doc.provenance.label}*", ""]
    out += [f"## 1. {TASK_SECTION}", ""]
    for entry in doc.task_entries:
        out.append(_render_entry(entry, _TASK_HEADING))
        out.append("")
    out.append(f"## 2. {IMAGE_SECTION}")
    out.append("")
    for entry in doc.image_entries:
        out.append(_render_entry(entry, _IMAGE_HEADING))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def resolve_frame(doc: InfoDocument, frame: Optional[str],
                  parameters: Optional[dict] = None) -> Optional[str]:
    if not frame:
        return None
    if os.path.isabs(frame):
        return frame
    storage_dir = load_parameters(parameters)["storage_dir"]
    return os.path.join(storage_dir, doc.frames_root, frame)


def dump_document(doc: InfoDocument, path: str) -> None:
    file_makedir(path)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        json.dump(doc.to_dict(), handle, indent=2)
    os.replace(tmp, path)


def load_document(path: str, parameters: Optional[dict] = None) -> InfoDocument:
    with open(path, "r") as handle:
        doc = InfoDocument.from_dict(json.load(handle))

    parameters = load_parameters(parameters)
    label = doc.provenance.label
    for entry in doc.task_entries + doc.image_entries:
        entry.source = label or None
        entry.resolved_frame = resolve_frame(doc, entry.frame, parameters)
        if entry.resolved_frame and not os.path.exists(entry.resolved_frame):
            log_error(f"Info document {path!r} records a frame for entry {entry.category!r} that is not on disk: {entry.frame!r} under frames_root {doc.frames_root!r} resolves to {entry.resolved_frame!r}. Either storage_dir does not point at the tree this document was built against, or its frames have been deleted. Rebuild it with vlm_scripts/build_info.py.", parameters=parameters)
    return doc
