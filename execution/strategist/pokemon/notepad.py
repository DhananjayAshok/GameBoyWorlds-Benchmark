from __future__ import annotations

import difflib
from typing import Any, Dict

from execution.artifact import StrategistArtifact
from execution.artifact import update as artifact_update


class ThoughtNotepad(StrategistArtifact):
    def __init__(self, text: str = "") -> None:
        self.text = text

    @property
    def length(self) -> int:
        return len(self.text.split())

    @artifact_update
    def clear(self) -> None:
        self.text = ""

    @artifact_update
    def update(self, text: str) -> None:
        self.text = text

    @artifact_update
    def add(self, text: str) -> None:
        self.text = f"{self.text}\n{text}" if self.text else text

    def diff(self, old: "ThoughtNotepad") -> Dict[str, Any]:
        if old.text == self.text:
            kind = "unchanged"
        elif not self.text:
            kind = "cleared"
        elif not old.text or self.text.startswith(old.text + "\n"):
            kind = "appended"
        else:
            kind = "rewritten"
        old_lines, new_lines = old.text.splitlines(), self.text.splitlines()
        added, removed = [], []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old_lines, b=new_lines).get_opcodes():
            if tag in ("delete", "replace"):
                removed.extend(old_lines[i1:i2])
            if tag in ("insert", "replace"):
                added.extend(new_lines[j1:j2])
        return {
            "kind": kind,
            "added_lines": added,
            "removed_lines": removed,
            "length_change": self.length - old.length,
            "unified": list(difflib.unified_diff(old_lines, new_lines, "old", "new", lineterm="")),
        }
