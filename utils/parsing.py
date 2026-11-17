from __future__ import annotations

import re
from typing import Optional

from utils.fundamental import depathify  # noqa: F401
from utils.lm_inference import parse_key_value


# A list item: "- foo", "* foo", "• foo", "1. foo", "1) foo", "1 foo".
_ITEM_RE = re.compile(r"^(?:[-*•]|\d+[.)]?)\s+(.*)$")

_ABSENT = {"NONE", "N/A", "NA"}


def _is_absent(value: str) -> bool:
    stripped = value.strip()
    return (not stripped
            or stripped.upper() in _ABSENT
            or not any(ch.isalnum() for ch in stripped))


def _heading_index(lines: list[str], marker: str) -> Optional[int]:
    mention = None
    for i, line in enumerate(lines):
        stripped = line.strip().lower().lstrip("*#->•+ \t")
        if stripped.startswith(marker):
            return i
        if mention is None and marker in line.lower():
            mention = i
    return mention


def parse_list(text: str, key: Optional[str] = None) -> list[str]:
    body = text or ""

    lowered = body.lower()
    if lowered.count("response:") == 1:
        body = body[lowered.index("response:") + len("response:"):]

    lines = body.splitlines()
    items: list[str] = []
    start = 0

    if key is not None:
        index = _heading_index(lines, f"{key.lower()}:")
        if index is None:
            return []
        marker = f"{key.lower()}:"
        line = lines[index]
        head = line[line.lower().index(marker) + len(marker):].strip()
        match = _ITEM_RE.match(head)
        head = match.group(1).strip() if match else head
        if not _is_absent(head):
            items.append(head)
        start = index + 1

    started = bool(items)
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        match = _ITEM_RE.match(stripped)
        if match is None:
            if started:
                break
            continue
        started = True
        value = match.group(1).strip()
        if not _is_absent(value):
            items.append(value)
    return items


def parse_int(text: str, key: str, lo: Optional[int] = None,
              hi: Optional[int] = None) -> Optional[int]:
    raw = (parse_key_value(text, key) or "").strip()
    if not raw or raw.lower().startswith(("n/a", "na", "none", "unknown")):
        return None

    def in_range(value: int) -> bool:
        return (lo is None or value >= lo) and (hi is None or value <= hi)

    for token in raw.replace(",", " ").split():
        if token.isdigit() and in_range(int(token)):
            return int(token)

    if lo is None and hi is None:
        digits = "".join(ch for ch in raw if ch.isdigit())
        return int(digits) if digits else None

    for ch in raw:
        if ch.isdigit() and in_range(int(ch)):
            return int(ch)
    return None


def parse_action_line(text) -> Optional[str]:
    if not isinstance(text, str):
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("action:"):
            value = stripped[len("action:"):].strip()
            return value or None
    return None


PLAN_SEPARATOR = "[STEP]"


def parse_steps(text: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in text.split(PLAN_SEPARATOR) if part.strip()]


def parse_plan(text: str) -> list[str]:
    body = text or ""

    lowered = body.lower()
    if lowered.count("response:") == 1:
        body = body[lowered.index("response:") + len("response:"):]

    lines = body.splitlines()
    marker = "plan:"
    index = _heading_index(lines, marker)
    if index is None:
        return []

    head = lines[index]
    rest = head[head.lower().index(marker) + len(marker):]
    block = "\n".join([rest, *lines[index + 1:]]).strip().lstrip("*# \t").strip()

    if not block or block.upper().startswith("NONE"):
        return []
    if PLAN_SEPARATOR in block:
        return parse_steps(block)
    steps = parse_list(text, "Plan")
    if steps:
        return steps
    return [] if _is_absent(block) else [" ".join(block.split())]
