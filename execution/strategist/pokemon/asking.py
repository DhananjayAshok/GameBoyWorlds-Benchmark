from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from execution.strategist.pokemon import prompts as P
from utils import log_warn
from utils.lm_inference import clean_value

Parser = Callable[[str], Tuple[Any, Optional[str]]]

MAX_PREVIOUS_CHARS = 1500


def ask(vlm_call: Callable[..., str], prompt: str, parse: Parser, stage: str, attempts: int,
        parameters: Optional[dict] = None, **call_kwargs: Any) -> Optional[Any]:
    """One VLM question; each retry tells the model what was wrong with its last reply."""
    text = prompt
    for attempt in range(attempts):
        output = vlm_call(texts=text, **call_kwargs)
        value, problem = parse(output)
        if problem is None:
            return value
        log_warn(f"{stage} reply unusable (attempt {attempt + 1} of {attempts}): {problem}", parameters)
        text = prompt + (P.RETRY_BLOCK
                         .replace("[PREVIOUS]", output[:MAX_PREVIOUS_CHARS])
                         .replace("[PROBLEM]", problem))
    return None


def field_values(output: str, key: str) -> List[str]:
    marker = f"{key.lower()}:"
    values = []
    for line in output.splitlines():
        line = line.replace("**", "").strip().lstrip("-*• ").strip()
        if line.lower().startswith(marker):
            values.append(line[len(marker):].strip())
    return values


def single_value(output: str, key: str) -> Tuple[Optional[str], Optional[str]]:
    values = field_values(output, key)
    if len(values) != 1:
        return None, f"There must be exactly one `{key}:` line, found {len(values)}."
    return clean_value(values[0]), None
