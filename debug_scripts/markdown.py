import os

import pandas as pd


def h1(text: str) -> str:
    return f"# {text}\n"


def h2(text: str) -> str:
    return f"## {text}\n"


def h3(text: str) -> str:
    return f"### {text}\n"


def para(text: str) -> str:
    return f"{text}\n"


def bullets(items) -> str:
    items = list(items)
    if not items:
        return "_(none)_\n"
    return "\n".join(f"- {item}" for item in items) + "\n"


def numbered(items) -> str:
    items = list(items)
    if not items:
        return "_(none)_\n"
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1)) + "\n"


def code(text: str, lang: str = "") -> str:
    body = "" if text is None else str(text)
    fence = "```"
    while fence in body:
        fence += "`"
    return f"{fence}{lang}\n{body}\n{fence}\n"


def rel(path: str, report_dir: str) -> str:
    return os.path.relpath(path, report_dir).replace(os.sep, "/")


def img(alt: str, path: str, report_dir: str) -> str:
    return f"![{alt}]({rel(path, report_dir)})\n"


def link(text: str, path: str, report_dir: str) -> str:
    return f"[{text}]({rel(path, report_dir)})"


def table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    if df is None or len(df) == 0:
        return "_(no rows)_\n"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(
                lambda v: "" if pd.isna(v) else floatfmt.format(v)
            )
        else:
            formatted[col] = formatted[col].map(lambda v: "" if v is None else str(v))
    header = "| " + " | ".join(str(c) for c in formatted.columns) + " |"
    sep = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    rows = [
        "| " + " | ".join(v.replace("|", "\\|").replace("\n", "<br>") for v in row) + " |"
        for row in formatted.astype(str).values
    ]
    return "\n".join([header, sep, *rows]) + "\n"


def details(summary: str, body: str) -> str:
    return f"<details>\n<summary>{summary}</summary>\n\n{body}\n</details>\n"


def note(text: str) -> str:
    return f"> **Note:** {text}\n"


def warn(text: str) -> str:
    return f"> ⚠ **{text}**\n"


def write_report(path: str, blocks) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = "\n".join(block for block in blocks if block is not None)
    with open(path, "w") as handle:
        handle.write(body)
    return path
