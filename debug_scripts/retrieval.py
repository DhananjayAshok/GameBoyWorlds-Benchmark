import ast
import json
import os
from collections import Counter

import click
import pandas as pd

from utils import log_info
from debug_scripts import markdown as md
from debug_scripts.benchmark import _load
from python_scripts.paths import Paths

RETRIEVAL_COLUMNS = ["selected_entry_ids", "insights_block"]


def _parse_ids(value) -> list:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text in ("[]", "nan"):
        return []
    for loads in (json.loads, ast.literal_eval):
        try:
            parsed = loads(text)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def _mean(frame, column):
    return frame[column].mean() if column in frame and len(frame) else float("nan")


def _split_table(frame, retrieved_mask) -> pd.DataFrame:
    rows = []
    for label, subset in [("retrieved something", frame[retrieved_mask]),
                          ("retrieved nothing (== subgoal arm)", frame[~retrieved_mask])]:
        rows.append({
            "episodes": label,
            "n": len(subset),
            "success": int(subset["success"].sum()) if len(subset) else 0,
            "success_%": subset["success"].mean() * 100 if len(subset) else float("nan"),
            "steps_per_ep": _mean(subset, "n_steps"),
            "invalid_per_ep": _mean(subset, "n_invalid"),
        })
    return pd.DataFrame(rows)


@click.command(name="retrieval")
@click.option("--csv", "csv_path", required=True,
              help="Path to an info_subgoal_* benchmark CSV.")
@click.option("--label", default=None, help="Short name for the run. Defaults to the CSV stem.")
@click.option("--n_examples", default=10, show_default=True,
              help="Episodes rendered in full, with their insights block (0 = all).")
@click.pass_obj
def debug_retrieval(obj, csv_path, label, n_examples):
    """What the retrieval arm selected per episode, and whether it selected anything at all."""
    paths = Paths(
        parameters=obj["parameters"], game=obj["game"], run_name=obj["run_name"],
        executor=obj["executor"], controller_variant=obj["controller_variant"],
        output_dir=obj["output_dir"], mode=obj["mode"],
    )
    report_dir = paths.debug_dir("retrieval")
    label = label or os.path.splitext(os.path.basename(csv_path))[0]

    frame = _load(paths.require(csv_path, "benchmark"))
    missing = [column for column in RETRIEVAL_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{csv_path} has no {', '.join(missing)} column — it was not written by an "
            f"info_subgoal_* arm, so there is no retrieval to audit. Point --csv at a "
            f"info_subgoal_retrieval_* or info_subgoal_parametric_* CSV."
        )

    selections = [_parse_ids(value) for value in frame["selected_entry_ids"]]
    frame = frame.assign(n_selected=[len(ids) for ids in selections])
    retrieved = frame["n_selected"] > 0
    log_info(f"[retrieval] {label}: {int(retrieved.sum())}/{len(frame)} episodes selected "
             f"at least one entry")

    counts = Counter(entry for ids in selections for entry in ids)
    per_entry = []
    for entry, count in counts.most_common():
        hits = [i for i, ids in enumerate(selections) if entry in ids]
        wins = int(frame.iloc[hits]["success"].sum())
        per_entry.append({
            "entry": entry,
            "episodes": count,
            "success": wins,
            "success_%": wins / len(hits) * 100 if hits else float("nan"),
        })
    entry_table = pd.DataFrame(per_entry)

    funnel = [column for column in
              ("n_insights_candidate", "n_insights_kept", "n_insights_distilled",
               "n_supervisor_calls")
              if column in frame.columns]

    ordered = list(frame.index)
    ordered.sort(key=lambda i: (frame.loc[i, "n_selected"] == 0, i))
    chosen = ordered[:n_examples] if n_examples else ordered
    episodes = []
    for i in chosen:
        row = frame.loc[i]
        ids = _parse_ids(row["selected_entry_ids"])
        block = str(row.get("insights_block") or "").strip()
        body = [
            md.h3(f"[{i}] {row['task']}"),
            md.bullets([
                f"success: `{bool(row['success'])}` · steps: {row['n_steps']} · "
                f"invalid: {row['n_invalid']}",
                f"entries selected: **{len(ids)}**"
                + (f" — {', '.join(f'`{e}`' for e in ids)}" if ids else
                   " — _nothing selected; this episode is the subgoal arm_"),
            ]),
        ]
        if block:
            body.append(md.details(
                f"insights block given to the planner ({len(block)} chars)",
                md.code(block)))
        episodes.append("\n".join(body))

    n_empty = int((~retrieved).sum())
    blocks = [
        md.h1(f"Retrieval audit — {paths.game} / {label}"),
        md.para(f"Source: `{csv_path}`"),
        md.bullets([
            f"episodes: **{len(frame)}**",
            f"selected at least one entry: **{int(retrieved.sum())}** "
            f"({retrieved.mean() * 100:.1f}%)",
            f"selected nothing: **{n_empty}** ({n_empty / max(len(frame), 1) * 100:.1f}%)",
            f"distinct entries ever selected: **{len(counts)}**",
            f"mean entries per episode: **{frame['n_selected'].mean():.2f}**",
        ]),
        md.note(
            "An episode that selects nothing is planned with an empty insights block, which is "
            "exactly what the `subgoal` arm does. Those rows carry no evidence about the "
            "document either way, so the arm's headline success rate is a blend of "
            "document-using and document-free episodes."
        ),
        md.warn(
            f"{n_empty} of {len(frame)} episodes ({n_empty / max(len(frame), 1) * 100:.0f}%) "
            f"silently ran as the subgoal arm."
        ) if n_empty else None,
        md.h2("Outcome split"),
        md.table(_split_table(frame, retrieved)),
        md.h2("Filter funnel"),
        md.table(frame[funnel].describe().T.reset_index().rename(
            columns={"index": "column"})) if funnel else md.para("_(no funnel columns)_"),
        md.note(
            "`n_insights_candidate` → `n_insights_kept` is the per-entry relevance filter "
            "(one VLM call per candidate entry, which is where this arm's supervisor-call cost "
            "comes from); `n_insights_distilled` is what survived into the planner's prompt."
        ),
        md.h2("Entries selected"),
        md.para("How often each document entry was chosen, and how those episodes ended. "
                "An entry that is chosen constantly and wins rarely is the one to read first."),
        md.table(entry_table),
        md.h2(f"Episodes ({len(chosen)} of {len(frame)})"),
        md.para("Episodes that retrieved something are listed first."),
        "\n\n".join(episodes),
    ]

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    path = md.write_report(os.path.join(report_dir, f"{stem}.md"), blocks)
    log_info(f"[retrieval] wrote {path}")
