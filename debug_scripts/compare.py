import os
import re

import click
import pandas as pd

from utils import log_info, log_warn
from debug_scripts import markdown as md
from debug_scripts.benchmark import _load, _paired_index
from python_scripts.paths import Paths

QUADRANTS = [
    ("a_only", "{a} passes, {b} fails"),
    ("b_only", "{b} passes, {a} fails"),
    ("both_pass", "Both pass"),
    ("both_fail", "Both fail"),
]


def _episode_block(label: str, row) -> str:
    lines = [md.para(
        f"**{label}** — success: `{bool(row['success'])}` · steps: {row['n_steps']} · "
        f"invalid: {row['n_invalid']} · "
        f"subgoals: {row['subgoals_reached_n']}/{row['all_subgoals_n']}"
    )]
    sessions = row.get("session_dirs")
    if isinstance(sessions, str) and sessions.strip():
        lines.append(md.para(f"session: `{sessions}`"))
    return "\n".join(lines)


@click.command(name="compare")
@click.option("--csv_a", required=True, help="Path to the first benchmark CSV.")
@click.option("--csv_b", required=True, help="Path to the second benchmark CSV.")
@click.option("--label_a", default=None, help="Short name for A. Defaults to the CSV stem.")
@click.option("--label_b", default=None, help="Short name for B. Defaults to the CSV stem.")
@click.option("--n_examples", default=2, show_default=True,
              help="Episodes rendered per quadrant (0 = all).")
@click.pass_obj
def debug_compare(obj, csv_a, csv_b, label_a, label_b, n_examples):
    """Four-quadrant comparison of two benchmark CSVs."""
    paths = Paths(
        parameters=obj["parameters"], game=obj["game"], run_name=obj["run_name"],
        executor=obj["executor"], controller_variant=obj["controller_variant"],
        output_dir=obj["output_dir"], mode=obj["mode"],
    )
    report_dir = paths.debug_dir("compare", f"{paths.executor}_{paths.controller_variant}")

    label_a = label_a or os.path.splitext(os.path.basename(csv_a))[0]
    label_b = label_b or os.path.splitext(os.path.basename(csv_b))[0]
    if label_a == label_b:
        label_a, label_b = f"{label_a} (A)", f"{label_b} (B)"

    a = _load(paths.require(csv_a, "benchmark"))
    b = _load(paths.require(csv_b, "benchmark"))
    log_info(f"[compare] {label_a}: {len(a)} tasks · {label_b}: {len(b)} tasks")

    games = set(a["game"]) | set(b["game"])
    if len(games) > 1:
        log_warn(f"[compare] CSVs span several games {sorted(games)}; "
                 f"the report is labelled '{paths.game}'.")

    paired = _paired_index(a, label_a, b, label_b, paths.parameters)
    fa, fb = a.loc[paired], b.loc[paired]
    only_a = [i for i in a.index if i >= len(paired)]
    only_b = [i for i in b.index if i >= len(paired)]

    quadrant_tasks = {
        "both_pass": [i for i in paired if fa.loc[i, "success"] and fb.loc[i, "success"]],
        "a_only": [i for i in paired if fa.loc[i, "success"] and not fb.loc[i, "success"]],
        "b_only": [i for i in paired if fb.loc[i, "success"] and not fa.loc[i, "success"]],
        "both_fail": [i for i in paired if not fa.loc[i, "success"] and not fb.loc[i, "success"]],
    }

    sections = []
    for key, title in QUADRANTS:
        rows = quadrant_tasks[key]
        chosen = rows[:n_examples] if n_examples else rows
        blocks = [md.h2(f"{title.format(a=label_a, b=label_b)} — {len(rows)} episode(s)")]
        if not rows:
            blocks.append(md.para("_(none)_"))
            sections.append("\n".join(blocks))
            continue
        blocks.append(md.para(
            "Episodes: " + ", ".join(f"`[{i}] {fa.loc[i, 'task']}`" for i in rows)))
        for i in chosen:
            blocks.append(md.h3(f"[{i}] {fa.loc[i, 'task']}"))
            for label, frame in [(label_a, fa), (label_b, fb)]:
                blocks.append(_episode_block(label, frame.loc[i]))
        if n_examples and len(rows) > len(chosen):
            blocks.append(md.para(
                f"_… {len(rows) - len(chosen)} further episode(s) in this quadrant "
                f"(--n_examples)_"
            ))
        sections.append("\n".join(blocks))

    n = len(paired)
    hits_a, hits_b = int(fa["success"].sum()), int(fb["success"].sum())
    summary = pd.DataFrame([
        {"run": label_a, "success": hits_a, "n": n, "success_%": hits_a / n * 100,
         "subgoal_frac_%": fa["subgoal_frac"].mean() * 100,
         "invalid_per_ep": fa["n_invalid"].mean(), "steps_per_ep": fa["n_steps"].mean()},
        {"run": label_b, "success": hits_b, "n": n, "success_%": hits_b / n * 100,
         "subgoal_frac_%": fb["subgoal_frac"].mean() * 100,
         "invalid_per_ep": fb["n_invalid"].mean(), "steps_per_ep": fb["n_steps"].mean()},
    ])

    blocks = [
        md.h1(f"Benchmark comparison — {paths.game}"),
        md.bullets([
            f"**A** = `{label_a}` — `{csv_a}`",
            f"**B** = `{label_b}` — `{csv_b}`",
        ]),
        md.h2("Pairing"),
        md.bullets([
            f"episodes compared: **{n}** (row for row over the common prefix)",
            f"only in A: **{len(only_a)}** "
            + (f"({', '.join(a.loc[i, 'task'] for i in only_a[:10])})" if only_a else ""),
            f"only in B: **{len(only_b)}** "
            + (f"({', '.join(b.loc[i, 'task'] for i in only_b[:10])})" if only_b else ""),
        ]),
        md.note(
            "Episodes are paired on row position, not task string: the benchmark anchors some "
            "tasks to two init_states, so a task string matches more than one episode. The two "
            "CSVs are checked to agree on every shared row before anything below is computed."
        ),
        md.h2("Summary"),
        md.table(summary),
        md.bullets([
            f"A success: **{hits_a}/{n}** ({hits_a / n * 100:.1f}%)",
            f"B success: **{hits_b}/{n}** ({hits_b / n * 100:.1f}%)",
            f"A only: **{len(quadrant_tasks['a_only'])}** · "
            f"B only: **{len(quadrant_tasks['b_only'])}** · "
            f"both pass: **{len(quadrant_tasks['both_pass'])}** · "
            f"both fail: **{len(quadrant_tasks['both_fail'])}**",
        ]),
        md.note(
            "No frames are rendered here. Each episode's frames live in the archived "
            "`report.pkl.gz` beside its video, under the `session_dirs` path recorded on the "
            "row — the supervisor report holds the images every call actually saw, which is "
            "strictly better than the emulator replay this command used to do."
        ),
    ] + sections

    stem_a = re.sub(r"[^\w.-]", "_", label_a)
    stem_b = re.sub(r"[^\w.-]", "_", label_b)
    path = md.write_report(os.path.join(report_dir, f"{stem_a}__vs__{stem_b}.md"), blocks)
    log_info(f"[compare] wrote {path}")
