import json
import os
import pickle

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import log_info, log_warn
from debug_scripts import markdown as md
from debug_scripts.frames import trajectory_strip
from python_scripts.paths import Paths


def _figure_tries(frame: pd.DataFrame, out_path: str):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    tries = sorted(frame["n_tries"].dropna().unique())
    successes = [int(((frame["n_tries"] == t) & (frame["success"])).sum()) for t in tries]
    failures = [int(((frame["n_tries"] == t) & (~frame["success"])).sum()) for t in tries]
    x = np.arange(len(tries))
    left.bar(x, successes, 0.6, label="succeeded", color="#54A24B")
    left.bar(x, failures, 0.6, bottom=successes, label="failed", color="#E45756")
    left.set_xticks(x)
    left.set_xticklabels([int(t) for t in tries])
    left.set_xlabel("attempts used (n_tries)")
    left.set_ylabel("tasks")
    left.set_title("Outcome by number of attempts")
    left.legend(frameon=False, fontsize=9)

    solved = frame[frame["success"]]
    counts = [int((solved["n_tries"] == t).sum()) for t in tries]
    cumulative = np.cumsum(counts) / max(len(solved), 1) * 100
    right.plot(x, cumulative, marker="o", color="#4C78A8")
    right.set_xticks(x)
    right.set_xticklabels([int(t) for t in tries])
    right.set_ylim(0, 100)
    right.set_xlabel("attempts used")
    right.set_ylabel("cumulative % of all successes")
    right.set_title("How much success is unaided?")
    right.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _figure_by_init_state(frame: pd.DataFrame, out_path: str):
    grouped = (
        frame.groupby("init_state")["success"]
        .agg(["sum", "count"])
        .sort_values("count", ascending=False)
    )
    grouped["rate"] = grouped["sum"] / grouped["count"] * 100
    fig, ax = plt.subplots(figsize=(11, max(3.5, 0.26 * len(grouped))))
    ax.barh(range(len(grouped)), grouped["rate"], color="#4C78A8")
    ax.set_yticks(range(len(grouped)))
    ax.set_yticklabels(
        [f"{state} (n={int(n)})" for state, n in zip(grouped.index, grouped["count"])],
        fontsize=7,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("attempt success rate (%)")
    ax.set_title("Attempt success by init_state")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _completion_check_blocks(frame: pd.DataFrame) -> list:
    if "termination_reason" not in frame.columns:
        return []
    known = frame[frame["termination_reason"].notna()]
    if known.empty:
        return []

    breakdown = (
        known.groupby("termination_reason")
        .agg(episodes=("success", "count"), judged_successful=("success", "sum"))
        .reset_index()
        .sort_values("episodes", ascending=False)
    )
    breakdown["judged_successful_%"] = (
        breakdown["judged_successful"] / breakdown["episodes"] * 100
    )

    done = known[known["termination_reason"] == "agent_done"]
    n_done = len(done)
    false_done = int((~done["success"]).sum())
    missed = known[(known["termination_reason"] != "agent_done") & (known["success"])]

    lines = [
        f"episodes ending `agent_done` (the check fired): **{n_done}** of {len(known)} "
        f"({n_done / len(known) * 100:.1f}%)",
        f"of those, the checker disagreed (`success == False`): **{false_done}** "
        f"({false_done / max(n_done, 1) * 100:.1f}%) — these are trajectories cut short",
        f"episodes the checker called successful where the check never fired: "
        f"**{len(missed)}** — the task was done and the executor kept playing",
    ]
    if n_done and {"n_env_steps", "max_steps"} <= set(frame.columns):
        saved = (done["max_steps"] - done["n_env_steps"]).dropna()
        if len(saved):
            lines.append(
                f"steps saved when it fired: **{saved.mean():.1f}** on average "
                f"(median {saved.median():.0f}) of a {done['max_steps'].max():.0f}-step budget"
            )

    return [
        md.h2("Completion check (self-termination)"),
        md.bullets(lines),
        md.para(
            "The check is one VLM call per environment step, so it roughly doubles this "
            "stage's call count. A **high false-`agent_done` rate is the failure to act on** "
            "— it truncates a trajectory before the task is done and then feeds it forward "
            "as a success. Note the two verdicts are not independent evidence of the same "
            "quality: the checker sees the whole trajectory described, the check sees two "
            "frames, so where they disagree the checker is usually the one to believe."
        ),
        md.table(breakdown.reset_index(drop=True)),
    ]


@click.command(name="attempt")
@click.option("--model_name", required=True, help="Full VLM name (e.g. google/gemma-4-31b-it)")
@click.option("--n_frames", default=8, show_default=True, help="Frames per trajectory strip.")
@click.option("--max_trajectories", default=40, show_default=True,
              help="Cap on rendered success-trajectory strips (0 = no cap).")
@click.pass_obj
def debug_attempt(obj, model_name, n_frames, max_trajectories):
    """Attempt funnel, hint-escalation analysis, and successful-trajectory strips."""
    paths = Paths(
        parameters=obj["parameters"], game=obj["game"], run_name=obj["run_name"],
        executor=obj["executor"], controller_variant=obj["controller_variant"],
        model_name=model_name, output_dir=obj["output_dir"], mode=obj["mode"],
    )
    overwrite = obj["overwrite"]
    report_dir = paths.debug_dir("attempt")
    images_dir = paths.debug_dir("attempt", "images")

    csv_path = paths.require(paths.all_trajectories_csv(), "attempts")
    frame = pd.read_csv(csv_path)
    frame["success"] = frame["success"].astype(bool)
    log_info(f"[attempt] {len(frame)} attempted tasks from {csv_path}")

    n_total = len(frame)
    n_success = int(frame["success"].sum())
    solved = frame[frame["success"]]
    unaided = int((solved["n_tries"] == 1).sum())
    hinted = n_success - unaided

    tasks_path = paths.tasks_file()
    n_proposed = 0
    if os.path.exists(tasks_path):
        proposals = pd.read_json(tasks_path, lines=True)
        n_proposed = int(sum(len(t) for t in proposals["tasks"]))

    tries_fig = os.path.join(images_dir, "n_tries.png")
    state_fig = os.path.join(images_dir, "by_init_state.png")
    _figure_tries(frame, tries_fig)
    _figure_by_init_state(frame, state_fig)

    by_state = (
        frame.groupby("init_state")
        .agg(attempted=("success", "count"),
             succeeded=("success", "sum"),
             mean_tries=("n_tries", "mean"))
        .reset_index()
    )
    by_state["success_rate_%"] = by_state["succeeded"] / by_state["attempted"] * 100
    by_state = by_state.sort_values("success_rate_%", ascending=False)

    never = frame[~frame["success"]][["init_state", "task_string", "n_tries"]]

    success_json = paths.success_trajectories_json()
    consistency = []
    success_keys = set()
    if os.path.exists(success_json):
        with open(success_json) as handle:
            success_keys = set(json.load(handle))
    csv_success_keys = set(solved["group_idx"].astype(str))
    consistency.append(f"`all_trajectories.csv` successes: **{len(csv_success_keys)}**")
    consistency.append(f"`success_trajectories.json` entries: **{len(success_keys)}**")
    only_csv = sorted(csv_success_keys - success_keys)
    only_json = sorted(success_keys - csv_success_keys)
    if only_csv:
        consistency.append(f"in the CSV but not in successes (**{len(only_csv)}**): "
                           f"`{', '.join(only_csv[:20])}`")
    if only_json:
        consistency.append(f"in successes but not in the CSV (**{len(only_json)}**): "
                           f"`{', '.join(only_json[:20])}`")

    strips = []
    pkl_path = paths.success_trajectories_pkl()
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as handle:
            trajectories = pickle.load(handle)
        task_of = {}
        if os.path.exists(success_json):
            with open(success_json) as handle:
                task_of = json.load(handle)
        keys = sorted(trajectories)
        if max_trajectories:
            keys = keys[:max_trajectories]
        for group_idx in keys:
            out = os.path.join(images_dir, "strips", f"{group_idx}.png")
            if trajectory_strip(trajectories[group_idx], out, n=n_frames, overwrite=overwrite):
                strips.append((group_idx, task_of.get(group_idx, ""), out))
        del trajectories
        log_info(f"[attempt] rendered {len(strips)} success-trajectory strips")
    else:
        log_warn(f"[attempt] {pkl_path} missing — no trajectory strips rendered.")

    strip_blocks = []
    for group_idx, task, out in strips:
        strip_blocks.append(f"**`{group_idx}`** — {task}\n")
        strip_blocks.append(md.img(str(group_idx), out, report_dir))

    blocks = [
        md.h1(f"Task attempts — {paths.game} / {paths.model_save_name}"),
        md.para(f"Source: `{paths.attempts_dir()}`"),
        md.h2("Funnel"),
        md.bullets([
            f"tasks proposed: **{n_proposed}**" if n_proposed else "tasks proposed: _(jsonl absent)_",
            f"tasks attempted: **{n_total}**",
            f"tasks succeeded: **{n_success}/{n_total}** "
            f"({n_success / max(n_total, 1) * 100:.1f}%)",
            f"succeeded on the **first, unaided** attempt: **{unaided}** "
            f"({unaided / max(n_success, 1) * 100:.1f}% of successes, "
            f"{unaided / max(n_total, 1) * 100:.1f}% of all tasks)",
            f"succeeded **only after hint escalation** (n_tries > 1): **{hinted}** "
            f"({hinted / max(n_success, 1) * 100:.1f}% of successes)",
        ]),
        md.para(
            "`attempt_tasks.py` derives a fresh hint from the failed trajectory between attempts, "
            "so every success with `n_tries > 1` was carried by information the agent was given "
            "rather than information it worked out. The saved success trajectories "
            "do not distinguish the two."
        ),
        *_completion_check_blocks(frame),
        md.h2("Attempts used"),
        md.img("n_tries breakdown", tries_fig, report_dir),
        md.h2("By init_state"),
        md.img("success by init_state", state_fig, report_dir),
        md.table(by_state),
        md.h2("Artifact consistency"),
        md.bullets(consistency),
        md.h2(f"Never solved in 5 attempts ({len(never)})"),
        md.para(
            "These are the proposed tasks the proposing model could not complete even with "
            "escalating hints — the sharpest available evidence about task quality."
        ),
        md.details(f"{len(never)} unsolved tasks", md.table(never.reset_index(drop=True))),
        md.h2(f"Successful trajectories ({len(strips)} shown)"),
        md.para(f"{n_frames} frames each, sampled evenly; frame index top-left, action bottom-left."),
        "\n".join(strip_blocks) if strip_blocks else md.para("_(no trajectories rendered)_"),
    ]

    report_path = md.write_report(os.path.join(report_dir, "report.md"), blocks)
    log_info(f"[attempt] wrote {report_path}")
