import os
import pickle

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import log_info, parse_action_line
from debug_scripts import markdown as md
from debug_scripts.frames import strip
from python_scripts.paths import Paths

SAFETY_MARGIN = 2  # must match create_dataset.py's --safety_margin default

BENCHMARK_SUPERVISOR = "subgoal"


def _practice_dir(paths, leg: str) -> str:
    # Must mirror how practice_tasks.py derives its output directory.
    guidance_path = paths.info_source_stem(leg) + "_guidance.json"
    return os.path.join(os.path.dirname(guidance_path), f"practice_{paths.executor}")


def _call_log_strip(call_log, out_path: str, n: int, overwrite: bool):
    frames = [image for record in call_log for image in getattr(record, "images", [])]
    if not frames:
        return None
    return strip(frames, out_path, n=n, overwrite=overwrite)


def _figure_safe_point(frame: pd.DataFrame, out_path: str):
    points = frame["safe_success_point"].dropna()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    if len(points):
        left.hist(points, bins=range(0, int(points.max()) + 2), color="#4C78A8")
        left.axvline(points.median(), color="#E45756", linestyle="--",
                     label=f"median {points.median():.0f}")
        left.legend(frameon=False, fontsize=9)
    left.set_xlabel("safe_success_point (vlm_call_log index)")
    left.set_ylabel("episodes")
    left.set_title("Where the judge says the task was done")

    kept = (points + SAFETY_MARGIN).clip(lower=0)
    if len(kept):
        right.hist(kept, bins=range(0, int(kept.max()) + 2), color="#54A24B")
    right.set_xlabel(f"calls kept per episode (point + safety_margin={SAFETY_MARGIN})")
    right.set_title("Training rows contributed per episode")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _figure_accept(decisions: pd.DataFrame, out_path: str):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    by_call = decisions.groupby("call_idx")["accept"].agg(["mean", "count"])
    by_call = by_call[by_call["count"] >= 5]
    left.plot(by_call.index, by_call["mean"] * 100, color="#4C78A8")
    left.set_ylim(0, 101)
    left.set_xlabel("call index within episode")
    left.set_ylabel("accept rate (%)")
    left.set_title("Clean-filter accept rate by call position")
    left.grid(alpha=0.3)

    per_task = decisions.groupby("group_idx")["accept"].mean() * 100
    right.hist(per_task, bins=30, color="#F58518")
    right.set_xlabel("accept rate per group (%)")
    right.set_ylabel("groups")
    right.set_title("Accept rate spread across groups")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _figure_task_success(frame: pd.DataFrame, out_path: str):
    per_task = (
        frame.groupby("task_string")["success"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(per_task)), per_task["mean"] * 100, color="#4C78A8", width=1.0)
    ax.set_xlabel(f"task, ranked ({len(per_task)} unique)")
    ax.set_ylabel("judged success (%)")
    ax.set_title("Judged success by task")
    ax.set_ylim(0, 101)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return per_task


@click.command(name="practice")
@click.option("--model_name", required=True, help="Full VLM name (e.g. google/gemma-4-31b-it)")
@click.option("--leg", default="both", show_default=True,
              type=click.Choice(["curiosity", "zeroshot", "both"]),
              help="Which data-collection leg's practice run to report. 'both' writes one "
                   "report per leg. The merged dataset is not a leg — it holds only the two "
                   "dataset CSVs, none of the practice artifacts this report reads.")
@click.option("--n_audit", default=40, show_default=True,
              help="Episodes rendered for the manual judge audit (half success, half fail).")
@click.option("--n_frames", default=8, show_default=True, help="Frames per audit strip.")
@click.option("--seed", default=0, show_default=True, help="Seed for the audit sample.")
@click.pass_obj
def debug_practice(obj, model_name, leg, n_audit, n_frames, seed):
    """Judge reliability, episode cutoffs, clean-filter behaviour, and an audit pack."""
    paths = Paths(
        parameters=obj["parameters"], game=obj["game"], run_name=obj["run_name"],
        executor=obj["executor"], controller_variant=obj["controller_variant"],
        extra_name=obj["extra_name"], model_name=model_name,
        output_dir=obj["output_dir"], mode=obj["mode"],
    )
    legs = ["curiosity", "zeroshot"] if leg == "both" else [leg]
    for name in legs:
        log_info(_practice_report(paths, obj, name, n_audit, n_frames, seed), obj["parameters"])


def _practice_report(paths, obj, leg, n_audit, n_frames, seed):
    overwrite = obj["overwrite"]
    report_dir = paths.debug_dir("practice")
    images_dir = paths.debug_dir("practice", "images", leg)
    practice_dir = _practice_dir(paths, leg)

    results_path = paths.require(os.path.join(practice_dir, "results.csv"), "practice")
    frame = pd.read_csv(results_path)
    score_mode = "success" in frame.columns and frame["success"].isna().all()
    if not score_mode:
        frame["success"] = frame["success"].astype(bool)
    else:
        frame["success"] = frame["score"] >= 6
    log_info(f"[practice/{leg}] {len(frame)} episodes from {results_path}")

    decisions_path = paths.require(os.path.join(practice_dir, "clean_decisions.csv"), "clean")
    decisions = pd.read_csv(decisions_path)
    decisions["accept"] = decisions["accept"].astype(bool)

    n_episodes = len(frame)
    n_success = int(frame["success"].sum())

    bench_path = paths.benchmark_csv(paths.game, paths.model_save_name, BENCHMARK_SUPERVISOR)
    bench_line = "_(no benchmark CSV for the base model — run scripts/benchmark.sh)_"
    if os.path.exists(bench_path):
        bench = pd.read_csv(bench_path)
        bench_rate = bench["success"].astype(bool).mean()
        ratio = (n_success / n_episodes) / bench_rate if bench_rate else float("inf")
        bench_line = (
            f"Practice judged success **{n_success / n_episodes * 100:.1f}%** "
            f"vs benchmark environment-verified success "
            f"**{bench_rate * 100:.1f}%** "
            f"({int(bench['success'].astype(bool).sum())}/{len(bench)}) "
            f"— an optimism ratio of **{ratio:.1f}x**."
        )

    safe_fig = os.path.join(images_dir, "safe_success_point.png")
    accept_fig = os.path.join(images_dir, "accept_rate.png")
    task_fig = os.path.join(images_dir, "task_success.png")
    _figure_safe_point(frame, safe_fig)
    _figure_accept(decisions, accept_fig)
    per_task = _figure_task_success(frame, task_fig)

    points = frame["safe_success_point"].dropna()
    tiny = int(((points + SAFETY_MARGIN) <= 3).sum())
    always = per_task[(per_task["mean"] == 1.0) & (per_task["count"] >= 3)]
    never = per_task[per_task["mean"] == 0.0]

    rng = np.random.default_rng(seed)
    audit_rows = []
    for label, subset in [("success", frame[frame["success"]]), ("fail", frame[~frame["success"]])]:
        take = min(n_audit // 2, len(subset))
        if take:
            picked = subset.iloc[rng.choice(len(subset), size=take, replace=False)]
            audit_rows.append(picked.assign(judged=label))
    audit = pd.concat(audit_rows) if audit_rows else frame.head(0)

    audit_entries = []
    for _, row in audit.iterrows():
        pkl_path = os.path.join(
            practice_dir, f"{row['group_idx']}_{int(row['attempt'])}.pkl")
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, "rb") as handle:
            call_log = pickle.load(handle)
        out = os.path.join(images_dir, "audit", f"{row['group_idx']}_{int(row['attempt'])}.png")
        if not _call_log_strip(call_log, out, n=n_frames, overwrite=overwrite):
            continue
        actions = [parse_action_line(getattr(r, "response", "")) for r in call_log]
        actions = [a for a in actions if a]
        audit_entries.append({
            "episode": f"{row['group_idx']}_{int(row['attempt'])}",
            "task": row["task_string"],
            "judged": row["judged"],
            "safe_point": row.get("safe_success_point"),
            "n_calls": len(call_log),
            "actions": " ".join(actions[:20]),
            "image": out,
            # TODO(legacy-cols): drop the .get fallbacks once every practice dir has these columns.
            "used_retry": row.get("used_retry"),
            "derived_hint": row.get("derived_hint", ""),
            "judge_reasoning": row.get("judge_reasoning", ""),
        })
    log_info(f"[practice/{leg}] rendered {len(audit_entries)} audit strips")

    audit_blocks = []
    for entry in audit_entries:
        # TODO(legacy-cols): None means the column is absent; collapse to a two-way check once every practice dir has it.
        retry = entry.get("used_retry")
        retry_note = ""
        if retry is True:
            retry_note = " · ⚠ **hint-carried retry**"
        elif retry is False:
            retry_note = " · unaided first draw"
        audit_blocks.append(
            f"**`{entry['episode']}`** — judged **{entry['judged']}**{retry_note} · "
            f"safe_point `{entry['safe_point']}` · {entry['n_calls']} calls · your verdict: ______\n\n"
            f"task: _{entry['task']}_\n\n"
            f"actions: `{entry['actions']}`\n"
        )
        if entry.get("derived_hint"):
            audit_blocks.append(md.details("hint the retry ran under", md.code(entry["derived_hint"])))
        if entry.get("judge_reasoning"):
            audit_blocks.append(md.details("judge's reasoning", md.code(entry["judge_reasoning"])))
        audit_blocks.append(md.img(entry["episode"], entry["image"], report_dir))

    blocks = [
        md.h1(f"Practice — {paths.game} / {paths.model_save_name} / leg={leg}"),
        md.para(f"Leg: **{leg}** · source: `{practice_dir}`"),
        md.h2("Judged success"),
        md.bullets([
            f"episodes: **{n_episodes}** over **{frame['task_string'].nunique()}** unique tasks",
            f"judged successful: **{n_success}** — "
            f"{n_success / max(n_episodes, 1) * 100:.1f}% of {n_episodes}",
            f"scoring mode: **{'1-10 score' if score_mode else 'binary success'}**",
        ]),
        md.para(bench_line),
        md.para(
            "The practice label comes from `SimpleCheckerSupervisor`, which is the same model "
            "grading its own rollout, is shown the correct-solution guidance, and is instructed "
            "*\"Do not be overly strict\"*. The benchmark number is the environment's own "
            "`termination_reason`. A large ratio between them is label noise, and every training "
            "row inherits it."
        ),
        md.h2("Judged success by task"),
        md.img("task success", task_fig, report_dir),
        md.bullets([
            f"tasks judged successful **every** time (≥3 attempts): **{len(always)}**",
            f"tasks **never** judged successful: **{len(never)}**",
        ]),
        md.details("tasks at 100% judged success",
                   md.table(always.reset_index().rename(columns={"mean": "rate", "count": "n"}))),
        md.h2("Episode cutoff"),
        md.img("safe success point", safe_fig, report_dir),
        md.para(
            f"`create_dataset` keeps `safe_success_point + {SAFETY_MARGIN}` calls per episode. "
            f"Episodes contributing **≤3 calls**: **{tiny}** of {len(points)} with a defined point "
            f"({tiny / max(len(points), 1) * 100:.1f}%). Mass at the low end means the dataset is "
            "dominated by opening moves rather than task completions."
        ),
        md.h2("Clean filter"),
        md.img("accept rates", accept_fig, report_dir),
        md.bullets([
            f"call-level decisions: **{len(decisions)}**",
            f"rejected: **{int((~decisions['accept']).sum())}** "
            f"({(~decisions['accept']).mean() * 100:.1f}%)",
        ]),
        md.details(
            "sample of rejected calls",
            md.table(
                decisions[~decisions["accept"]]
                .head(25)[["group_idx", "attempt", "call_idx", "reason"]]
                .reset_index(drop=True)
            ),
        ),
        md.h2(f"Judge audit pack ({len(audit_entries)} episodes)"),
        md.warn(
            "This section is for **manual** scoring. For each episode below, look at the frames "
            "and decide whether the task was actually completed, then compare against the "
            "`judged` label. The resulting confusion matrix is the single most decisive number "
            "in this whole debug suite."
        ),
        md.note(
            # TODO(legacy-cols): drop this caveat once every practice dir has the column.
            "`used_retry` distinguishes a first-draw success from a hint-carried retry. It is "
            "absent from practice runs made before that column was added — those rows show "
            "neither annotation."
        ),
        "\n".join(audit_blocks) if audit_blocks else md.para("_(no audit strips rendered)_"),
    ]

    report_path = md.write_report(os.path.join(report_dir, f"report_{leg}.md"), blocks)
    log_info(f"[practice/{leg}] wrote {report_path}")
    return report_path
