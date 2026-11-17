import json
import os
import pickle

import click
import numpy as np
import pandas as pd

from utils import log_info, log_warn
from debug_scripts import markdown as md
from debug_scripts.frames import trajectory_strip
from python_scripts.paths import Paths


def _canonical_task(task: str) -> str:
    # Must match infer_tasks._canonical_task; not imported, to keep the VLM stack out of this report.
    return " ".join(str(task).lower().split()).strip(" .!?\"'")


def _traj_length(trajectory) -> int | None:
    try:
        observations = trajectory[0]
        if isinstance(observations, str) or not hasattr(observations, "__len__"):
            return None
        return len(observations)
    except (TypeError, IndexError):
        return None


@click.command(name="infer")
@click.option("--model_name", required=True, help="Full VLM name (e.g. google/gemma-4-31b-it)")
@click.option("--n_frames", default=5, show_default=True, help="Frames per trajectory strip.")
@click.option("--max_tasks", default=0, show_default=True,
              help="Cap tasks rendered with frame strips (0 = no cap). The summary table is "
                   "always complete.")
@click.option("--max_traj_per_task", default=3, show_default=True,
              help="Trajectory strips rendered per task.")
@click.pass_obj
def debug_infer(obj, model_name, n_frames, max_tasks, max_traj_per_task):
    """Distilled task strings per curiosity group, with the trajectories each came from."""
    paths = Paths(
        parameters=obj["parameters"], game=obj["game"], run_name=obj["run_name"],
        executor=obj["executor"], controller_variant=obj["controller_variant"],
        model_name=model_name, output_dir=obj["output_dir"],
        mode=obj["mode"],
    )
    overwrite = obj["overwrite"]
    report_dir = paths.debug_dir("infer")
    images_dir = paths.debug_dir("infer", "images")

    json_path = paths.require(paths.curiosity_annotation(), "curiosity")
    with open(json_path) as handle:
        # json keys are strings; the pkl keeps int group indices.
        annotation = {int(gid): task for gid, task in json.load(handle).items()}

    pkl_path = json_path.replace(".json", ".pkl")
    trajectories_by_group = {}
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as handle:
            raw = pickle.load(handle)
        trajectories_by_group = {int(gid): trajs for gid, trajs in raw.items()}
    else:
        log_warn(
            f"[infer] no trajectory_annotation.pkl beside {json_path} — rendering task "
            "strings without frame strips."
        )

    summary_rows = []
    for gid in sorted(annotation):
        used = trajectories_by_group.get(gid, [])
        lengths = [length for length in (_traj_length(t) for t in used) if length is not None]
        summary_rows.append({
            "group": gid,
            "task": annotation[gid],
            "trajectories": len(used),
            "median_len": float(np.median(lengths)) if lengths else 0.0,
        })
    summary = pd.DataFrame(summary_rows)

    canon = {}
    duplicates = {}
    for gid in sorted(annotation):
        key = _canonical_task(annotation[gid])
        if key in canon:
            duplicates.setdefault(canon[key], []).append(gid)
        else:
            canon[key] = gid

    sections = []
    for rank, gid in enumerate(sorted(annotation)):
        if max_tasks and rank >= max_tasks:
            break
        used = trajectories_by_group.get(gid, [])
        body = [md.h3(f"group {gid} — {annotation[gid]}")]
        strips = []
        for traj_i, trajectory in enumerate(used[:max_traj_per_task]):
            out = os.path.join(images_dir, f"group_{gid}_traj_{traj_i}.png")
            if trajectory_strip(trajectory, out, n=n_frames, overwrite=overwrite):
                strips.append(md.img(f"group {gid} traj {traj_i}", out, report_dir))
        body.append(
            "\n".join(strips) if strips
            else md.para(f"_({len(used)} trajectory(ies), none renderable)_")
        )
        sections.append("\n".join(body))

    n_tasks = len(annotation)
    total_traj = int(summary["trajectories"].sum()) if len(summary) else 0
    n_unique = len(canon)

    blocks = [
        md.h1(f"Task inference — {paths.game} / {paths.model_save_name} / {paths.run_name}"),
        md.para(
            f"**{n_tasks}** tasks distilled from the curiosity groups, from "
            f"**{total_traj}** used trajectories."
        ),
        md.para(f"Source: `{json_path}`"),
        md.note(
            "Groups the VLM answered NO TASK for (or whose output could not be parsed) are "
            "absent from this artifact and cannot be counted here — cross-reference the "
            "`curiosity` report's group count to see how many groups produced no task."
        ),
    ]

    if duplicates:
        detail = "; ".join(
            f"group {keep} ≡ {folded}" for keep, folded in duplicates.items()
        )
        blocks.append(md.warn(
            f"{n_tasks - n_unique} duplicate task string(s) survived dedup "
            f"({n_unique} unique of {n_tasks}): {detail}. infer_tasks merges these before "
            "saving, so their presence means --no_dedup_tasks was set or dedup did not run."
        ))
    else:
        blocks.append(md.para(
            f"All **{n_unique}** task strings are distinct (exact-match dedup held)."
        ))

    blocks += [
        md.h2("Distilled tasks"),
        md.table(summary),
        md.para(
            "`trajectories` is how many trajectories from that group were actually used "
            "during inference (up to infer_tasks' `--max_trajectories_per_group`); "
            "`median_len` is their median frame count."
        ),
        md.h2("Task frame strips"),
        md.para(
            f"Up to {max_traj_per_task} trajectories per task, {n_frames} frames each "
            "(frame index top-left, action bottom-left). These are the exact trajectories "
            "the task string was distilled from — if the strips for a task look unrelated to "
            "its wording, the distillation drifted from the evidence."
        ),
        "\n\n".join(sections) if sections else md.para("_(no trajectories to render)_"),
    ]

    report_path = md.write_report(os.path.join(report_dir, "report.md"), blocks)
    log_info(f"[infer] wrote {report_path}")
