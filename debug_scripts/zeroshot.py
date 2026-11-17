import os

import click
import pandas as pd

from utils import log_info, log_warn
from debug_scripts import markdown as md
from debug_scripts.frames import to_pil
from python_scripts.paths import Paths


def _load_tasks(path: str) -> dict[str, list[str]]:
    frame = pd.read_json(path, lines=True)
    if len(frame) == 0:
        return {}
    return {row["init_state"]: list(row["tasks"]) for _, row in frame.iterrows()}


def _load_benchmark_tasks(paths: Paths) -> pd.DataFrame:
    series_csv = paths.benchmark_series_csv()
    frame = pd.read_csv(series_csv)
    return frame[frame["game"] == paths.game].reset_index(drop=True)


def _render_first_frame(paths: Paths, init_state: str, images_dir: str,
                        controller_variant: str, overwrite: bool) -> str | None:
    out_path = os.path.join(images_dir, f"{init_state}.jpg")
    if os.path.exists(out_path) and not overwrite:
        return out_path
    # Imported, not reimplemented, so the frame is exactly what proposal saw.
    from vlm_scripts.propose_tasks_zeroshot import get_first_frame_and_actions
    try:
        first_frame, _ = get_first_frame_and_actions(
            init_state, paths.game, controller_variant=controller_variant
        )
    except Exception as exc:
        log_warn(f"[zeroshot] could not load init_state '{init_state}': {exc}")
        return None
    to_pil(first_frame).save(out_path, "JPEG", quality=92)
    return out_path


@click.command(name="zeroshot")
@click.option("--model_name", required=True, help="Full VLM name (e.g. google/gemma-4-31b-it)")
@click.pass_obj
def debug_zeroshot(obj, model_name):
    """Proposed tasks per init_state, beside that state's real benchmark tasks."""
    controller_variant = obj["controller_variant"]
    paths = Paths(
        parameters=obj["parameters"], game=obj["game"], run_name=obj["run_name"],
        executor=obj["executor"], controller_variant=controller_variant,
        model_name=model_name, output_dir=obj["output_dir"], mode=obj["mode"],
    )
    overwrite = obj["overwrite"]
    report_dir = paths.debug_dir("zeroshot")
    images_dir = paths.debug_dir("zeroshot", "init_state_images")

    tasks_path = paths.require(paths.tasks_file(), "tasks")

    benchmark = _load_benchmark_tasks(paths)
    benchmark_by_state: dict[str, pd.DataFrame] = {
        state: rows for state, rows in benchmark.groupby("init_state")
    }
    log_info(f"[zeroshot] {len(benchmark)} benchmark tasks over "
             f"{len(benchmark_by_state)} init_states")

    proposed = _load_tasks(tasks_path)
    if not proposed:
        log_warn(f"[zeroshot] {tasks_path} is empty — nothing to report.")
        return

    states = sorted(set(proposed) | set(benchmark_by_state))

    summary_rows, sections = [], []
    for state in states:
        state_tasks = proposed.get(state, [])
        bench_rows = benchmark_by_state.get(state)
        n_bench = 0 if bench_rows is None else len(bench_rows)

        summary_rows.append({
            "init_state": state,
            "proposed": len(state_tasks),
            "benchmark_tasks": n_bench,
            "in_proposal": state in proposed,
            "in_benchmark": state in benchmark_by_state,
        })

        image_path = _render_first_frame(
            paths, state, images_dir, controller_variant, overwrite
        )
        body = [md.h2(state)]
        if image_path:
            body.append(md.img(state, image_path, report_dir))
        else:
            body.append(md.para("_(first frame unavailable — environment failed to load)_"))

        body.append(md.h3(f"Proposed tasks ({len(state_tasks)})"))
        body.append(md.bullets(state_tasks))

        body.append(md.h3(f"True tasks — benchmark ({n_bench})"))
        if n_bench:
            body.append(md.bullets(
                f"**{row['task']}**  _({row['task_category']})_"
                for _, row in bench_rows.iterrows()
            ))
        else:
            body.append(md.para(
                "_No benchmark task is anchored to this init_state._ Proposals here are "
                "trained on but never evaluated."
            ))
        sections.append("\n".join(body))

    summary = pd.DataFrame(summary_rows)
    proposed_states = set(proposed)
    bench_states = set(benchmark_by_state)
    covered = proposed_states & bench_states
    total_proposed = int(summary["proposed"].sum())

    blocks = [
        md.h1(f"Zero-shot proposals — {paths.game} / {paths.model_save_name}"),
        md.para(
            f"**{total_proposed}** tasks proposed across **{len(proposed_states)}** init_states. "
            f"The benchmark defines **{len(benchmark)}** tasks across "
            f"**{len(bench_states)}** init_states."
        ),
        md.para(f"Source: `{tasks_path}`"),
        md.para(f"Benchmark: `{paths.benchmark_series_csv()}`"),
        md.h2("Coverage"),
        md.bullets([
            f"init_states with proposals **and** benchmark tasks: **{len(covered)}**",
            f"init_states proposed for but never benchmarked: "
            f"**{len(proposed_states - bench_states)}** "
            f"({', '.join(sorted(proposed_states - bench_states)) or 'none'})",
            f"benchmark init_states with no proposals: "
            f"**{len(bench_states - proposed_states)}** "
            f"({', '.join(sorted(bench_states - proposed_states)) or 'none'})",
            f"mean tasks proposed per init_state: "
            f"**{total_proposed / max(len(proposed_states), 1):.1f}**",
        ]),
        md.para(
            "Proposals on init_states with no benchmark task can only ever contribute training "
            "data that the evaluation never probes; benchmark init_states with no proposals are "
            "evaluated states the pipeline never practised."
        ),
        md.h2("Per-init_state summary"),
        md.table(summary),
        md.h2("Init states"),
        md.para(
            "Each section shows the exact first frame the proposer saw, the tasks it proposed "
            "from it, and the benchmark tasks that actually anchor to that state."
        ),
        "\n\n".join(sections),
    ]

    report_path = md.write_report(os.path.join(report_dir, "report.md"), blocks)
    log_info(f"[zeroshot] wrote {report_path}")
