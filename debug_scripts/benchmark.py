import ast
import gzip
import json
import os
import pickle

import click
import pandas as pd

from execution.report import (ACTION_TAGS, EnvironmentStepRecord,
                              SupervisorVLMCallRecord, _step_summary,
                              summarize_world_model)
from utils import log_error, log_info, log_warn, parse_action_line
from debug_scripts import markdown as md
from debug_scripts.frames import to_pil
from python_scripts.paths import BENCHMARK_SUPERVISORS, REPORT_FILENAME, Paths


def _as_list(value):
    # subgoals_reached and all_subgoals are stringified python lists in the CSV.
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def _load(path: str) -> pd.DataFrame:
    # Row position is the episode's identity, so the index must stay positional.
    frame = pd.read_csv(path).reset_index(drop=True)
    frame["success"] = frame["success"].astype(bool)
    frame["subgoals_reached_n"] = frame["subgoals_reached"].map(lambda v: len(_as_list(v)))
    frame["all_subgoals_n"] = frame["all_subgoals"].map(lambda v: len(_as_list(v)))
    frame["subgoal_frac"] = frame.apply(
        lambda r: r["subgoals_reached_n"] / r["all_subgoals_n"] if r["all_subgoals_n"] else 0.0,
        axis=1,
    )
    return frame


def _paired_index(a: pd.DataFrame, label_a: str, b: pd.DataFrame, label_b: str,
                  parameters: dict) -> list[int]:
    # Paired on row position, not task string: the same task can anchor to more than one init_state.
    n = min(len(a), len(b))
    if n == 0:
        log_error(
            f"{label_a} has {len(a)} row(s) and {label_b} has {len(b)} — nothing to compare.",
            parameters,
        )
    mismatched = [i for i in range(n) if a.loc[i, "task"] != b.loc[i, "task"]]
    if mismatched:
        first = mismatched[0]
        log_error(
            f"{label_a} and {label_b} disagree at row {first}: "
            f"{a.loc[first, 'task']!r} vs {b.loc[first, 'task']!r} "
            f"({len(mismatched)} of {n} shared rows differ). A paired comparison needs two "
            "runs over the same prefix of the same benchmark table.",
            parameters,
        )
    return list(range(n))


def _session_dir(row):
    raw = row.get("session_dirs")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        dirs = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return dirs[0] if dirs else None


def _load_report(row):
    session = _session_dir(row)
    if session is None:
        return None
    path = os.path.join(session, REPORT_FILENAME)
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)["report"]


def _require_archives(frame: pd.DataFrame, csv_path: str, parameters: dict) -> dict:
    if "session_dirs" not in frame.columns:
        log_error(
            f"{csv_path} has no session_dirs column, so its episodes cannot be located. It "
            "was written before the report archive existed — re-run the benchmark to get one.",
            parameters,
        )
    reports = {}
    for index, row in frame.iterrows():
        report = _load_report(row)
        if report is not None:
            reports[index] = report
    if not reports:
        log_error(
            f"No report.pkl.gz found for any episode in {csv_path}. The session directories "
            "it names hold no archive, so this CSV predates report archiving — re-run the "
            "benchmark.",
            parameters,
        )
    if len(reports) < len(frame):
        log_warn(f"[benchmark] {len(frame) - len(reports)} of {len(frame)} episodes have no "
                 f"archive; they are rendered from their CSV row alone.")
    return reports


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(text).lower()).strip("_")[:60]


def _storage_link(report_dir: str, name: str, target: str, parameters: dict) -> str:
    # Markdown renderers refuse images outside the workspace root, so storage is symlinked into the report dir.
    link = os.path.join(report_dir, name)
    if os.path.islink(link):
        if os.path.realpath(link) == os.path.realpath(target):
            return link
        os.unlink(link)
    elif os.path.exists(link):
        log_error(
            f"{link} exists and is not a symlink, so the report cannot point it at "
            f"{target}. Move it aside.",
            parameters,
        )
    os.symlink(target, link)
    return link


def _via_link(path: str, target_root: str, link_root: str) -> str:
    return os.path.join(link_root, os.path.relpath(path, target_root))


def _video_path(row) -> str | None:
    session = _session_dir(row)
    if session is None:
        return None
    path = os.path.join(session, "videos", "0.mp4")
    return path if os.path.exists(path) else None


def _video_markdown(row, video_roots, report_dir: str) -> str | None:
    video = _video_path(row)
    if video is None:
        return None
    if video_roots is not None:
        target_root, link_root = video_roots
        if os.path.realpath(video).startswith(os.path.realpath(target_root) + os.sep):
            return md.link(os.path.basename(video),
                           _via_link(video, target_root, link_root), report_dir)
    return f"`{video}`"


def _save_frame(frame, directory: str, name: str, overwrite: bool):
    path = os.path.join(directory, f"{name}.png")
    if os.path.exists(path) and not overwrite:
        return path
    try:
        to_pil(frame).save(path)
    except Exception as error:  # noqa: BLE001 - a bad frame must not kill the report
        log_warn(f"[benchmark] could not write {path}: {error}")
        return None
    return path


def _executor_call_blocks(call, index: int, frames_dir: str, report_dir: str,
                          prefix: str, overwrite: bool) -> list:
    lines = [md.para(f"**call {index}** · tag `{call.tag}`")]
    # getattr: a ScriptedActionRecord shares this log and has no world_model field.
    decision = getattr(call, "world_model", None)

    for i, image in enumerate(call.images):
        path = _save_frame(image, frames_dir, f"{prefix}_call{index}_saw{i}", overwrite)
        if path:
            caption = decision.image_caption(i) if decision else f"input {i}"
            if decision:
                lines.append(md.para(f"input {i}: {caption}"))
            lines.append(md.img(f"call {index} {caption}", path, report_dir))
    if not call.images:
        lines.append(md.para("_(no images on this call)_"))

    lines.append(md.details(f"call {index} prompt", md.code(call.prompt)))
    lines.append(md.para("output"))
    lines.append(md.code(call.response))

    if decision:
        lines.append(md.para(decision.describe()))
    elif call.tag in ACTION_TAGS:
        action = parse_action_line(call.response)
        lines.append(md.para(f"parsed action: `{action if action else 'none'}`"))

    # Every step, not just the last: one call can own several.
    if call.steps:
        lines.append(md.code("\n".join(
            f"{i + 1:>3}  {_step_summary(step)}" for i, step in enumerate(call.steps)
        )))
    else:
        lines.append(md.para("_(this call took no steps)_"))

    env_steps = [s for s in call.steps if isinstance(s, EnvironmentStepRecord)]
    if env_steps:
        path = _save_frame(env_steps[-1].frame_after, frames_dir,
                           f"{prefix}_call{index}_after", overwrite)
        if path:
            lines.append(md.img(f"call {index} next frame", path, report_dir))
    return lines


def _event_blocks(report, frames_dir: str, report_dir: str, overwrite: bool) -> list:
    lines = []
    leg = 0
    for position, event in enumerate(report.event_log):
        if isinstance(event, SupervisorVLMCallRecord):
            lines.append(md.h3(f"supervisor · {event.stage}"))
            for i, image in enumerate(event.images):
                path = _save_frame(image, frames_dir, f"sup{position}_saw{i}", overwrite)
                if path:
                    lines.append(md.img(f"supervisor {event.stage} input {i}", path, report_dir))
            lines.append(md.details(f"{event.stage} prompt", md.code(event.prompt)))
            lines.append(md.para("output"))
            lines.append(md.code(str(event.response)))
        else:
            leg += 1
            hint = (event.init_kwargs or {}).get("hint")
            lines.append(md.h3(f"executor leg {leg} · task: {event.task!r}"))
            lines.append(md.bullets([
                f"termination: `{event.termination_reason}`",
                f"step budget: **{event.max_steps}**",
                f"hint: {f'`{hint}`' if hint else '_none_'}",
                f"VLM calls: **{len(event.vlm_call_log)}**",
            ]))
            for index, call in enumerate(event.vlm_call_log, 1):
                lines += _executor_call_blocks(
                    call, index, frames_dir, report_dir, f"leg{leg}", overwrite)
    return lines


def _episode_section(index: int, row, report, model, report_dir,
                     frames_root: str, video_roots, overwrite: bool) -> str:
    subgoals = _as_list(row["subgoals_reached"])
    all_subgoals = _as_list(row["all_subgoals"])
    task = row["task"]

    blocks = [
        md.h2(f"[{index}] {task}"),
        md.bullets([
            f"success: **{row['success']}**",
            f"steps: **{row['n_steps']}** · invalid: **{row['n_invalid']}**",
            f"subgoals: **{len(subgoals)}/{len(all_subgoals)}** "
            f"({', '.join(subgoals) if subgoals else 'none reached'})",
        ]),
    ]

    video = _video_markdown(row, video_roots, report_dir)
    blocks.append(md.para(f"video: {video}" if video else "_(no video recorded)_"))

    if report is None:
        blocks.append(md.warn("No archived report for this episode — nothing to walk."))
        return "\n".join(blocks)

    blocks.append(md.bullets([
        f"supervisor: `{report.supervisor_name}`",
        f"events: **{len(report.event_log)}** "
        f"({len(report.supervisor_calls)} supervisor call(s), "
        f"{len(report.executor_reports)} executor leg(s))",
    ]))
    world_model_summary = summarize_world_model(report.world_model_decisions)
    if world_model_summary:
        blocks.append(md.code(world_model_summary))
    # Prefixed with the row number: episodes sharing a task string must not share a frames directory.
    frames_dir = os.path.join(frames_root, _slug(model), f"{index:03d}_{_slug(task)}")
    os.makedirs(frames_dir, exist_ok=True)
    blocks += _event_blocks(report, frames_dir, report_dir, overwrite)
    return "\n".join(blocks)


@click.command(name="benchmark")
@click.option("--model_name", required=True, help="Full VLM name (e.g. google/gemma-4-31b-it)")
@click.option("--compare_model", default="none", show_default=True,
              help="Fine-tuned served-model name. 'none' derives "
                   "<model_save_name>-<game>-<run_name> as serve_and_benchmark.sh does.")
@click.option("--bench_game", default="none", show_default=True,
              help="Game whose benchmark CSVs to read. 'none' uses --game.")
@click.option("--max_episodes", default=0, show_default=True,
              help="Cap episodes rendered in episodes_*.md (0 = all).")
@click.option("--supervisor", default="dummy", show_default=True,
              type=click.Choice(BENCHMARK_SUPERVISORS),
              help="Which supervisor's CSV to read. Part of the filename, so this is not "
                   "optional in practice — every supervisor writes its own file for the same "
                   "game, executor and model.")
@click.option("--n_tasks", default=None, type=int,
              help="Set if the run used --n_tasks, which gives it its own _firstN file.")
@click.pass_obj
def debug_benchmark(obj, model_name, compare_model, bench_game, max_episodes, supervisor,
                    n_tasks):
    """Per-episode frame-by-frame trajectories plus a paired base-vs-fine-tuned comparison."""
    paths = Paths(
        parameters=obj["parameters"], game=obj["game"], run_name=obj["run_name"],
        executor=obj["executor"], controller_variant=obj["controller_variant"],
        extra_name=obj["extra_name"],
        model_name=model_name, output_dir=obj["output_dir"],
        mode=obj["mode"],
    )
    arm = supervisor if obj["extra_name"] is None else f"{supervisor}_{obj['extra_name']}"
    report_dir = paths.debug_dir("benchmark", arm)
    overwrite = obj["overwrite"]
    game = paths.game if bench_game == "none" else bench_game

    frames_root = _storage_link(
        report_dir, "frames", paths.debug_frames_dir("benchmark"), paths.parameters)
    gbw = paths.gameboy_worlds_storage()
    video_roots = None
    if gbw is not None:
        sessions_root = os.path.join(gbw, "sessions", game)
        if os.path.isdir(sessions_root):
            video_roots = (sessions_root,
                           _storage_link(report_dir, "videos", sessions_root,
                                         paths.parameters))
    base_model = paths.model_save_name
    ft_model = paths.finetuned_model_name if compare_model == "none" else compare_model

    base_path = paths.require(
        paths.benchmark_csv(game, base_model, supervisor=supervisor, n_tasks=n_tasks),
        "benchmark")
    base = _load(base_path)
    base_reports = _require_archives(base, base_path, paths.parameters)
    log_info(f"[benchmark] base: {len(base)} tasks from {base_path} "
             f"({len(base_reports)} archived)")

    ft_path = paths.benchmark_csv(game, ft_model, supervisor=supervisor, n_tasks=n_tasks)
    finetuned, ft_reports = None, {}
    if os.path.exists(ft_path):
        finetuned = _load(ft_path)
        ft_reports = _require_archives(finetuned, ft_path, paths.parameters)
        log_info(f"[benchmark] finetuned: {len(finetuned)} tasks from {ft_path}")
    else:
        log_warn(f"[benchmark] no fine-tuned CSV at {ft_path} — comparison will be skipped.")

    written = []

    for model, frame, reports in [(base_model, base, base_reports),
                                  (ft_model, finetuned, ft_reports)]:
        if frame is None:
            continue
        rows = frame.head(max_episodes) if max_episodes else frame
        sections = [
            _episode_section(index, row, reports.get(index), model, report_dir,
                             frames_root, video_roots, overwrite)
            for index, row in rows.iterrows()
        ]
        n_success = int(frame["success"].sum())
        blocks = [
            md.h1(f"Benchmark episodes — {game} / {paths.executor} / {model}"),
            md.para(f"Source: `{paths.benchmark_csv(game, model, supervisor=supervisor, n_tasks=n_tasks)}`"),
            md.bullets([
                f"tasks: **{len(frame)}**",
                f"success: **{n_success}/{len(frame)}** "
                f"({n_success / max(len(frame), 1) * 100:.1f}%)",
                f"mean subgoal fraction: **{frame['subgoal_frac'].mean() * 100:.1f}%**",
                f"mean invalid actions per episode: **{frame['n_invalid'].mean():.2f}**",
            ]),
            md.note(
                "Every call below is read from the archived supervisor report beside that "
                f"episode's video, not reconstructed from text. Frames are written to "
                f"`{frames_root}` — on storage, since a full report is thousands of PNGs."
            ),
            "\n\n".join(sections),
        ]
        path = md.write_report(os.path.join(report_dir, f"episodes_{model}.md"), blocks)
        log_info(f"[benchmark] wrote {path}")
        written.append(path)

    if finetuned is not None:
        paired = _paired_index(base, base_path, finetuned, ft_path, paths.parameters)
        b, f = base.loc[paired], finetuned.loc[paired]
        dropped_base = [i for i in base.index if i >= len(paired)]
        dropped_ft = [i for i in finetuned.index if i >= len(paired)]

        base_only = [i for i in paired if b.loc[i, "success"] and not f.loc[i, "success"]]
        ft_only = [i for i in paired if f.loc[i, "success"] and not b.loc[i, "success"]]
        both = [i for i in paired if b.loc[i, "success"] and f.loc[i, "success"]]
        neither = [i for i in paired if not b.loc[i, "success"] and not f.loc[i, "success"]]

        n = len(paired)
        base_hits, ft_hits = int(b["success"].sum()), int(f["success"].sum())

        summary = pd.DataFrame([
            {"model": base_model, "success": base_hits, "n": n,
             "success_%": base_hits / n * 100,
             "subgoal_frac_%": b["subgoal_frac"].mean() * 100,
             "invalid_per_ep": b["n_invalid"].mean(), "steps_per_ep": b["n_steps"].mean()},
            {"model": ft_model, "success": ft_hits, "n": n,
             "success_%": ft_hits / n * 100,
             "subgoal_frac_%": f["subgoal_frac"].mean() * 100,
             "invalid_per_ep": f["n_invalid"].mean(), "steps_per_ep": f["n_steps"].mean()},
        ])

        def _step_lines(report) -> str:
            if report is None:
                return "_(no archive)_"
            steps = [_step_summary(step)
                     for leg in report.executor_reports
                     for call in leg.vlm_call_log
                     for step in call.steps]
            return md.code("\n".join(f"{i + 1:>3}  {s}" for i, s in enumerate(steps)))

        def _diff_sections(indices, left_model, right_model):
            out = []
            for i in indices:
                lines = [
                    md.h3(f"[{i}] {b.loc[i, 'task']}"),
                    md.table(pd.DataFrame([
                        {"model": left_model, "success": bool(b.loc[i, "success"]),
                         "steps": b.loc[i, "n_steps"], "invalid": b.loc[i, "n_invalid"],
                         "subgoals": f"{b.loc[i, 'subgoals_reached_n']}/{b.loc[i, 'all_subgoals_n']}"},
                        {"model": right_model, "success": bool(f.loc[i, "success"]),
                         "steps": f.loc[i, "n_steps"], "invalid": f.loc[i, "n_invalid"],
                         "subgoals": f"{f.loc[i, 'subgoals_reached_n']}/{f.loc[i, 'all_subgoals_n']}"},
                    ])),
                ]
                for model, side, reports in [(left_model, b, base_reports),
                                             (right_model, f, ft_reports)]:
                    report = reports.get(i)
                    video = _video_markdown(side.loc[i], video_roots, report_dir)
                    link = f" · video: {video}" if video else ""
                    lines.append(md.para(f"**{model}** steps{link}"))
                    lines.append(_step_lines(report))
                    calls = [c for leg in (report.executor_reports if report else [])
                             for c in leg.vlm_call_log]
                    if calls:
                        lines.append(md.details(f"{model} — final VLM output",
                                                md.code(calls[-1].response)))
                out.append("\n".join(lines))
            return "\n\n".join(out) if out else md.para("_(none)_")

        blocks = [
            md.h1(f"Benchmark comparison — {game} / {paths.executor}"),
            md.para(f"`{base_model}` (base) vs `{ft_model}` (fine-tuned)"),
            md.h2("Pairing"),
            md.bullets([
                f"episodes compared: **{n}** (row for row over the common prefix)",
                f"dropped, only in base: **{len(dropped_base)}** "
                + (f"({', '.join(base.loc[i, 'task'] for i in dropped_base[:10])})"
                   if dropped_base else ""),
                f"dropped, only in fine-tuned: **{len(dropped_ft)}** "
                + (f"({', '.join(finetuned.loc[i, 'task'] for i in dropped_ft[:10])})"
                   if dropped_ft else ""),
            ]),
            md.note(
                "Episodes are paired on row position, not task string: the benchmark anchors "
                "some tasks to two init_states, so a task string matches more than one episode "
                "and cannot key the join. Row *i* of a results CSV is row *i* of the benchmark "
                "table, and the two CSVs are checked to agree on every shared row before "
                "anything below is computed."
            ),
            md.h2("Summary"),
            md.table(summary),
            md.bullets([
                f"solved by base only: **{len(base_only)}**",
                f"solved by fine-tuned only: **{len(ft_only)}**",
                f"solved by both: **{len(both)}** · by neither: **{len(neither)}**",
            ]),
            md.para(
                "`subgoal_frac_%` uses the partial progress already recorded in every row and is "
                "the lower-variance signal to judge by. `invalid_per_ep` tests whether "
                "unparseable or non-executable training targets leaked into behaviour."
            ),
            md.h2(f"Solved by base only ({len(base_only)})"),
            md.para("Regressions introduced by fine-tuning — the debugging shortlist."),
            _diff_sections(base_only, base_model, ft_model),
            md.h2(f"Solved by fine-tuned only ({len(ft_only)})"),
            _diff_sections(ft_only, base_model, ft_model),
            md.h2("Both / neither"),
            md.table(pd.DataFrame({
                "outcome": ["both solved"] * len(both) + ["neither solved"] * len(neither),
                "row": both + neither,
                "task": [b.loc[i, "task"] for i in both + neither],
            })),
        ]
        path = md.write_report(os.path.join(report_dir, "comparison.md"), blocks)
        log_info(f"[benchmark] wrote {path}")
        written.append(path)
