import gc
import json
import os
import pickle

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import log_info, log_warn, log_error
from debug_scripts import markdown as md
from debug_scripts.frames import trajectory_strip
from python_scripts.paths import Paths


GB = 1024 ** 3


def gini(values) -> float:
    array = np.sort(np.asarray(values, dtype=float))
    n = len(array)
    if n == 0 or array.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * index - n - 1).dot(array) / (n * array.sum()))


def lorenz_points(values):
    array = np.sort(np.asarray(values, dtype=float))
    if array.sum() == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    cumulative = np.cumsum(array) / array.sum()
    x = np.arange(1, len(array) + 1) / len(array)
    return np.insert(x, 0, 0.0), np.insert(cumulative, 0, 0.0)


def _histogram(values, bins=60):
    # Pre-binned so the cache never stores per-trajectory arrays.
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"counts": [], "edges": []}
    counts, edges = np.histogram(values, bins=bins)
    return {"counts": counts.tolist(), "edges": edges.tolist()}


def _is_trajectory(obj) -> bool:
    # Discriminated on the trailing init_state string, since a group of five trajectories is also length 5.
    return (
        isinstance(obj, (tuple, list))
        and len(obj) == 5
        and isinstance(obj[4], str)
        and not isinstance(obj[0], str)
        and hasattr(obj[0], "__len__")
    )


def classify(payload):
    if isinstance(payload, dict):
        payload = list(payload.values())
    if not isinstance(payload, (list, tuple)) or len(payload) == 0:
        return "unknown", None
    first = payload[0]
    if isinstance(first, str):
        return "manifest", list(payload)
    if _is_trajectory(first):
        return "trajectories", list(payload)
    if isinstance(first, (list, tuple)) and len(first) > 0 and _is_trajectory(first[0]):
        return "groups", list(payload)
    return "unknown", None


def _summarise(groups) -> dict:
    sizes = [len(group) for group in groups]
    lengths, total_rewards, max_rewards = [], [], []
    malformed = 0
    for group in groups:
        for trajectory in group:
            if not _is_trajectory(trajectory):
                malformed += 1
                continue
            observations, rewards = trajectory[0], trajectory[3]
            lengths.append(len(observations))
            rewards = np.asarray(rewards, dtype=float).ravel()
            total_rewards.append(float(rewards.sum()) if rewards.size else 0.0)
            max_rewards.append(float(rewards.max()) if rewards.size else 0.0)
    if malformed:
        log_warn(f"[curiosity] skipped {malformed} malformed trajectory entries")
    return {
        "n_groups": len(groups),
        "n_trajectories": int(sum(sizes)),
        "group_sizes": sizes,
        "gini_group_size": gini(sizes),
        "median_length": float(np.median(lengths)) if lengths else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "n_malformed": malformed,
        "n_degenerate": int(sum(1 for length in lengths if length <= 2)),
        "reward_total_mean": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "reward_total_std": float(np.std(total_rewards)) if total_rewards else 0.0,
        "reward_max_mean": float(np.mean(max_rewards)) if max_rewards else 0.0,
        "hist_length": _histogram(lengths, bins=40),
        "hist_reward_total": _histogram(total_rewards),
        "hist_reward_max": _histogram(max_rewards),
    }


def _expected_strips(images_dir, state, group_sizes, max_groups, max_traj_per_group):
    expected = []
    for group_i, size in enumerate(group_sizes):
        if max_groups and group_i >= max_groups:
            break
        for traj_i in range(min(size, max_traj_per_group)):
            expected.append((group_i, traj_i, os.path.join(
                images_dir, "strips", state, f"group_{group_i}_traj_{traj_i}.png")))
    return expected


def _plot_cached_hist(ax, hist, colour, xlabel, title):
    counts, edges = hist["counts"], hist["edges"]
    if not counts:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    centres = 0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))
    ax.bar(centres, counts, width=np.diff(edges), color=colour, align="center")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_title(title)


def _merge_hists(stats, key):
    usable = [stats[s][key] for s in stats if stats[s][key]["counts"]]
    if not usable:
        return {"counts": [], "edges": []}
    lo = min(h["edges"][0] for h in usable)
    hi = max(h["edges"][-1] for h in usable)
    edges = np.linspace(lo, hi if hi > lo else lo + 1, 61)
    counts = np.zeros(len(edges) - 1)
    for hist in usable:
        centres = 0.5 * (np.asarray(hist["edges"][:-1]) + np.asarray(hist["edges"][1:]))
        idx = np.clip(np.digitize(centres, edges) - 1, 0, len(counts) - 1)
        np.add.at(counts, idx, hist["counts"])
    return {"counts": counts.tolist(), "edges": edges.tolist()}


def _figure_group_sizes(stats, out_path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    states = sorted(stats, key=lambda s: stats[s]["n_trajectories"], reverse=True)

    left.barh(range(len(states)), [stats[s]["n_trajectories"] for s in states], color="#4C78A8")
    left.set_yticks(range(len(states)))
    left.set_yticklabels(states, fontsize=7)
    left.invert_yaxis()
    left.set_xscale("log")
    left.set_xlabel("trajectories (log)")
    left.set_title("Trajectories per init_state")

    for state in states:
        x, y = lorenz_points(stats[state]["group_sizes"])
        right.plot(x, y, alpha=0.55, linewidth=1)
    right.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect equality")
    right.set_xlabel("cumulative share of groups")
    right.set_ylabel("cumulative share of trajectories")
    right.set_title("Group-size inequality (one line per init_state)")
    right.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _figure_gini(stats, out_path):
    states = sorted(stats, key=lambda s: stats[s]["gini_group_size"], reverse=True)
    values = [stats[s]["gini_group_size"] for s in states]
    fig, ax = plt.subplots(figsize=(11, max(3.5, 0.24 * len(states))))
    ax.barh(range(len(states)), values,
            color=["#E45756" if v >= 0.6 else "#4C78A8" for v in values])
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(states, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.axvline(0.6, color="#E45756", linestyle="--", linewidth=1,
               label="0.6 (highly concentrated)")
    ax.set_xlabel("Gini of group sizes")
    ax.set_title("Group-size concentration per init_state")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _figure_rewards(stats, out_path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5))
    _plot_cached_hist(left, _merge_hists(stats, "hist_reward_total"), "#4C78A8",
                      "total reward per trajectory", "Curiosity reward — trajectory totals")
    left.set_ylabel("count (log)")
    _plot_cached_hist(right, _merge_hists(stats, "hist_reward_max"), "#F58518",
                      "max single-step reward", "Curiosity reward — per-trajectory peak")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _figure_lengths(stats, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _plot_cached_hist(ax, _merge_hists(stats, "hist_length"), "#54A24B",
                      "frames per trajectory", "Trajectory lengths (all init_states)")
    ax.set_ylabel("count (log)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


@click.command(name="curiosity")
@click.option("--init_state", default="all_states", show_default=True,
              help="Restrict to one init_state, or 'all_states' for every one found.")
@click.option("--n_frames", default=5, show_default=True, help="Frames per trajectory strip.")
@click.option("--max_groups", default=0, show_default=True,
              help="Cap groups rendered per init_state (0 = no cap).")
@click.option("--max_traj_per_group", default=3, show_default=True,
              help="Trajectory strips rendered per group.")
@click.option("--max_gb", default=8.0, show_default=True,
              help="Skip grouped pickles larger than this (GB). They are loaded whole; "
                   "the largest in deja_vu_1 is 23 GB.")
@click.pass_obj
def debug_curiosity(obj, init_state, n_frames, max_groups, max_traj_per_group, max_gb):
    """Summarise curiosity exploration: group counts, inequality, rewards, frame strips."""
    paths = Paths(
        parameters=obj["parameters"], game=obj["game"], run_name=obj["run_name"],
        executor=obj["executor"], controller_variant=obj["controller_variant"],
        output_dir=obj["output_dir"], mode=obj["mode"],
    )
    overwrite = obj["overwrite"]
    report_dir = paths.debug_dir("curiosity")
    images_dir = paths.debug_dir("curiosity", "images")
    cache_path = os.path.join(report_dir, "summary.json")

    states = paths.init_states()
    if init_state != "all_states":
        if init_state not in states:
            log_error(
                f"init_state '{init_state}' has no grouped trajectories under "
                f"{paths.grouped_dir()}. Available: {', '.join(states)}",
                paths.parameters,
            )
        states = [init_state]

    cache = {}
    if os.path.exists(cache_path) and not overwrite:
        with open(cache_path) as handle:
            cache = json.load(handle)

    stats, skipped, not_trajectories, rendered_by_state = {}, [], [], {}
    for state in states:
        path = paths.require(paths.grouped_file(state), "grouped")
        size = os.path.getsize(path)
        mtime = os.path.getmtime(path)
        key = f"{state}:{size}:{int(mtime)}"

        cached = cache.get(key)
        strips = None
        if cached is not None:
            strips = _expected_strips(images_dir, state, cached["group_sizes"],
                                      max_groups, max_traj_per_group)
            if all(os.path.exists(p) for _, _, p in strips):
                stats[state] = cached
                rendered_by_state[state] = strips
                log_info(f"[curiosity] {state}: cached ({cached['n_trajectories']} trajectories)")
                continue

        if size > max_gb * GB:
            log_warn(
                f"[curiosity] skipping {state}: {size / GB:.1f} GB exceeds --max_gb {max_gb}."
            )
            skipped.append((state, size / GB))
            continue

        log_info(f"[curiosity] loading {state} ({size / GB:.2f} GB)...")
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        kind, groups = classify(payload)
        if kind != "groups":
            if kind == "trajectories":
                groups = [groups]  # flat file: treat the whole thing as one group
            else:
                log_warn(f"[curiosity] {state}: file holds no trajectories ({kind}) — skipping.")
                not_trajectories.append((state, kind, len(payload) if hasattr(payload, "__len__") else 0))
                del payload
                gc.collect()
                continue
        stats[state] = _summarise(groups)
        cache[key] = stats[state]

        strips = []
        for group_i, group in enumerate(groups):
            if max_groups and group_i >= max_groups:
                break
            for traj_i, trajectory in enumerate(group[:max_traj_per_group]):
                out = os.path.join(images_dir, "strips", state,
                                   f"group_{group_i}_traj_{traj_i}.png")
                if trajectory_strip(trajectory, out, n=n_frames, overwrite=overwrite):
                    strips.append((group_i, traj_i, out))
        rendered_by_state[state] = strips

        del groups
        gc.collect()
        log_info(f"[curiosity] {state}: {stats[state]['n_groups']} groups, "
                 f"{stats[state]['n_trajectories']} trajectories, {len(strips)} strips")

        with open(cache_path, "w") as handle:
            json.dump(cache, handle)

    if not stats:
        log_error(
            "No grouped trajectory file could be read. Every candidate exceeded --max_gb "
            f"({max_gb} GB); raise it or pass --init_state to pick a smaller one.",
            paths.parameters,
        )

    summary = pd.DataFrame([
        {
            "init_state": state,
            "groups": stats[state]["n_groups"],
            "trajectories": stats[state]["n_trajectories"],
            "median_len": stats[state]["median_length"],
            "degenerate(<=2f)": stats[state]["n_degenerate"],
            "reward_mean": stats[state]["reward_total_mean"],
            "reward_std": stats[state]["reward_total_std"],
            "gini_group_size": stats[state]["gini_group_size"],
            "largest_group_share": (
                max(stats[state]["group_sizes"]) / stats[state]["n_trajectories"]
                if stats[state]["n_trajectories"] else 0.0
            ),
            "written": pd.Timestamp(
                os.path.getmtime(paths.grouped_file(state)), unit="s"
            ).strftime("%Y-%m-%d %H:%M"),
        }
        for state in stats
    ]).sort_values("trajectories", ascending=False)

    size_fig = os.path.join(images_dir, "group_sizes.png")
    gini_fig = os.path.join(images_dir, "gini.png")
    reward_fig = os.path.join(images_dir, "rewards.png")
    length_fig = os.path.join(images_dir, "lengths.png")
    _figure_group_sizes(stats, size_fig)
    _figure_gini(stats, gini_fig)
    _figure_rewards(stats, reward_fig)
    _figure_lengths(stats, length_fig)

    strip_sections = []
    for state in sorted(rendered_by_state):
        body, current_group = [], None
        for group_i, traj_i, out in rendered_by_state[state]:
            if group_i != current_group:
                current_group = group_i
                size = stats[state]["group_sizes"][group_i]
                body.append(f"\n**group {group_i}** ({size} trajectories)\n")
            body.append(md.img(f"{state} g{group_i} t{traj_i}", out, report_dir))
        strip_sections.append(md.details(
            f"{state} — {stats[state]['n_groups']} groups, "
            f"{stats[state]['n_trajectories']} trajectories",
            "\n".join(body) if body else "_(no renderable trajectories)_",
        ))

    total_groups = int(summary["groups"].sum())
    total_traj = int(summary["trajectories"].sum())
    concentrated = summary[summary["gini_group_size"] >= 0.6]["init_state"].tolist()
    flat_reward = summary[summary["reward_std"] < 1e-6]["init_state"].tolist()

    blocks = [
        md.h1(f"Curiosity exploration — {paths.game} / {paths.run_name}"),
        md.para(
            f"**{len(stats)}** init_states analysed · **{total_groups}** groups · "
            f"**{total_traj}** high-reward trajectories."
        ),
        md.para(f"Source: `{paths.grouped_dir()}`"),
        md.note(
            "Z-scores are not reported. They live in `replay_buffers/`, which "
            "`scripts/rl/create_traj.sh` deletes once grouping succeeds, so they are gone for any "
            "completed run. The per-trajectory reward arrays inside the grouped pickles are always "
            "present and are used as the durable curiosity signal instead."
        ),
    ]

    if skipped:
        blocks.append(md.warn(
            f"{len(skipped)} init_state(s) skipped — larger than --max_gb {max_gb} GB and pickle "
            "cannot be read partially: "
            + ", ".join(f"`{state}` ({size:.1f} GB)" for state, size in skipped)
            + ". Re-run with a higher --max_gb on a big-memory node to include them."
        ))

    if not_trajectories:
        blocks.append(md.note(
            "Skipped as already-counted or non-trajectory: "
            + ", ".join(f"`{state}` ({kind}, {n} entries)" for state, kind, n in not_trajectories)
            + ". A `manifest` is a list of *paths* to the per-init_state pickles, written by "
            "`python_funcs.py combine_trajectories` so one `infer_tasks` call can "
            "annotate every state without materialising tens of GB. This is intended, and "
            "`infer_tasks.load_grouped_trajectories` reads both formats — it is skipped here only "
            "because its contents are the other init_states, which would double-count."
        ))

    blocks += [
        md.h2("Per-init_state inventory"),
        md.table(summary),
        md.para(
            "`largest_group_share` is the fraction of an init_state's trajectories sitting in its "
            "single biggest group — a value near 1 means the clusterer collapsed everything into "
            "one bucket. `written` is the grouped-pickle mtime; an init_state written much earlier "
            "than the rest was probably reused by `create_traj.sh`'s skip-if-exists guard rather "
            "than regenerated."
        ),
        md.h2("Group-size inequality"),
        md.img("group sizes and Lorenz curves", size_fig, report_dir),
        md.img("Gini per init_state", gini_fig, report_dir),
        md.para(
            "Init_states with Gini ≥ 0.6 (highly concentrated): "
            + (", ".join(f"`{s}`" for s in concentrated) if concentrated else "_none_")
            + "."
        ),
        md.h2("Curiosity reward"),
        md.img("reward distributions", reward_fig, report_dir),
        md.para(
            "Near-zero variance means the curiosity signal saturated — the agent stopped finding "
            "anything novel, and every 'high-reward' trajectory is high-reward only relative to an "
            "equally flat baseline. Init_states with zero reward spread: "
            + (", ".join(f"`{s}`" for s in flat_reward) if flat_reward else "_none_")
            + "."
        ),
        md.h2("Trajectory lengths"),
        md.img("length histogram", length_fig, report_dir),
        md.para(
            f"Trajectories of ≤2 frames across the analysed init_states: "
            f"**{int(summary['degenerate(<=2f)'].sum())}**."
        ),
        md.h2("Group frame strips"),
        md.para(
            f"Up to {max_traj_per_group} trajectories per group, {n_frames} frames each "
            "(frame index top-left, action bottom-left). If the strips within one init_state all "
            "look like the same screen, the clustering is not separating game states."
        ),
        "\n".join(strip_sections),
    ]

    report_path = md.write_report(os.path.join(report_dir, "report.md"), blocks)
    log_info(f"[curiosity] wrote {report_path}")
