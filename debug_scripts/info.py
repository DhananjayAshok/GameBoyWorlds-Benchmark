import glob
import json
import os
import re

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from debug_scripts import markdown as md
from python_scripts.paths import Paths
from execution.info_doc import TASK_SECTION, InfoDocument
from python_scripts.paths import INFO_DOC_FILENAME
from utils import log_info

VAGUE_PHRASES = [
    "carefully", "make sure to", "pay attention", "as needed", "appropriately",
    "be aware", "in general", "if necessary", "keep in mind", "properly",
    "various", "some", "explore the area", "as appropriate",
]
CONCRETE_ANCHORS = [
    r"\bpress\b", r"\bbutton\b", r"\bA\b", r"\bB\b", r"\bstart\b", r"\bselect\b",
    r"\bup\b", r"\bdown\b", r"\bleft\b", r"\bright\b", r"\bnorth\b", r"\bsouth\b",
    r"\beast\b", r"\bwest\b", r"\bmenu\b", r"\bdoor\b", r"\bchest\b", r"\bNPC\b",
]


def _count_hits(text: str, patterns, regex: bool) -> int:
    lowered = text.lower()
    if not regex:
        return sum(lowered.count(p) for p in patterns)
    return sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns)


def _load_insights(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _round_dirs(info_dir: str) -> list[str]:
    root = os.path.join(info_dir, "merge")
    if not os.path.isdir(root):
        return []
    dirs = [d for d in glob.glob(os.path.join(root, "round_*")) if os.path.isdir(d)]
    return sorted(dirs, key=lambda d: int(os.path.basename(d).split("_")[1]))


def _round_documents(round_dir: str) -> list:
    docs = []
    # Node documents only: the sibling .matches.json and .meta.json would fail InfoDocument.from_dict.
    for doc_path in sorted(glob.glob(os.path.join(round_dir, "[0-9]*.json"))):
        if doc_path.endswith((".matches.json", ".meta.json")):
            continue
        with open(doc_path, "r") as handle:
            docs.append(InfoDocument.from_dict(json.load(handle)))
    return docs


def _figure_entry_counts(per_round: pd.DataFrame, n_leaves: int, out_path: str):
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(per_round["round"], per_round["task_entries"], marker="o",
              color="#4C78A8", label="task entries")
    axis.plot(per_round["round"], per_round["image_entries"], marker="o",
              color="#F58518", label="image entries")
    axis.axhline(n_leaves, ls="--", color="#888888",
                 label=f"never matched (= {n_leaves} leaves)")
    axis.set_xlabel("merge round")
    axis.set_ylabel("entries in the surviving documents")
    axis.set_title("Entry count per round\n(tracking the dashed line = matcher never fires)")
    axis.legend(frameon=False, fontsize=9)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _figure_drift(per_round: pd.DataFrame, out_path: str):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    left.plot(per_round["round"], per_round["mean_insight_words"], marker="o", color="#54A24B")
    left.set_xlabel("merge round")
    left.set_ylabel("mean words per insight")
    left.set_title("Insight length\n(shrinking = smoothing away detail)")
    left.grid(alpha=0.3)

    right.plot(per_round["round"], per_round["vague_per_insight"], marker="o",
               color="#E45756", label="vague phrases")
    right.plot(per_round["round"], per_round["concrete_per_insight"], marker="o",
               color="#4C78A8", label="concrete anchors")
    right.set_xlabel("merge round")
    right.set_ylabel("hits per insight")
    right.set_title("Anti-generality check\n(vague rising over concrete = drift)")
    right.legend(frameon=False, fontsize=9)
    right.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _document_stats(docs: list) -> dict:
    task_entries = sum(len(d.task_entries) for d in docs)
    image_entries = sum(len(d.image_entries) for d in docs)
    insights, words, vague, concrete = [], 0, 0, 0
    for doc in docs:
        for entry in doc.task_entries + doc.image_entries:
            for insight in entry.insights:
                insights.append(insight)
                words += len(insight.split())
                vague += _count_hits(insight, VAGUE_PHRASES, regex=False)
                concrete += _count_hits(insight, CONCRETE_ANCHORS, regex=True)
    n = max(len(insights), 1)
    return {
        "task_entries": task_entries,
        "image_entries": image_entries,
        "n_insights": len(insights),
        "mean_insight_words": words / n,
        "vague_per_insight": vague / n,
        "concrete_per_insight": concrete / n,
    }


@click.command()
@click.option("--model_name", required=True, help="VLM that built the document.")
@click.option("--source", default="attempt", show_default=True,
              type=click.Choice(["attempt", "curiosity"]),
              help="Which vertical's info dir to read; selects the paths.py accessor.")
@click.option("--max_entries", default=40, show_default=True,
              help="Cap on entries rendered in full (0 = no cap).")
@click.pass_obj
def debug_info(obj, model_name, source, max_entries):
    """Info-document build diagnostics: funnel, coverage, merge curve, drift, match audit."""
    paths = Paths(
        parameters=obj["parameters"], game=obj["game"], run_name=obj["run_name"],
        executor=obj["executor"], controller_variant=obj["controller_variant"],
        model_name=model_name, output_dir=obj["output_dir"],
        mode=obj["mode"],
    )
    report_dir = paths.debug_dir("info", model_name, source)
    images_dir = paths.debug_dir("info", model_name, source, "images")

    info_dir = paths.source_info_dir(source)
    insights_path = paths.require(os.path.join(info_dir, "insights.jsonl"), "insights")
    rows = _load_insights(insights_path)
    log_info(f"[info] {len(rows)} stage-A leaves from {insights_path}")

    leaf_docs = [InfoDocument.from_dict(row["document"]) for row in rows]
    leaf_stats = _document_stats(leaf_docs)

    leaf_render_blocks = [
        md.h2("Raw insights (pre-merge leaves)"),
        md.para(f"**{len(rows)}** leaves — one per stage-A pair, before any merge round "
                f"consolidates them."),
    ]
    leaf_render_rows = rows[:max_entries] if max_entries else rows
    for i, row in enumerate(leaf_render_rows):
        doc = leaf_docs[i]
        title = f"leaf {i} — init_state={row.get('init_state')}"
        if row.get("task"):
            title += f" — {row['task']}"
        parts = []
        for label, entries in (("Task entry", doc.task_entries), ("Image entry", doc.image_entries)):
            for entry in entries:
                parts.append(
                    f"**{label}: {entry.category}**\n\n"
                    f"Description: {entry.description}\n\n"
                    f"Examples:\n{md.bullets(entry.examples)}\n"
                    f"Insights:\n{md.bullets(entry.insights)}"
                )
        leaf_render_blocks.append(md.details(title, "\n\n".join(parts)))
    if max_entries and len(rows) > max_entries:
        leaf_render_blocks.append(md.warn(
            f"Showing {max_entries}/{len(rows)} leaves — pass --max_entries 0 for all."
        ))

    # insights.jsonl only holds the survivors, so the gap to the annotation json is the NONE rate.
    n_pairs = None
    for candidate in (paths.success_trajectories_json() if source == "attempt"
                      else paths.curiosity_annotation(),):
        if os.path.exists(candidate):
            with open(candidate) as handle:
                n_pairs = len(json.load(handle))

    funnel = [
        f"pairs available upstream: **{n_pairs}**" if n_pairs
        else "pairs available upstream: _(annotation json absent)_",
        f"leaves extracted (non-NONE): **{len(rows)}**",
    ]
    if n_pairs:
        dropped = n_pairs - len(rows)
        funnel.append(f"dropped as NONE / unparseable: **{dropped}** "
                      f"({100.0 * dropped / max(n_pairs, 1):.0f}%)")
    funnel.append(f"leaf entries: **{leaf_stats['task_entries']}** task / "
                  f"**{leaf_stats['image_entries']}** image")
    funnel.append(f"leaf insights: **{leaf_stats['n_insights']}**")

    coverage_block = []
    try:
        from gameboy_worlds import get_benchmark_tasks

        bench = get_benchmark_tasks(game=paths.game)
        counts = {}
        for row in rows:
            counts[row.get("init_state")] = counts.get(row.get("init_state"), 0) + 1
        coverage = (
            bench.groupby("init_state").size().reset_index(name="benchmark_tasks")
        )
        coverage["stage_a_records"] = coverage["init_state"].map(lambda s: counts.get(s, 0))
        coverage = coverage.sort_values("stage_a_records")
        n_zero = int((coverage["stage_a_records"] == 0).sum())
        coverage_block = [
            md.para(
                f"**{len(coverage) - n_zero}/{len(coverage)}** benchmark init states are "
                f"represented in stage A. Zero-coverage states are the episodes this "
                f"document has nothing first-hand to say about."
            ),
            md.table(coverage),
        ]
        if n_zero:
            coverage_block.append(md.warn(
                f"{n_zero} benchmark init state(s) have no records. Retrieval will still "
                f"fire on them — on entries learned elsewhere — so their episodes are the "
                f"ones to read first when a plan looks confidently wrong."
            ))
    except Exception as error:
        coverage_block = [md.warn(f"Could not compute benchmark coverage: {error}")]

    round_dirs = _round_dirs(info_dir)
    per_round_rows = [dict(round=-1, **leaf_stats)]  # round -1 is the leaves themselves
    match_records = []
    for round_dir in round_dirs:
        round_idx = int(os.path.basename(round_dir).split("_")[1])
        docs = _round_documents(round_dir)
        if docs:
            per_round_rows.append(dict(round=round_idx, **_document_stats(docs)))
        for matches_path in sorted(glob.glob(os.path.join(round_dir, "*.matches.json"))):
            with open(matches_path) as handle:
                matches = json.load(handle)
            node = os.path.basename(matches_path).split(".")[0]
            for section, entries in matches.items():
                for record in entries:
                    match_records.append({
                        "round": round_idx,
                        "node": node,
                        "section": "task" if section == TASK_SECTION else "image",
                        "candidate": record.get("candidate"),
                        "matched": record.get("matched") or "— (appended fresh)",
                        "reason": (record.get("reason") or "")[:160],
                    })

    per_round = pd.DataFrame(per_round_rows).sort_values("round")
    merge_blocks = []
    if len(round_dirs):
        counts_fig = os.path.join(images_dir, "entry_counts.png")
        drift_fig = os.path.join(images_dir, "insight_drift.png")
        _figure_entry_counts(per_round, leaf_stats["task_entries"] + leaf_stats["image_entries"],
                             counts_fig)
        _figure_drift(per_round, drift_fig)
        merge_blocks = [
            md.h2("Merge tree"),
            md.para(f"{len(round_dirs)} round(s) under `{os.path.join(info_dir, 'merge')}`. "
                    "Round -1 is the unmerged leaves."),
            md.img("entry counts per round", counts_fig, report_dir),
            md.img("insight drift", drift_fig, report_dir),
            md.table(per_round.round(2)),
        ]
        if len(match_records):
            match_frame = pd.DataFrame(match_records)
            n_matched = int((match_frame["matched"] != "— (appended fresh)").sum())
            merge_blocks += [
                md.h2("Match audit"),
                md.para(f"**{n_matched}/{len(match_frame)}** candidate entries matched an "
                        "existing entry; the rest were appended verbatim. A misfiled entry is "
                        "found here."),
                md.table(match_frame.head(200)),
            ]
    else:
        merge_blocks = [md.h2("Merge tree"),
                        md.warn("No `merge/` tree yet — stage B has not been run, so there "
                                "is no `info.json` and `--knowledge_mode retrieval` has "
                                "nothing to read. Re-run build_info with `--stage all`.")]

    doc_blocks = []
    doc_path = os.path.join(info_dir, INFO_DOC_FILENAME)
    if os.path.exists(doc_path):
        with open(doc_path) as handle:
            document = InfoDocument.from_dict(json.load(handle))
        summary = pd.DataFrame([
            {"section": "task", "category": e.category,
             "examples": len(e.examples), "insights": len(e.insights),
             "init_states": ", ".join(e.init_states)}
            for e in document.task_entries
        ] + [
            {"section": "image", "category": e.category,
             "examples": len(e.examples), "insights": len(e.insights),
             "init_states": ", ".join(e.init_states)}
            for e in document.image_entries
        ])
        doc_blocks = [
            md.h2("Final document"),
            md.para(f"`{doc_path}` — **{len(document.task_entries)}** task / "
                    f"**{len(document.image_entries)}** image entries."),
            md.table(summary),
        ]
        entries = document.task_entries + document.image_entries
        if max_entries:
            entries = entries[:max_entries]
        for entry in entries:
            body = (f"**Description:** {entry.description}\n\n"
                    f"**Examples:**\n{md.bullets(entry.examples)}\n"
                    f"**Insights:**\n{md.bullets(entry.insights)}")
            doc_blocks.append(md.details(f"{entry.category}", body))
    else:
        doc_blocks = [md.h2("Final document"),
                      md.warn(f"No `{INFO_DOC_FILENAME}` at {doc_path} — stage B has not been run.")]

    blocks = [
        md.h1(f"Info document — {paths.game} / {paths.model_save_name} / source={source}"),
        md.para(f"Source: `{info_dir}`"),
        *leaf_render_blocks,
        md.h2("Yield funnel"),
        md.bullets(funnel),
        md.note("If most pairs return NONE, or the final document has a handful of entries, "
                "nothing downstream can work and the fix is the stage-A prompt, not a "
                "benchmark run."),
        md.h2("Benchmark init_state coverage"),
        *coverage_block,
        *merge_blocks,
        *doc_blocks,
    ]
    out = md.write_report(os.path.join(report_dir, "report.md"), blocks)
    log_info(f"[info] wrote {out}")
