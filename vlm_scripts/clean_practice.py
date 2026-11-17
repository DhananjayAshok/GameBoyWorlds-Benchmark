import json
import os
import pickle
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import pandas as pd
from tqdm import tqdm

from execution.report import ACTION_TAGS, EnvironmentStepRecord
from utils import (VLM, HuggingFaceModel, log_info, log_error, parse_key_value,
                   parse_list)
from create_dataset import select_successful, _episode_cutoff


HINT_RE = re.compile(r'\n?\[HINT_START\].*?\[HINT_END\]', re.DOTALL)
STEP_INFO_RE = re.compile(r'\n?\[STEP_INFO\].*?\[STEP_INFO_END\]', re.DOTALL)


AUGMENT_PARAPHRASE_PROMPT = """You are given a canonical task string:
"[CORE_TASK]"

Generate at least [N] diverse, valid paraphrases of this task. Vary the wording,
phrasing style, and structure but preserve the exact core meaning and level of specificity.
Use imperative tone throughout.

Respond in exactly this format (one paraphrase per line):
- <paraphrase 1>
- <paraphrase 2>
...
[STOP]"""


CLEAN_PROMPT = """You are reviewing a single decision an agent made while playing [GAME].

The agent was working toward this task:
"[TASK]"

It was shown the game frame(s) provided as image(s) and given this prompt context:
[AGENT_PROMPT]

The agent's reasoning and action (its response) was:
[AGENT_RESPONSE]

Your job is to catch only CRITICAL errors. Consider two things:
- Is the agent's description of the frame clearly wrong given the visual evidence?
- Is the chosen action clearly poor and unlikely to contribute toward completing the task?

Err strongly on the side of ACCEPT. Only REJECT if there is clear visual evidence that the
response badly misdescribes the frame, or that the action is likely counterproductive or
unlikely to advance the task. If you are unsure, ACCEPT.

Respond in exactly this format:
Reason: <one short sentence justifying your decision>
Decision: <ACCEPT or REJECT>
[STOP]"""


# Keep the Reason/Decision footer identical to CLEAN_PROMPT.
CLEAN_PROMPT_TRANSITION = """You are reviewing a single decision an agent made while playing [GAME].

The agent was working toward this task:
"[TASK]"

It was shown the game frame(s) provided as image(s) (every image EXCEPT the last) and given this prompt context:
[AGENT_PROMPT]

The agent's reasoning and action (its response) was:
[AGENT_RESPONSE]

The FINAL image is the game frame AFTER the agent's action was executed. Use the change from
the agent's frame(s) to this resulting frame as your main evidence of whether the action helped.

Your job is to catch only CRITICAL errors. Consider:
- Is the agent's description of the frame clearly wrong given the visual evidence?
- Does the action's visible effect (the before -> after change) clearly fail to advance the task, or move away from it?

Err strongly on the side of ACCEPT. Only REJECT if there is clear visual evidence that the
response badly misdescribes the frame, or that the resulting transition shows the action was
counterproductive or unlikely to advance the task. If you are unsure, ACCEPT.

Respond in exactly this format:
Reason: <one short sentence justifying your decision>
Decision: <ACCEPT or REJECT>
[STOP]"""


def _strip_blocks(text: str) -> str:
    return STEP_INFO_RE.sub('', HINT_RE.sub('', text))


def _paraphrase_task(task: str, vlm: VLM, max_new_tokens: int, k: int) -> list[str]:
    prompt = (
        AUGMENT_PARAPHRASE_PROMPT
        .replace("[CORE_TASK]", task)
        .replace("[N]", str(k))
    )
    output = vlm.infer(texts=prompt, max_new_tokens=max_new_tokens)["output"]
    paraphrases = [p for p in parse_list(output) if p and p != task]
    return paraphrases[:k]


def _filter_record(
    task: str,
    record,
    vlm: VLM,
    game: str,
    max_new_tokens: int,
) -> tuple[bool, str]:
    images = list(record.images) if record.images else []
    env_steps = [s for s in record.steps if isinstance(s, EnvironmentStepRecord)]
    next_frame = env_steps[-1].frame_after if env_steps else None
    use_transition = next_frame is not None and len(images) > 0
    if use_transition:
        images = images + [next_frame]
    template = CLEAN_PROMPT_TRANSITION if use_transition else CLEAN_PROMPT
    prompt = (
        template
        .replace("[GAME]", game)
        .replace("[TASK]", task)
        .replace("[AGENT_PROMPT]", _strip_blocks(record.prompt))
        .replace("[AGENT_RESPONSE]", record.response or "")
    )
    output = vlm.infer(texts=prompt, max_new_tokens=max_new_tokens, images=images or None)["output"]
    reason = parse_key_value(output, "Reason") or ""
    decision_raw = parse_key_value(output, "Decision") or ""
    accept = "reject" not in decision_raw.lower()
    return accept, reason


def _atomic_write_json(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _run_paraphrase_pass(
    practice_path, successful, vlm, k, max_new_tokens, effective_workers, overwrite, parameters
):
    out_path = os.path.join(practice_path, "paraphrases.json")
    checkpoint_path = os.path.join(practice_path, "paraphrases_checkpoint.json")

    if os.path.exists(out_path) and not overwrite:
        log_info(f"paraphrases.json already exists at {out_path} — skipping paraphrase pass.")
        return

    if os.path.exists(checkpoint_path) and not overwrite:
        with open(checkpoint_path, "r") as f:
            task_to_paraphrases = json.load(f)
    else:
        task_to_paraphrases = {}

    unique_tasks = [t for t in successful["task_string"].dropna().unique()]
    todo = [t for t in unique_tasks if t not in task_to_paraphrases]
    log_info(
        f"paraphrase pass: {len(unique_tasks) - len(todo)}/{len(unique_tasks)} tasks done — "
        f"running {len(todo)}.",
        parameters,
    )

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(_paraphrase_task, task, vlm, max_new_tokens, k): task
            for task in todo
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="paraphrasing tasks"):
            task = futures[future]
            try:
                paraphrases = future.result()
            except Exception:
                # Abort, never skip: tasks done so far are checkpointed.
                log_error(
                    f"clean paraphrase aborting on task {task!r}: VLM inference failed "
                    f"(is the server up?). Tasks done so far are checkpointed; re-run to "
                    f"resume.\n{traceback.format_exc()}",
                    parameters,
                )
            if len(paraphrases) < k:
                log_info(f"Warning: only {len(paraphrases)} paraphrase(s) for task {task!r}.", parameters)
            task_to_paraphrases[task] = paraphrases
            _atomic_write_json(task_to_paraphrases, checkpoint_path)

    _atomic_write_json(task_to_paraphrases, out_path)
    log_info(f"Saved paraphrases -> {out_path}")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


def _run_filter_pass(
    practice_path, successful, vlm, game, max_new_tokens, safety_margin,
    effective_workers, overwrite, verbose, parameters
):
    out_path = os.path.join(practice_path, "clean_decisions.csv")
    checkpoint_path = os.path.join(practice_path, "clean_decisions_checkpoint.json")

    if os.path.exists(out_path) and not overwrite:
        log_info(f"clean_decisions.csv already exists at {out_path} — skipping filter pass.")
        return

    if os.path.exists(checkpoint_path) and not overwrite:
        with open(checkpoint_path, "r") as f:
            rows = json.load(f)
        done = {(r["group_idx"], r["attempt"], r["call_idx"]) for r in rows}
    else:
        rows = []
        done = set()

    jobs = []
    missing_pkls = 0
    for _, row in successful.iterrows():
        # group_idx is a string id like "10_0": never int() it, since int("10_0") == 100.
        group_idx = str(row["group_idx"])
        attempt = int(row["attempt"])
        pkl_path = os.path.join(practice_path, f"{group_idx}_{attempt}.pkl")
        if not os.path.exists(pkl_path):
            missing_pkls += 1
            continue
        with open(pkl_path, "rb") as f:
            vlm_call_log = pickle.load(f)
        cutoff = _episode_cutoff(row.get("safe_success_point"), len(vlm_call_log), safety_margin)
        for call_idx, record in enumerate(vlm_call_log[:cutoff]):
            if (group_idx, attempt, call_idx) in done:
                continue
            # Non-action calls are not candidate training rows, so there is nothing to judge.
            if record.tag not in ACTION_TAGS:
                continue
            jobs.append((group_idx, attempt, call_idx, row["task_string"], record))

    if missing_pkls:
        log_info(f"Warning: {missing_pkls} pkl files not found (skipped).", parameters)
    log_info(f"filter pass: {len(done)} records done — running {len(jobs)}.", parameters)

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_job = {
            executor.submit(_filter_record, task, record, vlm, game, max_new_tokens):
                (group_idx, attempt, call_idx)
            for group_idx, attempt, call_idx, task, record in jobs
        }
        for future in tqdm(as_completed(future_to_job), total=len(future_to_job), desc="filtering"):
            group_idx, attempt, call_idx = future_to_job[future]
            try:
                accept, reason = future.result()
            except Exception:
                # An infer exception is an outage, not a judgment: parse failures never raise.
                log_error(
                    f"clean filter aborting at [{group_idx}_{attempt}] call {call_idx}: VLM "
                    f"inference failed (is the server up?). Records judged so far are "
                    f"checkpointed; re-run to resume.\n{traceback.format_exc()}",
                    parameters,
                )
            if verbose:
                log_info(f"[{group_idx}_{attempt}] call {call_idx}: "
                         f"{'accept' if accept else 'reject'} ({reason})", parameters)
            rows.append({
                "group_idx": group_idx,
                "attempt": attempt,
                "call_idx": call_idx,
                "accept": accept,
                "reason": reason,
            })
            _atomic_write_json(rows, checkpoint_path)

    n_reject = sum(1 for r in rows if not r["accept"])
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log_info(f"Saved {len(rows)} decisions ({n_reject} reject) -> {out_path}")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


@click.command(name="clean_practice")
@click.option(
    "--practice_path",
    required=True,
    help="Path to the practice output directory (contains results.csv and *.pkl).",
)
@click.option(
    "--k",
    default=3,
    show_default=True,
    help="Number of paraphrases to generate per unique task.",
)
@click.option(
    "--safety_margin",
    default=2,
    show_default=True,
    help="Extra VLM calls past the safe-success cutoff to also judge "
         "(must match create_dataset's --safety_margin).",
)
@click.option(
    "--max_concurrency",
    default=16,
    show_default=True,
    help="Max concurrent VLM calls. Forced to 1 for --verbose or a huggingface vlm_kind.",
)
@click.pass_obj
def clean_practice_cmd(obj, practice_path, k, safety_margin, max_concurrency):
    """Precompute paraphrases and accept/reject filter decisions for a practice dir."""
    parameters = obj["parameters"]
    game = obj["game"]
    model_name = obj["model_name"]
    vlm_kind = obj["vlm_kind"]
    overwrite = obj["overwrite"]
    verbose = obj["verbose"]
    max_new_tokens = obj["max_new_tokens"]

    if not os.path.isdir(practice_path):
        log_error(f"practice_path '{practice_path}' does not exist.", parameters)

    results_csv = os.path.join(practice_path, "results.csv")
    if not os.path.exists(results_csv):
        log_error(f"results.csv not found in {practice_path}", parameters)

    df = pd.read_csv(results_csv)
    successful = select_successful(df)
    log_info(f"Total episodes: {len(df)} | Successful: {len(successful)}", parameters)

    vlm = VLM(model_name, vlm_kind)
    effective_workers = 1 if (verbose or isinstance(vlm._vlm, HuggingFaceModel)) else max_concurrency

    _run_paraphrase_pass(
        practice_path, successful, vlm, k, max_new_tokens, effective_workers, overwrite, parameters
    )
    _run_filter_pass(
        practice_path, successful, vlm, game, max_new_tokens, safety_margin,
        effective_workers, overwrite, verbose, parameters
    )
