import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
import click

from utils import log_info, log_warn, log_error, VLM, parse_key_value, HuggingFaceModel
from python_scripts.paths import Paths
from show_trajectories import plot_transitions

SAVE_FRAMES_DIR = "save_frames"


def save_frames(frames, save_name, high_level_actions=None):
    os.makedirs(SAVE_FRAMES_DIR, exist_ok=True)
    plot_transitions(
        frames,
        os.path.join(SAVE_FRAMES_DIR, f"{save_name}.png"),
        high_level_actions=high_level_actions,
    )


DESCRIBE_PROMPT = """You are observing two consecutive screenshots from a game of [GAME].

Image 1 is the INITIAL frame. Image 2 is the NEXT frame.

Your job is to describe both frames carefully. Be precise and conservative — only state details you are confident about and that are clearly visible. Do not hallucinate or guess.

Respond in exactly this format:
Initial Frame Description: <describe what is visible in the initial frame>
Differences: <what has changed between frame 1 and frame 2>
[STOP]"""

INFER_PROMPT = """You are analysing multiple screenshots in sequence from a game of [GAME].

[DESC_AND_CHANGES]

Over the course of some of these frames, a single primary task may have been performed by the player, with the task being completed either at the very end or in some frame close to the end. 
Describe, with a single phrase, the action or task the player performed over the course these frames? Do not use conjunctions like "and" or "while" in your description. If there are multiple distinct tasks that seem to be happening, try to describe the whole subtrajectory wholistically and omit the less important subtasks. If there is no clear task, say "NO TASK".
Be specific but concise, each task should be a single, specific and meaningful action and not trivial. Describe only what is clearly supported by the evidence above. Make the task description as unambiguous as possible — include enough distinguishing detail that it cannot be confused with other similar tasks that could occur in the same game.

Always try to pick the longest horizon, most multistep version of the task that is present in the trajectory. If there is no clear task, respond with "NO TASK"
Otherwise, respond in exactly this format:
Visual Description: <a description of the individual frames and changes that occur from leftmost frame to rightmost frame>
Reasoning: <one single, short sentence describing your thinking. Reference visual evidence of the key frames and overall actions that led you to infer this task.>
Task: <one or two sentence description of what the player did or is doing>
Start: <integer index of starting frame> this is to indicate when the player seems to be acting with the intent to perform the task, not simply the step right before the task is executed. This may be several frames before the task is completed, and will depend on the specific task and context.
End: <integer index of frame where task is performed or executed or completed or first detected>
[STOP]"""

REFINE_PROMPT = """You are given a description of what a player did over the course of some game frames:
"[CANDIDATE_TASK]"

First judge whether this is a well-formed task. A VALID task is SPECIFIC and CONCRETE — a
clearly-defined action with an unambiguous completion state that could be checked from the
screen (e.g. "select the diamond from the inventory", "open the cellar door", "exit the taxi").
Judge it INVALID if it is too generic or vague to complete exactly — e.g. "navigate the game
environment", "interact with an object", "explore the area", "manage inventory", "pick up an
object" — cases where many different behaviours would all satisfy it.

If VALID, rewrite it as a concise imperative instruction:
- Use second-person imperative tone (no subject).
- Keep it short (under 10 words if possible).
- Do not add any detail that was not in the original description.

Respond in exactly this format:
Verdict: <VALID or INVALID>
Task: <imperative task string if VALID, or NONE if INVALID>
[STOP]"""

CONSOLIDATE_PROMPT = """You are given several candidate descriptions of a task performed in a game of [GAME], all inferred from similar game states:
[CANDIDATE_LIST]

A representative final frame is provided as an image.

Identify the single most accurate, concise imperative task string that best captures what all of these trajectories have in common. Ignore noise or outliers.

Respond in exactly this format:
Task: <single canonical imperative task string>
[STOP]"""

DISTILL_PROMPT = """You are given several candidate descriptions of a task performed in a game of [GAME], all inferred from similar game states:

[CANDIDATE_LIST]

Generate a single, unifying task string that captures the core commonality between all of these candidates, while omitting any extraneous detail or noise. Use imperative tone.

The distilled task must stay SPECIFIC, CONCRETE and UNAMBIGUOUS — a clearly-defined action with a checkable completion state (e.g. "select the diamond from the inventory", "open the cellar door"). Do not over-generalise into a vague or generic instruction (e.g. "interact with an object", "navigate the environment", "manage inventory") that many different behaviours would satisfy. Keep the most specific meaning shared by the candidates.

Respond in exactly this format:
Reasoning: <one single, short sentence describing your thinking. Reference the commonalities between the candidates that led you to infer this distilled task.>
Task: <single distilled imperative task string>
[STOP]
"""


def _parse_infer_block(
    text: str, window_offset: int = 0, n_obs: int = None
) -> dict | None:
    # start and end are returned as absolute observation indices.
    text = text.lower()

    task = parse_key_value(text, "Task")
    if task is None or "no task" in task:
        return None
    start_str = parse_key_value(text, "Start")
    end_str = parse_key_value(text, "End")
    try:
        start = int(start_str) - 1 + window_offset
    except (TypeError, ValueError):
        start = None
    try:
        end = int(end_str) - 1 + window_offset
        if n_obs is not None and end >= n_obs - 1:
            end -= 1
    except (TypeError, ValueError):
        end = None
    if n_obs is not None:
        if start is not None:
            start = max(0, min(start, n_obs - 1))
        if end is not None:
            end = max(0, min(end, n_obs - 2))
    return {"task": task, "start": start, "end": end}


def got_bigger(subject, refined_subject, multiplier=1.5):
    if refined_subject is None:
        return None
    subject_count = subject.count(" ")
    refined_count = refined_subject.count(" ")
    if refined_count > multiplier * subject_count:
        return True
    else:
        return False


def describe_pairwise(
    window,
    action_window,
    vlm: VLM,
    game: str,
    max_new_tokens: int,
    verbose: bool = False,
) -> str:
    k = len(window)
    frame_descs = {}
    diffs = {}
    describe_prompt_template = DESCRIBE_PROMPT.replace("[GAME]", game)
    pair_indices = list(range(k - 1))
    if pair_indices:
        describe_outputs = vlm.infer(
            texts=[describe_prompt_template for _ in pair_indices],
            images=[[window[i], window[i + 1]] for i in pair_indices],
            max_new_tokens=max_new_tokens,
        )["output"]
        for i, describe_output in zip(pair_indices, describe_outputs):
            describe_output = describe_output.lower()
            if verbose:
                save_frames(
                    [window[i], window[i + 1]],
                    f"describe_pair_{i}",
                    high_level_actions=[action_window[i]],
                )
                log_info(f"DESCRIBE output (pair {i}→{i+1}):\n{describe_output}\n---")
            frame_descs[i] = parse_key_value(describe_output, "Initial Frame Description") or ""
            diffs[i] = parse_key_value(describe_output, "Differences") or ""

    desc_lines = []
    for i in range(k - 1):
        if i == 0:
            desc_lines.append(f"Frame {i}: {frame_descs[i]}")
        desc_lines.append(f"Changes {i}→{i+1}: {diffs[i]}")
    if desc_lines:
        return (
            "Here is a description of the changes that occured in between each frame:\n"
            + "\n".join(desc_lines),
            frame_descs,
        )
    return "", frame_descs


def infer_task(
    trajectory,
    vlm: VLM,
    game: str,
    max_new_tokens: int,
    lookback: int = 5,
    verbose: bool = False,
    describe_pairs: bool = False,
) -> dict | None:
    observations, actions, high_level_actions, rewards, init_state = trajectory
    n = len(observations)
    k = min(lookback, n)
    window = observations[n - k :]
    action_window = high_level_actions[
        n - k :
    ]  # action_window[i] is the action causing window[i]→window[i+1]

    frame_descs = {}

    if describe_pairs:
        desc_and_changes, frame_descs = describe_pairwise(
            window, action_window, vlm, game, max_new_tokens, verbose=verbose
        )
    else:
        desc_and_changes = ""

    infer_prompt = INFER_PROMPT.replace("[GAME]", game).replace(
        "[DESC_AND_CHANGES]", desc_and_changes
    )
    if verbose:
        save_frames(list(window), "infer_window", high_level_actions=action_window)
        log_info(f"INFER prompt:\n{infer_prompt}\n---")
    infer_output = vlm.infer(
        texts=infer_prompt,
        images=list(window),
        max_new_tokens=max_new_tokens,
    )["output"].lower()
    if verbose:
        log_info(f"INFER output:\n{infer_output}\n---")

    if "no task" in infer_output:
        if verbose:
            log_info("INFER returned NO TASK.")
        return None

    parsed_block = _parse_infer_block(infer_output, window_offset=n - k, n_obs=n)
    if parsed_block is None:
        log_warn(
            f"infer_task failed to parse Task block from INFER stage. Output was:\n{infer_output}"
        )
        return None
    task_description = parse_key_value(infer_output, "Visual Description")

    refine_prompt = REFINE_PROMPT.replace("[CANDIDATE_TASK]", parsed_block["task"])
    refine_output = vlm.infer(
        texts=refine_prompt,
        max_new_tokens=max_new_tokens,
    )["output"].lower()
    if verbose:
        log_info(f"REFINE output:\n{refine_output}\n---")

    # "invalid" contains "valid", so test for "invalid" first, then require an explicit "valid".
    verdict = parse_key_value(refine_output, "Verdict") or ""
    if "invalid" in verdict or "valid" not in verdict:
        if verbose:
            log_info(f"REFINE rejected task (verdict={verdict!r}) — treating as NO TASK.")
        return None

    refined_task = parse_key_value(refine_output, "Task")
    if refined_task is None or refined_task.strip() in ("", "none"):
        if verbose:
            log_info("REFINE gave no usable Task despite VALID verdict — treating as NO TASK.")
        return None
    task = parsed_block["task"]
    if got_bigger(task, refined_task):
        refined_task = task

    if verbose:
        log_info(
            f"Final inferred task: {refined_task}, start: {parsed_block['start']}, end: {parsed_block['end']}"
        )
        start_idx = parsed_block["start"] if parsed_block["start"] is not None else None
        end_idx = parsed_block["end"] if parsed_block["end"] is not None else None
        if start_idx is not None and end_idx is not None:
            save_frames([observations[start_idx], observations[end_idx]], "task_frames")
        log_info(
            "Exiting after one inference for verbose demonstration. Set VERBOSE = False to run full inference."
        )
        breakpoint()
    return {
        "task": {
            "task": refined_task,
            "start": parsed_block["start"],
            "end": parsed_block["end"],
        },
        "task_description": task_description,
        "frame_descriptions": frame_descs if frame_descs else None,
        "used_lookback": k,
    }


def _select_group_trajectories(group: list, max_trajectories_per_group: int) -> list:
    use_traj_idxes = list(range(len(group)))
    if len(group) > max_trajectories_per_group:
        use_traj_idxes = list(
            np.random.choice(use_traj_idxes, max_trajectories_per_group, replace=False)
        )
    return [group[traj_idx] for traj_idx in use_traj_idxes]


def _distill_tasks(all_tasks: list[str], vlm: VLM, game: str, max_new_tokens: int) -> str:
    distill_prompt = DISTILL_PROMPT.replace("[GAME]", game).replace(
        "[CANDIDATE_LIST]", "\n".join(f"- {t}" for t in all_tasks)
    )
    distill_output = vlm.infer(
        texts=distill_prompt,
        max_new_tokens=max_new_tokens,
    )["output"].lower()
    distilled_task = parse_key_value(distill_output, "Task")
    if distilled_task is None:
        distilled_task = all_tasks[0]
    return distilled_task


def _canonical_task(task: str) -> str:
    return " ".join(task.lower().split()).strip(" .!?\"'")


def _dedup_task_groups(trajectory_output: dict, trajectory_data_output: dict):
    canon_to_keep: dict[str, int] = {}
    deduped_output: dict[int, str] = {}
    deduped_data: dict[int, list] = {}
    merges: dict[int, list] = {}
    for gid in sorted(trajectory_output):
        task = trajectory_output[gid]
        canon = _canonical_task(task)
        if canon in canon_to_keep:
            keep = canon_to_keep[canon]
            deduped_data[keep].extend(trajectory_data_output.get(gid, []))
            merges[keep].append(gid)
        else:
            canon_to_keep[canon] = gid
            deduped_output[gid] = task
            deduped_data[gid] = list(trajectory_data_output.get(gid, []))
            merges[gid] = []
    return deduped_output, deduped_data, merges


def load_grouped_trajectories(path):
    # The pkl is either a manifest (a list of paths to grouped pkls) or a plain list of groups.
    with open(path, "rb") as f:
        items = pickle.load(f)
    for item in items:
        if isinstance(item, str):
            with open(item, "rb") as g:
                for group in pickle.load(g):
                    yield group
        else:
            yield item


@click.command(name="infer_task")
@click.option(
    "--trajectory_path",
    required=True,
    help="Path to grouped_high_reward_trajectories.pkl",
)
@click.option(
    "--run_name",
    default="all",
    help="Name under which to save",
)
@click.option(
    "--lookback",
    default=8,
    show_default=True,
    help="Number of frames from the end of each trajectory to analyse",
)
@click.option(
    "--max_trajectories_per_group",
    default=3,
    show_default=True,
    help="Max trajectories to sample per group",
)
@click.option(
    "--describe_pairs",
    is_flag=True,
    default=False,
    help="Run pairwise DESCRIBE stage before INFER.",
)
@click.option(
    "--dedup_tasks/--no_dedup_tasks",
    default=True,
    show_default=True,
    help="Merge groups that distilled to an identical task string into one group "
         "(concatenating trajectories) before saving. Exact-match only.",
)
@click.option(
    "--max_concurrency",
    default=16,
    show_default=True,
    help="Max concurrent infer_task pipelines (across groups and trajectories). Forced to 1 for --verbose or a huggingface vlm_kind.",
)
@click.pass_obj
def infer_task_cmd(
    obj, trajectory_path, run_name, lookback, max_trajectories_per_group, describe_pairs, dedup_tasks, max_concurrency
):
    """Infer task strings for each trajectory group."""
    max_new_tokens = obj["max_new_tokens"]
    vlm_kind = obj["vlm_kind"]
    game = obj["game"]
    parameters = obj["parameters"]
    model_name = obj["model_name"]
    vlm = VLM(model_name, vlm_kind)
    paths = Paths(parameters=parameters, game=game, model_name=model_name,
                  run_name=run_name)
    traj_path = paths.curiosity_annotation()
    out_dir = os.path.dirname(traj_path)
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = traj_path.replace(".json", "_checkpoint.json")

    if os.path.exists(traj_path) and not obj["overwrite"]:
        log_info(
            f"Skipping infer — output already exists at {traj_path}. Use --overwrite to rerun."
        )
        return

    pkl_path = traj_path.replace(".json", ".pkl")
    pkl_checkpoint_path = checkpoint_path.replace(".json", ".pkl")

    if os.path.exists(checkpoint_path) and not obj["overwrite"]:
        with open(checkpoint_path, "r") as f:
            trajectory_output = {int(k): v for k, v in json.load(f).items()}
        trajectory_data_output = {}
        if os.path.exists(pkl_checkpoint_path):
            with open(pkl_checkpoint_path, "rb") as f:
                trajectory_data_output = pickle.load(f)
        # A group is done only when both its annotation and its trajectories are on disk: the two writes are not atomic.
        unpaired = [
            group_idx for group_idx in trajectory_output
            if group_idx not in trajectory_data_output
        ]
        for group_idx in unpaired:
            del trajectory_output[group_idx]
        if unpaired:
            log_warn(
                f"checkpoint mismatch: {len(unpaired)} group(s) had a saved annotation but "
                f"no saved trajectories ({unpaired[:5]}{' ...' if len(unpaired) > 5 else ''}) "
                "— rerunning them."
            )
        log_info(
            f"Resuming infer from checkpoint — {len(trajectory_output)} groups already done."
        )
    else:
        trajectory_output = {}
        trajectory_data_output = {}

    grouped_trajectories = load_grouped_trajectories(trajectory_path)

    # HuggingFaceModel is not safe for concurrent generate() calls.
    effective_workers = (
        1 if (obj["verbose"] or isinstance(vlm._vlm, HuggingFaceModel)) else max_concurrency
    )

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        pending = []
        for group_idx, group in enumerate(grouped_trajectories):
            if group_idx in trajectory_output:
                continue
            trajectories = _select_group_trajectories(group, max_trajectories_per_group)
            futures = [
                executor.submit(
                    infer_task,
                    trajectory,
                    vlm,
                    game,
                    max_new_tokens,
                    lookback,
                    verbose=obj["verbose"],
                    describe_pairs=describe_pairs,
                )
                for trajectory in trajectories
            ]
            pending.append((group_idx, trajectories, futures))

        for group_idx, trajectories, futures in tqdm(pending, desc="Processing groups"):
            all_tasks = []
            used_trajectories = []
            for trajectory, future in zip(trajectories, futures):
                result = future.result()
                if result is not None:
                    all_tasks.append(result["task"]["task"])
                    used_trajectories.append(trajectory)

            if not all_tasks:
                log_warn(
                    f"skipping group {group_idx} — could not infer any task strings."
                )
                continue

            distilled_task = _distill_tasks(all_tasks, vlm, game, max_new_tokens)
            trajectory_output[group_idx] = distilled_task
            trajectory_data_output[group_idx] = used_trajectories
            with open(checkpoint_path, "w") as f:
                json.dump(trajectory_output, f, indent=2)
            with open(pkl_checkpoint_path, "wb") as f:
                pickle.dump(trajectory_data_output, f)

    if dedup_tasks:
        before = len(trajectory_output)
        trajectory_output, trajectory_data_output, merges = _dedup_task_groups(
            trajectory_output, trajectory_data_output
        )
        for keep, folded in merges.items():
            if folded:
                log_info(
                    f"Merged groups {folded} into group {keep} "
                    f"(task: {trajectory_output[keep]!r})"
                )
        log_info(f"Dedup: {before} groups → {len(trajectory_output)} unique tasks.")

    # Checked before writing: traj_path existing is this command's skip-if-exists marker.
    if not trajectory_output:
        log_error(
            f"infer_tasks produced 0 task annotations from {trajectory_path}. "
            "Nothing was written, so downstream build_info cannot run. "
            "Check that the grouped trajectory file actually contains trajectories "
            "(a 'manifest' pkl holds paths, not frames) and that the VLM returned parseable "
            "output.",
            parameters,
        )

    with open(traj_path, "w") as f:
        json.dump(trajectory_output, f, indent=2)
    with open(pkl_path, "wb") as f:
        pickle.dump(trajectory_data_output, f)
    log_info(f"Saved trajectory annotations → {traj_path}")
    log_info(f"Saved trajectories → {pkl_path}")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    if os.path.exists(pkl_checkpoint_path):
        os.remove(pkl_checkpoint_path)
