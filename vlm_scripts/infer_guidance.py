"""
Called by scripts/vlm/infer_guidance.sh (via vlm.py infer_guidance). Use --help for CLI options.
"""
# Reads from the output JSON and pickle files produced by infer_task.py.
# For each group, generates a gold step-by-step guidance description on how
# to accomplish the inferred task, then saves the enriched data to a sibling JSON.
#
# Input files (derived from trajectory_path, same as infer_task_cmd):
#   <trajectory_path>.json  -> {group_idx: task_string}
#   <trajectory_path>.pkl   -> {group_idx: [trajectory, ...]}
#
# Output file:
#   <trajectory_path>_guidance.json    -> {group_idx: {task, init_state, guidance, goal_condition}}

import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import click
from tqdm import tqdm

from python_scripts import paths
from utils import log_info, log_warn, log_error, VLM, parse_key_value, HuggingFaceModel

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SLICE_GUIDANCE_PROMPT = """You are an expert player of [GAME]. You are given frames [START_IDX]-[END_IDX] (out of [TOTAL] total frames) from a game trajectory, along with the task being accomplished:

Task: "[TASK]"

Describe what the player does in this section of the trajectory. Base your guidance strictly on what you can observe — do not invent details not visible.

Guidelines:
- Write each step as a short, clear imperative instruction (e.g. "Press A to speak to the NPC", "Walk left towards the door").
- Order steps chronologically within this section.
- Be specific about directions, button presses and targets when clearly visible.
- If the exact button labels are intuitive, describe the action instead.
- You must make sure that for each step, you explicitly refer to the visual cues in the frame that indicate WHEN to do a particular step, and when to move on to the next. For example, instead of saying "walk upwards", say "when the character is past the tree line, walk upwards until the door is visible at the top of the frame." Be very specific about the visual cues that indicate what to do and when.

Respond in exactly this format:
Summary: <one sentence describing what happens in frames [START_IDX]-[END_IDX]>
Goal condition: <a visual description of the final frame in this section>
Steps:
- <step 1>
- <step 2>
...
[STOP]"""

CONSOLIDATE_GUIDANCE_PROMPT = """You are consolidating partial guidance from multiple sections of a trajectory in a game of [GAME].

Task: "[TASK]"

Here are the descriptions of each section, in chronological order:
[SECTION_DESCRIPTIONS]

Combine these into a single, complete, coherent set of step-by-step guidance that a new player could follow from start to finish to accomplish the task.

Respond in exactly this format:
Summary: <one sentence describing the overall approach to complete the task>
Goal condition: <a visual description of the state that confirms task completion>
Steps:
- <step 1>
- <step 2>
...
[STOP]"""

# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------


def _parse_guidance(text: str) -> dict | None:
    """Parse Summary, Goal condition, and bullet Steps from GUIDANCE_PROMPT output."""
    text_lower = text.lower()
    stop_idx = text_lower.find("[stop]")
    if stop_idx != -1:
        text_lower = text_lower[:stop_idx]

    summary = parse_key_value(text_lower, "Summary") or ""
    goal_condition = parse_key_value(text_lower, "Goal condition") or ""

    steps = []
    in_steps = False
    for line in text_lower.splitlines():
        stripped = line.strip()
        if stripped.startswith("steps:"):
            in_steps = True
            continue
        if in_steps and stripped.startswith("- "):
            steps.append(stripped[2:].strip())

    if not steps:
        return None
    return {"summary": summary, "goal_condition": goal_condition, "steps": steps}


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def infer_guidance_for_trajectory(
    trajectory,
    task: str,
    vlm: VLM,
    game: str,
    max_new_tokens: int,
    max_obs_at_once: int = 8,
    verbose: bool = False,
) -> dict | None:
    """Chunk the full trajectory into non-overlapping slices, get structured guidance
    for each, then consolidate into a single guidance with a text-only VLM call."""
    observations, actions, high_level_actions, rewards, init_state = trajectory
    total = len(observations)

    slice_ranges = []
    slice_prompts = []
    slice_images = []
    for start in range(0, total, max_obs_at_once):
        end = min(start + max_obs_at_once, total)
        frames = list(observations[start:end])

        prompt = (
            SLICE_GUIDANCE_PROMPT
            .replace("[GAME]", game)
            .replace("[TASK]", task)
            .replace("[START_IDX]", str(start))
            .replace("[END_IDX]", str(end - 1))
            .replace("[TOTAL]", str(total))
        )

        if verbose:
            log_info(f"SLICE prompt (frames {start}-{end - 1} / {total}):\n{prompt}\n---")

        slice_ranges.append((start, end))
        slice_prompts.append(prompt)
        slice_images.append(frames)

    slice_descriptions = []
    if slice_prompts:
        outputs = vlm.infer(texts=slice_prompts, images=slice_images, max_new_tokens=max_new_tokens)["output"]
        for (start, end), output in zip(slice_ranges, outputs):
            if verbose:
                log_info(f"SLICE output:\n{output}\n---")

            parsed = _parse_guidance(output)
            if parsed is None:
                log_warn(f"failed to parse guidance for frames {start}-{end - 1}, skipping slice.")
                continue

            slice_descriptions.append(
                f"Frames {start}-{end - 1} / {total}:\n"
                f"Summary: {parsed['summary']}\n"
                f"Steps:\n" + "\n".join(f"- {s}" for s in parsed["steps"])
            )

    if not slice_descriptions:
        return None

    consolidate_prompt = (
        CONSOLIDATE_GUIDANCE_PROMPT
        .replace("[GAME]", game)
        .replace("[TASK]", task)
        .replace("[SECTION_DESCRIPTIONS]", "\n\n".join(slice_descriptions))
    )

    if verbose:
        log_info(f"CONSOLIDATE prompt:\n{consolidate_prompt}\n---")

    output = vlm.infer(texts=consolidate_prompt, max_new_tokens=max_new_tokens)["output"]

    if verbose:
        log_info(f"CONSOLIDATE output:\n{output}\n---")

    parsed = _parse_guidance(output)
    if parsed is None:
        log_warn(f"failed to parse consolidated guidance output:\n{output}")
        return None
    return parsed


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------


@click.command(name="infer_guidance")
@click.option(
    "--trajectory_path",
    required=True,
    help="Path to trajectory annotation, as saved by infer_task.py",
)
@click.option(
    "--max_obs_at_once",
    default=8,
    show_default=True,
    help="Max frames per slice when chunking the trajectory.",
)
@click.option(
    "--max_concurrency",
    default=16,
    show_default=True,
    help="Max concurrent infer_guidance_for_trajectory pipelines (across groups). Forced to 1 for --verbose or a huggingface vlm_kind.",
)
@click.pass_obj
def infer_guidance_cmd(obj, trajectory_path, max_obs_at_once, max_concurrency):
    """Generate gold step-by-step guidance for each inferred task group."""
    max_new_tokens = obj["max_new_tokens"]
    vlm_kind = obj["vlm_kind"]
    game = obj["game"]
    model_name = obj["model_name"]
    overwrite = obj["overwrite"]
    verbose = obj["verbose"]
    parameters = obj["parameters"]

    vlm = VLM(model_name, vlm_kind)
    model_save_name = paths.model_save_name(model_name)
    # have a guard to ensure we aren't using different models
    if model_save_name not in trajectory_path.lower():
        log_error(
            f"Model name '{model_save_name}' not found in trajectory_path '{trajectory_path}'. "
            "Ensure the trajectory was produced by the same model.",
            parameters,
        )

    annotation_json = trajectory_path + ".json"
    annotation_pkl = annotation_json.replace(".json", ".pkl")

    if not os.path.exists(annotation_json):
        log_error(
            f"trajectory_annotation.json not found at {annotation_json}.", parameters
        )
    if not os.path.exists(annotation_pkl):
        log_error(
            f"trajectory_annotation.pkl not found at {annotation_pkl}.", parameters
        )

    out_json = trajectory_path + "_guidance.json"
    checkpoint_path = out_json.replace(".json", "_checkpoint.json")

    if os.path.exists(out_json) and not overwrite:
        log_info(
            f"Skipping infer_guidance — output already exists at {out_json}. Use --overwrite to rerun."
        )
        return

    with open(annotation_json, "r") as f:
        task_map = json.load(f)
    with open(annotation_pkl, "rb") as f:
        traj_map = {str(k): v for k, v in pickle.load(f).items()}

    if os.path.exists(checkpoint_path) and not overwrite:
        with open(checkpoint_path, "r") as f:
            guidance_output = json.load(f)
        log_info(
            f"Resuming infer_guidance from checkpoint — {len(guidance_output)} groups already done."
        )
    else:
        guidance_output = {}

    # infer_guidance_for_trajectory pipelines are independent across groups, so
    # submit them all to one shared pool. --verbose runs breakpoint()-style
    # debugging and HuggingFaceModel isn't safe for concurrent generate() calls —
    # both fall back to max_workers=1, which processes submitted jobs one at a
    # time in submission order (i.e. identical to the old sequential loop).
    effective_workers = (
        1 if (verbose or isinstance(vlm._vlm, HuggingFaceModel)) else max_concurrency
    )

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_info = {}
        for group_idx in sorted(task_map.keys()):
            if group_idx in guidance_output:
                continue

            task = task_map[group_idx]
            trajectories = traj_map.get(group_idx)
            if not trajectories:
                log_warn(f"no trajectories for group {group_idx}, skipping.")
                continue

            # Use the first available trajectory as the representative example.
            # infer_tasks.py stores a list of trajectories; attempt_tasks.py stores a single tuple.
            trajectory = trajectories[0] if isinstance(trajectories, list) else trajectories
            _, _, _, _, init_state = trajectory
            future = executor.submit(
                infer_guidance_for_trajectory,
                trajectory,
                task,
                vlm,
                game,
                max_new_tokens,
                max_obs_at_once=max_obs_at_once,
                verbose=verbose,
            )
            future_info[future] = (group_idx, task, init_state)

        for future in tqdm(as_completed(future_info), total=len(future_info), desc="Generating guidance"):
            group_idx, task, init_state = future_info[future]
            guidance = future.result()
            if guidance is None:
                log_warn(f"could not generate guidance for group {group_idx}.")
                continue

            goal_condition = guidance.pop("goal_condition", "")
            guidance_output[group_idx] = {
                "task": task,
                "init_state": init_state,
                "goal_condition": goal_condition,
                "guidance": guidance,
            }

            with open(checkpoint_path, "w") as f:
                json.dump(guidance_output, f, indent=2)

    with open(out_json, "w") as f:
        json.dump(guidance_output, f, indent=2)
    log_info(f"Saved guidance annotations → {out_json}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
