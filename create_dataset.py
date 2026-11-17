import json
import os
import pickle
import re
from collections import defaultdict

import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from execution.registry import AVAILABLE_EXECUTORS
from execution.report import ACTION_TAGS
from utils import log_error, log_info, parse_action_line


HINT_RE = re.compile(r'\n?\[HINT_START\].*?\[HINT_END\]', re.DOTALL)
STEP_INFO_RE = re.compile(r'\n?\[STEP_INFO\].*?\[STEP_INFO_END\]', re.DOTALL)


def _strip_blocks(text: str) -> str:
    return STEP_INFO_RE.sub('', HINT_RE.sub('', text))


# The low_level controller's legal buttons.
VALID_ACTIONS = {"A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"}


def _episode_cutoff(safe_success_point, n_calls: int, safety_margin: int) -> int:
    # safe_success_point is a vlm_call_log slice index, not a frame number.
    if safe_success_point is None or pd.isna(safe_success_point):
        return n_calls
    return min(n_calls, int(safe_success_point) + safety_margin)


def select_successful(df: pd.DataFrame) -> pd.DataFrame:
    if 'success' not in df.columns:
        log_error("results.csv has no 'success' column — it was not written by practice_tasks.")
    return df[df['success'] == True]


def _save_image(img_array: np.ndarray, path: str) -> None:
    arr = img_array
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 255)).astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode='L').convert('RGB')
    elif arr.shape[-1] == 1:
        img = Image.fromarray(arr[..., 0]).convert('RGB')
    else:
        img = Image.fromarray(arr)
    img.save(path, 'JPEG')


def _load_decisions(practice_path: str) -> dict:
    path = os.path.join(practice_path, 'clean_decisions.csv')
    if not os.path.exists(path):
        log_error(
            f"clean_decisions.csv not found at {path}. Run clean_practice on this "
            "practice dir before create_dataset."
        )
    df = pd.read_csv(path)
    return {
        (str(r['group_idx']), int(r['attempt']), int(r['call_idx'])): bool(r['accept'])
        for _, r in df.iterrows()
    }


def _load_paraphrases(practice_path: str) -> dict:
    path = os.path.join(practice_path, 'paraphrases.json')
    if not os.path.exists(path):
        log_error(
            f"paraphrases.json not found at {path}. Run clean_practice on this "
            "practice dir before create_dataset."
        )
    with open(path, 'r') as f:
        return json.load(f)


def _split_episodes_by_task(episodes_by_task: dict, val_frac: float, seed: int) -> set:
    rng = np.random.default_rng(seed)
    val_episodes = set()
    for task, episodes in episodes_by_task.items():
        episodes = sorted(episodes)
        n_val = int(len(episodes) * val_frac)
        if n_val < 1:
            continue
        perm = rng.permutation(len(episodes))
        for i in perm[:n_val]:
            val_episodes.add(episodes[i])
    return val_episodes


@click.group()
def cli():
    """Dataset construction from practice output."""


@cli.command(name='create_dataset')
@click.option('--practice_path', required=True, type=str,
              help='Path to the practice output directory (contains results.csv and *.pkl).')
@click.option('--overwrite', is_flag=True, default=False,
              help='Overwrite existing output files.')
@click.option('--safety_margin', default=2, show_default=True, type=int,
              help='Extra VLM calls kept past the safe-success cutoff (see _episode_cutoff). '
                   'Should match clean_practice --safety_margin.')
@click.option('--val_frac', default=0.2, show_default=True, type=float,
              help='Fraction of each task\'s episodes held out for validation.')
@click.option('--seed', default=0, show_default=True, type=int,
              help='Seed for the per-task episode shuffle.')
def create_dataset(practice_path, overwrite, safety_margin, val_frac, seed):
    """Build train/validation VLA datasets from successful practice episodes."""
    results_csv = os.path.join(practice_path, 'results.csv')
    if not os.path.exists(results_csv):
        log_error(f"results.csv not found in {practice_path}")

    images_dir = os.path.join(practice_path, 'images')
    train_path = os.path.join(practice_path, 'train_dataset.csv')
    val_path = os.path.join(practice_path, 'validation_dataset.csv')

    if (os.path.exists(train_path) or os.path.exists(val_path)) and not overwrite:
        log_info(f"Dataset already exists in {practice_path}. Use --overwrite to regenerate.")
        return

    df = pd.read_csv(results_csv)
    successful = select_successful(df)
    log_info(f"Total episodes: {len(df)} | Successful: {len(successful)}")

    decisions = _load_decisions(practice_path)
    paraphrases = _load_paraphrases(practice_path)

    os.makedirs(images_dir, exist_ok=True)

    rows = []
    episodes_by_task = defaultdict(set)
    missing_pkls = 0
    n_rejected = 0
    n_unparseable = 0
    n_illegal = 0
    for _, row in tqdm(successful.iterrows(), total=len(successful), desc='episodes'):
        # group_idx is a string id like "10_0": never int() it, since int("10_0") == 100.
        group_idx = str(row['group_idx'])
        attempt = int(row['attempt'])
        episode_id = (group_idx, attempt)
        pkl_path = os.path.join(practice_path, f'{group_idx}_{attempt}.pkl')

        if not os.path.exists(pkl_path):
            missing_pkls += 1
            continue

        with open(pkl_path, 'rb') as f:
            vlm_call_log = pickle.load(f)

        cutoff = _episode_cutoff(row.get('safe_success_point'), len(vlm_call_log), safety_margin)

        for call_idx, record in enumerate(vlm_call_log[:cutoff]):
            if decisions.get((group_idx, attempt, call_idx)) is False:
                n_rejected += 1
                continue

            if record.tag not in ACTION_TAGS:
                continue

            # Checked before saving images so dropped rows leave no orphan files behind.
            action = parse_action_line(record.response)
            if action is None:
                n_unparseable += 1
                continue
            if action.strip().upper() not in VALID_ACTIONS:
                n_illegal += 1
                continue

            image_paths = []
            for img_idx, img_array in enumerate(record.images):
                img_fname = f'{group_idx}_{attempt}_{call_idx}_{img_idx}.jpg'
                img_path = os.path.join(images_dir, img_fname)
                _save_image(img_array, img_path)
                image_paths.append(img_path)

            episodes_by_task[row['task_string']].add(episode_id)
            rows.append({
                'episode_id': episode_id,
                'task_string': row['task_string'],
                'input': _strip_blocks(record.prompt),
                'output': record.response,
                'image': ', '.join(image_paths),
            })

    val_episodes = _split_episodes_by_task(episodes_by_task, val_frac, seed)

    fields = ['task_string', 'input', 'output', 'image']
    train_rows, val_rows = [], []
    for row in rows:
        clean_row = {k: row[k] for k in fields}
        task = row['task_string']
        task_paraphrases = paraphrases.get(task, [])
        # The last paraphrase is reserved for validation: train never uses it.
        test_paraphrase = task_paraphrases[-1] if task_paraphrases else None
        train_paraphrases = task_paraphrases[:-1]

        if row['episode_id'] in val_episodes:
            if test_paraphrase is not None:
                val_row = dict(clean_row)
                val_row['task_string'] = test_paraphrase
                val_row['input'] = row['input'].replace(task, test_paraphrase)
                val_rows.append(val_row)
            else:
                val_rows.append(clean_row)
        else:
            train_rows.append(clean_row)
            for paraphrase in train_paraphrases:
                aug = dict(clean_row)
                aug['task_string'] = paraphrase
                aug['input'] = row['input'].replace(task, paraphrase)
                train_rows.append(aug)

    pd.DataFrame(train_rows, columns=fields).to_csv(train_path, index=False)
    pd.DataFrame(val_rows, columns=fields).to_csv(val_path, index=False)

    if missing_pkls:
        log_info(f"Warning: {missing_pkls} pkl files not found (skipped).")
    log_info(f"Dropped {n_rejected} rejected calls.")
    log_info(f"Dropped {n_unparseable} calls with no parseable Action and "
             f"{n_illegal} calls with an illegal action.")
    log_info(f"Episodes: {len(val_episodes)} validation / "
             f"{sum(len(e) for e in episodes_by_task.values()) - len(val_episodes)} train.")
    log_info(f"Saved {len(train_rows)} train rows -> {train_path}")
    log_info(f"Saved {len(val_rows)} validation rows -> {val_path}")
    log_info(f"Images -> {images_dir}/")


TRAIN_NAME = 'train_dataset.csv'
VALIDATION_NAME = 'validation_dataset.csv'


def _source_label(practice_path: str) -> str:
    parts = os.path.normpath(practice_path).split(os.sep)
    if len(parts) >= 3 and parts[-3] == 'curiosity':
        return 'curiosity'
    stem = parts[-2] if len(parts) >= 2 else os.path.basename(practice_path)
    if stem.startswith('zeroshot_tasks') and stem.endswith('_attempts'):
        middle = stem[len('zeroshot_tasks'):-len('_attempts')].strip('_')
        matches = [
            name for name in AVAILABLE_EXECUTORS
            if middle == name or middle.startswith(f'{name}_')
        ]
        if matches:
            return f'zeroshot_{max(matches, key=len)}'
        return 'zeroshot'
    return stem or 'unknown'


def _load_split(practice_path: str, filename: str, label: str) -> pd.DataFrame:
    path = os.path.join(practice_path, filename)
    if not os.path.exists(path):
        log_error(
            f"{filename} not found at {path}. Run create_dataset on this practice dir first."
        )
    frame = pd.read_csv(path)
    frame['source'] = label
    return frame


def _balance_equal(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    counts = frame['source'].value_counts()
    target = int(counts.min())
    return pd.concat([
        group.sample(n=target, random_state=seed)
        for _, group in frame.groupby('source', sort=False)
    ])


@cli.command(name='merge_practices')
@click.option('--practice_paths', required=True, type=str,
              help='Comma-separated practice dirs, each holding train/validation_dataset.csv.')
@click.option('--save_path', required=True, type=str,
              help='Directory to write the merged train/validation pair into.')
@click.option('--balance', default='none', show_default=True,
              type=click.Choice(['none', 'equal']),
              help="'equal' downsamples every source to the smallest source's row count, "
                   'applied to train and validation independently.')
@click.option('--seed', default=0, show_default=True, type=int,
              help='Seed for balancing and the final shuffle.')
def merge_practices(practice_paths, save_path, balance, seed):
    """Merge the datasets of several practice dirs into one train/validation pair."""
    paths = [p.strip() for p in practice_paths.split(',') if p.strip()]
    if not paths:
        log_error('--practice_paths is empty.')
    for path in paths:
        if not os.path.isdir(path):
            log_error(f'practice path does not exist: {path}')

    train_path = os.path.join(save_path, TRAIN_NAME)
    val_path = os.path.join(save_path, VALIDATION_NAME)

    labels = [_source_label(p) for p in paths]
    if len(set(labels)) != len(labels):
        log_info(f'Warning: duplicate source labels {labels} — rows will share a source value.')

    train_parts, val_parts = [], []
    for path, label in zip(paths, labels):
        train = _load_split(path, TRAIN_NAME, label)
        validation = _load_split(path, VALIDATION_NAME, label)
        log_info(f'{label}: {len(train)} train / {len(validation)} validation  <- {path}')
        train_parts.append(train)
        val_parts.append(validation)

    merged = {}
    for split, parts in (('train', train_parts), ('validation', val_parts)):
        frame = pd.concat(parts, ignore_index=True)
        before = len(frame)
        # image is part of the dedup key: distinct frames can share an identical prompt and response.
        frame = frame.drop_duplicates(subset=['input', 'output', 'image'], keep='first')
        log_info(f'{split}: {before} rows -> {len(frame)} after dedup '
                 f'({before - len(frame)} duplicates dropped)')
        if balance == 'equal':
            frame = _balance_equal(frame, seed)
            log_info(f'{split}: balanced to {len(frame)} rows '
                     f'({frame["source"].value_counts().to_dict()})')
        merged[split] = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    os.makedirs(save_path, exist_ok=True)
    merged['train'].to_csv(train_path, index=False)
    merged['validation'].to_csv(val_path, index=False)

    log_info(f'Saved {len(merged["train"])} train rows      -> {train_path}')
    log_info(f'Saved {len(merged["validation"])} validation rows -> {val_path}')
    log_info(f'Sources: {merged["train"]["source"].value_counts().to_dict()}')


if __name__ == '__main__':
    cli()
