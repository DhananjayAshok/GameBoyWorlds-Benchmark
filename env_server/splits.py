import json
import os
import random
from typing import Optional

import yaml
from gameboy_worlds import get_benchmark_tasks, get_benchmark_tasks_dfs

from utils import load_parameters
from utils.log_handling import log_error, log_info

# test: held-back tasks of the training games. test_env: every task of a non-training game.
SPLITS = ("train", "val", "test", "test_env")
DEV_SPLITS = ("train", "val")
TEST_SPLITS = ("test", "test_env")

CONFIG_PATH = os.path.join("configs", "splits", "split_config.yaml")
SPLIT_PATH = os.path.join("configs", "splits", "benchmark_splits.json")

_cache: Optional[dict] = None


def task_id(*, game: str, index: int) -> str:
    return f"gb.{game}.{index:03d}"


def game_of(*, task_id: str) -> str:
    return task_id.split(".", 2)[1]


def index_of(*, task_id: str) -> int:
    return int(task_id.split(".", 2)[2])


def load_config(*, parameters: dict) -> dict:
    with open(os.path.join(parameters["project_root"], CONFIG_PATH)) as f:
        config = yaml.safe_load(f)
    known = benchmark_games()
    unknown = [g for g in config["train_games"] if g not in known]
    if unknown:
        log_error(f"split_config.yaml names games with no benchmark tasks: {unknown}. "
                  f"Available: {known}", parameters=parameters)
    if config["val_fraction"] + config["test_fraction"] >= 1.0:
        log_error("val_fraction + test_fraction must leave room for train", parameters=parameters)
    return config


def benchmark_games(*, min_tasks: int = 2) -> list:
    # min_tasks drops one-row placeholder games, which cannot be split.
    counts = {}
    for df in get_benchmark_tasks_dfs().values():
        for game, n in df["game"].value_counts().items():
            counts[game] = counts.get(game, 0) + int(n)
    return sorted(g for g, n in counts.items() if n >= min_tasks)


def generate(*, parameters: dict = None, write: bool = True) -> dict:
    parameters = load_parameters(parameters)
    config = load_config(parameters=parameters)
    rng = random.Random(config["seed"])
    splits = {}
    for game in benchmark_games():
        rows = get_benchmark_tasks(game=game).reset_index(drop=True)
        if game not in config["train_games"]:
            for i, row in rows.iterrows():
                splits[task_id(game=game, index=i)] = {"split": "test_env", "text": row["task"]}
            continue
        # Tasks sharing an init_state stay in one split, so free play never starts where a test task does.
        by_state = {}
        for i, row in rows.iterrows():
            by_state.setdefault(row["init_state"], []).append(i)
        groups = sorted(by_state.values(), key=lambda g: g[0])
        rng.shuffle(groups)
        n_test = int(round(len(rows) * config["test_fraction"]))
        n_val = int(round(len(rows) * config["val_fraction"]))
        assigned, taken = {}, 0
        for group in groups:
            if taken < n_test:
                split = "test"
            elif taken < n_test + n_val:
                split = "val"
            else:
                split = "train"
            for i in group:
                assigned[i] = split
            taken += len(group)
        for i, row in rows.iterrows():
            splits[task_id(game=game, index=i)] = {"split": assigned[i], "text": row["task"]}
    if write:
        path = os.path.join(parameters["project_root"], SPLIT_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(splits, f, indent=1, sort_keys=True)
        counts = {s: sum(1 for v in splits.values() if v["split"] == s) for s in SPLITS}
        log_info(f"wrote {len(splits)} tasks to {SPLIT_PATH}: {counts}", parameters=parameters)
    return splits


def load_splits(*, parameters: dict = None, check: bool = True) -> dict:
    global _cache
    if _cache is not None:
        return _cache
    parameters = load_parameters(parameters)
    path = os.path.join(parameters["project_root"], SPLIT_PATH)
    if not os.path.exists(path):
        log_error(f"{SPLIT_PATH} does not exist; generate it with "
                  f"python run_env_server.py generate_splits", parameters=parameters)
    with open(path) as f:
        splits = json.load(f)
    for entry in splits.values():
        if entry["split"] not in SPLITS:
            log_error(f"unknown split {entry['split']!r} in {SPLIT_PATH}", parameters=parameters)
    if check:
        verify(splits=splits, parameters=parameters)
    _cache = splits
    return splits


def verify(*, splits: dict, parameters: dict) -> None:
    for game in sorted({game_of(task_id=t) for t in splits}):
        rows = get_benchmark_tasks(game=game).reset_index(drop=True)
        ids = [t for t in splits if game_of(task_id=t) == game]
        if len(ids) != len(rows):
            log_error(f"{game}: the split file has {len(ids)} tasks, the benchmark has {len(rows)}. "
                      f"Regenerate the split file.", parameters=parameters)
        for t in ids:
            i = index_of(task_id=t)
            if splits[t]["text"] != rows.iloc[i]["task"]:
                log_error(f"{t}: the split file says {splits[t]['text']!r} but the benchmark says "
                          f"{rows.iloc[i]['task']!r}. The benchmark changed under the split file; regenerate it.",
                          parameters=parameters)


def row_of(*, task_id: str):
    rows = get_benchmark_tasks(game=game_of(task_id=task_id)).reset_index(drop=True)
    return rows.iloc[index_of(task_id=task_id)]


def listed(*, splits_wanted: tuple, parameters: dict = None) -> list:
    parameters = load_parameters(parameters)
    splits = load_splits(parameters=parameters)
    return [{"task_id": t, "split": v["split"], "text": v["text"]}
            for t, v in sorted(splits.items()) if v["split"] in splits_wanted]


def free_play_states(*, parameters: dict = None) -> dict:
    parameters = load_parameters(parameters)
    splits = load_splits(parameters=parameters)
    config = load_config(parameters=parameters)
    out = {}
    for game in config["train_games"]:
        rows = get_benchmark_tasks(game=game).reset_index(drop=True)
        states = set()
        for i, row in rows.iterrows():
            if splits[task_id(game=game, index=i)]["split"] not in DEV_SPLITS:
                continue
            if not bool(row["can_train_from_init_state"]):
                continue
            states.add(row["init_state"])
        if states:
            out[game] = sorted(states)
    return out
