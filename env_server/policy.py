from dataclasses import dataclass

from env_server import codec, splits

KINDS = ("dev", "test", "playthrough", "train")

# run_benchmark.py's --max_steps: the budget a benchmark task is scored against.
TASK_MAX_STEPS = 200
# GameBoyWorlds' gameboy_hard_max_steps: the emulator clamps anything larger.
PLAYTHROUGH_MAX_STEPS = 1_000_000


@dataclass(frozen=True)
class Policy:
    kind: str
    splits_open: tuple
    splits_listed: tuple
    free_play: bool
    state_mode: str
    state_optional: bool
    save_load: bool
    # A test episode is one attempt: a second reset would turn the score into best-of-N.
    reset_again: bool
    max_steps: int = TASK_MAX_STEPS
    # A playthrough session runs for days and a million steps: it is never reaped for being idle, its
    # recording is a video rather than a PNG per step, and it can be checkpointed to disk.
    long_running: bool = False
    # GameBoyWorlds controller variant: "low_level" takes button names, "state_wise" takes the game's
    # high-level action strings, which every observation lists under obs.actions.
    controller: str = "low_level"
    transitions: bool = False
    train_games_only: bool = False
    named_saves: bool = False
    opening_replaces: bool = False

    @classmethod
    def of(cls, *, kind: str) -> "Policy":
        if kind == "dev":
            return cls(kind=kind, splits_open=splits.DEV_SPLITS, splits_listed=splits.DEV_SPLITS,
                       free_play=True, state_mode="full", state_optional=True, save_load=True,
                       reset_again=True)
        if kind == "test":
            return cls(kind=kind, splits_open=splits.TEST_SPLITS, splits_listed=(),
                       free_play=False, state_mode="scored", state_optional=False, save_load=False,
                       reset_again=False)
        if kind == "playthrough":
            return cls(kind=kind, splits_open=splits.PLAYTHROUGH_SPLITS,
                       splits_listed=splits.PLAYTHROUGH_SPLITS, free_play=False, state_mode="scored",
                       state_optional=False, save_load=False, reset_again=False,
                       max_steps=PLAYTHROUGH_MAX_STEPS, long_running=True, controller="state_wise",
                       transitions=True)
        if kind == "train":
            return cls(kind=kind, splits_open=splits.PLAYTHROUGH_SPLITS,
                       splits_listed=splits.PLAYTHROUGH_SPLITS, free_play=False, state_mode="full",
                       state_optional=True, save_load=True, reset_again=True,
                       max_steps=PLAYTHROUGH_MAX_STEPS, long_running=True, controller="state_wise",
                       transitions=True, train_games_only=True, named_saves=True, opening_replaces=True)
        raise ValueError(f"unknown server kind {kind!r}; one of {KINDS}")

    def __post_init__(self) -> None:
        if self.state_mode not in codec.STATE_MODES:
            raise ValueError(f"unknown state mode {self.state_mode!r}; one of {codec.STATE_MODES}")

    def visible(self, *, task_id: str, parameters: dict) -> bool:
        entry = splits.load_splits(parameters=parameters).get(task_id)
        if entry is None or entry["split"] not in self.splits_open:
            return False
        return not self.train_games_only or \
            splits.game_of(task_id=task_id) in splits.train_sandbox_games(parameters=parameters)
