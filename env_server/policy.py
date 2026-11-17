from dataclasses import dataclass

from env_server import codec, splits

KINDS = ("dev", "test")


@dataclass(frozen=True)
class Policy:
    """What a server instance shows. Fixed at launch; no request can change it."""
    kind: str
    splits_open: tuple
    splits_listed: tuple
    free_play: bool
    state_mode: str
    state_optional: bool
    save_load: bool
    # A test episode is one attempt: a second reset would turn the score into best-of-N.
    reset_again: bool

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
        raise ValueError(f"unknown server kind {kind!r}; one of {KINDS}")

    def __post_init__(self) -> None:
        if self.state_mode not in codec.STATE_MODES:
            raise ValueError(f"unknown state mode {self.state_mode!r}; one of {codec.STATE_MODES}")

    def visible(self, *, task_id: str, parameters: dict) -> bool:
        entry = splits.load_splits(parameters=parameters).get(task_id)
        return entry is not None and entry["split"] in self.splits_open
