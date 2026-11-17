"""
What a server instance shows. Fixed at launch; a request cannot change it.

    dev   lists train + val, opens them, free play from train/val start states, the full tracker report on
          request, save/load, resets at will
    test  opens test + test_env only, lists nothing (ids come from our harness), serves task success and subgoal
          progress and nothing else, no free play, no save/load, and one reset per session -- the one that
          starts the episode. A test episode is one attempt: a second reset would let the client try, watch
          `success` and `subgoals`, and try again, which turns the score into best-of-N.

A held-out game is unreachable on dev because its tasks are not visible and its start states are not free-play
states — GameBoy games are separate ROMs, so unlike CUSI's Android apps and websites there is nothing to undo
mid-episode.
"""
from dataclasses import dataclass

from env_server import codec, splits

KINDS = ("dev", "test")


@dataclass(frozen=True)
class Policy:
    kind: str
    splits_open: tuple
    splits_listed: tuple
    free_play: bool
    #: "full" (the whole tracker report) or "scored" (success and subgoals only). See codec.STATE_MODES.
    state_mode: str
    #: Whether a client may ask for state at all. On test the scored block is served unconditionally, because it
    #: is what the run is graded on and withholding it would not make the grade any more secret.
    state_optional: bool
    save_load: bool
    #: Whether a session may be reset after its episode has started. The first reset, which starts the episode,
    #: is always allowed.
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
