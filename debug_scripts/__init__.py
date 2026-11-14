"""
Debug/diagnostic commands for the GameBoyRL pipeline. Registered on the click group in
debug.py, mirroring vlm.py / vlm_scripts.

Every command is read-only over artifacts the pipeline is guaranteed to have written, and
raises (naming the producing script) when a required artifact is absent. Nothing here
reads logs/, wandb or slurm.

Two commands need more than saved artifacts: `zeroshot` opens an emulator for each
init_state's first frame, and `compare` opens one and *does* take steps, replaying each
episode's recorded action sequence to reconstruct frames. Neither constructs a VLM.
"""

from debug_scripts.check_paths import check_paths
from debug_scripts.curiosity import debug_curiosity
from debug_scripts.infer import debug_infer
from debug_scripts.zeroshot import debug_zeroshot
from debug_scripts.attempt import debug_attempt
from debug_scripts.benchmark import debug_benchmark
from debug_scripts.compare import debug_compare
from debug_scripts.info import debug_info
from debug_scripts.retrieval import debug_retrieval
from debug_scripts.practice import debug_practice
from debug_scripts.playthrough import debug_playthrough

__all__ = [
    "check_paths",
    "debug_curiosity",
    "debug_infer",
    "debug_zeroshot",
    "debug_attempt",
    "debug_benchmark",
    "debug_compare",
    "debug_info",
    "debug_retrieval",
    "debug_practice",
    "debug_playthrough",
]
