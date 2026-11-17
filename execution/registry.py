from execution.executors import Executor
from execution.executors.executor import make_executor_class
from execution.executors.policies import (AVAILABLE_ACTION_POLICIES,
                                          AVAILABLE_HISTORY_POLICIES)
from execution.executors.world_model import (WORLD_MODEL_EXECUTOR, WorldModelExecutor,
                                             make_world_model_executor_class)
from execution.supervisors import (DummySupervisor, InfoSubgoalSupervisor,
                                   RevisingSupervisor, SubgoalSupervisor, Supervisor)

AVAILABLE_EXECUTORS: dict[str, type[Executor]] = {
    f"{action}_{history}": make_executor_class(action, history)
    for action in AVAILABLE_ACTION_POLICIES
    for history in AVAILABLE_HISTORY_POLICIES
}
""" Registry of available executors. Keys are ``<action>_<history>``, values the classes.
"""

# Names a family, not a runnable arm: run_benchmark.py resolves it per run name.
AVAILABLE_EXECUTORS[WORLD_MODEL_EXECUTOR] = WorldModelExecutor

KNOWLEDGE_MODES = ("retrieval", "parametric")


def make_info_subgoal_class(knowledge_mode: str) -> type:
    name = f"info_subgoal_{knowledge_mode}"
    return type(name, (InfoSubgoalSupervisor,), {
        "KNOWLEDGE_MODE": knowledge_mode,
        "__doc__": f"Plan arm, {knowledge_mode} knowledge.",
    })


AVAILABLE_SUPERVISORS: dict[str, type[Supervisor]] = {
    "dummy": DummySupervisor,
    "revision": RevisingSupervisor,
    "subgoal": SubgoalSupervisor,
    **{f"info_subgoal_{mode}": make_info_subgoal_class(mode)
       for mode in KNOWLEDGE_MODES},
}
""" Registry of available supervisors.

The key is the full identity of an arm, and is what names its benchmark CSV and its emulator
session directory (see ``python_scripts.paths.benchmark_stem``). ``info_subgoal`` is not a key
on its own — it is always qualified by a knowledge mode, because the unqualified form does not
describe a runnable experiment.

Note ``dummy`` is the control arm, whose ``run_benchmark.py`` subcommand word is ``baseline``.
"""
