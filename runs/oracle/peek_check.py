import sys

import numpy as np
from gameboy_worlds import get_benchmark_tasks, get_test_environment

games = sys.argv[1].split(",")
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
rng = np.random.default_rng(0)
all_ok = True
for game in games:
    row = get_benchmark_tasks(game=game).iloc[0]
    env = get_test_environment(row=row, controller_variant="low_level", headless=True, save_video=True,
                               session_name=f"oracle_peek_check/{game}", max_steps=200, wait_ticks=20)
    emu = env._emulator
    ctrl = env._controller
    cands = [(cls, kw) for cls, a in zip(ctrl.ACTIONS, ctrl.actions) for kw in a.get_all_valid_parameters()]
    stats = dict(rewind_frame=0, rewind_steps=0, match=0, changed=0, total=0)
    for _ in range(n_steps):
        f0 = emu.get_current_frame()
        sc0, steps0 = emu.step_count, env.get_info()["core"]["steps"]
        peeks = [env.peek_high_level_action(cls, **kw) for cls, kw in cands]
        stats["rewind_frame"] += np.array_equal(emu.get_current_frame(), f0)
        stats["rewind_steps"] += (emu.step_count == sc0 and env.get_info()["core"]["steps"] == steps0)
        i = int(rng.integers(len(cands)))
        cls, kw = cands[i]
        env.step_high_level_action(cls, **kw)
        after = emu.get_current_frame()
        stats["match"] += np.array_equal(after, peeks[i])
        stats["changed"] += not np.array_equal(after, f0)
        stats["total"] += 1
    video_frames = getattr(emu.video_writer, "n_frames", None)
    ok = stats["rewind_frame"] == stats["rewind_steps"] == stats["match"] == stats["total"]
    all_ok &= ok
    print(f"{game}: {'OK' if ok else 'FAIL'} {stats} final_step_count={emu.step_count} video_frames={video_frames}",
          flush=True)
    env.close()
print("ALL OK" if all_ok else "SOME FAILED")
