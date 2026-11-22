"""
A global grid map assembled from screens alone.

The screen is a 10 x 9 grid of 16 px cells with the player at local (0, 0); x grows to the right
and y grows upward. After each move the new screen is placed on the global grid and merged in.
How far the player moved is read off the screen, by finding the shift along the move direction at
which the new frame best matches the last placed one; the action's own step count is not trusted.
When no shift matches clearly, the player is looked for in cells already mapped, and failing that
a new map segment is started, so a door warp or cut-scene never corrupts an existing map.
"""

from __future__ import annotations

from .registration import PX_SLACK, SCREEN_UNIT, grid_offset, grid_residual
from .model import BUILDING, CAVE, DIRECTIONS, OCCUPIED, OUTDOORS, GlobalMap, MapCell, SegmentInfo
from .env import MappingEnvironment, StepEvent
from .render import render_segment, segment_bounds
from .text import (LEGEND, ascii_window, door_lines, door_uses, entrance, entrance_text, frontier_targets, glyph,
                   sector)
from .judges import judge_merge, judge_stuck, name_segment, navigate_map
from .exploration import (FAILURE_EPISODES, explore_text, exploring, keep_exploring, prompt_block,
                          stuck_signs)
