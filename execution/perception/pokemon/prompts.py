from typing import Optional, Tuple

TILE_TAGS = (
    "ground", "ground decoration", "ledge", "obstacle", "entrance", "water", "tall grass",
    "interactable object", "item", "npc", "sign", "staircase", "elevator",
    "special ground tile", "unknown",
)

CLASSIFY_PROMPT ="""You are looking at a screen from a Game Boy Pokemon game. The screen is divided into a grid of 16x16 pixel cells.

You are given two images:
1. The full game screen, with one grid cell outlined by a red box.
2. The contents of that grid cell on its own.

Classify what is in the outlined cell into exactly one of these categories:
- Ground: plain navigable ground the player can walk on, including ground with a repeating texture
- Ground Decoration: walkable ground with small decorations on it, such as flowers or little tufts that look like tiny Digletts
- Ledge: a ledge the player can jump down from one side
- Obstacle: a boundary or obstacle that blocks movement and cannot be interacted with (trees, rocks, walls, fences, buildings)
- Entrance: a door or entrance the player can walk into to enter or leave a building, cave or another area, including doorways, cave mouths, gates between routes, and the exit mats on the floor at the edge of a room
- Water
- Tall Grass: tall grass where wild Pokemon can appear
- Interactable Object: an object the player can interact with (machines, computers, bookshelves, etc.)
- Item: an item lying on the ground that the player can pick up. An item on the ground always looks like a Poke Ball
- NPC: a non-player character
- Sign: a sign the player can read
- Staircase: stairs, a ladder, or a hole in the floor that takes the player up or down to another floor or level
- Elevator: an elevator, or the panel or floor of an elevator, that moves the player between floors
- Special Ground Tile: a tile the player can walk on that may move them around, such as arrow or spinner tiles that push the player, ice floors the player slides across, or teleportation pads
- Unknown: none of the above, or you cannot tell
[PLAYER_NOTE]
If the category is NPC, also give a short name or description of the NPC and where it is on the screen.
If the category is Sign, also give where it is on the screen.

Respond in exactly this format:
Reasoning: <brief description of what is in the cell and its surroundings>
Category: <one category from the list>
Description: <a few words naming the whole object or thing this cell is part of, specific enough to tell it apart from similar things. Name the whole object, not which part of it this cell shows: write "wooden signpost", never "top-left corner of a signpost". Other examples: "Poke Ball on the floor", "door of the Poke Mart", "Pokemon Center nurse">
Name: <NPC name or description, or None>
Location: <where the NPC or sign is on the screen, or None>
[STOP]"""

TILE_CLASSIFY_PROMPT = (
    CLASSIFY_PROMPT.replace("a grid of 16x16 pixel cells", "a grid of 8x8 pixel tiles")
    .replace("one grid cell outlined", "one tile outlined")
    .replace("that grid cell", "that tile")
    .replace("in the outlined cell", "in the outlined tile")
    .replace("what is in the cell", "what is in the tile")
    .replace("this cell", "this tile")
)

TILE_IN_CELL_PROMPT = TILE_CLASSIFY_PROMPT.replace(
    "You are given two images:\n1. The full game screen, with one tile outlined by a red box.\n2. The contents of that tile on its own.",
    "You are given three images:\n1. The full game screen, with one tile outlined by a red box.\n2. The contents of that tile on its own.\n"
    "3. The 16x16 pixel grid cell that contains the tile. The tile is the [QUARTER] quarter of this cell.",
)

PLAYER_NOTE = """
Note: the player character stands in the grid cell [DIRECTION] this one. The player sprite is drawn slightly larger than its own cell, so part of it (for example the top of the player's cap) may spill into the [SIDE] of this cell. Ignore any part of the player sprite: it is not an NPC or an object, and you should classify only what is underneath or around it.
"""

PLAYER_SIDES = {
    (0, 1): ("directly below", "bottom edge"),
    (0, -1): ("directly above", "top edge"),
    (1, 0): ("directly to the left of", "left edge"),
    (-1, 0): ("directly to the right of", "right edge"),
    (1, 1): ("diagonally below and to the left of", "bottom-left corner"),
    (-1, 1): ("diagonally below and to the right of", "bottom-right corner"),
    (1, -1): ("diagonally above and to the left of", "top-left corner"),
    (-1, -1): ("diagonally above and to the right of", "top-right corner"),
}


def player_note(cell: Tuple[int, int]) -> Optional[str]:
    if cell not in PLAYER_SIDES:
        return None
    direction, side = PLAYER_SIDES[cell]
    return PLAYER_NOTE.replace("[DIRECTION]", direction).replace("[SIDE]", side)


def fill_player_note(prompt: str, cell: Tuple[int, int]) -> str:
    return prompt.replace("[PLAYER_NOTE]", player_note(cell) or "")
