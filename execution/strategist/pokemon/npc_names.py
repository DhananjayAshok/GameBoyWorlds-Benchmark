import random
import re
from typing import Iterable, Set

ADJECTIVES = (
    "Amber", "Ashen", "Azure", "Beige", "Birch", "Bronze", "Cedar", "Cobalt", "Copper", "Coral", "Crimson", "Dusky",
    "Ebony", "Emerald", "Fawn", "Flint", "Golden", "Hazel", "Ivory", "Jade", "Khaki", "Lilac", "Linen",
    "Maple", "Maroon", "Mint", "Mossy", "Navy", "Oaken", "Olive", "Onyx", "Opal", "Pearl", "Plum", "Quartz",
    "Rosy", "Rusty", "Sage", "Sandy", "Scarlet", "Silver", "Slate", "Smoky", "Tawny", "Teal", "Topaz",
    "Umber", "Velvet", "Violet", "Walnut", "Willow", "Argent", "Auburn", "Cinder", "Cream", "Dapple", "Ginger",
    "Heather", "Juniper", "Lemon", "Marble", "Nutmeg", "Peach", "Russet", "Sepia",
)

NOUNS = (
    "Heron", "Finch", "Wren", "Sparrow", "Falcon", "Plover", "Swift", "Lark", "Robin", "Thrush", "Crane", "Egret",
    "Ibis", "Kestrel", "Magpie", "Osprey", "Petrel", "Quail", "Raven", "Starling", "Tern", "Warbler", "Jay",
    "Linnet", "Pipit", "Rook", "Siskin", "Teal", "Dove", "Owl", "Birch", "Cedar", "Fir", "Hazel", "Larch",
    "Maple", "Pine", "Rowan", "Spruce", "Willow", "Alder", "Aspen", "Beech", "Cypress", "Hickory", "Laurel",
    "Poplar", "Yew", "Agate", "Basalt", "Beryl", "Garnet", "Granite", "Jasper", "Marl", "Pebble", "Shale",
    "Cobble", "Gravel", "Obsidian", "Pumice", "Tuff", "Fennel", "Clover", "Thistle", "Bracken", "Sorrel",
)


def norm(name: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", name.lower()).split())


def draw_name(game: str, run_name: str, npc_id: str, taken: Iterable[str]) -> str:
    used: Set[str] = {norm(name) for name in taken if name}
    rng = random.Random(f"{game}:{run_name}:{npc_id}")
    for _ in range(200):
        candidate = f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}"
        if norm(candidate) not in used:
            return candidate
    base, k = f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}", 2
    while norm(f"{base} {k}") in used:
        k += 1
    return f"{base} {k}"
