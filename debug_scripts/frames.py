import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    _FONT = ImageFont.load_default(size=14)
except TypeError:  # older Pillow
    _FONT = ImageFont.load_default()

SEPARATOR_WIDTH = 2


def to_pil(frame: np.ndarray) -> Image.Image:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    return Image.fromarray(array).convert("RGB")


def sample_indices(total: int, n: int) -> list[int]:
    if total <= 0:
        return []
    if total <= n:
        return list(range(total))
    return [int(round(i * (total - 1) / (n - 1))) for i in range(n)]


def action_label(high_level_action) -> str:
    # Entries are [action_class, kwargs] pairs, stored as numpy object arrays once through the replay buffer.
    try:
        pair = high_level_action
        if isinstance(pair, np.ndarray):
            pair = pair.tolist()
        if isinstance(pair, (tuple, list)) and len(pair) > 1:
            kwargs = pair[1]
            if isinstance(kwargs, dict) and "low_level_action" in kwargs:
                raw = str(kwargs["low_level_action"])
                return (
                    raw.replace("LowLevelActions.PRESS_BUTTON_", "")
                    .replace("LowLevelActions.PRESS_ARROW_", "")
                )
        return str(pair)[:12]
    except Exception:
        return ""


def strip(frames, out_path: str, labels=None, n: int = 5, overwrite: bool = False) -> str | None:
    if not overwrite and os.path.exists(out_path):
        return out_path
    frames = list(frames)
    indices = sample_indices(len(frames), n)
    if not indices:
        return None

    images = [to_pil(frames[i]) for i in indices]
    height = max(im.height for im in images)
    width = sum(im.width for im in images) + SEPARATOR_WIDTH * (len(images) - 1)

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x = 0
    for panel_i, (frame_i, image) in enumerate(zip(indices, images)):
        canvas.paste(image, (x, 0))
        draw.text((x + 3, 2), str(frame_i), fill=(255, 0, 0), font=_FONT)
        if labels is not None and frame_i < len(labels):
            text = str(labels[frame_i])
            if text:
                draw.text((x + 3, height - 16), text, fill=(255, 0, 0), font=_FONT)
        x += image.width
        if panel_i < len(images) - 1:
            draw.rectangle([x, 0, x + SEPARATOR_WIDTH - 1, height], fill=(0, 128, 255))
            x += SEPARATOR_WIDTH

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, "PNG")
    return out_path


def trajectory_strip(trajectory, out_path: str, n: int = 5, overwrite: bool = False) -> str | None:
    observations = trajectory[0]
    high_level_actions = trajectory[2] if len(trajectory) > 2 else None
    labels = None
    if high_level_actions is not None:
        labels = [action_label(a) for a in high_level_actions]
    return strip(observations, out_path, labels=labels, n=n, overwrite=overwrite)


