# Videos are re-encoded to H.264/yuv420p because browsers cannot decode the emulator's mp4v recordings.

import base64
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile

import click
import markdown

from python_scripts.common import MAYBE_NONE
from utils.log_handling import log_error, log_info, log_warn

IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)\)")
HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)
VIDEO_SUFFIXES = (".mp4", ".webm", ".gif")

CSS = """
:root { --bg:#ffffff; --fg:#1a1a1a; --muted:#666; --line:#e3e3e3; --code:#f6f6f6; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg:#141414; --fg:#e8e8e8; --muted:#9a9a9a;
                                    --line:#2e2e2e; --code:#1d1d1d; }
}
html { background: var(--bg); }
body { background: var(--bg); color: var(--fg); margin: 0 auto; padding: 2rem 1.5rem 6rem;
       max-width: 60rem; font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI",
       Roboto, sans-serif; }
h1 { font-size: 1.7rem; border-bottom: 2px solid var(--line); padding-bottom: .4rem; }
h2 { font-size: 1.3rem; margin-top: 2.5rem; border-bottom: 1px solid var(--line);
     padding-bottom: .3rem; }
h3 { font-size: 1.05rem; margin-top: 1.8rem; color: var(--muted); }
img { image-rendering: pixelated; max-width: 100%; border: 1px solid var(--line);
      border-radius: 3px; }
video { image-rendering: pixelated; max-width: 100%; display: block; margin: .4rem 0;
        border: 1px solid var(--line); border-radius: 3px; }
pre { background: var(--code); padding: .8rem 1rem; border-radius: 4px; overflow-x: auto;
      font-size: 13px; }
code { background: var(--code); padding: .1rem .3rem; border-radius: 3px; font-size: 13px; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; display: block; overflow-x: auto; max-width: 100%; }
th, td { border: 1px solid var(--line); padding: .35rem .6rem; text-align: left; }
details { margin: .5rem 0; }
summary { cursor: pointer; color: var(--muted); }
blockquote { border-left: 3px solid var(--line); margin-left: 0; padding-left: 1rem;
             color: var(--muted); }
"""


def find_ffmpeg():
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:  # noqa: BLE001 - absence is a normal outcome, not an error
        pass
    return shutil.which("ffmpeg")


def b64(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def data_uri(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as handle:
        return b64(handle.read(), mime)


def transcoded_uri(path: str, ffmpeg: str, scale: int):
    # The scale filter rounds to even dimensions because H.264 4:2:0 cannot encode odd ones.
    handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    handle.close()
    try:
        chain = (f"scale=iw*{scale}:ih*{scale}:flags=neighbor,"
                 "scale=trunc(iw/2)*2:trunc(ih/2)*2") if scale > 1 else \
                "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        result = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", path,
             "-vf", chain, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", handle.name],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or os.path.getsize(handle.name) == 0:
            log_warn(f"  ffmpeg failed on {os.path.basename(path)}: "
                     f"{result.stderr.strip()[:160]}")
            return None
        with open(handle.name, "rb") as done:
            return b64(done.read(), "video/mp4")
    finally:
        os.unlink(handle.name)


def select_tasks(text: str, wanted: list) -> str:
    marks = list(HEADING_RE.finditer(text))
    if not marks:
        return text
    kept = [text[: marks[0].start()]]
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        if any(w.lower() in mark.group(1).lower() for w in wanted):
            kept.append(text[mark.start():end])
    return "".join(kept)


@click.command(name="embed_markdown")
@click.argument("report")
@click.option("--out", default=None,
              help="Output HTML path. Defaults to the report with a .html suffix.")
@click.option("--tasks", default=None, type=MAYBE_NONE,
              help="Comma-separated substrings; keep only matching '## ' sections.")
@click.option("--video_scale", "--video-scale", "video_scale", type=int, default=3,
              show_default=True,
              help="Nearest-neighbour upscale for embedded clips (1 = none).")
@click.option("--no_transcode", "--no-transcode", "no_transcode", is_flag=True, default=False,
              help="Embed video bytes as recorded. They will not play in a browser.")
def embed_markdown_cmd(report, out, tasks, video_scale, no_transcode) -> int:
    """Inline a markdown report's local images and videos into one standalone HTML file."""
    if not os.path.exists(report):
        log_error(f"no such report: {report}")
    base_dir = os.path.dirname(os.path.abspath(report))
    out_path = out or os.path.splitext(report)[0] + ".html"

    ffmpeg = None if no_transcode else find_ffmpeg()
    if not no_transcode and ffmpeg is None:
        log_warn("no ffmpeg found. Videos are embedded as recorded (mp4v), which no "
                 "browser will play.")

    text = open(report).read()
    if tasks:
        text = select_tasks(text, [t.strip() for t in tasks.split(",") if t.strip()])

    cache: dict = {}
    stats = {"img": 0, "video": 0, "transcoded": 0, "raw_video": 0,
             "missing": 0, "skipped": 0}

    def resolve(target: str, is_video: bool):
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
            stats["skipped"] += 1
            return None
        full = os.path.realpath(os.path.join(base_dir, target))
        if not os.path.isfile(full):
            stats["missing"] += 1
            return None
        if full not in cache:
            uri = None
            if is_video and ffmpeg is not None:
                uri = transcoded_uri(full, ffmpeg, video_scale)
                if uri is not None:
                    stats["transcoded"] += 1
            if uri is None:
                uri = data_uri(full)
                if is_video:
                    stats["raw_video"] += 1
            cache[full] = uri
        return cache[full]

    def sub_image(match):
        alt, target = match.group(1), match.group(2)
        uri = resolve(target, is_video=False)
        if uri is None:
            return match.group(0)
        stats["img"] += 1
        return f'<img loading="lazy" alt="{alt}" src="{uri}">'

    def sub_link(match):
        target = match.group(2)
        if not target.lower().endswith(VIDEO_SUFFIXES):
            return match.group(0)
        uri = resolve(target, is_video=True)
        if uri is None:
            return match.group(0)
        stats["video"] += 1
        # A bare <video>, not <figure>: these links sit inside a <p>, where <figure> is invalid.
        return f'<video controls preload="none" src="{uri}"></video>'

    text = IMAGE_RE.sub(sub_image, text)
    text = LINK_RE.sub(sub_link, text)

    body = markdown.markdown(
        text,
        # md_in_html so fenced code inside <details> blocks still renders as code.
        extensions=["fenced_code", "tables", "md_in_html", "sane_lists"],
    )
    title = os.path.basename(os.path.splitext(out_path)[0])
    html = (f"<!doctype html>\n<html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{title}</title><style>{CSS}</style></head><body>\n{body}\n</body></html>\n")

    with open(out_path, "w") as handle:
        handle.write(html)

    size = os.path.getsize(out_path) / 1e6
    log_info(f"wrote {out_path}  ({size:.1f} MB)")
    log_info(f"  embedded {stats['img']} image(s), {stats['video']} video(s) "
             f"[{stats['transcoded']} transcoded to H.264, {stats['raw_video']} left as recorded]")
    log_info(f"  {stats['missing']} unresolved, {stats['skipped']} external left alone")
    return 0
