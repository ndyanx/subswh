import datetime
import textwrap
from pathlib import Path
from typing import List

from subtitles_generator.config import SUBTITLE_LINE_WIDTH
from subtitles_generator.core import SubtitleChunk


def format_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    if seconds is None:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def create_srt(
    subtitles_path: Path,
    chunks: List[SubtitleChunk],
    line_width: int = SUBTITLE_LINE_WIDTH,
) -> int:
    """Write chunks (with real start/end timestamps) to a .srt file.

    Returns the number of subtitle frames written.

    Differences from the previous version:
      * uses the model's real timestamps instead of `frame_index * chunk_size`,
        so timing doesn't drift on long files.
      * wraps long lines with `textwrap` instead of silently truncating
        text at 250 characters.
      * if the model didn't return an end timestamp for the last chunk
        (rare, can happen at end-of-audio), falls back to start + 2s
        instead of crashing or leaving `None` in the file.
    """
    subtitles_path = Path(subtitles_path)
    if subtitles_path.suffix != ".srt":
        subtitles_path = subtitles_path.with_suffix(".srt")

    frame_count = 0
    with open(subtitles_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            start = chunk["start"] or 0.0
            end = chunk["end"]
            if end is None or end <= start:
                end = start + 2.0

            wrapped_text = "\n".join(textwrap.wrap(chunk["text"], width=line_width)) or chunk["text"]

            frame_count += 1
            f.write(
                f"{frame_count}\n"
                f"{format_timestamp(start)} --> {format_timestamp(end)}\n"
                f"{wrapped_text}\n\n"
            )

    return frame_count
