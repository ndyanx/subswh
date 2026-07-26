from pathlib import Path

from subtitles_generator.utils import create_srt, format_timestamp


def test_format_timestamp_basic():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(61.5) == "00:01:01,500"
    assert format_timestamp(3661.001) == "01:01:01,001"


def test_format_timestamp_none_defaults_to_zero():
    assert format_timestamp(None) == "00:00:00,000"


def test_create_srt_writes_expected_frames(tmp_path: Path):
    chunks = [
        {"text": "Hello world", "start": 0.0, "end": 2.0},
        {"text": "Second line", "start": 2.0, "end": 4.5},
    ]
    out_path = tmp_path / "out.srt"

    n_frames = create_srt(out_path, chunks)

    assert n_frames == 2
    content = out_path.read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:02,000\nHello world" in content
    assert "2\n00:00:02,000 --> 00:00:04,500\nSecond line" in content


def test_create_srt_falls_back_when_end_missing(tmp_path: Path):
    chunks = [{"text": "No end timestamp", "start": 5.0, "end": None}]
    out_path = tmp_path / "out.srt"

    create_srt(out_path, chunks)

    content = out_path.read_text(encoding="utf-8")
    # falls back to start + 2s instead of crashing or writing "None"
    assert "00:00:05,000 --> 00:00:07,000" in content


def test_create_srt_wraps_long_lines(tmp_path: Path):
    long_text = " ".join(["word"] * 40)  # way over line_width
    chunks = [{"text": long_text, "start": 0.0, "end": 2.0}]
    out_path = tmp_path / "out.srt"

    create_srt(out_path, chunks, line_width=42)

    content = out_path.read_text(encoding="utf-8")
    lines = content.strip().split("\n")[2:]  # skip index + timestamp lines
    assert all(len(line) <= 42 for line in lines)
    # nothing got silently truncated/dropped
    assert content.count("word") == 40
