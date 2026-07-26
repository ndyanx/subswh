"""Audio decoding helpers.

We shell out to the system `ffmpeg` binary directly (same approach
`transformers` uses internally) instead of adding moviepy/librosa as
dependencies. This works for both audio and video containers: ffmpeg
demuxes the audio track from video automatically.
"""

import subprocess
from pathlib import Path
from typing import Iterator, Tuple, Union

import numpy as np

SAMPLING_RATE = 16000  # Whisper's expected input sampling rate


def load_audio(path: Union[str, Path], sampling_rate: int = SAMPLING_RATE) -> np.ndarray:
    """Decode any ffmpeg-readable audio/video file to mono float32 PCM."""
    command = [
        "ffmpeg",
        "-i", str(path),
        "-ac", "1",
        "-ar", str(sampling_rate),
        "-f", "f32le",
        "-hide_banner",
        "-loglevel", "error",
        "-",
    ]
    process = subprocess.run(command, capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to read '{path}'. Is ffmpeg installed and is this a "
            f"valid audio/video file?\n{process.stderr.decode(errors='ignore')}"
        )
    return np.frombuffer(process.stdout, dtype=np.float32)


def iter_windows(
    audio: np.ndarray,
    window_s: int,
    sampling_rate: int = SAMPLING_RATE,
) -> Iterator[Tuple[np.ndarray, float]]:
    """Split audio into fixed-size windows.

    Yields (window_array, offset_seconds) pairs. `offset_seconds` is how
    far into the full audio this window starts, so callers can shift the
    window-relative timestamps returned by Whisper back into absolute time.
    """
    window_size = window_s * sampling_rate
    n_samples = audio.shape[0]
    for start in range(0, n_samples, window_size):
        end = min(start + window_size, n_samples)
        yield audio[start:end], start / sampling_rate
