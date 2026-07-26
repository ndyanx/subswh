import logging
from pathlib import Path
from typing import List, Optional, TypedDict, Union

logger = logging.getLogger(__name__)


class SubtitleChunk(TypedDict):
    text: str
    start: float
    end: Optional[float]


class Model:
    """Thin wrapper around the Hugging Face `automatic-speech-recognition`
    pipeline, configured for long-form transcription with real timestamps.

    Compared to the previous implementation this:
      * lets `transformers` handle chunking + batching internally
        (`chunk_length_s`) instead of a hand-rolled `np.array_split`
        that could cut a word in half at chunk boundaries.
      * returns real per-segment timestamps from the model instead of
        a fixed `timer += interval_size` approximation, so subtitles
        stay in sync on long videos.
      * accepts video files directly. The pipeline reads audio via
        ffmpeg internally, which demuxes the audio track from any
        container ffmpeg supports (mp4, webm, avi, mov, mkv, ...), so
        we never need to extract/write an intermediate .wav file.
      * uses `model.generate(language=..., task=...)` semantics via
        `generate_kwargs`, which is the current, non-deprecated way to
        pin language/task (the old `forced_decoder_ids` approach is
        deprecated in recent `transformers` versions).
    """

    def __init__(
        self,
        model_name: str,
        lang: Optional[str] = None,
        chunk_length_s: int = 30,
        batch_size: int = 8,
        device: Optional[Union[int, str]] = None,
        torch_dtype=None,
    ):
        import torch
        from transformers import pipeline

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        if torch_dtype is None:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        generate_kwargs = {"task": "transcribe"}
        if lang and lang != "auto":
            generate_kwargs["language"] = lang

        logger.info("Loading %s on device=%s (dtype=%s)...", model_name, device, torch_dtype)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )

    def transcribe(self, media_path: Union[str, Path]) -> List[SubtitleChunk]:
        """Transcribe an audio or video file and return timed chunks.

        `end` can be `None` for the very last chunk if the model didn't
        emit a closing timestamp (happens occasionally with very short
        trailing audio); callers should handle that case.
        """
        result = self.pipe(str(media_path))
        chunks: List[SubtitleChunk] = []
        for c in result.get("chunks", []):
            start, end = c["timestamp"]
            text = c["text"].strip()
            if not text:
                continue
            chunks.append({"text": text, "start": start, "end": end})
        return chunks
