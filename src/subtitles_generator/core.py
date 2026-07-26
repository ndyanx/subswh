import logging
import warnings
from pathlib import Path
from typing import List, Optional, TypedDict, Union

from subtitles_generator.audio import SAMPLING_RATE, iter_windows, load_audio

logger = logging.getLogger(__name__)

# A handful of known-benign notices from transformers/Whisper that aren't
# actionable for users of this package:
#  - "has been passed to `.generate()`, but it was also created in
#    `.generate()`" - logits-processor precedence notice, raised via
#    warnings.warn() in transformers/generation, NOT the HF logger, so
#    `transformers.utils.logging.set_verbosity_error()` alone doesn't
#    catch it. Needs an explicit warnings filter.
#  - "clean_up_tokenization_spaces" - BPE-vs-WordPiece tokenizer notice,
#    raised via the HF logger.
#  - "did not predict an ending timestamp" - expected on the last window
#    if audio is cut off mid-word; create_srt already has a fallback for it.
for _pattern in (
    r".*has been passed to `\.generate\(\)`.*",
    r".*clean_up_tokenization_spaces.*",
    r".*did not predict an ending timestamp.*",
):
    warnings.filterwarnings("ignore", message=_pattern)


class SubtitleChunk(TypedDict):
    text: str
    start: float
    end: Optional[float]


class Model:
    """Thin wrapper around a Hugging Face Whisper ASR pipeline.

    Instead of relying on the pipeline's built-in `chunk_length_s`
    long-form chunking (which `transformers` itself flags as "very
    experimental" for seq2seq models like Whisper, recommending
    `generate()`'s own chunking instead), we do our own fixed-window
    chunking at exactly Whisper's native 30s context length. Each
    window is transcribed as a normal, non-chunked, single-window
    inference call, so:
      * we avoid the experimental-chunking warning entirely, since we
        never ask the pipeline to chunk internally.
      * we get a real per-window progress bar (tqdm), instead of one
        opaque call that only prints a result once everything is done.
      * we can offset each window's local timestamps back into
        absolute time ourselves, in a way we fully control and can test.
    """

    def __init__(
        self,
        model_name: str,
        lang: Optional[str] = None,
        chunk_length_s: int = 30,
        device: Optional[Union[int, str]] = None,
        torch_dtype=None,
    ):
        import torch
        from transformers import pipeline
        from transformers.utils import logging as hf_logging

        # Quiets HF-logger-based notices (e.g. the BPE tokenization one).
        # The logits-processor precedence warnings are handled separately
        # above via warnings.filterwarnings(), since they come from
        # Python's warnings module, not this logger.
        hf_logging.set_verbosity_error()

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        if torch_dtype is None:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        generate_kwargs = {"task": "transcribe"}
        if lang and lang != "auto":
            generate_kwargs["language"] = lang

        if "distil-whisper" in model_name and lang not in (None, "auto", "english"):
            warnings.warn(
                f"'{model_name}' is an English-only Distil-Whisper checkpoint. It will "
                f"silently produce English output for lang='{lang}' instead of raising "
                f"an error. Use a multilingual model (e.g. openai/whisper-large-v3) for "
                f"non-English audio.",
                stacklevel=2,
            )

        self.chunk_length_s = chunk_length_s
        logger.info("Loading %s on device=%s (dtype=%s)...", model_name, device, torch_dtype)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=torch_dtype,
            device=device,
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )

    def transcribe(self, media_path: Union[str, Path]) -> List[SubtitleChunk]:
        """Transcribe an audio or video file and return timed chunks.

        Shows a tqdm progress bar over the audio's duration while it runs.
        `end` can be `None` for the very last sub-chunk of the whole file
        if the model didn't emit a closing timestamp; callers should
        handle that case (create_srt already does).
        """
        from tqdm import tqdm

        audio = load_audio(media_path, sampling_rate=SAMPLING_RATE)
        total_seconds = audio.shape[0] / SAMPLING_RATE

        chunks: List[SubtitleChunk] = []
        with tqdm(total=round(total_seconds, 1), unit="s", desc="Transcribing") as bar:
            for window, offset_s in iter_windows(audio, self.chunk_length_s, SAMPLING_RATE):
                result = self.pipe({"raw": window, "sampling_rate": SAMPLING_RATE})
                for c in result.get("chunks", []):
                    start, end = c["timestamp"]
                    text = c["text"].strip()
                    if not text:
                        continue
                    chunks.append({
                        "text": text,
                        "start": start + offset_s,
                        "end": (end + offset_s) if end is not None else None,
                    })
                bar.update(round(window.shape[0] / SAMPLING_RATE, 1))

        return chunks
