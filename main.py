import argparse
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from subtitles_generator import config
from subtitles_generator.core import Model
from subtitles_generator.utils import create_srt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")


def validate_model_lang_combo(model_size: str, lang: str) -> None:
    """Raise ValueError if an English-only model is paired with another language."""
    if model_size in config.ENGLISH_ONLY_MODEL_SIZES and lang not in ("english", "auto"):
        raise ValueError(
            f"Model size '{model_size}' ({config.MODEL_NAMES[model_size]}) is "
            f"English-only: it accepts --lang without erroring but silently produces "
            f"English output for any other language. Use --lang english/auto with this "
            f"model, or pick a multilingual model size (large, medium, small, base, tiny) "
            f"for '{lang}'."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate subtitles (.srt) from a video or audio file."
    )
    parser.add_argument(
        "--model_size", type=str, default="medium", choices=list(config.MODEL_NAMES),
        help="Whisper model size to use.",
    )
    parser.add_argument(
        "--lang", type=str, default="auto", choices=config.SUPPORTED_LANGUAGES,
        help="Language spoken in the audio/video, or 'auto' to auto-detect.",
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Path to an audio or video file.",
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Path to the resulting .srt file (defaults to <input_file>.srt).",
    )
    parser.add_argument(
        "--chunk_length_s", type=int, default=config.DEFAULT_CHUNK_LENGTH_S,
        help="Length in seconds of each audio window (30s is Whisper's native window).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE,
        help="Number of windows transcribed together per GPU forward pass. Raise this "
             "if your GPU RAM usage is low (check Colab's resource panel); lower it if "
             "you hit an out-of-memory error.",
    )

    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file {input_file} does not exist")

    supported = config.SUPPORTED_MEDIA_FORMATS["video"] + config.SUPPORTED_MEDIA_FORMATS["audio"]
    if input_file.suffix.lower() not in supported:
        raise ValueError(
            f"Unsupported file extension '{input_file.suffix}'. Supported: {supported}"
        )

    args.input_file = input_file
    args.output_file = (
        Path(args.output_file) if args.output_file else input_file.with_suffix(".srt")
    )
    validate_model_lang_combo(args.model_size, args.lang)
    return args


def app() -> None:
    args = parse_args()

    model = Model(
        config.MODEL_NAMES[args.model_size],
        lang=args.lang,
        chunk_length_s=args.chunk_length_s,
    )

    logger.info("Generating subtitles...")
    chunks = model.transcribe(args.input_file, batch_size=args.batch_size)

    n_frames = create_srt(args.output_file, chunks)
    logger.info(f"Wrote {n_frames} subtitle frames to {args.output_file}")


if __name__ == "__main__":
    app()
