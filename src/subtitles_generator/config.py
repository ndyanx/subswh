"""Static configuration for the package.

Kept as a plain python module (no Hydra/OmegaConf) so the package has
zero configuration-management dependencies. This avoids version
conflicts with whatever `omegaconf`/`antlr4-python3-runtime` versions
happen to be installed in the environment (e.g. Google Colab).
"""

# Whisper checkpoints on the Hugging Face Hub, keyed by "size".
MODEL_NAMES = {
    "large-distil": "distil-whisper/distil-large-v3",
    "large": "openai/whisper-large-v3",
    "medium": "openai/whisper-medium",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "tiny": "openai/whisper-tiny",
}

# Media containers that ffmpeg (used internally by the ASR pipeline)
# can demux. Video files are supported directly: ffmpeg extracts the
# audio track on the fly, so we never need to create an intermediate
# .wav file on disk.
SUPPORTED_MEDIA_FORMATS = {
    "video": [".mp4", ".avi", ".webm", ".mov", ".mkv"],
    "audio": [".wav", ".mp3", ".m4a", ".flac", ".ogg"],
}

# "auto" lets Whisper detect the spoken language itself instead of
# forcing one, which is the recommended approach for mixed/unknown
# language audio.
SUPPORTED_LANGUAGES = [
    "auto", "english", "chinese", "german", "spanish", "russian", "korean",
    "french", "japanese", "portuguese", "turkish", "polish", "catalan",
    "dutch", "arabic", "swedish", "italian", "indonesian", "hindi",
    "finnish", "vietnamese", "hebrew", "ukrainian", "greek", "malay",
    "czech", "romanian", "danish", "hungarian", "tamil", "norwegian",
    "thai", "urdu", "croatian", "bulgarian", "lithuanian", "latin",
    "maori", "malayalam", "welsh", "slovak", "telugu", "persian",
    "latvian", "bengali", "serbian", "azerbaijani", "slovenian",
    "kannada", "estonian", "macedonian", "breton", "basque", "icelandic",
    "armenian", "nepali", "mongolian", "bosnian", "kazakh", "albanian",
    "swahili", "galician", "marathi", "punjabi", "sinhala", "khmer",
    "shona", "yoruba", "somali", "afrikaans", "occitan", "georgian",
    "belarusian", "tajik", "sindhi", "gujarati", "amharic", "yiddish",
    "lao", "uzbek", "faroese", "creole", "pashto", "turkmen", "nynorsk",
    "maltese", "sanskrit", "luxembourgish", "myanmar", "tibetan",
    "tagalog", "malagasy", "assamese", "tatar", "hawaiian", "lingala",
    "hausa", "bashkir", "javanese", "sundanese",
]

# Default chunking / decoding parameters, still overridable from the CLI.
DEFAULT_CHUNK_LENGTH_S = 30
DEFAULT_BATCH_SIZE = 8
SUBTITLE_LINE_WIDTH = 42  # standard-ish subtitle line width, in characters
