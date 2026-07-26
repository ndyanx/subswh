## Multilingual Subtitles Generator

Generates subtitles from a video or audio file and saves them as a `.srt` file, using Whisper. 99 languages supported, or `auto` to let Whisper detect the language itself.

## Get Started

### Installation

```bash
git clone https://github.com/konverner/subtitles-generator.git
cd subtitles-generator
pip install .
```

This only installs `transformers` and `accelerate`. It deliberately does **not** pin (or reinstall) `torch`: if you're on Google Colab, torch is already installed with the right CUDA build, and reinstalling it here would be more likely to break things than help. If you don't have torch yet, install it first following https://pytorch.org, or run `pip install ".[torch]"`.

You also need `ffmpeg` on your system (Colab already has it; on Ubuntu/Debian: `sudo apt install ffmpeg`).

### Running

```bash
python main.py --model_size medium --lang german --input_file /content/test.mp4
```

Arguments:

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input_file` | yes | – | path to an audio or video file |
| `--model_size` | no | `medium` | `tiny`, `base`, `small`, `medium`, `large`, `large-distil` — larger is more accurate but slower |
| `--lang` | no | `auto` | language spoken (e.g. `english`, `german`, `french`, ... see below), or `auto` to detect it |
| `--output_file` | no | `<input_file>.srt` | where to write the subtitles |
| `--chunk_length_s` | no | `30` | seconds per audio chunk fed to the model |

> ⚠️ **`large-distil` (`distil-whisper/distil-large-v3`) is English-only.** Despite accepting a `--lang` value for any language without erroring, it silently produces English output for non-English audio — this is a known limitation of the Distil-Whisper checkpoints themselves, not something specific to this tool. `main.py` refuses this combination and raises an error instead of quietly mistranscribing; if you're transcribing non-English audio, use `large`, `medium`, `small`, `base`, or `tiny` instead.

The `.srt` file is written next to the input file unless `--output_file` is given. Video files (`.mp4`, `.avi`, `.webm`, `.mov`, `.mkv`) are read directly — there's no intermediate audio-extraction step, ffmpeg handles that internally.

### Using it as a library

```python
from subtitles_generator import Model, create_srt

model = Model("openai/whisper-medium", lang="auto")
chunks = model.transcribe("video.mp4")
create_srt("video.srt", chunks)
```

### Supported languages

```
english, chinese, german, spanish, russian, korean, french, japanese,
portuguese, turkish, polish, catalan, dutch, arabic, swedish, italian,
indonesian, hindi, finnish, vietnamese, hebrew, ukrainian, greek, malay,
czech, romanian, danish, hungarian, tamil, norwegian, thai, urdu,
croatian, bulgarian, lithuanian, latin, maori, malayalam, welsh, slovak,
telugu, persian, latvian, bengali, serbian, azerbaijani, slovenian,
kannada, estonian, macedonian, breton, basque, icelandic, armenian,
nepali, mongolian, bosnian, kazakh, albanian, swahili, galician, marathi,
punjabi, sinhala, khmer, shona, yoruba, somali, afrikaans, occitan,
georgian, belarusian, tajik, sindhi, gujarati, amharic, yiddish, lao,
uzbek, faroese, creole, pashto, turkmen, nynorsk, maltese, sanskrit,
luxembourgish, myanmar, tibetan, tagalog, malagasy, assamese, tatar,
hawaiian, lingala, hausa, bashkir, javanese, sundanese
```

## What changed from the previous version

- Timestamps in the `.srt` now come from the model itself instead of a fixed `chunk_size` arithmetic — subtitles stay in sync on long videos.
- Video files are transcribed directly (ffmpeg extracts audio internally); no more writing a temporary `.wav` next to your input, and no more risk of that cleanup step deleting your original file.
- Dropped Hydra/OmegaConf/moviepy/librosa — fewer dependencies, fewer version conflicts, especially on Colab.
- Long subtitle lines are wrapped instead of truncated at 250 characters.
- Real progress bar (`tqdm`) over the audio duration while transcribing, instead of a silent wait.
- Chunking is done manually in fixed 30s windows (Whisper's native context length) instead of via the pipeline's own `chunk_length_s`, which `transformers` itself flags as experimental for Whisper — this also gets rid of that warning, plus a couple of other benign but noisy log lines that are now suppressed.
