import numpy as np

from subtitles_generator.core import Model


class _FakePipe:
    """Mimics the HF pipeline's generator+batch_size interface: consumes
    an iterable of inputs and yields one result per input, in order,
    regardless of how batch_size groups them internally."""

    def __call__(self, stream, batch_size):
        for i, _item in enumerate(stream):
            yield {
                "chunks": [
                    {"text": f"chunk {i}", "timestamp": (0.0, 1.0)},
                ]
            }


def test_transcribe_offsets_match_windows_under_batching(monkeypatch):
    model = Model.__new__(Model)  # skip __init__ (no real model load)
    model.chunk_length_s = 30
    model.pipe = _FakePipe()

    sr = 16000
    audio = np.zeros(75 * sr, dtype=np.float32)  # -> 3 windows: 0s, 30s, 60s
    monkeypatch.setattr("subtitles_generator.core.load_audio", lambda path, sampling_rate: audio)

    chunks = model.transcribe("fake_path.wav", batch_size=8)

    assert [c["text"] for c in chunks] == ["chunk 0", "chunk 1", "chunk 2"]
    assert [c["start"] for c in chunks] == [0.0, 30.0, 60.0]
    assert [c["end"] for c in chunks] == [1.0, 31.0, 61.0]
