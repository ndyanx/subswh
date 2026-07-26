import numpy as np

from subtitles_generator.audio import iter_windows


def test_iter_windows_splits_and_offsets_correctly():
    sr = 16000
    # 75 seconds of audio -> with 30s windows: [0-30), [30-60), [60-75)
    audio = np.zeros(75 * sr, dtype=np.float32)

    windows = list(iter_windows(audio, window_s=30, sampling_rate=sr))

    assert len(windows) == 3
    assert [offset for _, offset in windows] == [0.0, 30.0, 60.0]
    assert [w.shape[0] for w, _ in windows] == [30 * sr, 30 * sr, 15 * sr]


def test_iter_windows_exact_multiple():
    sr = 16000
    audio = np.zeros(60 * sr, dtype=np.float32)

    windows = list(iter_windows(audio, window_s=30, sampling_rate=sr))

    assert len(windows) == 2
    assert [offset for _, offset in windows] == [0.0, 30.0]


def test_iter_windows_shorter_than_one_window():
    sr = 16000
    audio = np.zeros(5 * sr, dtype=np.float32)

    windows = list(iter_windows(audio, window_s=30, sampling_rate=sr))

    assert len(windows) == 1
    assert windows[0][1] == 0.0
    assert windows[0][0].shape[0] == 5 * sr
