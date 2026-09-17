from __future__ import annotations

import numpy as np

from open_eeg_synth.artifacts.patterns import load_eog_patterns

HEOG_SD_MAX = 0.45  # measured 0.42 over 24 of 30 subjects: the heog map varies more than blink


def _idx(names, label):
    return [str(n) for n in names].index(label)


def test_blink_map_shape_and_orientation():
    f = load_eog_patterns()
    names = f["channel_names"]
    assert len(names) == 64 and f["blink_mean"].shape == (64,)
    b = f["blink_mean"]
    top2 = set(np.argsort(-b)[:2])
    assert top2 == {_idx(names, "Fp1"), _idx(names, "Fp2")}
    for ch in ("F3", "Fz", "F4", "F7", "F8"):
        assert b[_idx(names, ch)] > 0
    assert abs(b[_idx(names, "O1")]) < 0.25 and abs(b[_idx(names, "O2")]) < 0.25
    assert f["blink_sd"].max() < 0.35


def test_heog_map_is_lateral():
    f = load_eog_patterns()
    names = f["channel_names"]
    h = f["heog_mean"]
    f7, f8 = _idx(names, "F7"), _idx(names, "F8")
    assert np.sign(h[f7]) == -np.sign(h[f8]) and h[f8] > 0
    assert set(np.argsort(-np.abs(h))[:2]) <= {f7, f8, _idx(names, "AF7"), _idx(names, "AF8")}
    assert f["heog_sd"].max() < HEOG_SD_MAX
    assert int(f["n_subjects"]) >= 15
    assert "10.13026/C28G6P" in str(f["attribution"])
