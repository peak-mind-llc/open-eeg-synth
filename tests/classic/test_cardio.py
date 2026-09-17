"""Cardiac and respiration mocks (the RR tests moved from Coherence Recorder)."""

from __future__ import annotations

import numpy as np
import pytest

from open_eeg_synth.classic.cardio import MockAccSource, MockEcgSource, MockRRSource


def test_mock_rr_values_are_physiological():
    src = MockRRSource(breath_bpm=5.5, rsa_amplitude_ms=40.0, seed=1)
    arr = np.array([src.next_rr() for _ in range(200)])
    assert arr.min() > 600 and arr.max() < 1100  # plausible RR band
    assert 750 < arr.mean() < 950


def test_mock_rr_is_deterministic_under_seed():
    a = [MockRRSource(seed=7).next_rr() for _ in range(20)]
    b = [MockRRSource(seed=7).next_rr() for _ in range(20)]
    assert a == pytest.approx(b)


def test_mock_rr_shows_breathing_oscillation():
    src = MockRRSource(breath_bpm=6.0, rsa_amplitude_ms=45.0, seed=2)
    arr = np.array([src.next_rr() for _ in range(300)])
    assert arr.std() > 15.0


def test_mock_ecg_beats_at_the_requested_rate():
    fs = 130.0
    src = MockEcgSource(bpm=70.0, fs=fs, seed=3)
    x = np.array(src.next_chunk(int(fs * 60)))  # one minute
    above = x > 450.0  # R waves reach ~900 µV; noise is ~8 µV
    beats = int(np.count_nonzero(above[1:] & ~above[:-1]))
    assert 64 <= beats <= 76


def test_mock_ecg_is_deterministic_under_seed():
    a = MockEcgSource(seed=4).next_chunk(500)
    b = MockEcgSource(seed=4).next_chunk(500)
    assert a == pytest.approx(b)


def test_mock_acc_breathes_at_the_requested_rate():
    fs = 25.0
    src = MockAccSource(fs=fs, breath_bpm=5.5, seed=5)
    xyz = np.array(src.next_chunk(int(fs * 120)))
    assert xyz.shape == (int(fs * 120), 3)
    x = xyz[:, 0] - xyz[:, 0].mean()
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    peak_hz = freqs[int(np.argmax(np.abs(np.fft.rfft(x))))]
    assert abs(peak_hz * 60.0 - 5.5) < 0.6
    assert abs(float(np.mean(xyz[:, 2])) - 1000.0) < 5.0  # gravity axis
