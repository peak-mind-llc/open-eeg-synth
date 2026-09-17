from __future__ import annotations

import numpy as np

from open_eeg_synth.dsp import OU, PinkCascade, lowpass_decimate, raised_cosine_envelope
from tests.helpers import band_power, psd_slope

FS = 256.0


def test_pink_cascade_slope_and_unit_variance():
    rng = np.random.default_rng(0)
    pk = PinkCascade(3, FS, beta=1.2)
    pk.warm_up(rng, 20.0)
    x = pk.process(rng.standard_normal((int(120 * FS), 3)).T)
    assert abs(psd_slope(x, FS) - 1.2) < 0.1
    assert 0.8 < x.std() < 1.25


def test_pink_cascade_chunk_invariance():
    def run(chunks):
        rng = np.random.default_rng(5)
        pk = PinkCascade(2, FS, beta=1.0)
        out = [pk.process(rng.standard_normal((n, 2)).T) for n in chunks]
        return np.concatenate(out, axis=1)

    assert np.allclose(run([1000]), run([7, 300, 693]), atol=1e-9)


def test_ou_is_stationary_with_requested_sd_and_mean():
    rng = np.random.default_rng(1)
    ou = OU(4, FS, mu=10.0, sigma=0.35, tau_s=4.0, rng=rng)
    x = ou.step(rng.standard_normal((int(600 * FS), 4)).T)
    assert abs(x.mean() - 10.0) < 0.05
    assert abs(x.std() - 0.35) < 0.05
    y = ou.step(np.zeros((4, 10)), mu=np.full(10, 20.0))  # time-varying mean is additive
    assert np.all(y > 15.0)


def test_lowpass_decimate_removes_tone_above_new_nyquist():
    fs_hi = 8 * FS
    t = np.arange(int(4 * fs_hi)) / fs_hi
    x = np.sin(2 * np.pi * 200.0 * t)  # would alias to 56 Hz at 256 Hz
    y = lowpass_decimate(x, 8)
    assert y.shape[0] == int(4 * FS)
    assert band_power(y, FS, 50.0, 62.0)[0] < 1e-3 * band_power(x, fs_hi, 194.0, 206.0)[0]


def test_raised_cosine_envelope_shape():
    e = raised_cosine_envelope(256, FS, 0.1, 0.2)
    assert e[0] == 0.0 and np.isclose(e[128], 1.0) and e[-1] < 0.01
    assert np.all(np.diff(e[:25]) >= 0)
