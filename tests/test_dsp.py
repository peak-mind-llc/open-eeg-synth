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


def test_pink_cascade_unit_variance_within_5pct_long_run():
    """Exact impulse-response calibration: 10 min, 6 rows, both sample rates, both betas."""
    for fs in (256.0, 500.0):
        for beta in (1.2, 2.0):
            rng = np.random.default_rng(int(fs) * 10 + int(beta * 100))
            pk = PinkCascade(6, fs, beta=beta)
            pk.warm_up(rng, 30.0)
            x = pk.process(rng.standard_normal((int(600 * fs), 6)).T)
            var = float((x**2).mean())  # zero-mean process: raw second moment is the variance
            assert abs(var - 1.0) < 0.05, (fs, beta, var)


def test_ou_is_stationary_with_requested_sd_and_mean():
    rng = np.random.default_rng(1)
    ou = OU(4, FS, mu=10.0, sigma=0.35, tau_s=4.0, rng=rng)
    x = ou.step(rng.standard_normal((int(600 * FS), 4)).T)
    assert abs(x.mean() - 10.0) < 0.05
    assert abs(x.std() - 0.35) < 0.05
    y = ou.step(np.zeros((4, 10)), mu=np.full(10, 20.0))  # time-varying mean is additive
    assert np.all(y > 15.0)


def test_ou_chunk_invariance_is_bit_identical():
    def run(chunks):
        rng = np.random.default_rng(11)
        ou = OU(3, FS, mu=2.0, sigma=0.5, tau_s=0.7, rng=rng)
        return np.concatenate([ou.step(rng.standard_normal((n, 3)).T) for n in chunks], axis=1)

    assert np.array_equal(run([2000]), run([1, 13, 986, 1000]))


def test_ou_lag1_autocorrelation_matches_tau():
    rng = np.random.default_rng(5)
    tau_s = 0.1
    ou = OU(1, FS, mu=0.0, sigma=1.0, tau_s=tau_s, rng=rng)
    z = ou.step(rng.standard_normal((200_000, 1)).T)[0]
    rho = np.corrcoef(z[:-1], z[1:])[0, 1]
    assert abs(rho - np.exp(-1.0 / (tau_s * FS))) < 0.01


def test_lowpass_decimate_removes_tone_above_new_nyquist():
    fs_hi = 8 * FS
    t = np.arange(int(4 * fs_hi)) / fs_hi
    x = np.sin(2 * np.pi * 200.0 * t)  # would alias to 56 Hz at 256 Hz
    y = lowpass_decimate(x, 8)
    assert y.shape[0] == int(4 * FS)
    assert band_power(y, FS, 50.0, 62.0)[0] < 1e-3 * band_power(x, fs_hi, 194.0, 206.0)[0]


def _decimated_tone_amplitude_db(frac: float, factor: int) -> float:
    """Amplitude (dB re. unit input) of whatever comes out for a unit-amplitude tone at
    ``frac * fs_out`` going through ``lowpass_decimate(x, factor)``. For ``frac > 0.5`` the only
    thing that can appear in a real-valued, single-tone output is the aliased image, so this one
    number measures the passband case and the aliased-image case alike.
    """
    fs_out = FS
    fs_hi = factor * fs_out
    t = np.arange(int(20 * fs_hi)) / fs_hi
    x = np.sin(2 * np.pi * frac * fs_out * t)
    edge = int(2 * fs_out)  # lowpass_decimate has no carried state; drop the zero-padded edges
    y = lowpass_decimate(x, factor)[edge:-edge]
    amp = np.sqrt(2) * y.std()
    return 20 * np.log10(max(amp, 1e-12))


def test_lowpass_decimate_passes_tone_near_new_nyquist_within_1db():
    for factor in (2, 4, 8):
        assert abs(_decimated_tone_amplitude_db(0.40, factor)) < 1.0


def test_lowpass_decimate_suppresses_images_at_and_above_new_nyquist_by_60db():
    for factor in (2, 4, 8):
        for frac in (0.52, 0.55):
            assert _decimated_tone_amplitude_db(frac, factor) < -60.0


def test_raised_cosine_envelope_shape():
    e = raised_cosine_envelope(256, FS, 0.1, 0.2)
    assert e[0] == 0.0 and np.isclose(e[128], 1.0) and e[-1] < 0.01
    assert np.all(np.diff(e[:25]) >= 0)
