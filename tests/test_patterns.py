from __future__ import annotations

import numpy as np
import pytest

from open_eeg_synth.artifacts import patterns
from open_eeg_synth.channels import CHANNELS_19, UnknownChannelError
from open_eeg_synth.headmodel import load_head_model

# sd is deliberately NOT proportional to mean (unlike a naive `k * |mean|`), so a wrong jitter
# formula (ignoring sd, flipping its sign, or drawing z at the wrong scale) changes the
# max-normalised output rather than cancelling out in the final normalisation.
_BLINK_MEAN = np.zeros(19)
_BLINK_MEAN[[0, 1]] = 1.0  # Fp1, Fp2
_BLINK_MEAN[[2, 3, 16]] = 0.4  # F3, F4, Fz
_BLINK_MEAN[[8, 9]] = -0.05  # O1, O2
_BLINK_SD = np.zeros(19)
_BLINK_SD[0], _BLINK_SD[1] = 0.05, 0.08  # Fp1, Fp2
_BLINK_SD[2], _BLINK_SD[3], _BLINK_SD[16] = 0.10, 0.06, 0.07  # F3, F4, Fz
_BLINK_SD[8], _BLINK_SD[9] = 0.02, 0.03  # O1, O2
_HEOG_MEAN = np.zeros(19)
_HEOG_MEAN[10], _HEOG_MEAN[11] = -1.0, 1.0  # F7, F8
_HEOG_MEAN[0], _HEOG_MEAN[1] = -0.4, 0.4  # Fp1, Fp2
_HEOG_SD = np.zeros(19)
_HEOG_SD[10], _HEOG_SD[11] = 0.3, 0.2  # F7, F8
_HEOG_SD[0], _HEOG_SD[1] = 0.1, 0.15  # Fp1, Fp2


@pytest.fixture
def fake_eog(tmp_path, monkeypatch):
    names = np.array(list(CHANNELS_19))
    p = tmp_path / "eog_patterns.npz"
    np.savez(
        p,
        channel_names=names,
        blink_mean=_BLINK_MEAN,
        blink_sd=_BLINK_SD,
        heog_mean=_HEOG_MEAN,
        heog_sd=_HEOG_SD,
        n_subjects=np.int64(3),
        attribution=np.array("test"),
    )
    monkeypatch.setattr(patterns, "_EOG_PATH", p)
    patterns.load_eog_patterns.cache_clear()
    yield p
    patterns.load_eog_patterns.cache_clear()


def test_analytic_focal_and_dipole_shapes():
    head = load_head_model()
    t3 = head.electrode_pos[CHANNELS_19.index("T3")]
    m = patterns.analytic_focal(t3, head.electrode_pos, 35.0)
    assert np.isclose(m.max(), 1.0) and np.argmax(m) == CHANNELS_19.index("T3")
    assert m[CHANNELS_19.index("O2")] < 0.05
    d = patterns.analytic_dipole(patterns.EYE_CENTRE, (0.0, 0.0, 1.0), head.electrode_pos)
    assert np.abs(d).max() == 1.0 and d[CHANNELS_19.index("Fp1")] > 0.5
    h = patterns.analytic_dipole(patterns.EYE_CENTRE, (1.0, 0.0, 0.0), head.electrode_pos)
    assert np.sign(h[CHANNELS_19.index("F7")]) == -np.sign(h[CHANNELS_19.index("F8")])


def test_analytic_focal_pinned_gaussian_value():
    """A point at exactly one sigma from the centre keeps exp(-1/2) of the peak (not e.g. e^-1)."""
    centre = np.array([0.0, 0.0, 0.0])
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.035]])  # 0 mm, 35 mm away
    w = patterns.analytic_focal(centre, pos, sigma_mm=35.0)
    assert np.allclose(w, [1.0, np.exp(-0.5)])


def test_analytic_dipole_pinned_and_inverse_square_falloff():
    """Along a fixed direction, doubling the distance divides the dipole value by exactly 4."""
    pos = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    v = patterns.analytic_dipole((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), pos)
    assert np.allclose(v, [1.0, 0.25])


def test_empirical_selects_channels_jitters_and_falls_back(fake_eog):
    head = load_head_model()
    m = patterns.empirical("blink", ["fp1", "T7", "O1"], z=0.0)
    assert np.allclose(m, [1.0, 0.0, -0.05])
    j = patterns.empirical("blink", ["Fp1", "F3"], z=1.0)
    # mean + z*sd at z=1: Fp1 -> 1.0+0.05=1.05, F3 -> 0.4+0.10=0.5; max-normalised by 1.05.
    assert np.allclose(j, [1.0, 0.5 / 1.05])
    with pytest.raises(UnknownChannelError):
        patterns.empirical("blink", ["Fp1", "AF7"])
    fb = patterns.empirical(
        "blink",
        ["Fp1", "Oz"],
        electrode_pos=np.array([head.electrode_pos[0], [0.0, -0.11, 0.0]]),
        z=0.0,
    )
    assert fb[0] == 1.0 and abs(fb[1]) < 0.3


def test_empirical_jitter_formula_is_mean_plus_z_times_sd_not_proportional(fake_eog):
    """sd is not proportional to mean here, so a wrong jitter sign or magnitude moves the ratio."""
    plus = patterns.empirical("blink", ["Fp1", "F3"], z=1.0)
    minus = patterns.empirical("blink", ["Fp1", "F3"], z=-1.0)
    # mean + z*sd: Fp1 = 1.0 +/- 0.05, F3 = 0.4 +/- 0.10; normalised by Fp1's value each time.
    assert np.allclose(plus, [1.0, (0.4 + 0.10) / (1.0 + 0.05)])
    assert np.allclose(minus, [1.0, (0.4 - 0.10) / (1.0 - 0.05)])
    # A sign-flipped or ignored jitter would make these two draws equal (or swap which is larger).
    assert plus[1] != pytest.approx(minus[1])
    assert plus[1] > minus[1]


def test_empirical_rng_path_matches_known_draw(fake_eog):
    """rng= must draw exactly one N(0, jitter_sd) sample, identical to a fresh generator's first."""
    seed, jitter_sd = 42, 0.6
    z_expected = float(np.random.default_rng(seed).normal(0.0, jitter_sd))
    got = patterns.empirical(
        "blink", ["Fp1", "F3"], rng=np.random.default_rng(seed), jitter_sd=jitter_sd
    )
    want = patterns.empirical("blink", ["Fp1", "F3"], z=z_expected)
    assert np.array_equal(got, want)
    # a different jitter_sd changes the draw and so the result
    got_wide = patterns.empirical(
        "blink", ["Fp1", "F3"], rng=np.random.default_rng(seed), jitter_sd=2.0
    )
    assert not np.array_equal(got, got_wide)


def test_empirical_fallback_least_squares_scaled_to_file_not_saturated(fake_eog):
    """A fallback channel is scaled to the file's own units, not maxed to +-1 on its own."""
    head = load_head_model()
    pos = dict(zip(CHANNELS_19, head.electrode_pos, strict=True))
    oz = np.array([0.0, -0.11, 0.0])
    posterior = patterns.empirical(
        "blink", ["Fp1", "O1", "Oz"], electrode_pos=np.array([pos["Fp1"], pos["O1"], oz]), z=0.0
    )
    # Oz sits with the small posterior channels, nowhere near the frontal peak that dominates
    # the requested set - a locally-maxed fallback would instead put it at +-1.
    assert posterior[0] == 1.0
    assert np.isclose(posterior[2], -0.0176, atol=0.002)
    assert abs(posterior[2]) < abs(posterior[1]) * 2  # comparable to, not orders above, O1

    f9 = np.array([-0.085, 0.030, -0.030])
    frontal = patterns.empirical(
        "blink",
        ["Fp1", "Fp2", "F3", "F9"],
        electrode_pos=np.array([pos["Fp1"], pos["Fp2"], pos["F3"], f9]),
        z=0.0,
    )
    assert frontal[0] == frontal[1] == 1.0
    assert np.isclose(frontal[3], -0.0884, atol=0.002)


def test_empirical_fallback_uses_max_normalisation_when_no_channel_is_in_file(fake_eog):
    """With no anchor to fit against, the fallback keeps the analytic model's own max-1 scale."""
    f9 = np.array([-0.085, 0.030, -0.030])
    out = patterns.empirical("blink", ["F9"], electrode_pos=np.array([f9]), z=0.0)
    assert out[0] == -1.0


def test_empirical_real_file_blink_peaks_frontal_heog_opposite_f7_f8():
    """Exercises the real Task-15 data file: no monkeypatching, no fake fixture."""
    patterns.load_eog_patterns.cache_clear()
    try:
        blink = patterns.empirical("blink", CHANNELS_19, z=0.0)
        assert np.argmax(np.abs(blink)) in (CHANNELS_19.index("Fp1"), CHANNELS_19.index("Fp2"))
        heog = patterns.empirical("heog", CHANNELS_19, z=0.0)
        f7, f8 = heog[CHANNELS_19.index("F7")], heog[CHANNELS_19.index("F8")]
        assert np.sign(f7) != np.sign(f8) and f8 > f7
    finally:
        patterns.load_eog_patterns.cache_clear()


def test_load_eog_patterns_returns_read_only_arrays():
    patterns.load_eog_patterns.cache_clear()
    try:
        f = patterns.load_eog_patterns()
        with pytest.raises(ValueError):
            f["blink_mean"][0] = 99.0
    finally:
        patterns.load_eog_patterns.cache_clear()
