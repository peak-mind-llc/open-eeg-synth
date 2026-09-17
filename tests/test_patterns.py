from __future__ import annotations

import numpy as np
import pytest

from open_eeg_synth.artifacts import patterns
from open_eeg_synth.channels import CHANNELS_19, UnknownChannelError
from open_eeg_synth.headmodel import load_head_model

# sd is deliberately NOT proportional to mean (unlike a naive `k * |mean|`), so a wrong jitter
# formula (ignoring sd, dropping sign(mean), or drawing z at the wrong scale) changes the
# max-normalised output rather than cancelling out in the final normalisation. Every sd here is
# below 0.9 x |mean| (the cap), so these fake maps exercise the ruled formula itself.
_FAKE_NAMES = (*CHANNELS_19, "T9", "T10")
_BLINK_MEAN = np.zeros(21)
_BLINK_MEAN[[0, 1]] = 1.0  # Fp1, Fp2
_BLINK_MEAN[[2, 3, 16]] = 0.4  # F3, F4, Fz
_BLINK_MEAN[[8, 9]] = -0.05  # O1, O2
_BLINK_SD = np.zeros(21)
_BLINK_SD[0], _BLINK_SD[1] = 0.05, 0.08  # Fp1, Fp2
_BLINK_SD[2], _BLINK_SD[3], _BLINK_SD[16] = 0.10, 0.06, 0.07  # F3, F4, Fz
_BLINK_SD[8], _BLINK_SD[9] = 0.02, 0.03  # O1, O2
_BLINK_SD[4] = 0.2  # C3: mean exactly 0 with a real spread - must never move
_HEOG_MEAN = np.zeros(21)
_HEOG_MEAN[10], _HEOG_MEAN[11] = -1.0, 1.0  # F7, F8
_HEOG_MEAN[0], _HEOG_MEAN[1] = -0.4, 0.4  # Fp1, Fp2
_HEOG_SD = np.zeros(21)
_HEOG_SD[10], _HEOG_SD[11] = 0.3, 0.2  # F7, F8
_HEOG_SD[0], _HEOG_SD[1] = 0.1, 0.15  # Fp1, Fp2

# Positions for eight channels absent from CHANNELS_19 (and from the eegmmidb-derived data
# file), used only for the mirror-symmetry fallback test below. Computed offline once (not at
# test time, so the fast suite needs no MNE dependency) via MNE's colin27_1020 montage, mapped
# into the head model's coordinate frame by a similarity (Umeyama) fit on the 19 shared channels
# (fit RMS ~5.4 mm). Left/right mirror pairs by construction (A1/A2, TP9/TP10, P9/P10, F9/F10).
_EXTRA_CHANNELS = ("A1", "A2", "TP9", "TP10", "P9", "P10", "F9", "F10")
_EXTRA_POS = np.array(
    [
        [-0.084897, -0.021419, -0.067570],  # A1
        [0.085371, -0.021619, -0.066517],  # A2
        [-0.084595, -0.045377, -0.048387],  # TP9
        [0.085586, -0.046055, -0.047512],  # TP10
        [-0.072140, -0.072763, -0.047111],  # P9
        [0.073396, -0.073508, -0.046468],  # P10
        [-0.069172, 0.041779, -0.041353],  # F9
        [0.071722, 0.042095, -0.040882],  # F10
    ]
)
_MIRROR_PAIRS = (("A1", "A2"), ("TP9", "TP10"), ("P9", "P10"), ("F9", "F10"))
# Same offline procedure, MNE standard_1005 / standard_1020 (identical at these four sites):
# AF9/AF10 are absent from the data file; AF7/AF8 are in it and are requested below under a
# disguised name so that they take the fallback path and can be checked against the real value.
_AF_POS = {
    "AF9": np.array([-0.048246, 0.063516, -0.036162]),
    "AF10": np.array([0.050235, 0.063235, -0.035869]),
    "AF7": np.array([-0.054300, 0.063257, 0.000814]),
    "AF8": np.array([0.055254, 0.064224, 0.001497]),
}


def _write_fake(tmp_path, monkeypatch, t9=0.0, t10=0.0):
    blink, heog = _BLINK_MEAN.copy(), _HEOG_MEAN.copy()
    blink[19], blink[20] = t9, t10
    heog[19], heog[20] = t9, t10
    p = tmp_path / "eog_patterns.npz"
    np.savez(
        p,
        channel_names=np.array(list(_FAKE_NAMES)),
        blink_mean=blink,
        blink_sd=_BLINK_SD,
        heog_mean=heog,
        heog_sd=_HEOG_SD,
        n_subjects=np.int64(3),
        attribution=np.array("test"),
    )
    monkeypatch.setattr(patterns, "_EOG_PATH", p)
    patterns.load_eog_patterns.cache_clear()


@pytest.fixture
def fake_eog(tmp_path, monkeypatch):
    _write_fake(tmp_path, monkeypatch)  # T9 = T10 = 0: referencing leaves the fake map alone
    yield
    patterns.load_eog_patterns.cache_clear()


@pytest.fixture
def real_eog():
    patterns.load_eog_patterns.cache_clear()
    yield
    patterns.load_eog_patterns.cache_clear()


def _full(name, z=0.0):
    """The real file's whole 64-channel map, as `empirical` sees it."""
    names = [str(c) for c in patterns.load_eog_patterns()["channel_names"]]
    return names, patterns.empirical(name, names, z=z)


def test_analytic_focal_and_dipole_shapes():
    head = load_head_model()
    t3 = head.electrode_pos[CHANNELS_19.index("T3")]
    m = patterns.analytic_focal(t3, head.electrode_pos, 35.0)
    assert np.isclose(m.max(), 1.0) and np.argmax(m) == CHANNELS_19.index("T3")
    assert m[CHANNELS_19.index("O2")] < 0.05
    d = patterns.analytic_dipole(patterns.EYE_CENTRE, (0.0, 0.0, 1.0), head.electrode_pos)
    assert np.isclose(np.abs(d).max(), 1.0) and d[CHANNELS_19.index("Fp1")] > 0.5
    h = patterns.analytic_dipole(patterns.EYE_CENTRE, (1.0, 0.0, 0.0), head.electrode_pos)
    assert np.sign(h[CHANNELS_19.index("F7")]) == -np.sign(h[CHANNELS_19.index("F8")])


def test_analytic_focal_pinned_gaussian_value():
    """A point at exactly one sigma from the centre keeps exp(-1/2) of the peak (not e.g. e^-1)."""
    centre = np.array([0.0, 0.0, 0.0])
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.035]])  # 0 mm, 35 mm away
    w = patterns.analytic_focal(centre, pos, sigma_mm=35.0)
    assert np.allclose(w, [1.0, np.exp(-0.5)])


def test_analytic_focal_is_the_raw_gaussian_whatever_is_requested():
    """P30: no normalisation over the requested electrodes - an electrode 10 mm from the centre
    reads exp(-(10/35)^2 / 2) even when it is the only one asked for."""
    centre = np.array([0.0, 0.0, 0.0])
    pos = np.array([[0.010, 0.0, 0.0], [0.0, 0.050, 0.0], [0.0, 0.0, 0.070]])
    w = patterns.analytic_focal(centre, pos, sigma_mm=35.0)
    assert np.allclose(w, np.exp(-0.5 * (np.array([10.0, 50.0, 70.0]) / 35.0) ** 2))
    assert np.allclose(patterns.analytic_focal(centre, pos[1:], 35.0), w[1:])


def test_analytic_dipole_pinned_and_inverse_square_falloff():
    """Along a fixed direction, doubling the distance divides the dipole value by exactly 4, and
    the unit is the dipole's largest |value| over the 19-channel template head (P30)."""
    head = load_head_model()
    origin, moment = np.zeros(3), np.array([1.0, 0.0, 0.0])
    pos = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    v = patterns.analytic_dipole(origin, moment, pos)
    r = head.electrode_pos - origin
    peak = np.abs((r @ moment) / np.linalg.norm(r, axis=1) ** 3).max()
    assert np.allclose(v, np.array([1.0 / 0.1**2, 1.0 / 0.2**2]) / peak)
    assert np.isclose(v[0] / v[1], 4.0)


def test_analytic_dipole_is_normalised_over_the_full_template_head():
    """P30: asking for a subset gives the subset's entries of the full-head answer, so two far
    posterior electrodes stay small instead of being stretched to +-1."""
    head = load_head_model()
    full = patterns.analytic_dipole(patterns.EYE_CENTRE, (0.0, 0.3, 1.0), head.electrode_pos)
    rows = [CHANNELS_19.index(c) for c in ("O1", "O2")]
    sub = patterns.analytic_dipole(patterns.EYE_CENTRE, (0.0, 0.3, 1.0), head.electrode_pos[rows])
    assert np.allclose(sub, full[rows])
    assert np.isclose(np.abs(full).max(), 1.0) and np.abs(sub).max() < 0.2


def test_eye_maps_are_referenced_to_t9_t10(tmp_path, monkeypatch):
    """P28: the mean of the file's T9 and T10 is subtracted from each mean map at load."""
    _write_fake(tmp_path, monkeypatch, t9=0.1, t10=0.3)
    try:
        chs = ["Fp1", "F3", "O1", "Cz", "T9", "T10"]
        got = patterns.empirical("blink", chs, z=0.0)
        mean = np.array([1.0, 0.4, -0.05, 0.0, 0.1, 0.3]) - 0.2
        assert np.allclose(got, mean / 0.8)  # full-map max |value| is Fp1/Fp2's 1.0 - 0.2
        assert np.isclose(got[4], -got[5])  # the reference pair is antisymmetric afterwards
    finally:
        patterns.load_eog_patterns.cache_clear()


def test_real_file_referenced_maps_match_the_ruling(real_eog):
    """P28's stated values for the real file, T9/T10-referenced and max-normalised."""
    names, blink = _full("blink")
    want = {"Fp1": 1.00, "Fp2": 0.99, "F7": 0.43, "F8": 0.37, "F3": 0.33, "Fz": 0.29}
    want |= {"F4": 0.31, "C3": 0.08, "O1": -0.03, "O2": -0.03, "Pz": 0.01}
    for ch, v in want.items():
        assert blink[names.index(ch)] == pytest.approx(v, abs=0.01), ch
    names, heog = _full("heog")
    for ch, v in {"F7": -0.91, "F8": 1.00, "T3": -0.29, "T4": 0.34}.items():
        assert heog[names.index(ch)] == pytest.approx(v, abs=0.01), ch
    raw = patterns.load_eog_patterns()  # the loader itself still returns the file as derived
    assert raw["blink_mean"][names.index("O1")] == pytest.approx(-0.234, abs=0.001)


def test_empirical_selects_channels_jitters_and_falls_back(fake_eog):
    head = load_head_model()
    m = patterns.empirical("blink", ["fp1", "T7", "O1"], z=0.0)
    assert np.allclose(m, [1.0, 0.0, -0.05])
    j = patterns.empirical("blink", ["Fp1", "F3"], z=1.0)
    # mean + z*sign(mean)*sd at z=1: Fp1 1.05, F3 0.5; the whole map's peak is Fp2's 1.08.
    assert np.allclose(j, [1.05 / 1.08, 0.5 / 1.08])
    with pytest.raises(UnknownChannelError):
        patterns.empirical("blink", ["Fp1", "AF7"])
    fb = patterns.empirical(
        "blink",
        ["Fp1", "Oz"],
        electrode_pos=np.array([head.electrode_pos[0], [0.0, -0.11, 0.0]]),
        z=0.0,
    )
    assert fb[0] == 1.0 and -0.1 < fb[1] < 0.0


def test_empirical_jitter_keeps_each_channels_sign(fake_eog):
    """P29: mean + z * sign(mean) * sd. A negative channel grows more negative with z, exactly
    as a positive one grows more positive (the old `mean + z * sd` shrank it instead)."""
    chs = ["Fp1", "F3", "O1", "F8", "F7"]
    plus = patterns.empirical("blink", chs[:3], z=1.0)
    minus = patterns.empirical("blink", chs[:3], z=-1.0)
    # z=+1: Fp1 1.05, F3 0.5, O1 -0.07; whole-map peak Fp2 1.08.
    assert np.allclose(plus, np.array([1.05, 0.5, -0.07]) / 1.08)
    # z=-1: Fp1 0.95, F3 0.3, O1 -0.03; whole-map peak Fp1 0.95 (Fp2 is 0.92).
    assert np.allclose(minus, np.array([0.95, 0.3, -0.03]) / 0.95)
    heog = patterns.empirical("heog", chs[3:], z=1.0)
    assert np.allclose(heog, np.array([1.2, -1.3]) / 1.3)  # both sides grow with z


def test_empirical_jitter_is_clipped_and_leaves_zero_mean_channels_alone(fake_eog):
    chs = ["Fp1", "F3", "O1", "C3"]
    assert np.array_equal(
        patterns.empirical("blink", chs, z=5.0), patterns.empirical("blink", chs, z=1.0)
    )
    assert np.array_equal(
        patterns.empirical("blink", chs, z=-3.0), patterns.empirical("blink", chs, z=-1.0)
    )
    wide = patterns.empirical("blink", chs, rng=np.random.default_rng(3), jitter_sd=50.0)
    assert any(np.array_equal(wide, patterns.empirical("blink", chs, z=s)) for s in (1.0, -1.0))
    for z in (-1.0, -0.4, 0.7, 1.0):
        assert patterns.empirical("blink", chs, z=z)[3] == 0.0  # C3: mean 0, sd 0.2


def test_empirical_rng_path_matches_known_draw(fake_eog):
    """rng= draws exactly one N(0, jitter_sd) sample, clipped to [-1, 1]."""
    seed, jitter_sd = 42, 0.6
    z_expected = float(np.clip(np.random.default_rng(seed).normal(0.0, jitter_sd), -1.0, 1.0))
    got = patterns.empirical(
        "blink", ["Fp1", "F3"], rng=np.random.default_rng(seed), jitter_sd=jitter_sd
    )
    want = patterns.empirical("blink", ["Fp1", "F3"], z=z_expected)
    assert np.array_equal(got, want)
    got_wide = patterns.empirical(
        "blink", ["Fp1", "F3"], rng=np.random.default_rng(seed), jitter_sd=2.0
    )
    assert not np.array_equal(got, got_wide)


@pytest.mark.parametrize("z", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_real_file_jitter_keeps_signs_and_heog_antisymmetry(real_eog, z):
    """P29 on the real file, whose spreads exceed |mean| on 30 blink and 58 heog channels: no z in
    [-1, 1] may flip (or zero) a channel, so opposite-sign mirror pairs stay opposite and F7/F8
    stay balanced (the old `mean + z * sd` gave |F7|/|F8| = 0.21 at z = 1)."""
    names, base = _full("heog", 0.0)
    _, h = _full("heog", z)
    assert np.array_equal(np.sign(h), np.sign(base)) and np.all(h != 0.0)
    f7, f8 = h[names.index("F7")], h[names.index("F8")]
    assert f7 < 0.0 < f8 and 0.8 <= abs(f7) / f8 <= 1.25
    for left, right in (("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"), ("C3", "C4"), ("T3", "T4")):
        assert h[names.index(left)] < 0.0 < h[names.index(right)], (left, right)
    for left, right in (("P3", "P4"), ("T5", "T6"), ("FC5", "FC6"), ("T9", "T10")):
        assert h[names.index(left)] < 0.0 < h[names.index(right)], (left, right)
    names, b0 = _full("blink", 0.0)
    _, b = _full("blink", z)
    assert np.array_equal(np.sign(b), np.sign(b0)) and np.all(b != 0.0)
    assert np.isclose(np.abs(b).max(), 1.0)


def test_subset_request_returns_full_map_values(real_eog, tmp_path, monkeypatch):
    """P30: a 4-channel request returns the numbers those channels have in the whole map, whether
    or not the map's peak is among them."""
    for name in ("blink", "heog"):
        for z in (-0.8, 0.0, 0.6):
            names, full = _full(name, z)
            chs = ["O1", "O2", "T3", "T4"]
            sub = patterns.empirical(name, chs, z=z)
            assert np.array_equal(sub, [full[names.index(c)] for c in chs])
            assert np.abs(sub).max() < 0.5
    _write_fake(tmp_path, monkeypatch)
    assert np.allclose(patterns.empirical("blink", ["O1", "F3"], z=0.0), [-0.05, 0.4])


def test_real_file_maps_hold_over_many_subjects(real_eog):
    """No seed luck: the blink and heog map properties hold for every one of 25 subject draws."""
    names = [str(c) for c in patterns.load_eog_patterns()["channel_names"]]
    ix = {c: names.index(c) for c in ("Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8", "O1", "O2")}
    worst_o = 0.0
    for seed in range(25):
        rng = np.random.default_rng(1000 + seed)
        b = patterns.empirical("blink", names, rng=rng)
        assert set(np.argsort(-np.abs(b))[:2]) == {ix["Fp1"], ix["Fp2"]}
        # frontal clearly positive and above the posterior bound (lowest: Fz 0.199 at z = -1)
        assert all(b[ix[c]] > 0.15 for c in ("F3", "F4", "Fz", "F7", "F8"))
        worst_o = max(worst_o, abs(b[ix["O1"]]), abs(b[ix["O2"]]))
        h = patterns.empirical("heog", names, rng=rng)
        assert h[ix["F7"]] < -0.5 and h[ix["F8"]] > 0.5
        assert set(np.argsort(-np.abs(h))[:2]) == {ix["F7"], ix["F8"]}
    assert worst_o < 0.15


def test_empirical_fallback_pinned_cases(fake_eog):
    """The re-review's cases, pinned for the smooth fit (fix round 4). A fallback is scaled to its
    nearest present channels, not max-normalised over the request.

    blink on [O1, O2, Oz], Oz missing: Oz keeps O1's size (-0.0530 vs -0.05; the old per-request
    normalisation saturated it at -1). heog on [Fp1, Fp2, F7, F8, F9], F9 missing: F9 is 0.766 of
    F7. The grid version anchored on F7 alone and gave the dipole's own F9/F7 ratio, 0.864; the
    smooth fit also gives Fp1 and Fp2 their 1/d^2 share (they are 3x further away), which moves
    it to 0.766, inside the +-0.1 the ruling allows."""
    head = load_head_model()
    pos = dict(zip(CHANNELS_19, head.electrode_pos, strict=True))
    oz = np.array([0.0, -0.11, 0.0])
    posterior = patterns.empirical(
        "blink", ["O1", "O2", "Oz"], electrode_pos=np.array([pos["O1"], pos["O2"], oz]), z=0.0
    )
    assert np.allclose(posterior, [-0.05, -0.05, -0.0530], atol=0.0001)

    f9 = np.array([-0.085, 0.030, -0.030])
    frontal = patterns.empirical(
        "heog",
        ["Fp1", "Fp2", "F7", "F8", "F9"],
        electrode_pos=np.array([pos["Fp1"], pos["Fp2"], pos["F7"], pos["F8"], f9]),
        z=0.0,
    )
    assert np.allclose(frontal[:4], [-0.4, 0.4, -1.0, 1.0])
    assert frontal[4] / frontal[2] == pytest.approx(0.766, abs=0.001)
    assert abs(frontal[4] / frontal[2] - 0.864) <= 0.1


def test_empirical_fallback_frontal_case_with_a_present_dominant_channel(fake_eog):
    """blink on [Fp1, Fp2, F3, F9], F9 missing. Re-pinned for fix round 4: -0.0864 under the
    grid version (Fp1 alone, the nearest anchor at 78 mm) and -0.1230 now, because F3 (86 mm)
    and Fp2 also get their 1/d^2 share, and F3 needs a larger scale than Fp1 once the dipole's
    distance falloff is divided out."""
    head = load_head_model()
    pos = dict(zip(CHANNELS_19, head.electrode_pos, strict=True))
    f9 = np.array([-0.085, 0.030, -0.030])
    frontal = patterns.empirical(
        "blink",
        ["Fp1", "Fp2", "F3", "F9"],
        electrode_pos=np.array([pos["Fp1"], pos["Fp2"], pos["F3"], f9]),
        z=0.0,
    )
    assert frontal[0] == frontal[1] == 1.0
    assert np.isclose(frontal[3], -0.1230, atol=0.0005)


def test_empirical_fallback_with_no_channel_in_file_is_the_analytic_dipole(fake_eog):
    """With no anchor to fit against, the fallback is the analytic dipole on its full-head unit
    (P30), checked across two missing channels at different distances from the source."""
    f9 = np.array([-0.085, 0.030, -0.030])
    oz = np.array([0.0, -0.11, 0.0])
    pos = np.array([f9, oz])
    out = patterns.empirical("blink", ["F9", "Oz"], electrode_pos=pos, z=0.0)
    centre, moment = patterns._FALLBACK["blink"]
    assert np.allclose(out, patterns.analytic_dipole(centre, moment, pos))
    assert out[0] < 0.0 and out[1] < 0.0 and abs(out[1]) < abs(out[0]) < 1.0
    # 10 mm in front of the eyes the dipole is ~10x its template peak: clipped to the peak
    near = np.array([[0.0, 0.095, -0.020], *pos])
    assert patterns.analytic_dipole(centre, moment, near[:1])[0] > 5.0
    out = patterns.empirical("blink", ["X1", "F9", "Oz"], electrode_pos=near, z=0.0)
    assert out[0] == 1.0 and np.allclose(out[1:], patterns.analytic_dipole(centre, moment, pos))


def test_empirical_fallback_never_uses_an_anchor_that_disagrees_in_sign(real_eog):
    """On the real blink map F7, T3 and T4 are positive where the dipole model is negative. With
    only those three present no anchor is usable, so A1 is exactly the analytic dipole (fix round
    5). The rule it replaces fitted through all three and gave A1 +0.349; on the full montage of
    the mirror tests A1 is negative (-0.107), and so is the dipole (-0.091)."""
    head = load_head_model()
    pos = dict(zip(CHANNELS_19, head.electrode_pos, strict=True))
    a1 = _EXTRA_POS[0]
    p = np.array([pos["F7"], pos["T3"], pos["T4"], a1])
    v = patterns.empirical("blink", ["F7", "T3", "T4", "A1"], electrode_pos=p, z=0.0)
    names, full = _full("blink")
    assert np.array_equal(v[:3], [full[names.index(c)] for c in ("F7", "T3", "T4")])
    centre, moment = patterns._FALLBACK["blink"]
    _, cosine = patterns._dipole_terms(centre, moment, p)
    assert np.all(np.sign(v[:3]) != np.sign(cosine[:3]))
    assert v[3] == patterns.analytic_dipole(centre, moment, p[3:])[0]
    chs = [*CHANNELS_19, *_EXTRA_CHANNELS]
    montage = patterns.empirical(
        "blink", chs, electrode_pos=np.vstack([head.electrode_pos, _EXTRA_POS]), z=0.0
    )
    assert v[3] < 0.0 and montage[chs.index("A1")] < 0.0


def test_empirical_real_file_blink_peaks_frontal_heog_opposite_f7_f8(real_eog):
    """Exercises the real Task-15 data file: no monkeypatching, no fake fixture."""
    blink = patterns.empirical("blink", CHANNELS_19, z=0.0)
    assert np.argmax(np.abs(blink)) in (CHANNELS_19.index("Fp1"), CHANNELS_19.index("Fp2"))
    heog = patterns.empirical("heog", CHANNELS_19, z=0.0)
    f7, f8 = heog[CHANNELS_19.index("F7")], heog[CHANNELS_19.index("F8")]
    assert np.sign(f7) != np.sign(f8) and f8 > f7


def _mirror_checks(name, chs, v):
    ix = {c: chs.index(c) for c in chs}
    for left, right in _MIRROR_PAIRS:
        ratio = abs(v[ix[left]]) / abs(v[ix[right]])
        assert 0.5 <= ratio <= 2.0, (name, left, right, ratio)
    if name == "blink":
        assert all(v[ix[c]] > 0.0 for c in ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"))
        assert all(abs(v[ix[c]]) < 0.15 for c in ("O1", "O2", *_EXTRA_CHANNELS[:6]))
    else:
        assert v[ix["F7"]] < 0.0 < v[ix["F8"]]
        for left, right in _MIRROR_PAIRS:
            assert v[ix[left]] < 0.0 < v[ix[right]], (name, left, right)


def test_empirical_fallback_mirror_symmetry_on_a_full_montage(real_eog):
    """A missing channel's fallback must not be lopsided at its mirror twin, in size or in sign
    (re-review round 3 finding 2). Real data file. Positions from `_EXTRA_POS`."""
    head = load_head_model()
    chs = list(CHANNELS_19) + list(_EXTRA_CHANNELS)
    pos = np.vstack([head.electrode_pos, _EXTRA_POS])
    for name in ("blink", "heog"):
        _mirror_checks(name, chs, patterns.empirical(name, chs, electrode_pos=pos, z=0.0))

        # Under +-5 mm position jitter of the extra channels, every pair stays within 2x in at
        # least 90 % of 200 draws, for several seeds (real electrode placement is not exact).
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            bad = dict.fromkeys(_MIRROR_PAIRS, 0)
            for _ in range(200):
                jittered = _EXTRA_POS + rng.normal(0.0, 0.005, _EXTRA_POS.shape)
                jv = patterns.empirical(
                    name, chs, electrode_pos=np.vstack([head.electrode_pos, jittered]), z=0.0
                )
                for left, right in _MIRROR_PAIRS:
                    ratio = abs(jv[chs.index(left)]) / abs(jv[chs.index(right)])
                    bad[(left, right)] += not (0.5 <= ratio <= 2.0)
            assert max(bad.values()) <= 20, (name, seed, bad)


def test_empirical_fallback_af9_af10_heog_is_balanced(real_eog):
    """The grid version put AF10 at 32x AF9 (a strong anchor won a distance tie on one side).
    The smooth fit gives AF9 -1.119 and AF10 +1.296 (ratio 1.158); both lie beyond the file's
    peak, so since fix round 5 both are clipped to it and the ratio is 1."""
    head = load_head_model()
    chs = [*CHANNELS_19, "AF9", "AF10"]
    pos = np.vstack([head.electrode_pos, _AF_POS["AF9"], _AF_POS["AF10"]])
    v = patterns.empirical("heog", chs, electrode_pos=pos, z=0.0)
    assert v[19] < 0.0 < v[20]
    assert 0.5 <= abs(v[20]) / abs(v[19]) <= 2.0
    assert v[19] == -1.0 and v[20] == 1.0
    centre, moment = patterns._FALLBACK["heog"]
    envelope, cosine = patterns._dipole_terms(centre, moment, pos)
    agree = [i for i in range(19) if np.sign(cosine[i]) == np.sign(v[i])]
    labels = [CHANNELS_19[i] for i in agree]
    for m, want in ((19, -1.119), (20, 1.296)):  # the fit before the clip
        dist = np.linalg.norm(pos[agree] - pos[m], axis=1)
        fit = patterns._fallback_scale(v[agree], envelope[agree], cosine[agree], dist, labels)
        assert fit[1] == 1.0  # a fully reliable anchor is in the fit: no blend with the dipole
        assert fit[0] * envelope[m] * cosine[m] == pytest.approx(want, abs=0.001)


def test_empirical_fallback_recovers_a_disguised_in_file_channel(real_eog):
    """AF7/AF8 are in the file; asked for under another name they take the fallback path, and the
    fit must land within 0.3 of their real (full-map) value (the grid version was 1.4 off)."""
    head = load_head_model()
    names, full = _full("heog")
    for real in ("AF7", "AF8"):
        v = patterns.empirical(
            "heog",
            [*CHANNELS_19, f"ZZ_{real}"],
            electrode_pos=np.vstack([head.electrode_pos, _AF_POS[real]]),
            z=0.0,
        )
        assert abs(v[19] - full[names.index(real)]) < 0.3, (real, v[19])
        assert np.sign(v[19]) == np.sign(full[names.index(real)])


def test_empirical_fallback_is_continuous_under_small_moves(real_eog):
    """No discrete ranking: moving a missing channel in 0.05 mm steps never jumps its value by
    more than 25 % (and 0.02) in one step (the grid version jumped at its 10 mm bucket edges)."""
    head = load_head_model()
    steps = np.arange(-5.0, 5.0, 0.05) / 1000.0
    for name in ("blink", "heog"):
        for ch, base in (("A1", _EXTRA_POS[0]), ("TP10", _EXTRA_POS[3]), ("AF9", _AF_POS["AF9"])):
            for axis in range(3):
                vals = []
                for s in steps:
                    p = base.copy()
                    p[axis] += s
                    pos = np.vstack([head.electrode_pos, p])
                    vals.append(patterns.empirical(name, [*CHANNELS_19, ch], electrode_pos=pos)[19])
                vals = np.array(vals)
                step = np.abs(np.diff(vals))
                jumps = (step > 0.25 * np.abs(vals[:-1])) & (step > 0.02)
                assert not jumps.any(), (name, ch, axis, vals[1:][jumps])


def test_load_eog_patterns_returns_read_only_arrays(real_eog):
    f = patterns.load_eog_patterns()
    with pytest.raises(ValueError):
        f["blink_mean"][0] = 99.0


def test_fallback_scale_is_inverse_square_distance_weighted_least_squares():
    """Two anchors at 1 and 2 (units cancel), direction cosines 1 and 0.5, envelope-divided values
    2 and 6: s = sum(w c y) / sum(w c^2) with w = 1/d^2."""
    values = np.array([2.0, 3.0])
    envelope = np.array([1.0, 0.5])
    cosine = np.array([1.0, 0.5])
    dist = np.array([1.0, 2.0])
    w = np.array([1.0, 0.25])
    y = values / envelope
    want = (w * cosine * y).sum() / (w * cosine**2).sum()
    got, trust = patterns._fallback_scale(values, envelope, cosine, dist, ["a", "b"])
    assert np.isclose(got, want) and trust == 1.0
    # with 1/d instead the answer would differ
    assert not np.isclose(got, (np.array([1.0, 0.5]) * cosine * y).sum() / (0.5 * 0.25 + 1.0))


def test_fallback_scale_uses_only_the_k_nearest_anchors():
    k = patterns._FALLBACK_K
    n = k + 2
    values = np.ones(n)
    values[k:] = 1000.0  # the two furthest anchors are wildly different
    ones = np.ones(n)
    dist = np.arange(1.0, n + 1.0)
    got = patterns._fallback_scale(values, ones, ones, dist, [str(i) for i in range(n)])
    assert np.allclose(got, (1.0, 1.0))
    # the same anchors in reverse request order give the same answer
    rev = patterns._fallback_scale(values[::-1], ones, ones, dist[::-1], [str(i) for i in range(n)])
    assert np.allclose(rev, (1.0, 1.0))


def test_fallback_scale_breaks_exact_distance_ties_by_label_not_request_order():
    k = patterns._FALLBACK_K
    n = k + 1
    ones = np.ones(n)
    dist = np.ones(n)  # every anchor equally far: the k kept are the first k labels
    values = np.ones(n)
    values[0] = 5.0
    labels = [f"c{i}" for i in range(n)]
    labels[0] = "zz"  # sorts last, so it is the one left out
    assert np.isclose(patterns._fallback_scale(values, ones, ones, dist, labels)[0], 1.0)
    order = np.arange(n)[::-1]
    got = patterns._fallback_scale(
        values[order], ones, ones, dist[order], [labels[i] for i in order]
    )
    assert np.isclose(got[0], 1.0)


def test_fallback_reliability_rises_smoothly_from_the_floor_to_three_times_the_floor():
    """r(|cos|) is 0 up to the floor (0.05), 1 from 3x the floor (the ruling said 2x; 3x is
    what the real-montage sweeps support, see `_FALLBACK_FULL`), a smoothstep between, and the
    same for either sign; no step on a fine grid is large (a hard cutoff would jump 0 -> 1)."""
    f = patterns._FALLBACK_FLOOR
    assert f == 0.05
    c = np.array([0.0, 0.5 * f, f, 1.5 * f, 2.0 * f, 2.5 * f, 3.0 * f, 0.5, 1.0])
    r = patterns._reliability(c)
    assert np.array_equal(r[:3], [0.0, 0.0, 0.0])
    assert r[3] == pytest.approx(0.25**2 * (3.0 - 2.0 * 0.25))  # smoothstep(0.25) = 0.15625
    assert r[4] == pytest.approx(0.5)
    assert r[5] == pytest.approx(0.75**2 * (3.0 - 2.0 * 0.75))  # smoothstep(0.75) = 0.84375
    assert np.array_equal(r[6:], [1.0, 1.0, 1.0])
    assert np.array_equal(patterns._reliability(-c), r)
    grid = patterns._reliability(np.linspace(0.0, 4.0 * f, 4001))
    assert np.all(np.diff(grid) >= 0.0) and np.diff(grid).max() < 0.002


def test_fallback_scale_weights_by_reliability_and_fits_the_true_cosine():
    """w = r(|cos|) / d^2, and the fit uses cos as it is (no floor): an anchor at 2x the floor
    (r = 0.5) with cos -0.1 enters with half its distance weight. The trust returned is the
    largest r in the fit; a lone anchor inside the ramp returns its own r."""
    f = patterns._FALLBACK_FLOOR
    values = np.array([0.3, 0.2])
    envelope = np.array([1.0, 2.0])
    cosine = np.array([0.5, -2.0 * f])
    dist = np.array([2.0, 1.0])
    w = np.array([1.0 / 4.0, 0.5 / 1.0])
    y = values / envelope
    want = (w * cosine * y).sum() / (w * cosine**2).sum()
    got, trust = patterns._fallback_scale(values, envelope, cosine, dist, ["a", "b"])
    assert np.isclose(got, want) and trust == 1.0
    unweighted = (cosine * y / dist**2).sum() / (cosine**2 / dist**2).sum()
    assert not np.isclose(got, unweighted)
    lone = patterns._fallback_scale(values[1:], envelope[1:], cosine[1:], dist[1:], ["b"])
    assert np.isclose(lone[0], y[1] / cosine[1]) and lone[1] == pytest.approx(0.5)


def test_fallback_scale_skips_near_null_anchors_entirely():
    """An anchor at or below the floor has no weight and takes none of the k nearest slots; with
    no other anchor there is no fit (None), and `empirical` takes the analytic dipole."""
    f = patterns._FALLBACK_FLOOR
    for c in (f, 1e-6, -0.5 * f, 0.0):
        assert (
            patterns._fallback_scale(
                np.array([0.1]), np.array([1.0]), np.array([c]), np.array([0.01]), ["a"]
            )
            is None
        )
    k = patterns._FALLBACK_K
    # nearest: a near-null anchor with a wild value; then k usable anchors (the k-th differs);
    # then a (k+1)-th usable one, wild again, that must stay out of the fit
    n = k + 2
    dist = np.arange(1.0, n + 1.0)
    values = np.ones(n)
    values[0], values[k], values[k + 1] = 1000.0, 3.0, 1000.0
    cosine = np.ones(n)
    cosine[0] = 0.5 * f
    ones = np.ones(n)
    labels = [f"c{i}" for i in range(n)]
    w = 1.0 / dist[1 : k + 1] ** 2
    want = (w * values[1 : k + 1]).sum() / w.sum()
    assert np.allclose(patterns._fallback_scale(values, ones, cosine, dist, labels), (want, 1.0))


# MNE's easycap-M1 F7/F8, mapped into the head frame as for `_EXTRA_POS` (computed offline).
# There the blink dipole's null runs just inside them: cos +0.056 (F7, which now agrees in sign
# with the map) and +0.064 (F8), both inside the reliability ramp.
_EASYCAP_F7_F8 = np.array([[-0.076915, 0.035511, 0.000287], [0.077822, 0.035728, 0.001051]])


def test_empirical_fallback_fades_near_null_anchors_on_a_real_montage(real_eog):
    """With easycap-M1 F7/F8 as the only anchors, the reliability weights alone cannot fade them
    (their ratio fixes the fit): A1 would read -1.473 (clipped to -1), 14x its value on the
    mirror-test montage (-0.107). Blended by the best anchor's reliability (0.050), A1 reads
    -0.160 and F9 -0.357 (montage -0.239)."""
    centre, moment = patterns._FALLBACK["blink"]
    _, cosine = patterns._dipole_terms(centre, moment, _EASYCAP_F7_F8)
    assert np.allclose(cosine, [0.0556, 0.0636], atol=0.0001)
    extra = dict(zip(_EXTRA_CHANNELS, _EXTRA_POS, strict=True))
    for chs, want in ((("A1", "A2"), (-0.1603, -0.1516)), (("F9", "F10"), (-0.3567, -0.3056))):
        pos = np.vstack([_EASYCAP_F7_F8, [extra[c] for c in chs]])
        v = patterns.empirical("blink", ["F7", "F8", *chs], electrode_pos=pos, z=0.0)
        assert np.allclose(v[2:], want, atol=0.0005), (chs, v)
        dipole = patterns.analytic_dipole(centre, moment, pos[2:])
        assert np.all(np.abs(v[2:]) > np.abs(dipole)) and np.all(np.abs(v[2:]) < 2 * np.abs(dipole))


def test_empirical_fallback_is_continuous_as_a_lone_anchor_crosses_the_floor(real_eog):
    """F8 as the only anchor for F10, moved forward from 5 mm behind the template position to
    10 mm ahead (its cos goes 0.024 -> 0.077 and crosses the floor at +2.65 mm). Below the floor
    F10 is exactly the dipole (-0.1813); above it F10 grows smoothly (-0.2274 at +5 mm, -0.4670
    at +10 mm). Without the blend it jumped straight to the clip, -1, at the crossing."""
    head = load_head_model()
    f8 = head.electrode_pos[CHANNELS_19.index("F8")]
    f10 = _EXTRA_POS[_EXTRA_CHANNELS.index("F10")]
    centre, moment = patterns._FALLBACK["blink"]
    dipole = patterns.analytic_dipole(centre, moment, f10[None])[0]
    steps = np.arange(-5.0, 10.0001, 0.05) / 1000.0
    vals, cos = [], []
    for s in steps:
        pos = np.array([f8 + [0.0, s, 0.0], f10])
        vals.append(patterns.empirical("blink", ["F8", "F10"], electrode_pos=pos, z=0.0)[1])
        cos.append(patterns._dipole_terms(centre, moment, pos[:1])[1][0])
    vals, cos = np.array(vals), np.array(cos)
    below = cos <= patterns._FALLBACK_FLOOR
    assert below[:150].all() and not below[160:].any()
    assert np.all(vals[below] == dipole)
    step = np.abs(np.diff(vals))
    assert not ((step > 0.25 * np.abs(vals[:-1])) & (step > 0.02)).any(), step.max()
    assert vals[200] == pytest.approx(-0.2274, abs=0.0005)
    assert vals[-1] == pytest.approx(-0.4670, abs=0.0005)


# --- fix round 5: sparse montages, the jitter cap, the referenced jitter sign -----------------


def _mirror_montage(name):
    """The montage of the mirror-symmetry tests (CHANNELS_19 on the template head plus
    `_EXTRA_CHANNELS` at `_EXTRA_POS`), at z = 0: each channel's value and position."""
    head = load_head_model()
    chs = [*CHANNELS_19, *_EXTRA_CHANNELS]
    pos = np.vstack([head.electrode_pos, _EXTRA_POS])
    v = patterns.empirical(name, chs, electrode_pos=pos, z=0.0)
    return dict(zip(chs, v, strict=True)), dict(zip(chs, pos, strict=True))


@pytest.mark.parametrize(
    ("name", "chs", "before"),
    [
        ("blink", ("F7", "F8", "F9", "F10"), {"F9": -3.336, "F10": -3.077}),
        ("blink", ("F7", "F8", "A1", "A2"), {"A1": -1.545, "A2": -1.523}),
        ("blink", ("T3", "T4", "A1", "A2"), {"A1": 0.174, "A2": 0.181}),
        ("heog", ("O1", "Pz", "F9"), {"F9": -6.858}),
    ],
)
def test_sparse_montage_fallbacks_keep_the_full_montage_sign_and_size(real_eog, name, chs, before):
    """The re-review's small montages (values before fix round 5 in `before`). In each, every
    present channel either disagrees in sign with the dipole (blink F7, T3, T4; heog O1) or sits
    at its null (blink F8, cos +0.041; heog Pz, cos +0.004), so no anchor is usable and each
    missing channel is exactly the analytic dipole. Against the mirror-test montage (see
    `_mirror_montage`) each keeps its sign and is at most 3x its size there. The heog F9
    dipole value (-0.238) is 0.30 of the montage value (-0.792): the dipole puts the heog peak at
    Fp1/Fp2, where the real map peaks at F7/F8, so the lower bound here is 1/4, not 1/3."""
    ref, where = _mirror_montage(name)
    pos = np.array([where[c] for c in chs])
    v = patterns.empirical(name, list(chs), electrode_pos=pos, z=0.0)
    assert np.all(np.abs(v) <= 1.0)
    centre, moment = patterns._FALLBACK[name]
    for i, c in enumerate(chs):
        if c not in before:
            continue
        assert v[i] == patterns.analytic_dipole(centre, moment, pos[i : i + 1])[0], c
        assert np.sign(v[i]) == np.sign(ref[c]), (c, v[i], ref[c])
        assert 0.25 * abs(ref[c]) <= abs(v[i]) <= 3.0 * abs(ref[c]), (c, v[i], ref[c])
        assert abs(v[i] - before[c]) > 0.1  # the old value was off in size or sign


def test_mirror_montage_reference_values(real_eog):
    """The montage the sparse tests compare against, pinned (z = 0). Fix round 5 moved blink A2
    (-0.106 -> -0.090) and F10 (-0.256 -> -0.213): F8, at the dipole's null (cos +0.041), no
    longer anchors them; F7, their mirror anchor, disagrees in sign and never did."""
    blink, _ = _mirror_montage("blink")
    heog, _ = _mirror_montage("heog")
    want_blink = {"A1": -0.107, "A2": -0.090, "F9": -0.239, "F10": -0.213}
    for c, x in want_blink.items():
        assert blink[c] == pytest.approx(x, abs=0.001), c
    assert heog["F9"] == pytest.approx(-0.792, abs=0.001)


# Random sparse requests: 2-6 channels of CHANNELS_19 plus 1-2 of `_EXTRA_CHANNELS`, z = 0,
# against `_mirror_montage` (the reviewer's sweep, another seed). Before fix round 5 the
# reviewer's seed gave (blink / heog) 71 / 27 fallbacks more than 3x too large and 18 / 16 sign
# flips; on this seed 57f3560 gives 70 / 34 and 12 / 16. Measured now, over 4455 fallbacks per
# map: no flips and 0 / 6 more than 3x too large. All 6 are heog TP9 or P9 (3.1x) with O2 as the
# only usable anchor: the referenced heog map is lopsided there (O2 +0.090, O1 +0.012); O2 is far
# from the model's null (cos +0.153). The clip to the map peak fires 0 / 405 times, all at heog
# F9 (178) and F10 (227), which lie nearer the eyes than F7/F8 (see `empirical`).
_SWEEP_REQUESTS = 3000
_SWEEP_MAX_OVER_3X = {"blink": 0, "heog": 6}


@pytest.mark.parametrize("name", ["blink", "heog"])
def test_sparse_montage_sweep_keeps_sign_range_and_size(real_eog, name):
    ref, where = _mirror_montage(name)
    rng = np.random.default_rng(20260917)
    over = 0
    for _ in range(_SWEEP_REQUESTS):
        present = [str(c) for c in rng.choice(CHANNELS_19, int(rng.integers(2, 7)), replace=False)]
        k = int(rng.integers(1, 3))
        missing = [str(c) for c in rng.choice(_EXTRA_CHANNELS, k, replace=False)]
        chs = present + missing
        v = patterns.empirical(name, chs, electrode_pos=np.array([where[c] for c in chs]), z=0.0)
        assert np.all(np.abs(v) <= 1.0), (chs, v)
        for c, x in zip(missing, v[len(present) :], strict=True):
            assert np.sign(x) == np.sign(ref[c]), (chs, c, x, ref[c])
            over += abs(x) > 3.0 * abs(ref[c])
    assert over <= _SWEEP_MAX_OVER_3X[name], over


def _referenced(name):
    """The raw file map for `name`, its T9/T10-referenced mean and its sd, computed here."""
    f = patterns.load_eog_patterns()
    names = [str(c) for c in f["channel_names"]]
    raw = np.asarray(f[f"{name}_mean"], float)
    mean = raw - raw[[names.index("T9"), names.index("T10")]].mean()
    return names, raw, mean, np.asarray(f[f"{name}_sd"], float)


def _jittered(name, z, ch, anchor):
    """`ch`'s jittered file value at z, before the division by the map peak, recovered from its
    ratio to `anchor`: a channel whose sd is below 0.9 x |mean| (so its step is its sd) and whose
    sign the T9/T10 reference does not change."""
    names, raw, mean, sd = _referenced(name)
    a = names.index(anchor)
    assert sd[a] < 0.9 * abs(mean[a]) and np.sign(raw[a]) == np.sign(mean[a])
    v = patterns.empirical(name, [ch, anchor], z=z)
    return v[0] / v[1] * (mean[a] + z * np.sign(mean[a]) * sd[a])


@pytest.mark.parametrize(("name", "ch", "anchor"), [("blink", "T4", "Fp1"), ("heog", "FT7", "F7")])
@pytest.mark.parametrize("z", [-1.0, 1.0])
def test_real_file_jitter_step_is_capped_at_0_9_of_the_mean(real_eog, name, ch, anchor, z):
    """A channel whose sd exceeds its |mean| (blink T4: sd 0.059, |mean| 0.044; heog FT7: sd
    0.327, |mean| 0.297) moves by exactly 0.9 x |mean| at z = +-1 (a cap of 0.5 or 0.99 would
    move it by 0.5 or 0.99 x |mean|)."""
    names, _, mean, sd = _referenced(name)
    i = names.index(ch)
    assert sd[i] > abs(mean[i])
    step = (_jittered(name, z, ch, anchor) - mean[i]) / (z * np.sign(mean[i]))
    assert step == pytest.approx(0.9 * abs(mean[i]), rel=1e-9)


@pytest.mark.parametrize(("name", "ch", "anchor"), [("blink", "FC5", "Fp1"), ("heog", "FC1", "F7")])
def test_real_file_jitter_direction_follows_the_referenced_mean(real_eog, name, ch, anchor):
    """The jitter moves a channel along the sign of its REFERENCED mean. The T9/T10 reference
    flips blink FC5 (raw -0.009, referenced +0.194; the largest such channel of 34) and heog FC1
    (raw -0.008, referenced +0.022; the largest of 5): as z grows both move up, away from zero,
    where the raw sign would move them down."""
    names, raw, mean, sd = _referenced(name)
    i = names.index(ch)
    assert raw[i] < 0.0 < mean[i]
    step = min(sd[i], 0.9 * abs(mean[i]))
    for z in (-1.0, -0.5, 0.5, 1.0):
        got = _jittered(name, z, ch, anchor)
        assert got == pytest.approx(mean[i] + z * step, rel=1e-9), z
        assert (got > mean[i]) == (z > 0.0)
