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


def test_empirical_fallback_uses_every_anchor_when_all_disagree_in_sign(real_eog):
    """On the real blink map F7, T3 and T4 are positive where the dipole model is negative. With
    only those three present, the sign filter has nothing to keep, so the fit uses all three."""
    head = load_head_model()
    pos = dict(zip(CHANNELS_19, head.electrode_pos, strict=True))
    a1 = _EXTRA_POS[0]
    p = np.array([pos["F7"], pos["T3"], pos["T4"], a1])
    v = patterns.empirical("blink", ["F7", "T3", "T4", "A1"], electrode_pos=p, z=0.0)
    envelope, cosine = patterns._dipole_terms(*patterns._FALLBACK["blink"], p)
    assert np.all(np.sign(v[:3]) != np.sign(cosine[:3]))
    dist = np.linalg.norm(p[:3] - a1, axis=1)
    scale = patterns._fallback_scale(v[:3], envelope[:3], cosine[:3], dist, ["F7", "T3", "T4"])
    assert np.isfinite(v[3]) and np.isclose(v[3], scale * envelope[3] * cosine[3])


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
    """The grid version put AF10 at 32x AF9 (a strong anchor won a distance tie on one side)."""
    head = load_head_model()
    chs = [*CHANNELS_19, "AF9", "AF10"]
    pos = np.vstack([head.electrode_pos, _AF_POS["AF9"], _AF_POS["AF10"]])
    v = patterns.empirical("heog", chs, electrode_pos=pos, z=0.0)
    assert v[19] < 0.0 < v[20]
    assert 0.5 <= abs(v[20]) / abs(v[19]) <= 2.0


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
    got = patterns._fallback_scale(values, envelope, cosine, dist, ["a", "b"])
    assert np.isclose(got, want)
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
    assert np.isclose(got, 1.0)
    # the same anchors in reverse request order give the same answer
    rev = patterns._fallback_scale(values[::-1], ones, ones, dist[::-1], [str(i) for i in range(n)])
    assert np.isclose(rev, 1.0)


def test_fallback_scale_breaks_exact_distance_ties_by_label_not_request_order():
    k = patterns._FALLBACK_K
    n = k + 1
    ones = np.ones(n)
    dist = np.ones(n)  # every anchor equally far: the k kept are the first k labels
    values = np.ones(n)
    values[0] = 5.0
    labels = [f"c{i}" for i in range(n)]
    labels[0] = "zz"  # sorts last, so it is the one left out
    assert np.isclose(patterns._fallback_scale(values, ones, ones, dist, labels), 1.0)
    order = np.arange(n)[::-1]
    got = patterns._fallback_scale(
        values[order], ones, ones, dist[order], [labels[i] for i in order]
    )
    assert np.isclose(got, 1.0)


def test_fallback_scale_floor_keeps_a_near_null_anchor_from_blowing_up():
    """An anchor whose direction cosine is near zero (at the model's null) is floored at 0.05, so
    alone it gives 0.1 / 0.05 = 2.0 rather than 0.1 / 1e-6 = 100000."""
    got = patterns._fallback_scale(
        np.array([0.1]), np.array([1.0]), np.array([1e-6]), np.array([0.01]), ["a"]
    )
    assert np.isclose(got, 2.0)
    neg = patterns._fallback_scale(
        np.array([-0.1]), np.array([1.0]), np.array([-1e-6]), np.array([0.01]), ["a"]
    )
    assert np.isclose(neg, 2.0)
