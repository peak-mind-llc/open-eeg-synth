from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from open_eeg_synth.artifacts import jaw_emg
from open_eeg_synth.artifacts.base import Occupancy, Remedy, RenderContext
from open_eeg_synth.artifacts.jaw_emg import JawEmg
from open_eeg_synth.artifacts.registry import make_artifact
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.case import ArtifactSpec, make_case
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.recipes import resting_case
from open_eeg_synth.seeds import stream_rng
from tests.helpers import band_power

FS = 256.0


def _bind(art, seed=1, head=None):
    head = head or load_head_model()
    art.bind(
        RenderContext(
            head.channels,
            FS,
            head.electrode_pos,
            head,
            StateTimeline.constant("eyes_open"),
            stream_rng(seed, "t"),
            stream_rng(seed, "s"),
            Occupancy(),
            "artifact:emg",
        )
    )


def test_burst_is_broadband_above_20hz_and_band_limited():
    w = JawEmg.burst(FS, 2.0, np.random.default_rng(0), 8, 80.0)
    assert len(w) == 512
    plateau = w[77:435]
    assert abs(plateau.std() - 1.0) < 0.15
    hi = band_power(plateau, FS, 30.0, 100.0)[0]
    lo = band_power(plateau, FS, 2.0, 12.0)[0]
    top = band_power(plateau, FS, 110.0, 128.0)[0]
    assert hi > 5 * lo and top < 0.5 * band_power(plateau, FS, 60.0, 100.0)[0]


def test_burst_beta_band_matches_the_generator_spectrum():
    """Nothing folds back into the visible band (DESIGN §5.6). The 13-30 Hz power of a burst,
    relative to its 30-100 Hz power (the anti-alias filter is flat below 100 Hz), matches what
    |H(f)|^2 predicts. Decimating the 8x signal by taking every 8th sample instead folds
    ~3.7x the legitimate power into 13-30 Hz and lands at 1.6-2.8x this prediction (40 seeds);
    JawEmg.burst lands at 0.73-1.19x."""
    peak_hz = 80.0

    def h2(f):
        return ((f / peak_hz) / (1.0 + (f / peak_hz) ** 2)) ** 2

    predicted = quad(h2, 13.0, 30.0)[0] / quad(h2, 30.0, 100.0)[0]
    ratios = []
    for seed in range(8):
        w = JawEmg.burst(FS, 10.0, np.random.default_rng(seed), 8, peak_hz)
        plateau = w[int(0.15 * len(w)) : int(0.85 * len(w))]
        beta = band_power(plateau, FS, 13.0, 30.0)[0]
        ref = band_power(plateau, FS, 30.0, 100.0)[0]
        ratios.append(beta / ref / predicted)
    assert all(0.6 <= r <= 1.4 for r in ratios), ratios
    assert 0.85 <= np.mean(ratios) <= 1.15, ratios


def test_pattern_is_the_anatomical_temporalis_placement():
    """Centre 5 mm lateral, 5 mm anterior and 7.5 mm inferior of T3/T4, sigma 50 mm; values are
    relative to the loudest full-head channel (T3 / T4)."""
    head = load_head_model()
    for side, t, sx in (("left", "T3", -1.0), ("right", "T4", 1.0)):
        centre = {"left": jaw_emg.LEFT_CENTRE, "right": jaw_emg.RIGHT_CENTRE}[side]
        offset_mm = (centre - head.electrode_pos[CHANNELS_19.index(t)]) * 1000.0
        assert np.allclose(offset_mm, [5.0 * sx, 5.0, -7.5], atol=1e-6)
    art = make_artifact("emg")
    assert art.sigma_mm == 50.0
    _bind(art)
    want = {
        "left": {"T3": 1.00, "F7": 0.72, "T5": 0.40, "C3": 0.21},
        "right": {"T4": 1.00, "F8": 0.59, "T6": 0.38, "C4": 0.29},
    }
    for side, values in want.items():
        p = art.pattern[side]
        assert np.isclose(p.max(), 1.0) and CHANNELS_19[int(np.argmax(p))] in values
        for ch, v in values.items():
            assert p[CHANNELS_19.index(ch)] == pytest.approx(v, abs=0.05), (side, ch)
    assert art.pattern["left"][CHANNELS_19.index("T4")] < 0.01


def test_pattern_keeps_its_full_head_scale_on_a_subset():
    """Without T3 in the recording, C3 stays at its full-head 0.21, not stretched to 1."""
    rows = [CHANNELS_19.index(c) for c in ("C3", "C4", "P3", "P4")]
    for side in ("left", "right", "both"):
        kw = {"side": side, "rate_by_state": {"eyes_open": 0.3}, "min_gap_s": 0.5}
        full, sub = make_artifact("emg", **kw), make_artifact("emg", **kw)
        _bind(full)
        _bind(sub, head=load_head_model().subset(("C3", "C4", "P3", "P4")))
        for s in ("left", "right"):
            assert np.allclose(sub.pattern[s], full.pattern[s][rows])
        assert sub.pattern["left"].max() < 0.3 and sub.pattern["right"].max() < 0.3
        x_full, x_sub = full.render(0, int(60 * FS)), sub.render(0, int(60 * FS))
        assert np.array_equal(x_sub, x_full[rows])
        # the truth's peak is the full-head peak too, not the peak of whatever was recorded
        tf, ts = full.truth(), sub.truth()
        assert len(tf) >= 5 and [t.peak_uv for t in ts] == [t.peak_uv for t in tf]
        for t in tf:
            a, b = round(t.onset_s * FS), round(t.offset_s * FS)
            assert t.peak_uv == pytest.approx(float(np.abs(x_full[:, a:b]).max()), rel=1e-5)
            assert t.peak_uv > 2 * np.abs(x_sub[:, a:b]).max()


def test_sides_channels_and_truth():
    for side, loud, quiet in (("left", "T3", "T4"), ("right", "T4", "T3")):
        art = make_artifact("emg", side=side, rate_by_state={"eyes_open": 0.2}, min_gap_s=0.5)
        _bind(art)
        x = art.render(0, int(60 * FS))
        li, qi = CHANNELS_19.index(loud), CHANNELS_19.index(quiet)
        assert x[li].std() > 5 * x[qi].std()
        t = art.truth()[0]
        assert t.kind == "emg" and t.subtype == "jaw" and t.side == side and loud in t.channels
        assert t.remedies == (Remedy.MASK_SEGMENT, Remedy.REMOVE_COMPONENT)
        assert 0.5 <= t.offset_s - t.onset_s <= 3.05
        assert set(t.channels) == ({"T3", "F7", "T5"} if side == "left" else {"T4", "F8", "T6"})
    both = make_artifact("emg", side="both", rate_by_state={"eyes_open": 0.2})
    _bind(both)
    y = both.render(0, int(60 * FS))
    assert 0.3 < y[CHANNELS_19.index("T3")].std() / y[CHANNELS_19.index("T4")].std() < 3.0
    assert y[CHANNELS_19.index("Pz")].std() < 0.3 * y[CHANNELS_19.index("T3")].std()


def test_plateau_rms_is_the_drawn_rms_at_the_loudest_channel():
    art = make_artifact("emg", side="left", rate_by_state={"eyes_open": 0.1}, min_gap_s=1.0)
    _bind(art, seed=3)
    x = art.render(0, int(120 * FS))
    t3 = CHANNELS_19.index("T3")
    got = []
    for t in art.truth():
        a, b = round(t.onset_s * FS), round(t.offset_s * FS)
        if b > x.shape[1]:
            continue  # runs past the rendered span
        n = b - a
        got.append(x[t3, a + int(0.15 * n) : a + int(0.85 * n)].std() / t.params["rms_uv"])
    assert len(got) >= 5 and all(0.98 < g < 1.02 for g in got), got  # rms_uv is rounded


def test_random_side_uses_all_three():
    art = make_artifact("emg", rate_by_state={"eyes_open": 1.0}, min_gap_s=0.0)
    _bind(art, seed=5)
    art.render(0, int(120 * FS))
    assert {t.side for t in art.truth()} == {"left", "right", "both"}


def test_explicit_empty_rates_mean_never():
    art = make_artifact("emg", rate_by_state={})
    assert art.rate_by_state == {}
    _bind(art)
    assert not art.render(0, int(120 * FS)).any() and art.truth() == []


def test_left_clench_rms_at_c3_is_the_same_on_a_subset_case():
    """Through make_case: C3's jaw layer on a C3/C4/P3/P4 case equals the 19-channel one."""
    arts = (ArtifactSpec("emg", {"side": "left", "rate_by_state": {"eyes_open": 0.1}}),)
    rms = {}
    for label, chs in (("sub", ("C3", "C4", "P3", "P4")), ("full", CHANNELS_19)):
        case = make_case(resting_case(13, duration_s=60.0, artifacts=arts, channels=chs))
        rec = case.recordings["eyes_open"]
        layer = rec.layers["artifact:emg"][rec.channels.index("C3")]
        rms[label] = float(np.sqrt(np.mean(layer.astype(float) ** 2)))
    assert rms["full"] > 1.0
    assert rms["sub"] == pytest.approx(rms["full"], rel=0.01)
