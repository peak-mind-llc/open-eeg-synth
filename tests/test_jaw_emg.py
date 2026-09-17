from __future__ import annotations

import numpy as np

from open_eeg_synth import dsp
from open_eeg_synth.artifacts.base import Occupancy, Remedy, RenderContext
from open_eeg_synth.artifacts.jaw_emg import JawEmg
from open_eeg_synth.artifacts.registry import make_artifact
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.seeds import stream_rng
from tests.helpers import band_power

FS = 256.0


def _bind(art, seed=1):
    head = load_head_model()
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


def test_burst_has_no_energy_aliased_from_above_nyquist():
    """The 8x generation is anti-alias low-passed before decimating (DESIGN §5.6): naively
    subsampling the high-rate signal instead (no low-pass first) would fold content from above
    the new Nyquist back down into the visible band, inflating power near the top of it."""
    oversample, peak_hz, dur_s = 8, 80.0, 2.0
    fs_hi = oversample * FS
    n_hi = int(round(dur_s * FS)) * oversample
    white = np.random.default_rng(7).standard_normal(n_hi)
    f = np.fft.rfftfreq(n_hi, 1.0 / fs_hi)
    h = (f / peak_hz) / (1.0 + (f / peak_hz) ** 2)
    x_hi = np.fft.irfft(np.fft.rfft(white) * h, n=n_hi)
    x_hi *= dsp.raised_cosine_envelope(n_hi, fs_hi, 0.1, 0.1)

    proper = dsp.lowpass_decimate(x_hi, oversample)  # what JawEmg.burst does internally
    naive = x_hi[::oversample]  # decimate without an anti-alias low-pass first

    top_proper = band_power(proper, FS, 110.0, 128.0)[0]
    top_naive = band_power(naive, FS, 110.0, 128.0)[0]
    assert top_naive > 3 * top_proper


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
    both = make_artifact("emg", side="both", rate_by_state={"eyes_open": 0.2})
    _bind(both)
    y = both.render(0, int(60 * FS))
    assert 0.3 < y[CHANNELS_19.index("T3")].std() / y[CHANNELS_19.index("T4")].std() < 3.0
    assert y[CHANNELS_19.index("Pz")].std() < 0.3 * y[CHANNELS_19.index("T3")].std()


def test_random_side_uses_all_three():
    art = make_artifact("emg", rate_by_state={"eyes_open": 1.0}, min_gap_s=0.0)
    _bind(art, seed=5)
    art.render(0, int(120 * FS))
    assert {t.side for t in art.truth()} == {"left", "right", "both"}
