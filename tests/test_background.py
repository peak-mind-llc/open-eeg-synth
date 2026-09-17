from __future__ import annotations

import numpy as np

from open_eeg_synth.brain.background import Background, BackgroundSpec
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.seeds import stream_seed
from tests.helpers import psd_slope, render_whole_and_chunked

FS = 256.0


def test_background_rms_slope_and_chunk_invariance():
    head = load_head_model()
    spec = BackgroundSpec(network_frac=0.0)
    mixing = head.smoothed_mixing(spec.smoothing_mm)

    def make():
        return Background(head, FS, spec, stream_seed(1, "eyes_closed:background"), mixing=mixing)

    whole, chunked = render_whole_and_chunked(make, int(60 * FS), np.random.default_rng(0))
    assert np.allclose(whole, chunked, atol=1e-4)
    rms = np.sqrt(np.mean(whole**2))
    assert 16.0 < rms < 24.0  # spec.rms_uv = 20 on average over channels
    assert abs(psd_slope(whole, FS) - 1.2) < 0.15


def test_network_frac_reduces_background_share():
    head = load_head_model()
    a = Background(head, FS, BackgroundSpec(network_frac=0.0), stream_seed(2, "x")).render(0, 2560)
    b = Background(head, FS, BackgroundSpec(network_frac=0.5), stream_seed(2, "x")).render(0, 2560)
    assert np.isclose(b.std() / a.std(), np.sqrt(0.5), atol=0.02)
