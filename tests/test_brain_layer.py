from __future__ import annotations

import numpy as np

from open_eeg_synth.brain.layer import BrainLayer, BrainSpec
from open_eeg_synth.brain.network import wire_network
from open_eeg_synth.brain.placement import placed_centres, region_centres
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.recipes import resting_brain
from open_eeg_synth.seeds import stream_rng
from tests.helpers import band_power, psd_slope, render_whole_and_chunked

FS = 256.0


def _build(spec: BrainSpec, timeline, seed=1):
    head = load_head_model()
    mixing = head.smoothed_mixing(spec.background.smoothing_mm)
    wiring = wire_network(head, spec.network, FS, stream_rng(seed, "subject:network"))
    placements, f0 = {}, {}
    for r in spec.rhythms:
        rng = stream_rng(seed, f"subject:rhythm:{r.name}")
        placements[r.name] = (
            region_centres(head, r.region, r.n_patches, rng)
            if r.region
            else placed_centres(head, r.sites, rng)
        )
        f0[r.name] = r.f0_hz + (float(rng.normal(0, r.f0_jitter_hz)) if r.f0_jitter_hz else 0.0)

    def make():
        return BrainLayer(
            spec.rhythms,
            spec,
            head,
            FS,
            timeline,
            mixing,
            wiring,
            placements,
            f0,
            seed,
            "eyes_closed",
        )

    return make


def test_resting_brain_spec_roundtrip_and_names():
    spec = resting_brain()
    assert [r.name for r in spec.rhythms] == ["alpha", "theta", "beta", "smr"]
    assert BrainSpec.from_dict(spec.to_dict()) == spec


def test_brain_layer_chunk_invariance_and_eyes_closed_alpha():
    make = _build(resting_brain(), StateTimeline.constant("eyes_closed"))
    whole, chunked = render_whole_and_chunked(make, int(60 * FS), np.random.default_rng(0))
    assert np.allclose(whole, chunked, atol=1e-3)
    # Amplitudes are judged after average referencing, as DESIGN §9.1 measures them: the lead
    # field's native reference carries a common component (DESIGN §2.3) that inflates raw RMS.
    x = whole - whole.mean(axis=0, keepdims=True)
    o1, fz = CHANNELS_19.index("O1"), CHANNELS_19.index("Fz")
    alpha = band_power(x, FS, 8, 13)
    total = band_power(x, FS, 1, 40)
    assert alpha[o1] / total[o1] > 0.3  # provisional; Task 12 pins the calibrated value (~0.6)
    assert alpha[o1] > 1.5 * alpha[fz]
    assert 0.9 < psd_slope(x, FS) < 1.5
    assert 6.0 < x[CHANNELS_19.index("Cz")].std() < 30.0  # provisional; Task 12 pins ~12 uV


def test_eyes_open_collapses_alpha():
    ec = _build(resting_brain(), StateTimeline.constant("eyes_closed"))().render(0, int(30 * FS))
    eo = _build(resting_brain(), StateTimeline.constant("eyes_open"))().render(0, int(30 * FS))
    ec, eo = ec - ec.mean(axis=0, keepdims=True), eo - eo.mean(axis=0, keepdims=True)
    o1 = CHANNELS_19.index("O1")
    assert band_power(eo, FS, 8, 13)[o1] < 0.5 * band_power(ec, FS, 8, 13)[o1]
