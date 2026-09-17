from __future__ import annotations

import numpy as np

from open_eeg_synth.brain.layer import BrainLayer, BrainSpec
from open_eeg_synth.brain.network import wire_network
from open_eeg_synth.brain.placement import placed_centres, region_centres
from open_eeg_synth.brain.rhythm import Rhythm, draw_patch_params
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.recipes import resting_brain
from open_eeg_synth.seeds import stream_rng, stream_seed
from tests.helpers import band_power, psd_slope, render_whole_and_chunked

FS = 256.0


def _build(spec: BrainSpec, timeline, seed=1):
    """The brain layer as make_subject/make_engine build it, on the nominal head."""
    head = load_head_model()
    mixing = head.smoothed_mixing(spec.background.smoothing_mm)
    wiring = wire_network(head, spec.network, FS, stream_rng(seed, "subject:network"))
    placements, f0, offsets, lags = {}, {}, {}, {}
    for r in spec.rhythms:
        rng = stream_rng(seed, f"subject:rhythm:{r.name}")
        placements[r.name] = (
            region_centres(head, r.region, r.n_patches, rng)
            if r.region
            else placed_centres(head, r.sites, rng)
        )
        f0[r.name] = r.f0_hz + (float(rng.normal(0, r.f0_jitter_hz)) if r.f0_jitter_hz else 0.0)
        offsets[r.name], lags[r.name] = draw_patch_params(r, len(placements[r.name]), rng)

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
            f0_offsets_hz=offsets,
            lags_ms=lags,
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
    assert alpha[o1] > 1.5 * alpha[fz]
    assert 0.9 < psd_slope(x, FS) < 1.5
    # Coarse only: an unfiltered std is dominated by the background's sub-1 Hz power. The
    # calibrated, band-limited amplitude is pinned as a multi-seed median in the next test.
    assert 6.0 < x[CHANNELS_19.index("Cz")].std() < 30.0


# Calibrated medians of this test's own measurement (brain layer only, nominal head, eyes closed,
# average reference, 60 s, the 9 seeds below) for the recipe calibrated in Task 12
# (recipes.py): O1 alpha share 0.567 (range 0.20-0.76), Cz 1-45 Hz RMS 11.7 uV (11.1-14.0).
CAL_O1_ALPHA_SHARE = 0.567
CAL_CZ_RMS_1_45_UV = 11.7


def test_calibrated_amplitudes_median_across_seeds():
    """The resting recipe's calibrated amplitudes, judged as medians over subjects.

    One seed is not representative (O1 alpha share spans 0.20-0.76 over these nine), so both
    numbers are medians. Alpha share = 8-13 / 1-40 Hz power at O1, pinned to at least 0.45 (the
    realism target is ~0.6, DESIGN §9.1). Cz RMS is read over 1-45 Hz, as §9.1 measures it (the
    unfiltered value is dominated by sub-1 Hz background), within +/-30 % of the calibrated median.
    """
    o1, cz = CHANNELS_19.index("O1"), CHANNELS_19.index("Cz")
    shares, rms = [], []
    for seed in (1, 2, 3, 100, 101, 102, 103, 104, 105):  # 9 seeds, >= the required 6
        x = _build(resting_brain(), StateTimeline.constant("eyes_closed"), seed=seed)().render(
            0, int(60 * FS)
        )
        x = x - x.mean(axis=0, keepdims=True)
        alpha, total = band_power(x, FS, 8, 13), band_power(x, FS, 1, 40)
        shares.append(alpha[o1] / total[o1])
        rms.append(float(np.sqrt(band_power(x, FS, 1, 45)[cz])))
    assert 0.45 < float(np.median(shares)) < 0.75
    assert 0.7 * CAL_CZ_RMS_1_45_UV < float(np.median(rms)) < 1.3 * CAL_CZ_RMS_1_45_UV
    assert abs(float(np.median(shares)) - CAL_O1_ALPHA_SHARE) < 0.1


def test_eyes_open_collapses_alpha():
    ec = _build(resting_brain(), StateTimeline.constant("eyes_closed"))().render(0, int(30 * FS))
    eo = _build(resting_brain(), StateTimeline.constant("eyes_open"))().render(0, int(30 * FS))
    ec, eo = ec - ec.mean(axis=0, keepdims=True), eo - eo.mean(axis=0, keepdims=True)
    o1 = CHANNELS_19.index("O1")
    assert band_power(eo, FS, 8, 13)[o1] < 0.5 * band_power(ec, FS, 8, 13)[o1]


def test_brain_layer_part_matches_standalone_rhythm_stream():
    """A BrainLayer's alpha part must be seeded exactly like a standalone Rhythm built
    from the DESIGN §8.1 stream ``<condition>:rhythm:alpha`` (fix round 1, item 3d, P17;
    mutation M5 points the rhythm stream at the background stream instead), with its
    per-patch offsets and lags drawn from ``subject:rhythm:alpha``."""
    head = load_head_model()
    spec = resting_brain()
    tl = StateTimeline.constant("eyes_closed")
    seed = 1
    make = _build(spec, tl, seed=seed)
    layer = make()
    alpha_spec = spec.rhythms[0]
    assert alpha_spec.name == "alpha"
    part = layer.parts[2]
    assert part.name == "brain.rhythm:alpha"

    rng = stream_rng(seed, "subject:rhythm:alpha")
    centres = region_centres(head, alpha_spec.region, alpha_spec.n_patches, rng)
    f0 = alpha_spec.f0_hz + float(rng.normal(0, alpha_spec.f0_jitter_hz))
    # per-patch offsets and lags are subject draws, continuing subject:rhythm:alpha (DESIGN §8.1)
    offsets, lags = draw_patch_params(alpha_spec, len(centres), rng)
    standalone = Rhythm(
        head,
        FS,
        alpha_spec,
        tl,
        centres,
        f0,
        stream_seed(seed, "eyes_closed:rhythm:alpha"),
        f0_offsets_hz=offsets,
        lags_ms=lags,
    )
    assert part.f0_offsets_hz == offsets and part.lags_ms == lags

    n = int(5 * FS)
    assert np.array_equal(part.render(0, n), standalone.render(0, n))


def test_rows_selects_full_output_columns():
    """rows= must equal the full 19-channel render indexed by those rows (fix round 1,
    item 3e, P3; mutation M6 ignores rows)."""
    make = _build(resting_brain(), StateTimeline.constant("eyes_closed"))
    n = int(10 * FS)
    full = make().render(0, n)
    rows = [CHANNELS_19.index(c) for c in ("O1", "Cz", "Fp2", "O2")]
    subset_layer = make()
    subset_layer.rows = rows
    subset = subset_layer.render(0, n)
    assert subset.shape == (len(rows), n)
    assert np.array_equal(subset, full[rows])


def test_network_layer_contributes_to_brain_output():
    """Dropping the network part must change the rendered output (fix round 1, item 3f;
    mutation M4 drops Network from ``parts`` entirely)."""
    make = _build(resting_brain(), StateTimeline.constant("eyes_closed"))
    layer = make()
    names = [p.name for p in layer.parts]
    assert "brain.network" in names
    n = int(5 * FS)
    full = layer.render(0, n)

    layer2 = make()
    idx = [p.name for p in layer2.parts].index("brain.network")
    layer2.parts[idx].render = lambda t0, n: np.zeros((len(CHANNELS_19), n), dtype=np.float32)
    reduced = layer2.render(0, n)
    assert not np.allclose(full, reduced)
