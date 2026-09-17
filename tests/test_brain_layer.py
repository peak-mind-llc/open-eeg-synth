from __future__ import annotations

import numpy as np

from open_eeg_synth.brain.layer import BrainLayer, BrainSpec
from open_eeg_synth.brain.network import wire_network
from open_eeg_synth.brain.placement import placed_centres, region_centres
from open_eeg_synth.brain.plants import LateralImbalance, apply_modifiers
from open_eeg_synth.brain.rhythm import Rhythm, draw_patch_params
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.recipes import resting_brain
from open_eeg_synth.seeds import stream_rng, stream_seed
from tests.helpers import band_power, psd_slope, render_whole_and_chunked

FS = 256.0


_MIXING: dict[float, np.ndarray] = {}  # the nominal head's background mixing, by smoothing_mm


def _build(spec: BrainSpec, timeline, seed=1):
    """The brain layer as make_subject/make_engine build it, on the nominal head.

    Returns ``make(rhythms=None)``; ``rhythms`` replaces the spec's (compiled) rhythms, e.g. with
    a plant's modifiers applied, while the subject draws (placements, lags) stay the spec's.
    """
    head = load_head_model()
    sm = spec.background.smoothing_mm
    if sm not in _MIXING:
        _MIXING[sm] = head.smoothed_mixing(sm)
    mixing = _MIXING[sm]
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

    def make(rhythms=None):
        return BrainLayer(
            spec.rhythms if rhythms is None else rhythms,
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
# average reference, 60 s, the 9 seeds below) for the recipe calibrated in Task 12 and re-tuned
# against the realism reference in Task 32 (recipes.py): O1 alpha share 0.605 (range 0.28-0.77),
# Cz 1-45 Hz RMS 11.6 uV (10.0-14.6). (Task 12's recipe read 0.567 (0.22-0.76) and 11.9 uV
# (11.1-13.9).)
#
# These pins belong to this fixed seed set, which reads low: the population medians of the same
# measurement over 270 seeds (0-269) are O1 share 0.65 and Cz 12.7 uV (Task 12: 0.63, 12.8). Four
# of the nine subjects (seeds 2, 3, 100, 101: shares 0.28-0.39) drew alpha patches that project
# weakly onto O1, which is where the population's lower tail comes from. Thirty disjoint 9-seed
# sets from those 270 have medians from 0.45 to 0.83, so a different seed set needs its own pinned
# value.
CAL_O1_ALPHA_SHARE = 0.605
CAL_CZ_RMS_1_45_UV = 11.6


def test_calibrated_amplitudes_median_across_seeds():
    """The resting recipe's calibrated amplitudes, judged as medians over subjects.

    One seed is not representative (O1 alpha share spans 0.22-0.76 over these nine), so both
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


# A left theta imbalance must stay visible in the recipe's brain layer (Task 32 fix round 1):
# LateralImbalance("theta", "left", 0.4) raises the F4/F3 theta (4-8 Hz) power ratio (brain layer,
# nominal head, eyes closed, average reference, 20 s) by a median 2.06 dB over the six seeds below
# (per seed 1.23-3.07 dB). Population, seeds 0-239: median +2.10 dB, p10 +1.29, min +0.60; the
# medians of 40 disjoint six-seed sets run from +1.51 to +2.69. The first tuning pass's 40 mm theta
# patches gave +0.96 (six-seed medians +0.53 to +1.37), which the floor of 1.5 rejects.
THETA_IMBALANCE_SHIFT_DB = 2.06


def test_theta_lateral_imbalance_is_visible_on_the_resting_brain():
    spec = resting_brain()
    plant = LateralImbalance("theta", "left", 0.4)
    gained = tuple(apply_modifiers(spec.rhythms, plant.modifiers()))
    f3, f4 = CHANNELS_19.index("F3"), CHANNELS_19.index("F4")
    shifts = []
    for seed in range(1, 7):
        make = _build(spec, StateTimeline.constant("eyes_closed"), seed=seed)
        balance = []
        for layer in (make(), make(gained)):
            x = layer.render(0, int(20 * FS)).astype(float)
            x -= x.mean(axis=0, keepdims=True)
            p = band_power(x, FS, 4, 8)
            balance.append(10 * np.log10(p[f4] / p[f3]))
        shifts.append(balance[1] - balance[0])
    median = float(np.median(shifts))
    assert min(shifts) > 1.0, shifts
    assert median > 1.5, shifts
    assert abs(median - THETA_IMBALANCE_SHIFT_DB) < 0.3, shifts


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
