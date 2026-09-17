from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy.signal import csd
from scipy.signal import welch as sp_welch

from open_eeg_synth.brain.placement import placed_centres, region_centres
from open_eeg_synth.brain.rhythm import BurstGate, Rhythm, RhythmSpec, _orient_patches, _Oscillator
from open_eeg_synth.brain.state import StateSegment, StateTimeline
from open_eeg_synth.channels import CHANNELS_19, MIRROR, canonical_label
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.recipes import resting_brain
from open_eeg_synth.seeds import stream_rng, stream_seed
from tests.helpers import band_power, render_whole_and_chunked, welch

FS = 256.0


def test_placed_centres_mirror_pairs_and_midline():
    head = load_head_model()
    c = placed_centres(head, ("C3", "C4", "Cz"), stream_rng(1, "subject:rhythm:smr"))
    assert len(c) == 3
    assert c[1] == head.mirror_source(c[0])
    assert abs(head.source_pos[c[2], 0]) < 0.012  # near the midline
    assert placed_centres(head, ("C3", "C4", "Cz"), stream_rng(1, "subject:rhythm:smr")) == c


def test_region_centres_posterior_mirrored():
    head = load_head_model()
    c = region_centres(head, "posterior", 4, stream_rng(1, "subject:rhythm:alpha"))
    assert len(c) == 4 and c[2] == head.mirror_source(c[0])
    assert all(head.source_pos[i, 1] < -0.04 for i in c)


def test_rhythm_peaks_at_f0_on_targets_and_is_chunk_invariant():
    head = load_head_model()
    spec = RhythmSpec("smr", f0_hz=13.5, amp_uv=6.0, sites=("C3", "C4", "Cz"))
    centres = placed_centres(head, spec.sites, stream_rng(1, "subject:rhythm:smr"))
    tl = StateTimeline.constant("eyes_closed")

    def make():
        return Rhythm(head, FS, spec, tl, centres, 13.5, stream_seed(1, "eyes_closed:rhythm:smr"))

    whole, chunked = render_whole_and_chunked(make, int(60 * FS), np.random.default_rng(0))
    assert np.allclose(whole, chunked, atol=1e-4)
    c3, o1 = CHANNELS_19.index("C3"), CHANNELS_19.index("O1")
    inband = band_power(whole, FS, 12.5, 14.5)
    around = band_power(whole, FS, 17.0, 19.0)
    assert inband[c3] > 20 * around[c3]
    assert inband[c3] > 3 * inband[o1]
    peak = np.sqrt(2) * whole[c3].std()
    assert 3.0 < peak < 12.0  # amp_uv 6 at the loudest target, envelope 0.25-1.8


def test_state_gain_and_burst_gate():
    head = load_head_model()
    tl = StateTimeline(
        [StateSegment(0, 10, "eyes_closed"), StateSegment(10, 20, "eyes_open")], ramp_s=0.0
    )
    spec = RhythmSpec("alpha", 10.0, 20.0, sites=("O1", "O2"), state_gain={"eyes_open": 0.3})
    centres = placed_centres(head, spec.sites, stream_rng(2, "p"))
    r = Rhythm(head, FS, spec, tl, centres, 10.0, stream_seed(2, "t"))
    x = r.render(0, int(20 * FS))
    o1 = CHANNELS_19.index("O1")
    assert x[o1, : int(10 * FS)].std() > 2.5 * x[o1, int(10 * FS) :].std()

    g = RhythmSpec(
        "b", 6.0, 30.0, sites=("Fz",), burst=BurstGate(on_s=(1, 1), off_s=(2, 2), ramp_s=0.0)
    )
    rb = Rhythm(
        head,
        FS,
        g,
        StateTimeline.constant("eyes_closed"),
        placed_centres(head, g.sites, stream_rng(3, "p")),
        6.0,
        stream_seed(3, "t"),
    )
    y = rb.render(0, int(9 * FS))[CHANNELS_19.index("Fz")]
    active = np.abs(y).reshape(9, int(FS)).max(axis=1) > 1.0
    assert 2 <= active.sum() <= 4  # on for 1 s every 3 s, starting after the first off period


def test_rhythm_spec_roundtrip():
    s = RhythmSpec(
        "theta", 6.0, 7.0, sites=("Fz", "Cz"), state_gain={"drowsy": 2.0}, burst=BurstGate()
    )
    assert RhythmSpec.from_dict(s.to_dict()) == s


def test_rhythm_spec_needs_exactly_one_of_sites_or_region():
    with pytest.raises(ValueError, match="exactly one"):
        RhythmSpec("x", 10.0, 1.0)  # neither sites nor region
    with pytest.raises(ValueError, match="exactly one"):
        RhythmSpec("x", 10.0, 1.0, sites=("Fz",), region="posterior")  # both
    with pytest.raises(ValueError, match="even"):
        RhythmSpec("x", 10.0, 1.0, region="posterior", n_patches=3)


def test_mirror_pairs_consistent_polarity_and_magnitude():
    """Every mirror pair's map has the same sign at its own electrode(s) and a similar
    magnitude there, for both placements (fix round 1, item 1: a mirror patch used to be
    oriented independently of its partner and could end up anti-correlated with it)."""
    head = load_head_model()
    mismatches = []
    ratios = []
    for spec in resting_brain().rhythms:
        for seed in range(1, 21):
            rng = stream_rng(seed, f"subject:rhythm:{spec.name}")
            centres = (
                region_centres(head, spec.region, spec.n_patches, rng)
                if spec.region
                else placed_centres(head, spec.sites, rng)
            )
            r = Rhythm(
                head,
                FS,
                spec,
                StateTimeline.constant("eyes_closed"),
                centres,
                spec.f0_hz,
                stream_seed(seed, f"eyes_closed:rhythm:{spec.name}"),
            )
            raw = [head.patch_map(c, spec.width_mm) for c in centres]
            if spec.sites:
                ref = [head.index(canonical_label(s, head.channels)) for s in spec.sites]
            else:  # region placement: each patch's own strongest channel
                ref = [int(np.argmax(np.abs(m))) for m in raw]
            paired: set[int] = set()
            for i, ci in enumerate(centres):
                if i in paired:
                    continue
                mirror_c = head.mirror_source(ci)
                if mirror_c == ci:
                    continue
                j = next(
                    (
                        k
                        for k, ck in enumerate(centres)
                        if k != i and k not in paired and ck == mirror_c
                    ),
                    None,
                )
                if j is None:
                    continue
                paired.add(i)
                paired.add(j)
                mirror_ch = head.index(MIRROR.get(head.channels[ref[i]], head.channels[ref[i]]))
                vi, vj = r.maps[i][ref[i]], r.maps[j][mirror_ch]
                if np.sign(vi) != np.sign(vj):
                    mismatches.append((spec.name, seed, i, j))
                ratios.append(abs(vj) / abs(vi))
    assert not mismatches
    ratios = np.array(ratios)
    assert ((ratios >= 0.5) & (ratios <= 2.0)).all(), ratios[(ratios < 0.5) | (ratios > 2.0)]


def test_hemisphere_gain_applied_after_scale():
    """hemisphere_gain must not change ``scale`` (fix round 1, item 2: scale used to be
    computed from the already-gained maps, so the gain partly undid itself)."""
    head = load_head_model()
    tl = StateTimeline.constant("eyes_closed")
    spec = resting_brain().rhythms[1]  # theta: clean site placement, no region overlap
    assert spec.name == "theta"
    f3, f4 = CHANNELS_19.index("F3"), CHANNELS_19.index("F4")
    l_factors, r_factors = [], []
    for seed in range(1, 6):
        centres = placed_centres(head, spec.sites, stream_rng(seed, "subject:rhythm:theta"))
        gained = dataclasses.replace(spec, hemisphere_gain={"left": 0.4})
        r0 = Rhythm(head, FS, spec, tl, centres, spec.f0_hz, stream_seed(seed, "t"))
        r1 = Rhythm(head, FS, gained, tl, centres, spec.f0_hz, stream_seed(seed, "t"))
        assert math.isclose(r1.scale, r0.scale, rel_tol=1e-9)
        x0 = r0.render(0, int(20 * FS)).astype(float)
        x1 = r1.render(0, int(20 * FS)).astype(float)
        p0 = band_power(x0, FS, spec.f0_hz - 2, spec.f0_hz + 2)
        p1 = band_power(x1, FS, spec.f0_hz - 2, spec.f0_hz + 2)
        l_factors.append(p1[f3] / p0[f3])
        r_factors.append(p1[f4] / p0[f4])
    l_factors, r_factors = np.array(l_factors), np.array(r_factors)
    assert (l_factors < 0.4).all()  # left (F3) clearly quieter, though not a clean 0.4**2
    assert (0.85 < r_factors).all() and (r_factors < 1.15).all()  # right (F4) ~unchanged


def test_indep_controls_inter_patch_correlation():
    """indep must change how correlated a rhythm's own patches are (fix round 1, item 3g;
    mutation M3 hardcodes the shared/own mix, making indep a no-op)."""
    head = load_head_model()
    tl = StateTimeline.constant("eyes_closed")

    def corr(indep):
        spec = RhythmSpec("smr", 13.5, 6.0, sites=("C3", "C4", "Cz"), lag_ms=0.0, indep=indep)
        centres = placed_centres(head, spec.sites, stream_rng(1, "subject:rhythm:smr"))
        r = Rhythm(head, FS, spec, tl, centres, 13.5, stream_seed(1, "eyes_closed:rhythm:smr"))
        x = r.render(0, int(30 * FS)).astype(float)
        c3, cz = CHANNELS_19.index("C3"), CHANNELS_19.index("Cz")
        return abs(np.corrcoef(x[c3], x[cz])[0, 1])

    shared, own = corr(0.0), corr(1.0)
    assert shared > 0.9
    assert own < 0.6
    assert shared - own > 0.3


def test_lag_produces_cross_spectrum_phase_between_mirror_sites():
    """A non-zero per-patch lag must show up as a non-zero cross-spectrum phase between
    the sites it drives (fix round 1, item 3a; mutation M1 zeroes every lag)."""
    head = load_head_model()
    tl = StateTimeline.constant("eyes_closed")
    spec = RhythmSpec("smr", 13.5, 6.0, sites=("C3", "C4", "Cz"), lag_ms=20.0)
    centres = placed_centres(head, spec.sites, stream_rng(1, "subject:rhythm:smr"))
    c3, c4 = CHANNELS_19.index("C3"), CHANNELS_19.index("C4")

    def phase_deg(zero_lags: bool) -> float:
        r = Rhythm(head, FS, spec, tl, centres, 13.5, stream_seed(1, "eyes_closed:rhythm:smr"))
        if zero_lags:
            r.lags = [0] * len(centres)
            r.maxlag = 0
            r.hist = np.zeros(0)
        x = r.render(0, int(120 * FS)).astype(float)
        f, pxy = csd(x[c3], x[c4], FS, nperseg=512)
        _, pxx = sp_welch(x[c3], FS, nperseg=512)
        _, pyy = sp_welch(x[c4], FS, nperseg=512)
        band = (f >= 12) & (f <= 15)
        coh = pxy[band].sum() / np.sqrt(pxx[band].sum() * pyy[band].sum())
        return float(np.degrees(np.angle(coh)))

    zero_phase = phase_deg(True)
    lagged_phase = phase_deg(False)
    assert abs(zero_phase) < 10.0
    assert abs(lagged_phase - zero_phase) > 10.0


def test_drowsy_state_shift_moves_alpha_peak():
    """state_f0_shift_hz must move the rendered spectral peak, not just the internal
    frequency target (fix round 1, item 3c; mutation M2 zeroes the shift)."""
    head = load_head_model()
    spec = resting_brain().rhythms[0]  # alpha: state_f0_shift_hz={"drowsy": -1.0}
    assert spec.name == "alpha" and spec.state_f0_shift_hz == {"drowsy": -1.0}
    centres = region_centres(
        head, spec.region, spec.n_patches, stream_rng(1, "subject:rhythm:alpha")
    )
    o1 = CHANNELS_19.index("O1")

    def peak_hz(state: str) -> float:
        r = Rhythm(
            head,
            FS,
            spec,
            StateTimeline.constant(state),
            centres,
            10.0,
            stream_seed(1, "eyes_closed:rhythm:alpha"),
        )
        x = r.render(0, int(120 * FS))[o1]
        f, p = welch(x, FS, int(8 * FS))
        m = (f > 5) & (f < 15)
        return float(f[m][np.argmax(p[0][m])])

    ec_peak = peak_hz("eyes_closed")
    drowsy_peak = peak_hz("drowsy")
    assert -1.5 < (drowsy_peak - ec_peak) < -0.5


def test_orient_patches_flips_to_positive_and_mirrors_partner():
    """Direct, geometry-independent check of the orientation helper itself: a patch
    that starts out negative at its own reference channel is flipped positive, and its
    mirror partner is then forced to agree there too. Real cortical geometry is almost
    always already positive at a patch's own reference channel (DESIGN §4.3's flip is a
    rare correction), so a test built only from real seeds could pass by luck even with
    the flip removed entirely (mutation M13); this constructs the negative case
    directly."""
    head = load_head_model()
    c3, c4 = head.index("C3"), head.index("C4")
    centres = placed_centres(head, ("C3", "C4"), stream_rng(1, "subject:rhythm:smr"))
    ref = [c3, c4]
    raw = [head.patch_map(centres[0], 12.0), head.patch_map(centres[1], 12.0)]
    raw = [-raw[0] if raw[0][c3] > 0 else raw[0], raw[1]]  # force patch 0 negative at C3
    assert raw[0][c3] < 0

    oriented = _orient_patches(head, centres, raw, ref)
    assert oriented[0][c3] >= 0  # flipped positive at its own reference channel
    assert oriented[1][c4] >= 0  # mirror partner made to agree there too


def test_own_oscillators_get_distinct_per_patch_frequency_jitter():
    """Each patch's own oscillator is centred at f0 + a per-patch draw (fix round 1,
    item 3; mutation M7 zeroes that draw, locking every patch to exactly f0)."""
    head = load_head_model()
    spec = RhythmSpec("smr", 13.5, 6.0, sites=("C3", "C4", "Cz"))
    centres = placed_centres(head, spec.sites, stream_rng(1, "subject:rhythm:smr"))
    r = Rhythm(
        head,
        FS,
        spec,
        StateTimeline.constant("eyes_closed"),
        centres,
        13.5,
        stream_seed(1, "eyes_closed:rhythm:smr"),
    )
    freqs = [osc.f0 for osc in r.own]
    assert len({round(f, 6) for f in freqs}) > 1
    assert any(abs(f - 13.5) > 0.05 for f in freqs)


def test_driver_history_is_not_zeroed_at_construction():
    """The driver's pre-render history is drawn from its stationary distribution, not
    zeroed, so there is no warm-up transient (fix round 1, item 3; mutation M12 zeroes
    it)."""
    head = load_head_model()
    spec = RhythmSpec("smr", 13.5, 6.0, sites=("C3", "C4", "Cz"), lag_ms=100.0)
    centres = placed_centres(head, spec.sites, stream_rng(1, "subject:rhythm:smr"))
    r = Rhythm(
        head,
        FS,
        spec,
        StateTimeline.constant("eyes_closed"),
        centres,
        13.5,
        stream_seed(1, "eyes_closed:rhythm:smr"),
    )
    assert r.maxlag > 0
    assert r.hist.std() > 0.01


def test_midline_site_is_always_the_nearest_source_deterministically():
    """A midline site takes the single source nearest the midline, not a random pick
    (fix round 1, item 5a, DESIGN §4.3; mutation-equivalent to disabling that branch)."""
    head = load_head_model()
    cand = head.sources_under("Cz", 25)
    expected = int(cand[np.argsort(np.abs(head.source_pos[cand, 0]))[0]])
    for seed in range(1, 11):
        c = placed_centres(head, ("Cz",), stream_rng(seed, "subject:rhythm:x"))
        assert c[0] == expected


def test_envelope_is_clipped_to_configured_range():
    """The log-envelope OU process is clipped to [env_lo, env_hi] before exponentiating
    (fix round 1, item 3; mutation M17 removes the clip)."""
    spec = RhythmSpec(
        "x", 10.0, 1.0, sites=("Fz",), env_lo=0.25, env_hi=1.8, env_sd=5.0, env_tau_s=0.05
    )
    osc = _Oscillator(FS, 10.0, spec, np.random.default_rng(0))
    y = osc.render(int(60 * FS))
    assert np.abs(y).max() <= spec.env_hi + 1e-9


def test_frequency_wander_produces_measurable_instantaneous_spread():
    """f_sd must make the instantaneous frequency wander around f0 (fix round 1, item 3;
    mutation M18 zeroes the OU draw, locking the oscillator to a pure tone)."""
    from scipy.signal import hilbert

    spec = RhythmSpec(
        "x", 10.0, 1.0, sites=("Fz",), f_sd=3.0, f_tau_s=0.1, env_sd=0.001, env_tau_s=5.0
    )
    osc = _Oscillator(FS, 10.0, spec, np.random.default_rng(1))
    y = osc.render(int(30 * FS))
    inst_phase = np.unwrap(np.angle(hilbert(y)))
    inst_freq = np.diff(inst_phase) / (2 * np.pi) * FS
    assert inst_freq[500:-500].std() > 0.5
