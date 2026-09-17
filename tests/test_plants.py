from __future__ import annotations

import numpy as np
import pytest

from open_eeg_synth.brain.plants import (
    PLANTS,
    FocalSlow,
    LateralImbalance,
    PeakShift,
    ReducedRhythm,
    RhythmicBursts,
    WidespreadExcess,
    apply_modifiers,
    plant_from_dict,
    plant_to_dict,
    register_plant,
)
from open_eeg_synth.case import compiled_rhythms, make_case, make_recording, make_subject
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.recipes import resting_brain, resting_case
from tests.helpers import band_power, welch

FS = 256.0


def _ec(seed, plants, subject=None):
    """``subject`` reuses an already-built Subject (head perturbation + smoothing matrix, the
    expensive part of ``make_subject``) instead of building a fresh one: safe only for a
    modifier-only plant whose modifier leaves every rhythm's ``sites``/``region``/``n_patches``/
    ``f0_hz`` exactly as the subject it was built from had them (hemisphere_gain/amp_scale with
    no f0_shift_hz or extra_sites), since those are the only RhythmSpec fields make_subject's own
    per-rhythm draws (placements, f0, patch offsets/lags) read."""
    spec = resting_case(seed, duration_s=40.0, plants=plants, artifacts=())
    subj = subject if subject is not None else make_subject(spec)
    return make_recording(spec, subj, spec.conditions[0]).mixed


def test_registry_and_roundtrip():
    assert set(PLANTS) == {
        "focal_slow",
        "rhythmic_bursts",
        "lateral_imbalance",
        "widespread_excess",
        "peak_shift",
        "reduced_rhythm",
    }
    for p in (
        FocalSlow("F7"),
        RhythmicBursts(("Fz",)),
        LateralImbalance("alpha", "left", 0.6),
        WidespreadExcess("theta", 2.2),
        PeakShift("alpha", -1.5),
        ReducedRhythm("smr", 0.2),
    ):
        assert plant_from_dict(plant_to_dict(p)) == p
        rec = p.record()
        assert rec.kind == p.kind and rec.description and rec.to_dict()["kind"] == p.kind


def test_modifiers_compile_onto_base_rhythms():
    base = resting_brain().rhythms
    mods = [
        m
        for p in (
            WidespreadExcess("theta", 2.0, extra_sites=("Pz",)),
            PeakShift("alpha", -1.0),
            LateralImbalance("alpha", "left", 0.5),
        )
        for m in p.modifiers()
    ]
    out = {r.name: r for r in apply_modifiers(base, mods)}
    base_theta = next(r for r in base if r.name == "theta")
    assert out["theta"].amp_uv == pytest.approx(2.0 * base_theta.amp_uv)
    assert "Pz" in out["theta"].sites
    assert out["alpha"].f0_hz == 9.0 and out["alpha"].hemisphere_gain == {"left": 0.5}
    with pytest.raises(KeyError):
        apply_modifiers(base, [PeakShift("gamma", 1.0).modifiers()[0]])


def test_focal_slow_raises_delta_at_site():
    f7 = CHANNELS_19.index("F7")
    clean, planted = _ec(21, ()), _ec(21, (FocalSlow("F7", f0_hz=2.5, amp_uv=45.0),))
    assert band_power(planted, FS, 1.5, 3.5)[f7] > 4 * band_power(clean, FS, 1.5, 3.5)[f7]
    spec = resting_case(21, plants=(FocalSlow("F7"),))
    assert [r.name for r in compiled_rhythms(spec)][-1] == "plant:focal_slow:F7:2.5hz"


@pytest.mark.slow
def test_focal_slow_is_spatially_specific():
    """F7 has the largest increase in the planted band of any channel (not a neighbour).

    test_focal_slow_raises_delta_at_site already gives the fast suite FocalSlow's core DESIGN
    §9.2 coverage (band power at its site increases by the expected factor); this is a stronger,
    additional claim (F7 specifically, not merely *a* channel), at 3 seeds x 2 fresh subjects
    each (FocalSlow adds its own compiled rhythm, so unlike the modifier-only plants above, its
    subject cannot be shared between the clean and planted variants)."""
    for seed in (100, 101, 102):
        clean, planted = _ec(seed, ()), _ec(seed, (FocalSlow("F7", f0_hz=2.5, amp_uv=45.0),))
        increase = band_power(planted, FS, 1.5, 3.5) - band_power(clean, FS, 1.5, 3.5)
        assert CHANNELS_19[int(np.argmax(increase))] == "F7"


def test_lateral_imbalance_and_reduced_rhythm():
    """>= 6 seeds, average-referenced, median threshold: a single seed can be lucky —
    e.g. seed 108's raw-referenced lateral-imbalance ratio alone is 0.487, above a 0.6 bound.

    One subject per seed, shared across the clean/LateralImbalance/ReducedRhythm variants:
    LateralImbalance and ReducedRhythm only scale amplitude/hemisphere gain at render time
    (`apply_modifiers` leaves every rhythm's sites/region/n_patches/f0_hz exactly as the base
    spec had them for these two plant kinds, since neither sets a modifier's f0_shift_hz or
    extra_sites), so a subject built from the plant-free spec has exactly the placements/f0/
    offsets/lags a fresh make_subject(spec_with_that_plant) would draw too. Building the subject
    (head perturbation + smoothing matrix) is most of this test's cost."""
    o1, o2, c3 = (CHANNELS_19.index(k) for k in ("O1", "O2", "C3"))
    seeds = range(100, 106)
    lat_ratios, red_ratios = [], []
    for s in seeds:
        subject = make_subject(resting_case(s, duration_s=40.0, artifacts=()))
        y, x, z = (
            _ec(s, (), subject=subject),
            _ec(s, (LateralImbalance("alpha", "left", 0.4),), subject=subject),
            _ec(s, (ReducedRhythm("smr", 0.1),), subject=subject),
        )
        y, x, z = (v - v.mean(axis=0, keepdims=True) for v in (y, x, z))
        a_x, a_y = band_power(x, FS, 8, 13), band_power(y, FS, 8, 13)
        lat_ratios.append((a_x[o1] / a_x[o2]) / (a_y[o1] / a_y[o2]))
        red_ratios.append(band_power(z, FS, 12.5, 14.5)[c3] / band_power(y, FS, 12.5, 14.5)[c3])
    assert np.median(lat_ratios) < 0.5
    assert np.median(red_ratios) < 0.4


def test_rhythmic_bursts_are_intermittent():
    fz = CHANNELS_19.index("Fz")
    x = _ec(
        23,
        (RhythmicBursts(("Fz",), f0_hz=6.5, amp_uv=30.0, burst_s=(1, 2), gap_s=(4, 6)),),
    )
    x = x - x.mean(axis=0, keepdims=True)  # average reference
    theta = np.array(
        [band_power(x[fz, s : s + int(FS)], FS, 5.5, 7.5)[0] for s in range(0, x.shape[1], int(FS))]
    )  # per-second theta power at Fz
    assert theta.max() > 4 * np.median(theta)  # burst seconds stand well above the gaps


def test_rhythmic_bursts_description_keeps_one_decimal():
    rb = RhythmicBursts(("Fz",), burst_s=(0.5, 1.5), gap_s=(8.0, 25.0))
    assert "0.5-1.5 s" in rb.record().description


def test_plant_state_gain_silences_a_plant_by_state():
    f7 = CHANNELS_19.index("F7")
    plant = FocalSlow("F7", f0_hz=2.5, amp_uv=45.0, state_gain={"eyes_open": 0.0})
    # Both conditions live on the one case each build already returns; build each case once
    # rather than once per condition checked below.
    clean_case = make_case(resting_case(24, duration_s=20.0, artifacts=()))
    planted_case = make_case(resting_case(24, duration_s=20.0, plants=(plant,), artifacts=()))
    for cond in ("eyes_open", "eyes_closed"):
        clean = clean_case.recordings[cond].mixed
        ratio = (
            band_power(planted_case.recordings[cond].mixed, FS, 1.5, 3.5)[f7]
            / band_power(clean, FS, 1.5, 3.5)[f7]
        )
        if cond == "eyes_open":
            assert (
                abs(ratio - 1.0) < 1e-3
            )  # silenced: the other streams are untouched (DESIGN §8.1)
        else:
            assert ratio > 4
    assert plant.record().to_dict()["params"]["state_gain"] == {"eyes_open": 0.0}
    assert plant_from_dict(plant_to_dict(plant)) == plant


def test_state_confined_plant_omitted_from_conditions_where_silent():
    plant = FocalSlow("F7", f0_hz=2.5, amp_uv=45.0, state_gain={"eyes_open": 0.0})
    case = make_case(resting_case(24, duration_s=4.0, plants=(plant,), artifacts=()))
    assert [p.kind for p in case.recordings["eyes_closed"].plants] == ["focal_slow"]
    assert case.recordings["eyes_open"].plants == []
    assert "absent in eyes_open" in case.recordings["eyes_closed"].plants[0].description


def test_peak_shift_moves_the_measured_alpha_peak():
    o1 = CHANNELS_19.index("O1")

    def peak(x):
        f, p = welch(x[o1], FS, int(8 * FS))
        m = (f >= 6) & (f <= 13)
        return f[m][np.argmax(p[0, m])]

    shifts = [peak(_ec(s, ())) - peak(_ec(s, (PeakShift("alpha", -1.5),))) for s in range(100, 106)]
    assert np.median(shifts) > 1.0


def test_modifier_only_plant_records_have_no_band_or_sites():
    """Records for the four modifier-only plants (no rhythm of their own) carry the target
    rhythm's name in ``params`` instead of a band/site of their own."""
    for p in (
        LateralImbalance("alpha", "left", 0.6),
        WidespreadExcess("theta", 2.2, extra_sites=("Pz",)),
        PeakShift("alpha", -1.5),
        ReducedRhythm("smr", 0.2),
    ):
        rec = p.record()
        assert rec.band_hz is None and rec.amp_uv is None and rec.sites == ()
        assert rec.params["rhythm"] == p.rhythm
        d = rec.to_dict()
        assert d["band_hz"] is None and d["sites"] == []


def test_plant_record_to_dict_does_not_alias_params():
    plant = FocalSlow("F7", state_gain={"eyes_open": 0.0})
    rec = plant.record()
    d = rec.to_dict()
    d["params"]["state_gain"]["eyes_open"] = 9.0
    assert rec.params["state_gain"] == {"eyes_open": 0.0}


def test_duplicate_compiled_rhythm_names_raise():
    same = (FocalSlow("F7", f0_hz=2.5, amp_uv=45.0), FocalSlow("F7", f0_hz=2.5, amp_uv=10.0))
    with pytest.raises(ValueError, match="plant:focal_slow:F7:2.5hz"):
        compiled_rhythms(resting_case(3, plants=same, artifacts=()))


def test_two_focal_slow_plants_at_one_site_different_frequencies():
    two = (FocalSlow("F7", f0_hz=2.5, amp_uv=45.0), FocalSlow("F7", f0_hz=6.0, amp_uv=45.0))
    spec = resting_case(3, duration_s=20.0, plants=two, artifacts=())
    names = [r.name for r in compiled_rhythms(spec) if r.name.startswith("plant")]
    assert names == ["plant:focal_slow:F7:2.5hz", "plant:focal_slow:F7:6hz"]
    clean, planted = _ec(3, ()), _ec(3, two)
    f7 = CHANNELS_19.index("F7")
    assert band_power(planted, FS, 1.5, 3.5)[f7] > 4 * band_power(clean, FS, 1.5, 3.5)[f7]
    assert band_power(planted, FS, 5.0, 7.0)[f7] > 4 * band_power(clean, FS, 5.0, 7.0)[f7]


def test_two_rhythmic_bursts_plants_at_one_site_different_frequencies():
    """The compiled rhythm name must carry the frequency, as FocalSlow's already does (see
    above), so two RhythmicBursts plants at the same site(s) but different frequencies compile
    to distinct rhythms instead of colliding."""
    two = (RhythmicBursts(("Fz",), f0_hz=6.5), RhythmicBursts(("Fz",), f0_hz=10.0))
    spec = resting_case(3, duration_s=1.0, plants=two, artifacts=())
    names = [r.name for r in compiled_rhythms(spec) if r.name.startswith("plant")]
    assert names == ["plant:rhythmic_bursts:Fz:6.5hz", "plant:rhythmic_bursts:Fz:10hz"]
    make_case(spec)  # renders without raising "duplicate compiled rhythm name(s)"


def test_lateral_imbalance_rejects_unknown_side():
    with pytest.raises(ValueError, match="side"):
        LateralImbalance("alpha", "Left", 0.4)


def test_value_checks_reject_nonsense_and_accept_edge_cases():
    with pytest.raises(ValueError, match="f0_hz"):
        FocalSlow("F7", f0_hz=-1.0)
    with pytest.raises(ValueError, match="f0_hz"):
        RhythmicBursts(("Fz",), f0_hz=0.0)
    with pytest.raises(ValueError, match="factor"):
        ReducedRhythm("smr", -1.0)
    with pytest.raises(ValueError, match="factor"):
        LateralImbalance("alpha", "left", -0.1)
    with pytest.raises(ValueError, match="finite"):
        LateralImbalance("alpha", "left", float("nan"))
    with pytest.raises(ValueError):
        compiled_rhythms(resting_case(1, plants=(PeakShift("theta", -7.0),), artifacts=()))
    # a factor of exactly 0 is legal: it silences the rhythm, it is not nonsense
    zeroed = {
        r.name: r
        for r in compiled_rhythms(resting_case(1, plants=(WidespreadExcess("theta", 0.0),)))
    }
    assert zeroed["theta"].amp_uv == 0.0


def test_registry_rejects_duplicate_kind_and_names_unknown_kind():
    from dataclasses import dataclass
    from typing import ClassVar

    @dataclass(frozen=True)
    class _Impostor:
        kind: ClassVar[str] = "focal_slow"

    with pytest.raises(ValueError, match="focal_slow"):
        register_plant(_Impostor)
    register_plant(FocalSlow)  # re-registering the real class is a no-op, not an error
    with pytest.raises(KeyError, match="not_a_real_plant"):
        plant_from_dict({"kind": "not_a_real_plant", "params": {}})
