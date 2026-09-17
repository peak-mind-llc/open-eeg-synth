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
)
from open_eeg_synth.case import compiled_rhythms, make_case
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.recipes import resting_brain, resting_case
from tests.helpers import band_power

FS = 256.0


def _ec(seed, plants):
    return (
        make_case(resting_case(seed, duration_s=40.0, plants=plants, artifacts=()))
        .recordings["eyes_closed"]
        .mixed
    )


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
    assert out["theta"].amp_uv == 14.0 and "Pz" in out["theta"].sites
    assert out["alpha"].f0_hz == 9.0 and out["alpha"].hemisphere_gain == {"left": 0.5}
    with pytest.raises(KeyError):
        apply_modifiers(base, [PeakShift("gamma", 1.0).modifiers()[0]])


def test_focal_slow_raises_delta_at_site():
    f7 = CHANNELS_19.index("F7")
    clean, planted = _ec(21, ()), _ec(21, (FocalSlow("F7", f0_hz=2.5, amp_uv=45.0),))
    assert band_power(planted, FS, 1.5, 3.5)[f7] > 4 * band_power(clean, FS, 1.5, 3.5)[f7]
    spec = resting_case(21, plants=(FocalSlow("F7"),))
    assert [r.name for r in compiled_rhythms(spec)][-1] == "plant:focal_slow:F7:2.5hz"


def test_lateral_imbalance_and_reduced_rhythm():
    o1, o2, c3 = (CHANNELS_19.index(k) for k in ("O1", "O2", "C3"))
    x = _ec(22, (LateralImbalance("alpha", "left", 0.4),))
    y = _ec(22, ())
    a_x, a_y = band_power(x, FS, 8, 13), band_power(y, FS, 8, 13)
    assert (a_x[o1] / a_x[o2]) < 0.6 * (a_y[o1] / a_y[o2])
    z = _ec(22, (ReducedRhythm("smr", 0.1),))
    assert band_power(z, FS, 12.5, 14.5)[c3] < 0.7 * band_power(y, FS, 12.5, 14.5)[c3]


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


def test_plant_state_gain_silences_a_plant_by_state():
    f7 = CHANNELS_19.index("F7")
    plant = FocalSlow("F7", f0_hz=2.5, amp_uv=45.0, state_gain={"eyes_open": 0.0})
    for cond in ("eyes_open", "eyes_closed"):
        clean = make_case(resting_case(24, duration_s=20.0, artifacts=())).recordings[cond].mixed
        planted = make_case(resting_case(24, duration_s=20.0, plants=(plant,), artifacts=()))
        ratio = (
            band_power(planted.recordings[cond].mixed, FS, 1.5, 3.5)[f7]
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
