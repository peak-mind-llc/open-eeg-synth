"""Golden fingerprint for the layered engine: a fixed seed keeps producing the same samples
unless SIGNAL_VERSION is bumped (DESIGN §8.3).

The fingerprint covers every path a change could hide in: the base case's eyes-closed mix (every
channel's RMS and Fz's first samples); its eyes-open Fz (the state gain that suppresses alpha);
its eyes-open artifact layers (blink at Fp1, eye movement at F8, jaw EMG at T3 and T4) and, per
artifact kind, the event count and the first event's onset and peak; a dead channel added to
that recording (its transform delta, and O1 inside the dead span); and a planted case's
eyes-closed F7 (the plant path). The case lasts 20 s because its first jaw-EMG burst starts at
17.6 s.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from open_eeg_synth import version
from open_eeg_synth.brain.plants import FocalSlow
from open_eeg_synth.case import ArtifactSpec, make_case, make_recording
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.recipes import resting_case

GOLDEN = Path(__file__).parent / "golden" / "resting_seed20260916.json"
SEED, DURATION_S = 20260916, 20.0
DEAD = ArtifactSpec("dead_channel", {"channel": "O1", "onset_s": 4.0, "offset_s": 9.0})
_ATOL = 0.05
_REGENERATE = "regenerate with scripts/update_golden.py"


def _rms(x: np.ndarray) -> float:
    return round(float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))), 2)


def fingerprint() -> dict:
    ch = CHANNELS_19.index
    spec = resting_case(SEED, duration_s=DURATION_S)
    base = make_case(spec)
    ec, eo = base.recordings["eyes_closed"], base.recordings["eyes_open"]
    events = {}
    for kind in ("blink", "eye_movement", "emg"):
        of_kind = [t for t in eo.truth if t.kind == kind]
        first = of_kind[0] if of_kind else None
        events[kind] = {
            "count": len(of_kind),
            "first_onset_s": None if first is None else round(first.onset_s, 4),
            "first_peak_uv": None if first is None else round(first.peak_uv, 2),
        }
    # A dead channel draws nothing per subject, so the case's own subject is the one a fresh
    # make_subject would draw for this spec, and the recording costs no second subject.
    dead_spec = dataclasses.replace(spec, artifacts=(*spec.artifacts, DEAD))
    dead = make_recording(dead_spec, base.subject, dead_spec.conditions[1])
    fs = int(dead.fs)
    planted_ec = (
        make_case(resting_case(SEED, duration_s=DURATION_S, plants=(FocalSlow("F7"),)))
        .recordings["eyes_closed"]
        .mixed
    )
    return {
        "signal_version": version.SIGNAL_VERSION,
        "rms_uv": [round(float(v), 2) for v in ec.mixed.std(axis=1)],
        "fz_head": [round(float(v), 2) for v in ec.mixed[ch("Fz"), :32]],
        "eyes_open_fz_head": [round(float(v), 2) for v in eo.mixed[ch("Fz"), :32]],
        "eyes_open_artifact_rms_uv": {
            "blink_fp1": _rms(eo.layers["artifact:blink"][ch("Fp1")]),
            "eye_movement_f8": _rms(eo.layers["artifact:eye_movement"][ch("F8")]),
            "emg_t3": _rms(eo.layers["artifact:emg"][ch("T3")]),
            "emg_t4": _rms(eo.layers["artifact:emg"][ch("T4")]),
        },
        "eyes_open_events": events,
        "dead_channel_rms_uv": {
            "delta_o1": _rms(dead.layers["transform:dead_channel"][ch("O1")]),
            "o1_while_dead": _rms(dead.mixed[ch("O1"), 4 * fs : 9 * fs]),
        },
        "plant_f7_rms_uv": round(float(planted_ec[ch("F7"), :].std()), 2),
    }


def test_same_seed_same_samples_unless_signal_version_bumped():
    want = json.loads(GOLDEN.read_text())
    if want["signal_version"] != version.SIGNAL_VERSION:
        pytest.fail(f"SIGNAL_VERSION changed: {_REGENERATE}", pytrace=False)
    got = fingerprint()
    assert set(got) == set(want), f"the fingerprint's contents changed: {_REGENERATE}"
    changed = "signal changed without a SIGNAL_VERSION bump"
    for key in ("rms_uv", "fz_head", "eyes_open_fz_head", "plant_f7_rms_uv"):
        assert np.allclose(got[key], want[key], atol=_ATOL), f"{changed}: {key}"
    for key in ("eyes_open_artifact_rms_uv", "dead_channel_rms_uv"):
        assert got[key].keys() == want[key].keys(), f"{changed}: {key}"
        for name, value in want[key].items():
            assert np.isclose(got[key][name], value, atol=_ATOL), f"{changed}: {key}.{name}"
    for kind, w in want["eyes_open_events"].items():
        g = got["eyes_open_events"][kind]
        assert g["count"] == w["count"], f"{changed}: {kind} count"
        assert g["first_onset_s"] == w["first_onset_s"], f"{changed}: {kind} onset"
        assert np.isclose(g["first_peak_uv"], w["first_peak_uv"], atol=_ATOL), f"{changed}: {kind}"
    assert got["eyes_open_events"].keys() == want["eyes_open_events"].keys()


def test_a_signal_version_change_fails_until_the_golden_is_regenerated(monkeypatch):
    """A bumped SIGNAL_VERSION is not a silent pass: the fingerprint test fails, before it
    renders anything, until the golden file is regenerated."""
    monkeypatch.setattr(version, "SIGNAL_VERSION", version.SIGNAL_VERSION + 1)
    with pytest.raises(pytest.fail.Exception, match=r"SIGNAL_VERSION changed: regenerate with"):
        test_same_seed_same_samples_unless_signal_version_bumped()
