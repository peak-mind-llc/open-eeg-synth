"""Golden fingerprint for the layered engine: a fixed seed keeps producing the same samples
unless SIGNAL_VERSION is bumped (DESIGN §8.3).

The fingerprint covers three paths so a change hiding in any one of them is caught: the base
case's eyes-closed brain-plus-artifact mix (every channel's RMS and Fz's first samples), the same
case's eyes-open Fz (the state-gain path that suppresses alpha), and a planted case's eyes-closed
F7 (the plant path).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from open_eeg_synth import version
from open_eeg_synth.brain.plants import FocalSlow
from open_eeg_synth.case import make_case
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.recipes import resting_case

GOLDEN = Path(__file__).parent / "golden" / "resting_seed20260916.json"
_ATOL = 0.05


def fingerprint() -> dict:
    base = make_case(resting_case(20260916, duration_s=10.0)).recordings
    ec, eo = base["eyes_closed"].mixed, base["eyes_open"].mixed
    planted_ec = (
        make_case(resting_case(20260916, duration_s=10.0, plants=(FocalSlow("F7"),)))
        .recordings["eyes_closed"]
        .mixed
    )
    fz, f7 = CHANNELS_19.index("Fz"), CHANNELS_19.index("F7")
    return {
        "signal_version": version.SIGNAL_VERSION,
        "rms_uv": [round(float(v), 2) for v in ec.std(axis=1)],
        "fz_head": [round(float(v), 2) for v in ec[fz, :32]],
        "eyes_open_fz_head": [round(float(v), 2) for v in eo[fz, :32]],
        "plant_f7_rms_uv": round(float(planted_ec[f7, :].std()), 2),
    }


def test_same_seed_same_samples_unless_signal_version_bumped():
    want = json.loads(GOLDEN.read_text())
    got = fingerprint()
    if got["signal_version"] != want["signal_version"]:
        return  # a deliberate change; scripts/update_golden.py regenerates the file
    assert np.allclose(got["rms_uv"], want["rms_uv"], atol=_ATOL), (
        "signal changed without a SIGNAL_VERSION bump"
    )
    assert np.allclose(got["fz_head"], want["fz_head"], atol=_ATOL)
    assert np.allclose(got["eyes_open_fz_head"], want["eyes_open_fz_head"], atol=_ATOL)
    assert np.allclose(got["plant_f7_rms_uv"], want["plant_f7_rms_uv"], atol=_ATOL)
