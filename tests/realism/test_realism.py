"""The resting recipe against the committed PhysioNet reference bands (DESIGN §9.1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mne")
pytest.importorskip("mne_connectivity")

from open_eeg_synth.case import make_case  # noqa: E402
from open_eeg_synth.recipes import resting_case  # noqa: E402
from tests.realism.measure import BANDS, measure, to_raw  # noqa: E402

REF = json.loads(
    (Path(__file__).parent / "reference" / "eegmmidb_baselines_s01-20.json").read_text()
)
TOL = {"average": (0.0, 0.0), "bipolar": (0.0, 0.0), "laplacian": (-0.04, 0.08)}  # DESIGN §9.1
BIN_TOL = 0.06  # true pair values (twice the feasibility test's), so twice its +/-0.03 allowance
N_SEEDS, DUR_S = 6, 60.0


def _synth_group(cond: str) -> list[dict]:
    out = []
    for k in range(N_SEEDS):
        rec = make_case(resting_case(100 + k, duration_s=DUR_S, artifacts=())).recordings[cond]
        out.append(measure(to_raw(rec.mixed, rec.fs, rec.channels)))
    return out


def _median(group, key):
    return np.nanmedian(np.array([g[key] for g in group], dtype=float), axis=0)


@pytest.mark.realism
@pytest.mark.parametrize("cond", ["eyes_closed", "eyes_open"])
def test_resting_recipe_matches_public_reference(cond):
    ref = REF["groups"][cond]
    syn = _synth_group(cond)
    failures = []
    for reference in ("average", "bipolar", "laplacian"):
        lo_t, hi_t = TOL[reference]
        for band in BANDS:
            for metric in ("coh_mean", "dwpli_mean"):
                key = f"{reference}/{band}/{metric}"
                v, lo, hi = float(_median(syn, key)), ref[key]["p10"] + lo_t, ref[key]["p90"] + hi_t
                if not lo <= v <= hi:
                    failures.append(f"{key}: {v:.3f} not in [{lo:.3f}, {hi:.3f}]")
            key = f"{reference}/{band}/coh_by_dist"
            v = _median(syn, key)
            lo = np.array(ref[key]["p10"], float) - BIN_TOL
            hi = np.array(ref[key]["p90"], float) + BIN_TOL
            bad = [(i, round(float(v[i]), 3)) for i in range(5) if not (lo[i] <= v[i] <= hi[i])]
            if bad:
                failures.append(f"{key}: bins {bad} outside [p10-{BIN_TOL}, p90+{BIN_TOL}]")
    checks = {
        "aperiodic_exponent": (0.8, 1.4),
        "rms_uv/Cz": (8.0, 20.0),
        "alpha_share/O1": (0.35, 0.79) if cond == "eyes_closed" else (0.05, 0.30),
    }
    for key, (lo, hi) in checks.items():
        v = float(_median(syn, key))
        if not lo <= v <= hi:
            failures.append(f"{key}: {v:.3f} not in [{lo}, {hi}]")
    assert not failures, "\n".join(failures)
