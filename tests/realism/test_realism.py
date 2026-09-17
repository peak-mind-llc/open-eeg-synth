"""The resting recipe against the committed PhysioNet reference bands (DESIGN §9.1)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mne")
pytest.importorskip("mne_connectivity")

from open_eeg_synth.case import make_case  # noqa: E402
from open_eeg_synth.recipes import resting_case  # noqa: E402
from tests.realism.gaps import Failure, Gap, passing_gaps, unexpected  # noqa: E402
from tests.realism.measure import BANDS, measure, to_raw  # noqa: E402

REF = json.loads(
    (Path(__file__).parent / "reference" / "eegmmidb_baselines_s01-20.json").read_text()
)
TOL = {"average": (0.0, 0.0), "bipolar": (0.0, 0.0), "laplacian": (-0.04, 0.08)}  # DESIGN §9.1
BIN_TOL = 0.06  # true pair values (twice the feasibility test's), so twice its +/-0.03 allowance
# 24 seeds, not six: a six-seed median is too noisy to gate on (on fresh seeds, 340-819, only
# 10-12 % of disjoint six-seed sets pass every line outside KNOWN_GAPS; see recipes.py).
N_SEEDS, DUR_S = 24, 60.0

# Lines the tuned recipe is known to fail, (condition, reference, band, metric) -> Gap. The test
# fails on any other failing line, and on a failing distance bin the gap does not name; it warns
# when a listed line passes. TOL and BIN_TOL are not widened for these.
KNOWN_GAPS = {
    ("eyes_closed", "laplacian", "Theta", "coh_by_dist"): Gap(
        "nearest-distance bin too coherent: 0.536 on the test seeds and 0.531-0.536 at every "
        "population median measured (seeds 100-339, 340-579, 580-819), limit 0.513. The "
        "frontal-midline theta patches under Fz, F3, F4 and Cz share a driver and stay focal "
        "(12 mm, lags <= 40 ms) so that theta and theta plants stay visible by eye",
        bins=(0,),
    ),
    ("eyes_open", "bipolar", "Beta", "coh_by_dist"): Gap(
        "population-level gap: the nearest-distance bin is too low on fresh seeds (population "
        "median 0.310 on 340-579 and 0.308 on 580-819, limit 0.317; fails in 63 % and 80 % of "
        "random 24-seed sets there). It passes on the test seeds (0.328), so a warning that it "
        "passes is expected",
        bins=(0,),
    ),
}
# Out of sample (fresh seeds the recipe values never saw; 200 random 24-seed sets per pool), with
# both gaps excused: 26 % of sets from 340-579 and 49 % from 580-819 pass every other line (other
# failing lines mean 1.18 and 0.63, max 4 and 2); excusing only the Laplacian theta gap, 9 % and
# 9 %. The lines that fail most often beyond the two gaps, with their failure rates in the two
# pools and their population-median margins on 340-579:
#   eyes_closed laplacian/Beta/coh_by_dist, > 150 mm or nearest   47 %, 44 %   +0.001, +0.007
#   eyes_closed laplacian/Alpha/coh_by_dist, 120-150 mm           33 %, 4 %    +0.006
#   eyes_open   average/Theta/dwpli_mean                          20 %, 7 %
# On the pool the values were chosen on (seeds 124-339), the same measure read 31 % passing
# outside the Laplacian theta gap and a bipolar beta margin of +0.002; those figures are in-sample.

_GROUPS: dict[str, list[dict]] = {}


def _synth_group(cond: str) -> list[dict]:
    """Per-seed measurements; each seed's two-condition case is rendered once for both."""
    if not _GROUPS:
        groups: dict[str, list[dict]] = {}
        for k in range(N_SEEDS):
            case = make_case(resting_case(100 + k, duration_s=DUR_S, artifacts=()))
            for name, rec in case.recordings.items():
                groups.setdefault(name, []).append(measure(to_raw(rec.mixed, rec.fs, rec.channels)))
        _GROUPS.update(groups)  # stored only once every seed has been measured
    return _GROUPS[cond]


def _median(group, key):
    return np.nanmedian(np.array([g[key] for g in group], dtype=float), axis=0)


def _failures(cond: str, syn: list[dict]) -> list[Failure]:
    """Every failing line, with its failing distance bins for a binned metric."""
    ref = REF["groups"][cond]
    failures = []
    for reference in ("average", "bipolar", "laplacian"):
        lo_t, hi_t = TOL[reference]
        for band in BANDS:
            for metric in ("coh_mean", "dwpli_mean"):
                key = f"{reference}/{band}/{metric}"
                v, lo, hi = float(_median(syn, key)), ref[key]["p10"] + lo_t, ref[key]["p90"] + hi_t
                if not lo <= v <= hi:
                    msg = f"{key}: {v:.3f} not in [{lo:.3f}, {hi:.3f}]"
                    failures.append(Failure((cond, reference, band, metric), (), msg))
            key = f"{reference}/{band}/coh_by_dist"
            v = _median(syn, key)
            lo = np.array(ref[key]["p10"], float) - BIN_TOL
            hi = np.array(ref[key]["p90"], float) + BIN_TOL
            bad = [(i, round(float(v[i]), 3)) for i in range(5) if not (lo[i] <= v[i] <= hi[i])]
            if bad:
                msg = f"{key}: bins {bad} outside [p10-{BIN_TOL}, p90+{BIN_TOL}]"
                line = (cond, reference, band, "coh_by_dist")
                failures.append(Failure(line, tuple(i for i, _ in bad), msg))
    checks = {
        "aperiodic_exponent": (0.8, 1.4),
        "rms_uv/Cz": (8.0, 20.0),
        "alpha_share/O1": (0.35, 0.79) if cond == "eyes_closed" else (0.05, 0.30),
    }
    for key, (lo, hi) in checks.items():
        v = float(_median(syn, key))
        if not lo <= v <= hi:
            failures.append(Failure((cond, "", "", key), (), f"{key}: {v:.3f} not in [{lo}, {hi}]"))
    return failures


def test_binned_known_gaps_name_their_bins():
    for (_, _, _, metric), gap in KNOWN_GAPS.items():
        assert gap.bins or metric != "coh_by_dist"


@pytest.mark.realism
@pytest.mark.parametrize("cond", ["eyes_closed", "eyes_open"])
def test_resting_recipe_matches_public_reference(cond):
    failures = _failures(cond, _synth_group(cond))
    for line in passing_gaps(failures, KNOWN_GAPS, cond):
        warnings.warn(
            f"known realism gap passes on the test seeds: {line}: {KNOWN_GAPS[line].reason}",
            stacklevel=1,
        )
    bad = unexpected(failures, KNOWN_GAPS)
    assert not bad, "\n".join(f.message for f in bad)
