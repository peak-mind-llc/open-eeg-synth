from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mne")
pytest.importorskip("mne_connectivity")

from open_eeg_synth.case import make_case  # noqa: E402
from open_eeg_synth.recipes import resting_case  # noqa: E402
from tests.realism.measure import measure, to_raw  # noqa: E402


@pytest.mark.slow
@pytest.mark.realism
def test_measure_returns_every_metric():
    rec = make_case(resting_case(1, duration_s=40.0, artifacts=())).recordings["eyes_closed"]
    m = measure(to_raw(rec.mixed, rec.fs, rec.channels))
    assert m["n_epochs"] >= 8
    for ref in ("average", "bipolar", "laplacian"):
        for band in ("Delta", "Theta", "Alpha", "Beta"):
            assert 0.0 <= m[f"{ref}/{band}/coh_mean"] <= 1.0
            assert len(m[f"{ref}/{band}/coh_by_dist"]) == 5
            assert np.isfinite(m[f"{ref}/{band}/dwpli_mean"])
    assert 0.3 < m["aperiodic_exponent"] < 2.0 and 0.0 < m["alpha_share/O1"] < 1.0
