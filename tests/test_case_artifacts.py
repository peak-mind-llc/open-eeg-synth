from __future__ import annotations

import numpy as np

from open_eeg_synth.case import ArtifactSpec, make_case
from open_eeg_synth.recipes import ordinary_artifacts, resting_case


def test_ordinary_artifacts_layers_and_truth():
    spec = resting_case(11, duration_s=30.0)
    assert [a.kind for a in ordinary_artifacts()] == ["blink", "eye_movement", "emg"]
    case = make_case(spec)
    ec, eo = case.recordings["eyes_closed"], case.recordings["eyes_open"]
    assert set(eo.layers) == {
        "brain",
        "artifact:blink",
        "artifact:eye_movement",
        "artifact:emg",
        "sensor",
    }
    assert np.allclose(eo.mixed, sum(eo.layers.values()), atol=1e-3)
    assert any(t.kind == "blink" for t in eo.truth)
    assert not any(t.kind == "blink" for t in ec.truth)  # eyes closed: no blinks
    assert all(t.layer in eo.layers for t in eo.truth)


def test_duplicate_kinds_get_numbered_layers():
    spec = resting_case(
        12,
        duration_s=5.0,
        artifacts=(ArtifactSpec("emg", {"side": "left"}), ArtifactSpec("emg", {"side": "right"})),
    )
    rec = make_case(spec).recordings["eyes_open"]
    assert {"artifact:emg#1", "artifact:emg#2"} <= set(rec.layers)
