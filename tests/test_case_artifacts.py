from __future__ import annotations

import numpy as np

from open_eeg_synth.case import ArtifactSpec, CaseSpec, make_case, make_engine, make_subject
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


def test_engine_chunking_does_not_change_a_case():
    spec = resting_case(13, duration_s=20.0)
    subject = make_subject(spec)
    cond = spec.conditions[1]
    whole = make_engine(spec, subject, cond).render_all(int(20 * 256), block=100000)
    eng = make_engine(spec, subject, cond)
    rng = np.random.default_rng(0)
    parts, t0 = [], 0
    while t0 < 20 * 256:
        n = int(min(20 * 256 - t0, rng.integers(1, 900)))
        parts.append(eng.render(t0, n).mixed)
        t0 += n
    assert np.allclose(whole.mixed, np.concatenate(parts, axis=1), atol=1e-3)
    assert [t.to_dict() for t in whole.truth] == [t.to_dict() for t in eng.truth()]


def test_spec_with_artifacts_roundtrips():
    spec = resting_case(14, duration_s=1.0)
    assert CaseSpec.from_dict(spec.to_dict()) == spec
