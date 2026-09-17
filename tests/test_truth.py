import json

import numpy as np
import pytest

from open_eeg_synth import version
from open_eeg_synth.brain.plants import FocalSlow
from open_eeg_synth.case import make_case
from open_eeg_synth.casefile.truth import (
    MAGIC,
    LayerMismatchError,
    case_truth,
    read_truth,
    render_layers,
    write_truth,
)
from open_eeg_synth.recipes import resting_case


def test_truth_dict_contents_and_sealed_roundtrip(tmp_path):
    case = make_case(resting_case(51, duration_s=6.0, plants=(FocalSlow("F7"),)))
    d = case_truth(case, {"eyes_closed": "a.edf", "eyes_open": "b.edf"})
    assert d["format"] == "open-eeg-synth/truth" and d["label"] == "synthetic"
    assert d["generator"]["signal_version"] == version.SIGNAL_VERSION
    assert d["case_id"] == case.case_id and d["seed"] == 51
    ec = d["recordings"]["eyes_closed"]
    assert ec["file"] == "a.edf" and ec["layers"][0] == "brain" and ec["layers"][-1] == "sensor"
    assert len(ec["layer_rms_uv"]["brain"]) == 19
    assert ec["plants"][0]["kind"] == "focal_slow"
    assert all(set(t) >= {"kind", "onset_s", "remedies", "layer"} for t in ec["truth"])
    json.dumps(d)  # JSON-safe
    p = write_truth(tmp_path / "c.truth", d)
    raw = p.read_bytes()
    assert raw.startswith(MAGIC) and b"focal_slow" not in raw  # compressed, not readable as text
    assert read_truth(p) == d
    with pytest.raises(ValueError):
        (tmp_path / "bad").write_bytes(b"nope")
        read_truth(tmp_path / "bad")


def test_render_layers_reproduces_and_detects_mismatch():
    case = make_case(resting_case(52, duration_s=4.0))
    d = case_truth(case, {"eyes_closed": "a.edf", "eyes_open": "b.edf"})
    rec = render_layers(d, "eyes_open")
    assert np.allclose(rec.layers["brain"], case.recordings["eyes_open"].layers["brain"], atol=1e-3)
    d["recordings"]["eyes_open"]["layer_rms_uv"]["brain"][0] *= 2.0
    with pytest.raises(LayerMismatchError):
        render_layers(d, "eyes_open")


def test_render_layers_plants_match_make_case_when_a_timeline_silences_one():
    """R1: render_layers must rebuild a condition through the same code path as make_case, so a
    plant that a condition's timeline silences everywhere (here, FocalSlow zeroed in both states
    of the drowsy eyes_closed condition) is dropped from render_layers' plant list exactly as it
    is from make_case's, and kept where the timeline does not silence it."""
    spec = resting_case(
        54,
        duration_s=4.0,
        drowsy_from_s=2.0,
        plants=(
            FocalSlow("F7", state_gain={"eyes_closed": 0.0, "drowsy": 0.0}),
            FocalSlow("F8", f0_hz=3.0),
        ),
    )
    case = make_case(spec)
    d = case_truth(case, {"eyes_closed": "a.edf", "eyes_open": "b.edf"})
    ec = render_layers(d, "eyes_closed")
    eo = render_layers(d, "eyes_open")
    assert ec.plants == case.recordings["eyes_closed"].plants
    assert eo.plants == case.recordings["eyes_open"].plants
    assert len(ec.plants) == 1 and ec.plants[0].sites == ("F8",)
    assert len(eo.plants) == 2 and {p.sites for p in eo.plants} == {("F7",), ("F8",)}


def test_render_layers_raises_on_layer_set_mismatch():
    """R2: a layer name that the truth file's ``layers`` list no longer carries (here, dropping
    "sensor") is a mismatch against the re-rendered layer set, not just a value mismatch."""
    case = make_case(resting_case(55, duration_s=2.0))
    d = case_truth(case, {"eyes_closed": "a.edf", "eyes_open": "b.edf"})
    del d["recordings"]["eyes_open"]["layers"][-1]
    with pytest.raises(LayerMismatchError):
        render_layers(d, "eyes_open")


def test_case_truth_raises_value_error_for_missing_condition_files():
    """R4: a missing ``files`` entry names the missing condition(s), not a bare KeyError."""
    case = make_case(resting_case(56, duration_s=1.0))
    with pytest.raises(ValueError, match="eyes_open"):
        case_truth(case, {"eyes_closed": "a.edf"})
