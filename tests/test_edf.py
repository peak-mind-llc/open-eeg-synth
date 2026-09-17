from __future__ import annotations

import numpy as np
import pytest

from open_eeg_synth.case import make_case
from open_eeg_synth.casefile.edf import write_edf
from open_eeg_synth.recipes import resting_case

edfio = pytest.importorskip("edfio")


def test_edf_roundtrip_in_microvolts_no_annotations(tmp_path):
    rec = make_case(resting_case(41, duration_s=4.5, artifacts=())).recordings["eyes_open"]
    p = write_edf(tmp_path / "x.edf", rec)
    e = edfio.read_edf(str(p))
    assert [s.label for s in e.signals] == list(rec.channels)
    assert e.signals[0].physical_dimension == "uV" and e.signals[0].physical_range == (
        -2000.0,
        2000.0,
    )
    assert len(e.annotations) == 0
    # A plain EDF's "reserved" header field is blank; an "EDF Annotations" signal (added by
    # edfio whenever `annotations=` is given) would make this an EDF+ file instead ("EDF+C").
    assert e.reserved == ""
    assert e.patient.code == "SYNTHETIC" and "synthetic" in e.patient.additional
    with pytest.raises(edfio.edf_header.AnonymizedDateError):
        _ = e.recording.startdate
    got = e.signals[8].data  # O1
    assert len(got) == 4 * 256  # trimmed to whole seconds
    assert np.allclose(got, rec.mixed[8, : 4 * 256], atol=0.07)  # 16-bit over 4000 uV


def test_edf_has_no_annotations_and_the_right_channel_names_read_back_through_mne(tmp_path):
    """The same file read back through MNE (a different EDF parser from edfio, and how most
    downstream tooling actually consumes these files) must agree: zero annotations, and the 19
    channel names in the recorded order."""
    mne = pytest.importorskip("mne")
    rec = make_case(resting_case(43, duration_s=4.5, artifacts=())).recordings["eyes_open"]
    p = write_edf(tmp_path / "z.edf", rec)
    raw = mne.io.read_raw_edf(str(p), preload=False, verbose="error")
    assert len(raw.annotations) == 0
    assert raw.ch_names == list(rec.channels)


def test_clipping_to_range(tmp_path):
    rec = make_case(resting_case(42, duration_s=2.0, artifacts=())).recordings["eyes_open"]
    rec.layers["brain"][0, :10] = 5000.0
    rec.layers["brain"][1, :10] = -5000.0
    p = write_edf(tmp_path / "y.edf", rec, physical_range_uv=(-500.0, 500.0))
    e = edfio.read_edf(str(p))
    assert e.signals[0].data.max() <= 500.0 + 0.02
    assert e.signals[1].data.min() >= -500.0 - 0.02
