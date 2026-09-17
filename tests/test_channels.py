from __future__ import annotations

import pytest

from open_eeg_synth.channels import (
    CHANNELS_19,
    MIDLINE,
    MIRROR,
    UnknownChannelError,
    canonical_label,
    is_heart_label,
)


def test_channels_19_order_matches_head_model_rows():
    assert CHANNELS_19 == (
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "T5",
        "T6",
        "Fz",
        "Cz",
        "Pz",
    )


def test_canonical_label_case_and_aliases():
    assert canonical_label("fp1") == "Fp1"
    assert canonical_label(" T7 ") == "T3"
    assert canonical_label("P8") == "T6"
    with pytest.raises(UnknownChannelError, match="known"):
        canonical_label("AF3")
    assert canonical_label("AF3", known=("AF3", "AF4")) == "AF3"


def test_mirror_is_symmetric():
    for a, b in MIRROR.items():
        assert MIRROR[b] == a


def test_midline_membership():
    """placed_centres (DESIGN §4.3) gives a midline channel a deterministic, nearest-to-the-
    midline source instead of a random pick among the candidates under the electrode; this pins
    exactly which channels get that treatment."""
    assert MIDLINE == frozenset({"Fz", "Cz", "Pz", "Fpz", "Oz", "FCz", "CPz", "POz", "AFz"})
    assert {"Fz", "Cz", "Pz"} <= MIDLINE  # the 19-channel montage's own midline sites
    assert not (set(CHANNELS_19) - {"Fz", "Cz", "Pz"}) & MIDLINE  # no other 19-channel overlap


def test_heart_labels():
    assert is_heart_label("HR") and is_heart_label("ecg ") and not is_heart_label("Cz")
