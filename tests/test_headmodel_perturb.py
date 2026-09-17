from __future__ import annotations

import numpy as np

from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.seeds import stream_rng


def _rel_change(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_perturbation_is_deterministic_and_moderate():
    head = load_head_model()
    p1 = head.perturbed(stream_rng(1, "subject:head"))
    p2 = head.perturbed(stream_rng(1, "subject:head"))
    p3 = head.perturbed(stream_rng(2, "subject:head"))
    assert np.array_equal(p1.gain_free, p2.gain_free)
    assert not np.array_equal(p1.gain_free, p3.gain_free)
    assert 0.05 < _rel_change(p1.gain, head.gain) < 0.25
    assert head.perturbation is None and p1.perturbation is not None
    assert 5.0 < p1.perturbation["median_tilt_deg"] < 15.0
    assert 0.03 <= p1.perturbation["blur_eps"] <= 0.10
    assert len(p1.perturbation["channel_gain"]) == 19


def test_perturbed_normals_stay_unit_and_subset_commutes():
    head = load_head_model()
    p = head.perturbed(stream_rng(3, "subject:head"))
    assert np.allclose(np.linalg.norm(p.source_normal, axis=1), 1.0)
    sub = p.subset(["O1", "O2"])
    assert np.allclose(sub.gain[0], p.gain[head.index("O1")])


def test_single_channel_subset_perturbs_without_nan():
    p = load_head_model().subset(["Cz"]).perturbed(stream_rng(4, "subject:head"))
    assert np.isfinite(p.gain).all() and p.gain.shape[0] == 1
