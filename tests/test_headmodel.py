from __future__ import annotations

import numpy as np
import pytest

from open_eeg_synth.channels import CHANNELS_19, UnknownChannelError
from open_eeg_synth.headmodel import head_model_channels, load_head_model


@pytest.fixture(scope="module")
def head():
    return load_head_model()


def test_head_model_channels_matches_load_head_model_without_the_lead_field():
    assert head_model_channels() == load_head_model().channels == CHANNELS_19


def test_gain_is_free_dot_normal(head):
    g = head.gain
    assert g.shape == (19, head.n_sources)
    manual = np.einsum("csk,sk->cs", head.gain_free, head.source_normal)
    assert np.allclose(g, manual, rtol=1e-5, atol=1e-6)


def test_subset_and_aliases(head):
    sub = head.subset(["O1", "o2", "T7", "T8"])
    assert sub.channels == ("O1", "O2", "T3", "T4")
    assert sub.gain.shape == (4, head.n_sources)
    assert np.allclose(sub.gain[2], head.gain[CHANNELS_19.index("T3")])
    with pytest.raises(UnknownChannelError):
        head.subset(["Oz"])


def test_patch_map_unit_rms_and_local_maximum(head):
    c = int(head.sources_under("O1", 1)[0])
    m = head.patch_map(c, 12.0)
    assert np.isclose(np.sqrt(np.mean(m**2)), 1.0)
    assert np.argmax(np.abs(m)) in (CHANNELS_19.index("O1"), CHANNELS_19.index("T5"))


def test_mirror_source_flips_x(head):
    i = int(head.sources_under("C3", 1)[0])
    j = head.mirror_source(i)
    assert head.source_pos[j][0] > 0
    assert np.linalg.norm(head.source_pos[j] * [-1, 1, 1] - head.source_pos[i]) < 0.006


def test_smoothed_mixing_unit_mean_variance_and_neighbour_structure(head):
    M = head.smoothed_mixing(20.0)
    assert M.shape == (19, 19)
    C = M @ M.T
    assert np.isclose(np.mean(np.diag(C)), 1.0)
    corr = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    o1, o2, fp1 = (CHANNELS_19.index(k) for k in ("O1", "O2", "Fp1"))
    assert corr[o1, o2] > corr[o1, fp1]  # near pairs more correlated than far pairs
