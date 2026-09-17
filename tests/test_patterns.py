from __future__ import annotations

import numpy as np
import pytest

from open_eeg_synth.artifacts import patterns
from open_eeg_synth.channels import CHANNELS_19, UnknownChannelError
from open_eeg_synth.headmodel import load_head_model


@pytest.fixture
def fake_eog(tmp_path, monkeypatch):
    names = np.array(list(CHANNELS_19))
    blink = np.zeros(19)
    blink[[0, 1]] = 1.0
    blink[[2, 3, 16]] = 0.4
    blink[[8, 9]] = -0.05
    heog = np.zeros(19)
    heog[10] = -1.0
    heog[11] = 1.0
    heog[0] = -0.4
    heog[1] = 0.4
    p = tmp_path / "eog_patterns.npz"
    np.savez(
        p,
        channel_names=names,
        blink_mean=blink,
        blink_sd=0.1 * np.abs(blink),
        heog_mean=heog,
        heog_sd=0.1 * np.abs(heog),
        n_subjects=np.int64(3),
        attribution=np.array("test"),
    )
    monkeypatch.setattr(patterns, "_EOG_PATH", p)
    patterns.load_eog_patterns.cache_clear()
    yield p
    patterns.load_eog_patterns.cache_clear()


def test_analytic_focal_and_dipole_shapes():
    head = load_head_model()
    t3 = head.electrode_pos[CHANNELS_19.index("T3")]
    m = patterns.analytic_focal(t3, head.electrode_pos, 35.0)
    assert np.isclose(m.max(), 1.0) and np.argmax(m) == CHANNELS_19.index("T3")
    assert m[CHANNELS_19.index("O2")] < 0.05
    d = patterns.analytic_dipole(patterns.EYE_CENTRE, (0.0, 0.0, 1.0), head.electrode_pos)
    assert np.abs(d).max() == 1.0 and d[CHANNELS_19.index("Fp1")] > 0.5
    h = patterns.analytic_dipole(patterns.EYE_CENTRE, (1.0, 0.0, 0.0), head.electrode_pos)
    assert np.sign(h[CHANNELS_19.index("F7")]) == -np.sign(h[CHANNELS_19.index("F8")])


def test_empirical_selects_channels_jitters_and_falls_back(fake_eog):
    head = load_head_model()
    m = patterns.empirical("blink", ["fp1", "T7", "O1"], z=0.0)
    assert np.allclose(m, [1.0, 0.0, -0.05])
    j = patterns.empirical("blink", ["Fp1", "F3"], z=1.0)
    assert np.isclose(j[1] / j[0], 0.44 / 1.1)  # mean + z * sd, then max-normalised
    with pytest.raises(UnknownChannelError):
        patterns.empirical("blink", ["Fp1", "AF7"])
    fb = patterns.empirical(
        "blink",
        ["Fp1", "Oz"],
        electrode_pos=np.array([head.electrode_pos[0], [0.0, -0.11, 0.0]]),
        z=0.0,
    )
    assert fb[0] == 1.0 and abs(fb[1]) < 0.3
