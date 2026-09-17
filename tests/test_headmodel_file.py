from __future__ import annotations

from importlib import resources

import numpy as np
import pytest

from open_eeg_synth.channels import CHANNELS_19

DATA = resources.files("open_eeg_synth.headmodel") / "data"


@pytest.fixture(scope="module")
def npz():
    with resources.as_file(DATA / "colin27_19ch.npz") as p:
        return np.load(p, allow_pickle=False)


def test_arrays_shapes_and_dtypes(npz):
    n_src = npz["source_pos"].shape[0]
    assert tuple(npz["channel_names"]) == CHANNELS_19
    assert npz["electrode_pos"].shape == (19, 3) and npz["electrode_pos"].dtype == np.float32
    assert npz["source_pos"].shape == (n_src, 3) and npz["source_pos"].dtype == np.float32
    assert npz["source_normal"].shape == (n_src, 3)
    assert npz["gain_free"].shape == (19, n_src, 3) and npz["gain_free"].dtype == np.float32
    assert npz["hemisphere"].shape == (n_src,) and npz["hemisphere"].dtype == np.int8
    assert 4000 < n_src < 6000


def test_normals_unit_and_hemispheres_by_x(npz):
    assert np.allclose(np.linalg.norm(npz["source_normal"], axis=1), 1.0, atol=1e-4)
    x = npz["source_pos"][:, 0]
    assert (x[npz["hemisphere"] == 0] < 0).mean() > 0.98
    assert (x[npz["hemisphere"] == 1] > 0).mean() > 0.98


def test_positions_in_metres_head_frame(npz):
    e = npz["electrode_pos"]
    assert np.abs(e).max() < 0.15  # metres, not mm
    o1, o2 = e[CHANNELS_19.index("O1")], e[CHANNELS_19.index("O2")]
    fp1 = e[CHANNELS_19.index("Fp1")]
    assert o1[1] < -0.09 and fp1[1] > 0.07  # y is anterior
    assert o1[0] < 0 < o2[0]  # x is right


def test_attribution_and_notice_present(npz):
    text = str(npz["attribution"])
    assert "Louis Collins" in text and "McGill" in text
    assert (DATA / "NOTICE-colin27.txt").read_text().strip() in text
