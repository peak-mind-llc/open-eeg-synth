import numpy as np
import pytest

from open_eeg_synth.engine import Engine, Recording
from open_eeg_synth.seeds import stream_seed
from open_eeg_synth.sensor import SensorNoise
from tests.helpers import render_whole_and_chunked


class Ramp:
    """A deterministic test layer: channel c carries c + t."""

    name = "ramp"

    def __init__(self, n_ch):
        self.n_ch = n_ch

    def render(self, t0, n):
        t = np.arange(t0, t0 + n, dtype=np.float32)
        return (np.arange(self.n_ch, dtype=np.float32)[:, None] + t[None, :]).astype(np.float32)

    def truth(self):
        return []


class Kill:
    name = "transform:kill"

    def render_transform(self, t0, n, mix):
        d = np.zeros_like(mix)
        d[1] = -mix[1]
        return d

    def truth(self):
        return []


def test_engine_sums_layers_and_records_transform_delta():
    eng = Engine(["A", "B", "C"], 100.0, [Ramp(3)], transforms=[Kill()])
    fr = eng.render(0, 10)
    assert set(fr.layers) == {"ramp", "transform:kill"}
    assert np.allclose(fr.mixed, sum(fr.layers.values()))
    assert np.all(fr.mixed[1] == 0.0) and fr.mixed[2, 3] == 5.0
    assert eng.position == 10


def test_engine_requires_contiguous_calls():
    eng = Engine(["A"], 100.0, [Ramp(1)])
    eng.render(0, 5)
    with pytest.raises(ValueError, match="contiguous"):
        eng.render(9, 5)
    with pytest.raises(ValueError):
        eng.render(5, 0)


def test_render_all_equals_chunked_and_recording_sums():
    eng = Engine(["A", "B"], 100.0, [Ramp(2)])
    rec = eng.render_all(1000, block=64)
    assert isinstance(rec, Recording) and rec.n_samples == 1000 and rec.duration_s == 10.0
    assert np.allclose(rec.mixed, rec.layers["ramp"])
    ref = Ramp(2).render(0, 1000)
    assert np.array_equal(rec.layers["ramp"], ref)


def test_sensor_noise_level_and_chunk_invariance():
    def make():
        return SensorNoise(4, 1.5, stream_seed(9, "eyes_closed:sensor"))

    whole, chunked = render_whole_and_chunked(make, 5000, np.random.default_rng(0))
    assert np.allclose(whole, chunked)
    assert abs(whole.std() - 1.5) < 0.1 and whole.dtype == np.float32
