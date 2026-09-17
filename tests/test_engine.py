from __future__ import annotations

import numpy as np
import pytest

from open_eeg_synth.dsp import PinkCascade
from open_eeg_synth.engine import Engine, Recording
from open_eeg_synth.seeds import stream_rng, stream_seed
from open_eeg_synth.sensor import SensorNoise
from tests.helpers import render_whole_and_chunked

FS = 256.0


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


def test_engine_copies_layer_output_so_buffer_reuse_is_safe():
    class Reuse:
        """A hostile layer: reuses one scratch buffer across calls, mutated in place."""

        name = "reuse"

        def __init__(self):
            self.buf = None
            self.k = 0

        def render(self, t0, n):
            if self.buf is None or self.buf.shape[1] != n:
                self.buf = np.zeros((1, n), dtype=np.float32)
            self.k += 1
            self.buf[:] = self.k
            return self.buf

        def truth(self):
            return []

    eng = Engine(["A"], 100.0, [Reuse()])
    rec = eng.render_all(40, block=10)
    assert np.array_equal(rec.layers["reuse"][0, ::10], [1.0, 2.0, 3.0, 4.0])


def test_engine_copies_transform_delta_so_returning_mix_is_safe():
    class One:
        name = "one"

        def render(self, t0, n):
            return np.ones((1, n), dtype=np.float32)

        def truth(self):
            return []

    class Echo:
        """A hostile transform: hands back the running mix itself (an aliasing hazard)."""

        name = "transform:echo"

        def render_transform(self, t0, n, mix):
            return mix

        def truth(self):
            return []

    class Bump:
        """Runs after Echo and changes the mix again; must not retroactively change Echo's layer."""

        name = "transform:bump"

        def render_transform(self, t0, n, mix):
            return np.ones_like(mix)

        def truth(self):
            return []

    fr = Engine(["A"], 100.0, [One()], transforms=[Echo(), Bump()]).render(0, 3)
    assert np.all(
        fr.layers["transform:echo"] == 1.0
    )  # mix was 1 (just the "one" layer) when Echo ran
    assert np.all(fr.layers["transform:bump"] == 1.0)
    assert np.all(fr.mixed == 3.0)  # 1 (one) + 1 (echo) + 1 (bump)


def test_engine_mix_passed_to_transforms_is_read_only():
    class Meddle:
        name = "transform:meddle"

        def render_transform(self, t0, n, mix):
            mix[0] = 0.0  # attempting to mutate the running mix in place must fail
            return np.zeros_like(mix)

        def truth(self):
            return []

    eng = Engine(["A"], 100.0, [Ramp(1)], transforms=[Meddle()])
    with pytest.raises(ValueError):
        eng.render(0, 3)


def test_engine_rejects_transform_with_wrong_shape():
    class Flat:
        name = "transform:flat"

        def render_transform(self, t0, n, mix):
            return np.ones(n)  # 1-D, not (n_ch, n)

        def truth(self):
            return []

    eng = Engine(["A", "B"], 100.0, [Ramp(2)], transforms=[Flat()])
    with pytest.raises(ValueError, match="transform:flat"):
        eng.render(0, 3)


def test_engine_rejects_no_layers():
    with pytest.raises(ValueError, match="at least one layer"):
        Engine(["A"], 100.0, [])


def test_render_all_equals_random_partitions_with_stateful_layer_and_transform():
    class Pink:
        """A stateful brain-like layer: PinkCascade output scaled to a plausible µV range."""

        name = "brain"

        def __init__(self, n_ch, seed):
            self.rng = stream_rng(seed, "eyes_closed:background")
            self.pk = PinkCascade(n_ch, FS)
            self.pk.warm_up(self.rng, 5.0)
            self.n_ch = n_ch

        def render(self, t0, n):
            return 20.0 * self.pk.process(self.rng.standard_normal((n, self.n_ch)).T)

        def truth(self):
            return []

    class Dead:
        name = "transform:dead_channel"

        def render_transform(self, t0, n, mix):
            d = np.zeros_like(mix)
            d[1] = -mix[1]
            return d

        def truth(self):
            return []

    def make():
        return Engine(["A", "B", "C"], FS, [Pink(3, 1)], transforms=[Dead()])

    n_total = 5000
    whole = make().render_all(n_total, block=4096)
    for trial in range(3):
        rng = np.random.default_rng(trial)
        eng = make()
        frames, t0 = [], 0
        while t0 < n_total:
            n = int(min(n_total - t0, rng.integers(1, 700)))
            frames.append(eng.render(t0, n))
            t0 += n
        for k in whole.layers:
            cat = np.concatenate([f.layers[k] for f in frames], axis=1)
            assert np.allclose(cat, whole.layers[k], atol=1e-4), (trial, k)
        mixc = np.concatenate([f.mixed for f in frames], axis=1)
        assert np.allclose(mixc, whole.mixed, atol=1e-4), trial
