from __future__ import annotations

import numpy as np

from open_eeg_synth.brain.background import BackgroundSpec
from open_eeg_synth.brain.network import Network, NetworkSpec, wire_network
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.seeds import stream_rng, stream_seed
from tests.helpers import render_whole_and_chunked

FS = 256.0


def test_wiring_is_deterministic_with_physical_lags():
    head = load_head_model()
    spec = NetworkSpec()
    w1 = wire_network(head, spec, FS, stream_rng(1, "subject:network"))
    w2 = wire_network(head, spec, FS, stream_rng(1, "subject:network"))
    assert w1.centres == w2.centres and w1.lags == w2.lags
    assert len(w1.centres) == 40 and all(len(s) == 3 for s in w1.sources)
    flat = [lag for row in w1.lags for lag in row]
    assert min(flat) >= int(round(0.005 * FS)) and max(flat) <= int(round(0.045 * FS))


def test_network_rms_and_chunk_invariance():
    head = load_head_model()
    bg = BackgroundSpec(rms_uv=20.0, network_frac=0.5)
    wiring = wire_network(head, NetworkSpec(), FS, stream_rng(1, "subject:network"))

    def make():
        return Network(head, FS, NetworkSpec(), bg, wiring, stream_seed(1, "eyes_closed:network"))

    whole, chunked = render_whole_and_chunked(make, int(40 * FS), np.random.default_rng(3))
    assert np.allclose(whole, chunked, atol=1e-4)
    rms = np.sqrt(np.mean(whole**2))
    assert 0.7 * 20.0 * np.sqrt(0.5) < rms < 1.3 * 20.0 * np.sqrt(0.5)


def test_coupled_nodes_are_lagged_copies():
    head = load_head_model()
    wiring = wire_network(head, NetworkSpec(), FS, stream_rng(4, "subject:network"))
    net = Network(head, FS, NetworkSpec(), BackgroundSpec(), wiring, stream_seed(4, "n"))
    x, y = net.render_nodes(int(30 * FS))
    j = 0
    k, lag = wiring.sources[j][0], wiring.lags[j][0]
    a = y[j] - y[j].mean()
    b = x[k] - x[k].mean()
    xc = [np.dot(a[m:], b[: len(b) - m]) for m in range(0, 20)]
    assert int(np.argmax(xc)) == lag
