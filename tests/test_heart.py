from __future__ import annotations

import numpy as np

from open_eeg_synth.heart import HeartSource
from open_eeg_synth.seeds import stream_seed

FS = 256.0


def _kurt(x):
    x = x - x.mean()
    return float(np.mean(x**4) / np.mean(x**2) ** 2 - 3.0)


def test_ecg_is_spiky_at_about_72_bpm_and_chunk_invariant():
    def run(chunks):
        h = HeartSource(FS, stream_seed(1, "stream:heart"))
        out, t0 = [], 0
        for n in chunks:
            out.append(h.render(t0, n))
            t0 += n
        return np.concatenate(out), h

    whole, h = run([int(60 * FS)])
    chunked, _ = run([32] * int(60 * FS / 32))
    assert np.allclose(whole, chunked, atol=1e-3)  # beat windows cut Gaussian tails at ~1e-4 uV
    assert 150 < whole.max() <= 205 and _kurt(whole) > 3.0
    beats = h.beats_between(0.0, 60.0)
    assert 62 <= len(beats) <= 82
    assert whole.dtype == np.float32
