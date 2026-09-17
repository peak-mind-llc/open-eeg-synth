from __future__ import annotations

import numpy as np

from open_eeg_synth.case import make_case
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.recipes import resting_case
from tests.helpers import band_power, welch

FS = 256.0


def test_drowsy_half_has_less_slower_alpha_more_theta_and_roving_eyes():
    case = make_case(resting_case(31, duration_s=120.0, drowsy_from_s=60.0))
    rec = case.recordings["eyes_closed"]
    x = rec.layers["brain"]
    alert, drowsy = x[:, : int(55 * FS)], x[:, int(65 * FS) :]
    o1, fz = CHANNELS_19.index("O1"), CHANNELS_19.index("Fz")
    assert band_power(drowsy, FS, 8, 13)[o1] < 0.4 * band_power(alert, FS, 8, 13)[o1]
    assert band_power(drowsy, FS, 4, 8)[fz] > 2.0 * band_power(alert, FS, 4, 8)[fz]

    # alpha peak moves down by about 1 Hz
    def peak(seg):  # Welch (4 s Hann) rather than a raw periodogram: the argmax is far less noisy
        f, p = welch(seg[o1], FS, int(4 * FS))
        m = (f >= 6) & (f <= 13)
        return f[m][np.argmax(p[0, m])]

    assert peak(alert) - peak(drowsy) > 0.5
    subtypes = [(t.onset_s, t.subtype) for t in rec.truth if t.kind == "eye_movement"]
    assert any(s == "slow_roving" and on > 60 for on, s in subtypes)
    assert rec.timeline.state_at(90.0) == "drowsy"
    assert [s["state"] for s in rec.timeline.to_dict()["segments"]] == ["eyes_closed", "drowsy"]
