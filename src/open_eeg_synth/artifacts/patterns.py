"""Scalp patterns for non-cortical signals: empirical eye maps, analytic focal/dipole maps.

DESIGN §5.5.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from importlib import resources

import numpy as np

from open_eeg_synth.channels import UnknownChannelError, canonical_label

_EOG_PATH = resources.files("open_eeg_synth.artifacts") / "data" / "eog_patterns.npz"
EYE_CENTRE = np.array([0.0, 0.085, -0.020])  # between the eyes, head frame, metres


@lru_cache(maxsize=1)
def load_eog_patterns() -> dict:
    """Load the derived eye-pattern arrays, cached; every array is read-only (shared cache)."""
    with resources.as_file(_EOG_PATH) as p:
        z = np.load(p, allow_pickle=False)
        out = {}
        for k in z.files:
            arr = np.array(z[k])
            arr.flags.writeable = False
            out[k] = arr
        return out


def analytic_focal(
    centre_m: np.ndarray, electrode_pos_m: np.ndarray, sigma_mm: float
) -> np.ndarray:
    d = np.linalg.norm(electrode_pos_m - np.asarray(centre_m, float), axis=1) * 1000.0
    w = np.exp(-0.5 * (d / sigma_mm) ** 2)
    return w / w.max()


def analytic_dipole(pos_m, moment, electrode_pos_m: np.ndarray) -> np.ndarray:
    r = electrode_pos_m - np.asarray(pos_m, float)
    rn = np.linalg.norm(r, axis=1)
    v = (r @ np.asarray(moment, float)) / rn**3
    return v / np.abs(v).max()


_FALLBACK = {
    "blink": lambda pos: analytic_dipole(EYE_CENTRE, (0.0, 0.3, 1.0), pos),
    "heog": lambda pos: analytic_dipole(EYE_CENTRE, (1.0, 0.0, 0.0), pos),
}


def empirical(
    name: str,
    channels: Sequence[str],
    *,
    rng: np.random.Generator | None = None,
    jitter_sd: float = 0.6,
    electrode_pos: np.ndarray | None = None,
    z: float | None = None,
) -> np.ndarray:
    """Subject-averaged ICA map for ``name`` on ``channels``; unknowns fall back to analytic.

    A fallback channel is scaled to the file's own units: least squares fits the analytic model
    to the empirical values at whichever requested channels *are* in the file, and that one scalar
    is applied to the analytic values at the missing channels, so a fallback sits on the same scale
    as its neighbours rather than being independently max-normalised over just the requested set.
    Only when none of the requested channels are in the file (no anchor to fit against) does the
    fallback fall back to the analytic model's own max-1 normalisation.
    """
    if z is None:
        z = float(rng.normal(0.0, jitter_sd)) if rng is not None else 0.0
    f = load_eog_patterns()
    names = [str(c) for c in f["channel_names"]]
    mean, sd = f[f"{name}_mean"], f[f"{name}_sd"]
    out = np.empty(len(channels))
    found: list[int] = []
    missing: list[int] = []
    for i, ch in enumerate(channels):
        try:
            j = names.index(canonical_label(ch, names))
            out[i] = mean[j] + z * sd[j]
            found.append(i)
        except UnknownChannelError:
            missing.append(i)
    if missing:
        if electrode_pos is None:
            raise UnknownChannelError(
                f"{name}: no empirical map for {[channels[i] for i in missing]} and no positions"
            )
        analytic = _FALLBACK[name](np.asarray(electrode_pos, float))
        if found:
            a, e = analytic[found], out[found]
            denom = float(a @ a)
            scale = float(a @ e) / denom if denom > 0.0 else 0.0
            out[missing] = scale * analytic[missing]
        else:
            out[missing] = analytic[missing]
    return out / np.abs(out).max()
