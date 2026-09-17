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

# A model value below this (the model is already max-1 normalised over the requested positions)
# is too close to that model's null to anchor a fit alone; widen to the next-nearest present
# channel instead, and never divide by less than this either.
_FALLBACK_FLOOR = 0.05

# Two candidate anchors this close together count as equally near: which one distance ranks
# first is then noise (electrode-position uncertainty is easily this large), so the tie is
# broken by preferring the smaller model magnitude instead (see _nearest_scale).
_TIE_TOLERANCE_M = 0.010


def _nearest_scale(
    analytic: np.ndarray, empirical_out: np.ndarray, found: list[int], m: int, pos: np.ndarray
) -> float:
    """Least-squares scale fit for missing channel `m`, anchored on its nearest present channels.

    Starts with just the single nearest present channel - the best local match to a missing
    channel's true response - and only widens to its 2nd- and 3rd-nearest present neighbours if
    the channels included so far all sit too close to the analytic model's null (DESIGN §5.5): a
    channel near a null carries almost no information about the fitted scale, and including one
    on its own would let a near-zero denominator blow the scale up.

    Ties in "nearest" (within _TIE_TOLERANCE_M) are broken toward the smaller model magnitude:
    two present channels can sit at nearly identical distance from a missing one, and which of
    them distance ranks first is then a coin flip an electrode-position fit easily perturbs. If
    that coin flip is between a modest, in-family anchor and a far stronger one that only ties on
    distance by chance, picking the strong one can swing the fit wildly from one draw to the next
    even though nothing meaningfully moved; preferring the smaller magnitude on a true tie keeps
    the fit's sensitivity to position noise in line with everywhere else it is not exactly tied.
    """
    dist = {i: float(np.linalg.norm(pos[i] - pos[m])) for i in found}
    order = sorted(found, key=lambda i: (round(dist[i] / _TIE_TOLERANCE_M), abs(analytic[i])))
    k = 1
    while k < min(3, len(order)) and abs(analytic[order[k - 1]]) < _FALLBACK_FLOOR:
        k += 1
    anchors = order[:k]
    a = analytic[anchors]
    a_safe = np.where(a >= 0.0, 1.0, -1.0) * np.maximum(np.abs(a), _FALLBACK_FLOOR)
    e = empirical_out[anchors]
    denom = float(a_safe @ a_safe)
    return float(a_safe @ e) / denom if denom > 0.0 else 0.0


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

    A fallback channel is scaled to the file's own units: for each missing channel, least squares
    fits the analytic model to the empirical values at its nearest present requested channels (up
    to 3, widening past the nearest one only when needed for a stable fit - see `_nearest_scale`),
    and applies that one scalar to the analytic value at the missing channel, so a fallback sits on
    the same scale as its neighbours rather than being independently max-normalised over just the
    requested set. Only when none of the requested channels are in the file (no anchor to fit
    against at all) does the fallback fall back to the analytic model's own max-1 normalisation.
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
        pos = np.asarray(electrode_pos, float)
        analytic = _FALLBACK[name](pos)
        if found:
            # An anchor whose analytic sign disagrees with its own empirical sign is not just
            # weak (the floor already handles that) but actively misleading: the dipole model
            # gets the *direction* of the response wrong there, most often right at a near-null
            # crossing where a tiny perturbation flips the model's sign but the real (measured)
            # signal has already settled on a side. Fitting through such an anchor anyway drags
            # a missing channel's scale toward whatever compromise fits a fundamentally wrong
            # sign constraint, which is what made mirror-symmetric fallbacks (e.g. two ear
            # channels) come out lopsided. Skip those anchors - widen past them - unless every
            # candidate for this channel disagrees, in which case there is nothing better to use.
            agree = [i for i in found if np.sign(analytic[i]) == np.sign(out[i])] or found
            for m in missing:
                scale = _nearest_scale(analytic, out, agree, m, pos)
                out[m] = scale * analytic[m]
        else:
            out[missing] = analytic[missing]
    return out / np.abs(out).max()
