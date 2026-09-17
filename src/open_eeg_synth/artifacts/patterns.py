"""Scalp patterns for non-cortical signals: empirical eye maps, analytic focal/dipole maps.

DESIGN §5.5, with rulings P28-P30 and fix round 4:

- The eye maps in the data file are average-referenced ICA topographies. They are re-referenced to
  the mean of the file's T9 and T10 when loaded (P28): the synthetic recording is referential,
  an average-reference view of the result is unchanged, and a linked-ears view no longer shows a
  false posterior blink.
- Every amplitude is referenced to the full head, never to the requested channels (P30): an
  empirical map is divided by the largest |value| of the whole (referenced, jittered) file map,
  a focal map is the raw Gaussian, and a dipole map is divided by its largest |value| over the
  19-channel template head. A four-channel request returns the same numbers those four channels
  have in the full answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from importlib import resources

import numpy as np

from open_eeg_synth.channels import UnknownChannelError, canonical_label
from open_eeg_synth.headmodel import load_head_model

_EOG_PATH = resources.files("open_eeg_synth.artifacts") / "data" / "eog_patterns.npz"
EYE_CENTRE = np.array([0.0, 0.085, -0.020])  # between the eyes, head frame, metres
REFERENCE_CHANNELS = ("T9", "T10")  # the eye maps are referenced to their mean at load (P28)

# A channel's jitter step is its sd, capped at this fraction of its own |mean|. P29 asks that no
# z in [-1, 1] flip a channel's sign; on the shipped file the sd exceeds |mean| on 30 of the 64
# blink channels and 58 of the 64 heog channels, so the uncapped `mean + z * sign(mean) * sd`
# would flip (for example) heog T3 but not T4 at z = -1. Below the cap the formula is exact.
_JITTER_CAP = 0.9

# Fallback fit (fix round 4): the k nearest sign-agreeing present channels, weighted 1/d^2.
_FALLBACK_K = 5
# Floor on an anchor's model magnitude, taken as the dipole's direction cosine at that anchor:
# an anchor this close to the model's null says almost nothing about the scale, and must not be
# allowed to divide by (nearly) zero.
_FALLBACK_FLOOR = 0.05
# The analytic stand-in for each empirical map: an equivalent dipole (centre, moment) at the eyes.
_FALLBACK = {
    "blink": (EYE_CENTRE, (0.0, 0.3, 1.0)),  # vertical
    "heog": (EYE_CENTRE, (1.0, 0.0, 0.0)),  # lateral
}


@lru_cache(maxsize=1)
def load_eog_patterns() -> dict:
    """The derived eye-pattern file exactly as shipped (average-referenced), cached; every array is
    read-only (shared cache). `empirical` applies the T9/T10 reference on top of this."""
    with resources.as_file(_EOG_PATH) as p:
        z = np.load(p, allow_pickle=False)
        out = {}
        for k in z.files:
            arr = np.array(z[k])
            arr.flags.writeable = False
            out[k] = arr
        return out


def _eye_map(name: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """The file's channel names, its T9/T10-referenced mean map for `name`, and its sd (P28)."""
    f = load_eog_patterns()
    names = [str(c) for c in f["channel_names"]]
    try:
        ref_rows = [names.index(c) for c in REFERENCE_CHANNELS]
    except ValueError as err:
        raise ValueError(
            f"eye pattern file lacks a reference channel {REFERENCE_CHANNELS}"
        ) from err
    mean = np.asarray(f[f"{name}_mean"], float)
    return names, mean - mean[ref_rows].mean(), np.asarray(f[f"{name}_sd"], float)


def _full_map(name: str, z: float) -> tuple[list[str], np.ndarray]:
    """The whole referenced, jittered file map for `name`, divided by its largest |value| (P29-P30).

    ``jittered = mean + z * sign(mean) * step`` with ``step = min(sd, 0.9 |mean|)``: every channel
    moves away from zero as z grows and toward it as z falls, so mirror channels of opposite sign
    grow and shrink together, and for z in [-1, 1] none changes sign. A zero-mean channel never
    moves.
    """
    names, mean, sd = _eye_map(name)
    step = np.minimum(sd, _JITTER_CAP * np.abs(mean))
    jittered = mean + z * np.sign(mean) * step
    return names, jittered / np.abs(jittered).max()


def analytic_focal(
    centre_m: np.ndarray, electrode_pos_m: np.ndarray, sigma_mm: float
) -> np.ndarray:
    """``exp(-d^2 / 2 sigma^2)`` for each electrode, d its distance from the centre (P30).

    Not normalised: the value is 1 only at the centre point itself, and is the same for an
    electrode whichever other electrodes are requested with it. A caller that wants its loudest
    channel at 1 divides by the value at that channel on the full head.
    """
    d = np.linalg.norm(electrode_pos_m - np.asarray(centre_m, float), axis=1) * 1000.0
    return np.exp(-0.5 * (d / sigma_mm) ** 2)


def _dipole_terms(pos_m, moment, electrode_pos_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two factors of a dipole's raw value ``(r . m) / |r|^3``: the envelope ``|m| / |r|^2``
    (how loud the dipole could be at that distance) and ``cos`` of the angle between r and m."""
    m = np.asarray(moment, float)
    r = np.asarray(electrode_pos_m, float) - np.asarray(pos_m, float)
    rn = np.linalg.norm(r, axis=1)
    mn = float(np.linalg.norm(m))
    return mn / rn**2, (r @ m) / (rn * mn)


def analytic_dipole(pos_m, moment, electrode_pos_m: np.ndarray) -> np.ndarray:
    """``(r . m) / |r|^3``, divided by its largest |value| over the 19-channel template head
    (``load_head_model()``), not over the requested electrodes (P30): template electrodes lie in
    [-1, 1], and a request returns the same number for an electrode whatever else it asks for."""
    env, cos = _dipole_terms(pos_m, moment, electrode_pos_m)
    t_env, t_cos = _dipole_terms(pos_m, moment, load_head_model().electrode_pos)
    return env * cos / np.abs(t_env * t_cos).max()


def _fallback_scale(
    values: np.ndarray,
    envelope: np.ndarray,
    cosine: np.ndarray,
    dist: np.ndarray,
    labels: Sequence[str],
) -> float:
    """Scale of the analytic dipole at a missing channel, from candidate anchor channels.

    One entry per candidate: its map value, its dipole envelope and direction cosine, its
    distance from the missing channel and its label. Keeps the `_FALLBACK_K` nearest (exact
    distance ties go to the smaller label, so request order never matters) and solves the
    inverse-distance-weighted least squares ``min sum w (values/envelope - s * cos)^2`` with
    ``w = 1 / d^2`` and ``|cos|`` floored at `_FALLBACK_FLOOR`.

    Dividing by the envelope takes the dipole's 1/r^2 falloff out of the fit. Fitted on the raw
    model values instead, the anchor nearest the eyes (Fp1 is ~60x louder in the model than O1)
    would outweigh every nearer one, which is the fault that made earlier versions fit a missing
    ear or temporal channel to Fp1. With the falloff removed, what the fit weighs is how far
    away each anchor is and how far from the model's null it sits. No step of this ranks anchors
    by magnitude, so the result moves smoothly with the missing channel's position, apart from
    the (small, 1/d^2-weighted) change when the 5th and 6th nearest anchors swap.
    """
    order = sorted(range(len(dist)), key=lambda j: (float(dist[j]), str(labels[j])))
    near = np.array(order[:_FALLBACK_K])
    w = 1.0 / np.maximum(np.asarray(dist, float)[near], 1e-9) ** 2
    c = cosine[near]
    c = np.where(c >= 0.0, 1.0, -1.0) * np.maximum(np.abs(c), _FALLBACK_FLOOR)
    y = values[near] / envelope[near]
    return float(np.sum(w * c * y) / np.sum(w * c * c))


def empirical(
    name: str,
    channels: Sequence[str],
    *,
    rng: np.random.Generator | None = None,
    jitter_sd: float = 0.6,
    electrode_pos: np.ndarray | None = None,
    z: float | None = None,
) -> np.ndarray:
    """Subject-jittered ICA map for ``name`` on ``channels``; unknowns fall back to a dipole.

    The map is the file's T9/T10-referenced mean (P28) jittered by one subject scalar
    ``z ~ N(0, jitter_sd)`` clipped to [-1, 1] (drawn from ``rng`` unless ``z`` is given; see
    `_full_map` for the sign-keeping formula, P29), divided by the largest |value| of the whole
    64-channel map (P30). A requested channel in the file gets exactly its full-map value.

    A channel absent from the file (``electrode_pos`` required) takes the analytic dipole's value
    there, scaled to the file by `_fallback_scale` over its nearest requested in-file channels,
    using only those whose model sign agrees with their map sign (all of them if none agrees).
    A fallback is on the same full-head scale as its neighbours; near the eyes it can exceed 1.
    If no requested channel is in the file, the fallback is the analytic dipole itself (on its
    own full-head unit, see `analytic_dipole`).
    """
    if z is None:
        z = float(rng.normal(0.0, jitter_sd)) if rng is not None else 0.0
    z = float(np.clip(z, -1.0, 1.0))
    names, full = _full_map(name, z)
    out = np.empty(len(channels))
    found: list[int] = []
    labels: dict[int, str] = {}
    missing: list[int] = []
    for i, ch in enumerate(channels):
        try:
            labels[i] = canonical_label(ch, names)
        except UnknownChannelError:
            missing.append(i)
            continue
        out[i] = full[names.index(labels[i])]
        found.append(i)
    if not missing:
        return out
    if electrode_pos is None:
        raise UnknownChannelError(
            f"{name}: no empirical map for {[channels[i] for i in missing]} and no positions"
        )
    pos = np.asarray(electrode_pos, float)
    centre, moment = _FALLBACK[name]
    if not found:
        out[missing] = analytic_dipole(centre, moment, pos[missing])
        return out
    envelope, cosine = _dipole_terms(centre, moment, pos)
    # An anchor whose model sign disagrees with its map sign is not just weak but misleading (the
    # model has the direction of the response wrong there, typically right at a near-null
    # crossing), and fitting through it drags the scale toward a wrong compromise; skip such
    # anchors unless every candidate disagrees.
    agree = [i for i in found if np.sign(cosine[i]) == np.sign(out[i])] or found
    cand = np.array(agree)
    cand_labels = [labels[i] for i in agree]
    for m in missing:
        dist = np.linalg.norm(pos[cand] - pos[m], axis=1)
        scale = _fallback_scale(out[cand], envelope[cand], cosine[cand], dist, cand_labels)
        out[m] = scale * envelope[m] * cosine[m]
    return out
