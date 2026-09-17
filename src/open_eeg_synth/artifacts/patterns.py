"""Scalp patterns for non-cortical signals: empirical eye maps, analytic focal/dipole maps.

DESIGN §5.5, with rulings P28-P30 and fix rounds 4-5:

- The eye maps in the data file are average-referenced ICA topographies. They are re-referenced to
  the mean of the file's T9 and T10 when loaded (P28): the synthetic recording is referential,
  an average-reference view of the result is unchanged, and a linked-ears view no longer shows a
  false posterior blink.
- Every amplitude is referenced to the full head, never to the requested channels (P30): an
  empirical map is divided by the largest |value| of the whole (referenced, jittered) file map,
  a focal map is the raw Gaussian, and a dipole map is divided by its largest |value| over the
  19-channel template head. A four-channel request returns the same numbers those four channels
  have in the full answer.
- A channel absent from the file takes the analytic dipole there, scaled to its nearest usable
  requested channels (fix rounds 4-5, `empirical`), and never exceeds the file map's peak.
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

# The per-subject jitter `empirical` draws (its `jitter_sd` default) and clips to, the one
# definition `case.make_subject` also draws from when it records a jitter for a plug-in's
# TruthRecord: a subject's recorded jitter must always describe exactly what got rendered.
JITTER_SD = 0.6
JITTER_CLIP = 1.0

# A channel's jitter step is its sd, capped at this fraction of its own |mean|. P29 asks that no
# z in [-1, 1] flip a channel's sign; on the shipped file the sd exceeds |mean| on 30 of the 64
# blink channels and 58 of the 64 heog channels, so the uncapped `mean + z * sign(mean) * sd`
# would flip (for example) heog T3 but not T4 at z = -1. Below the cap the formula is exact.
_JITTER_CAP = 0.9

# Fallback fit (fix rounds 4-5): the k nearest usable present channels, weighted r / d^2.
_FALLBACK_K = 5
# An anchor's reliability r depends on the dipole's direction cosine there: 0 at or below
# _FALLBACK_FLOOR, 1 from _FALLBACK_FULL, a smoothstep between. Near the model's null an anchor
# says almost nothing about the scale (blink F8 sits at cos +0.041 on the template head, +0.064
# on easycap-M1; heog Pz at +0.004), and a floor that kept it in the fit turned it into a scale
# ~20x its real one on small montages; the ramp fades it out instead, so a small electrode move
# changes the result a little. The ramp ends at 3x the floor, not 2x: on real montage positions
# 2x still let F8 (in 9 cases with F7) blow blink fallbacks up (easycap-M1: 25 of 4,486 sweep
# values over 3x, worst 5.3x; 3x: none, worst 2.5x), and no pinned result moves.
_FALLBACK_FLOOR = 0.05
_FALLBACK_FULL = 3.0 * _FALLBACK_FLOOR
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


def _reliability(cosine: np.ndarray) -> np.ndarray:
    """How far an anchor's dipole direction cosine is from the model's null, as a weight: 0 for
    ``|cos| <= _FALLBACK_FLOOR``, 1 for ``|cos| >= _FALLBACK_FULL``, a smoothstep between."""
    t = (np.abs(np.asarray(cosine, float)) - _FALLBACK_FLOOR) / (_FALLBACK_FULL - _FALLBACK_FLOOR)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _fallback_scale(
    values: np.ndarray,
    envelope: np.ndarray,
    cosine: np.ndarray,
    dist: np.ndarray,
    labels: Sequence[str],
) -> tuple[float, float] | None:
    """``(scale, trust)`` of the analytic dipole at a missing channel, from candidate anchor
    channels, or None when no candidate is usable.

    One entry per candidate: its map value, its dipole envelope and direction cosine, its
    distance from the missing channel and its label. A candidate is usable when its reliability
    `_reliability(cos)` is above 0. Keeps the `_FALLBACK_K` nearest usable ones (exact distance
    ties go to the smaller label, so request order never matters) and solves the weighted least
    squares ``min sum w (values/envelope - s * cos)^2`` with ``w = r(|cos|) / d^2`` and the true
    ``cos``: a near-null anchor fades out of the fit rather than being floored into it. ``trust``
    is the largest r among the anchors kept; `empirical` blends the fit with the pure dipole by
    it, because the weights alone cannot fade a lone anchor (with one anchor, s = y / cos
    whatever its weight).

    Dividing by the envelope takes the dipole's 1/r^2 falloff out of the fit. Fitted on the raw
    model values instead, the anchor nearest the eyes (Fp1 is ~60x louder in the model than O1)
    would outweigh every nearer one, which is the fault that made earlier versions fit a missing
    ear or temporal channel to Fp1. With the falloff removed, what the fit weighs is how far
    away each anchor is and how far from the model's null it sits. No step of this ranks anchors
    by magnitude, so the result moves smoothly with the missing channel's position, apart from
    the (small, 1/d^2-weighted) change when the 5th and 6th nearest anchors swap.
    """
    r = _reliability(cosine)
    usable = [j for j in range(len(dist)) if r[j] > 0.0]
    if not usable:
        return None
    order = sorted(usable, key=lambda j: (float(dist[j]), str(labels[j])))
    near = np.array(order[:_FALLBACK_K])
    w = r[near] / np.maximum(np.asarray(dist, float)[near], 1e-9) ** 2
    c = np.asarray(cosine, float)[near]
    y = np.asarray(values, float)[near] / np.asarray(envelope, float)[near]
    return float(np.sum(w * c * y) / np.sum(w * c * c)), float(r[near].max())


def empirical(
    name: str,
    channels: Sequence[str],
    *,
    rng: np.random.Generator | None = None,
    jitter_sd: float = JITTER_SD,
    electrode_pos: np.ndarray | None = None,
    z: float | None = None,
) -> np.ndarray:
    """Subject-jittered ICA map for ``name`` on ``channels``; unknowns fall back to a dipole.

    The map is the file's T9/T10-referenced mean (P28) jittered by one subject scalar
    ``z ~ N(0, jitter_sd)`` clipped to [-1, 1] (drawn from ``rng`` unless ``z`` is given; see
    `_full_map` for the sign-keeping formula, P29), divided by the largest |value| of the whole
    64-channel map (P30). A requested channel in the file gets exactly its full-map value.

    A channel absent from the file (``electrode_pos`` required) takes the analytic dipole's value
    there, scaled to the file by `_fallback_scale` over its nearest requested in-file channels.
    An anchor is used only if its model sign agrees with its map sign, and it is weighted by how
    far it sits from the model's null. With no usable anchor (none requested, none agreeing in
    sign, or all at the null) the fallback is the analytic dipole itself, on its own full-head
    unit (see `analytic_dipole`). Otherwise it is ``trust * fit + (1 - trust) * dipole``, where
    trust is the reliability of the best anchor used: exactly the fit whenever one anchor is
    fully reliable, and sliding to the dipole as the last ones approach the null, so an
    electrode moving across the floor does not switch the result.

    A fallback is on the same full-head scale as its neighbours, and is clipped to [-1, 1], the
    file map's peak. Next to the eyes a fit can exceed it and is held at it: on standard_1005
    positions, blink at Fp1h/Fp2h and along the AFp row, and heog at AF9/AF10, along the AFF
    row and (on small montages) at F9/F10, all nearer the eyes than the heog peak at F7/F8.
    """
    if z is None:
        z = float(rng.normal(0.0, jitter_sd)) if rng is not None else 0.0
    z = float(np.clip(z, -JITTER_CLIP, JITTER_CLIP))
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
    envelope, cosine = _dipole_terms(centre, moment, pos)
    # An anchor whose model sign disagrees with its map sign is not just weak but misleading (the
    # model has the direction of the response wrong there, typically right at a near-null
    # crossing), and fitting through it drags the scale toward a wrong compromise: never use it.
    agree = [i for i in found if np.sign(cosine[i]) == np.sign(out[i])]
    cand = np.array(agree, dtype=int)
    cand_labels = [labels[i] for i in agree]
    dipole = analytic_dipole(centre, moment, pos[missing])
    for k, m in enumerate(missing):
        dist = np.linalg.norm(pos[cand] - pos[m], axis=1)
        fit = _fallback_scale(out[cand], envelope[cand], cosine[cand], dist, cand_labels)
        if fit is None:
            out[m] = dipole[k]
        else:
            scale, trust = fit
            out[m] = trust * (scale * envelope[m] * cosine[m]) + (1.0 - trust) * dipole[k]
    out[missing] = np.clip(out[missing], -1.0, 1.0)
    return out
