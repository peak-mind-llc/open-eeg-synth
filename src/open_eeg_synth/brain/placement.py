"""Where a rhythm's cortical patches sit (DESIGN §4.3)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from open_eeg_synth.channels import MIDLINE, MIRROR, canonical_label
from open_eeg_synth.headmodel import HeadModel

REGIONS = {"posterior": {"y_pct_max": 12.0, "z_pct_min": 30.0}}


def placed_centres(
    head: HeadModel, sites: Sequence[str], rng: np.random.Generator, n_near: int = 25
) -> list[int]:
    """One source under each site; a mirror pair gets mirror-image sources."""
    names = [canonical_label(s, head.channels) for s in sites]
    centres: dict[str, int] = {}
    for ch in names:
        if ch in centres:
            continue
        cand = head.sources_under(ch, n_near)
        if ch in MIDLINE or ch not in MIRROR:
            cand = cand[np.argsort(np.abs(head.source_pos[cand, 0]))[:5]]
        c = int(rng.choice(cand))
        centres[ch] = c
        partner = MIRROR.get(ch)
        if partner is not None and partner in names:
            centres[partner] = head.mirror_source(c)
    return [centres[ch] for ch in names]


def region_centres(
    head: HeadModel, region: str, n_patches: int, rng: np.random.Generator
) -> list[int]:
    """n_patches/2 sources in the left half of a named region, plus their mirror images."""
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}; known: {sorted(REGIONS)}")
    r = REGIONS[region]
    rr = head.source_pos
    cand = np.where(
        (rr[:, 1] < np.percentile(rr[:, 1], r["y_pct_max"]))
        & (rr[:, 2] > np.percentile(rr[:, 2], r["z_pct_min"]))
    )[0]
    left = cand[rr[cand, 0] < -0.01]
    half = [int(c) for c in rng.choice(left, max(1, n_patches // 2), replace=False)]
    return half + [head.mirror_source(c) for c in half]
