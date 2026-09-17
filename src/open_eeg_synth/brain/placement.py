"""Where a rhythm's cortical patches sit (DESIGN §4.3)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from open_eeg_synth.channels import MIDLINE, MIRROR, canonical_labels
from open_eeg_synth.headmodel import HeadModel

REGIONS = {"posterior": {"y_pct_max": 12.0, "z_pct_min": 30.0}}


def placed_centres(
    head: HeadModel, sites: Sequence[str], rng: np.random.Generator, n_near: int = 25
) -> list[int]:
    """One source under each site; a mirror pair gets mirror-image sources. Two sites that name
    one channel raise ``ValueError``: they would share one source and count it twice."""
    names = list(canonical_labels(sites, head.channels, what="sites"))
    centres: dict[str, int] = {}
    for ch in names:
        if ch in centres:
            continue
        cand = head.sources_under(ch, n_near)
        if ch in MIDLINE or ch not in MIRROR:
            # DESIGN §4.3: the midline takes the single source nearest the midline,
            # not a random pick among the candidates under the electrode.
            c = int(cand[np.argsort(np.abs(head.source_pos[cand, 0]))[0]])
        else:
            c = int(rng.choice(cand))
        centres[ch] = c
        partner = MIRROR.get(ch)
        if partner is not None and partner in names:
            centres[partner] = head.mirror_source(c)
    return [centres[ch] for ch in names]


def region_centres(
    head: HeadModel, region: str, n_patches: int, rng: np.random.Generator
) -> list[int]:
    """n_patches/2 sources in the left half of a named region, plus their mirror images.

    Mirror images are nearest-source matches, so two left sources can share one; a left draw
    whose mirror image is already taken is redrawn from the left candidates whose images are
    still free, so no source carries two patches. Draws without a collision are unchanged.
    """
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
    mirrors = [head.mirror_source(c) for c in half]
    taken: set[int] = set()
    for i in range(len(half)):
        if mirrors[i] in taken:
            free = [
                int(c)
                for c in left
                if int(c) not in half and head.mirror_source(int(c)) not in taken
            ]
            if not free:
                raise ValueError(f"region {region!r} cannot hold {n_patches} distinct patches")
            half[i] = int(rng.choice(free))
            mirrors[i] = head.mirror_source(half[i])
        taken.add(mirrors[i])
    return half + mirrors
