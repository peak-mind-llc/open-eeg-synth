"""Matching failing realism lines against named known gaps (no MNE, so it is unit-tested fast)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import NamedTuple

Line = tuple[str, str, str, str]  # (condition, reference, band, metric)


class Gap(NamedTuple):
    """Why a line is known to fail; ``bins`` names the distance bins excused (binned metrics)."""

    reason: str
    bins: tuple[int, ...] = ()


class Failure(NamedTuple):
    line: Line
    bins: tuple[int, ...]  # the failing distance bins; () for an unbinned metric
    message: str


def is_excused(failure: Failure, gaps: Mapping[Line, Gap]) -> bool:
    """A failure is excused only if its line is listed and every failing bin is a listed bin."""
    gap = gaps.get(failure.line)
    return gap is not None and set(failure.bins) <= set(gap.bins)


def unexpected(failures: Iterable[Failure], gaps: Mapping[Line, Gap]) -> list[Failure]:
    return [f for f in failures if not is_excused(f, gaps)]


def passing_gaps(
    failures: Iterable[Failure], gaps: Mapping[Line, Gap], condition: str
) -> list[Line]:
    """Listed lines of ``condition`` that did not fail (binned: none of their bins failed)."""
    failing = {f.line: set(f.bins) for f in failures}
    out = []
    for line, gap in gaps.items():
        if line[0] != condition:
            continue
        bad = failing.get(line)
        if bad is None or (gap.bins and not bad & set(gap.bins)):
            out.append(line)
    return out
