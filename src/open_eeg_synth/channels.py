"""Channel names, aliases and the mirror map used by the whole package."""

from __future__ import annotations

from collections.abc import Sequence

CHANNELS_19: tuple[str, ...] = (
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
)
MIRROR: dict[str, str] = {
    "Fp1": "Fp2",
    "F3": "F4",
    "F7": "F8",
    "C3": "C4",
    "T3": "T4",
    "P3": "P4",
    "T5": "T6",
    "O1": "O2",
}
MIRROR.update({v: k for k, v in list(MIRROR.items())})
MIDLINE: frozenset[str] = frozenset({"Fz", "Cz", "Pz", "Fpz", "Oz", "FCz", "CPz", "POz", "AFz"})
HEART_LABELS: frozenset[str] = frozenset({"HR", "ECG", "EKG"})
_ALIASES: dict[str, str] = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


class UnknownChannelError(ValueError):
    """A label that no head model or pattern file knows."""


def is_heart_label(label: str) -> bool:
    return label.strip().upper() in HEART_LABELS


def canonical_label(label: str, known: Sequence[str] = CHANNELS_19) -> str:
    """Map a device label onto one of ``known`` (case-insensitive, 10-10 aliases applied)."""
    key = label.strip().upper()
    key = _ALIASES.get(key, key)
    lookup = {k.upper(): k for k in known}
    if key in lookup:
        return lookup[key]
    raise UnknownChannelError(f"unknown channel {label!r}; known: {', '.join(known)}")


def canonical_labels(
    labels: Sequence[str], known: Sequence[str] = CHANNELS_19, *, what: str = "channels"
) -> tuple[str, ...]:
    """``labels`` canonicalised against ``known``; two labels naming one channel raise
    ``ValueError`` (``("T3", "t7")`` must not silently become ``("T3", "T3")``)."""
    canon = tuple(canonical_label(lb, known) for lb in labels)
    dupes = sorted({c for c in canon if canon.count(c) > 1})
    if dupes:
        raise ValueError(f"duplicate {what} after alias canonicalisation: {dupes}")
    return canon
