"""Artifact kinds by name; third parties add entry points in group "open_eeg_synth.artifacts"."""

from __future__ import annotations

import warnings
from importlib.metadata import entry_points

ARTIFACTS: dict[str, type] = {}
_discovered = False
_failed_entry_points: dict[str, str] = {}


def register(cls):
    if "kind" not in cls.__dict__:
        raise TypeError(
            f"{cls.__name__} must define its own 'kind' (not inherit "
            f"{cls.kind!r} from {cls.__mro__[1].__name__})"
        )
    kind = cls.kind
    if kind in ARTIFACTS and ARTIFACTS[kind] is not cls:
        raise ValueError(f"artifact kind {kind!r} already registered by {ARTIFACTS[kind]}")
    ARTIFACTS[kind] = cls
    return cls


def discover() -> None:
    """Load every third-party entry point once. A broken one is skipped, not silent: loading is
    only attempted once per entry point (the "already discovered" flag is only set after every
    entry point has had its chance, so a broken one never blocks the ones after it, and a later
    call never re-attempts ones that already succeeded or failed), but a failure is remembered
    and re-warned about on *every* call to discover(), not just the one that first hit it, and
    named in make_artifact's KeyError, so it stays visible rather than being reported once and
    then going quiet."""
    global _discovered
    if not _discovered:
        for ep in entry_points(group="open_eeg_synth.artifacts"):
            try:
                register(ep.load())
            except Exception as exc:  # noqa: BLE001 - one broken plug-in must not break the rest
                _failed_entry_points[getattr(ep, "name", repr(ep))] = str(exc)
        _discovered = True
    for name, err in _failed_entry_points.items():
        warnings.warn(
            f"open_eeg_synth.artifacts: failed to load entry point {name!r}: {err}",
            stacklevel=2,
        )


def make_artifact(kind: str, **params):
    discover()
    if kind not in ARTIFACTS:
        msg = f"unknown artifact kind {kind!r}; known: {sorted(ARTIFACTS)}"
        if _failed_entry_points:
            msg += f"; failed entry points: {_failed_entry_points}"
        raise KeyError(msg)
    return ARTIFACTS[kind](**params)
