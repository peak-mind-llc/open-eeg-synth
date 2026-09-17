"""Artifact kinds by name; third parties add entry points in group "open_eeg_synth.artifacts"."""

from __future__ import annotations

import warnings
from importlib.metadata import entry_points

ARTIFACTS: dict[str, type] = {}
_discovered = False


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
    """Load every third-party entry point once. A broken one is skipped, not silent: it is
    warned about and does not stop the other entry points from registering, and the "already
    discovered" flag is only set after every entry point has had its chance, so a later call
    never needs to (and does not) retry ones that already succeeded."""
    global _discovered
    if _discovered:
        return
    for ep in entry_points(group="open_eeg_synth.artifacts"):
        try:
            register(ep.load())
        except Exception as exc:  # noqa: BLE001 - a broken third-party plug-in must not break the rest
            name = getattr(ep, "name", repr(ep))
            warnings.warn(
                f"open_eeg_synth.artifacts: failed to load entry point {name!r}: {exc}",
                stacklevel=2,
            )
    _discovered = True


def make_artifact(kind: str, **params):
    discover()
    if kind not in ARTIFACTS:
        raise KeyError(f"unknown artifact kind {kind!r}; known: {sorted(ARTIFACTS)}")
    return ARTIFACTS[kind](**params)
