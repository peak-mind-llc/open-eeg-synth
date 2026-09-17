"""Artifact kinds by name; third parties add entry points in group "open_eeg_synth.artifacts"."""

from __future__ import annotations

from importlib.metadata import entry_points

ARTIFACTS: dict[str, type] = {}
_discovered = False


def register(cls):
    kind = cls.kind
    if kind in ARTIFACTS and ARTIFACTS[kind] is not cls:
        raise ValueError(f"artifact kind {kind!r} already registered by {ARTIFACTS[kind]}")
    ARTIFACTS[kind] = cls
    return cls


def discover() -> None:
    global _discovered
    if _discovered:
        return
    _discovered = True
    for ep in entry_points(group="open_eeg_synth.artifacts"):
        register(ep.load())


def make_artifact(kind: str, **params):
    discover()
    if kind not in ARTIFACTS:
        raise KeyError(f"unknown artifact kind {kind!r}; known: {sorted(ARTIFACTS)}")
    return ARTIFACTS[kind](**params)
