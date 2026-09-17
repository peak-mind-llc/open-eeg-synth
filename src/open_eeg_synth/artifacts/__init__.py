"""Artifact plug-ins. Importing this package registers the built-in kinds."""

from __future__ import annotations

from open_eeg_synth.artifacts.base import (  # noqa: F401
    Event,
    EventArtifact,
    Occupancy,
    Remedy,
    RenderContext,
    TransformArtifact,
    TruthRecord,
)
from open_eeg_synth.artifacts.registry import ARTIFACTS, make_artifact, register  # noqa: F401

# built-in kinds (each module registers itself on import) — extended in Tasks 16-19
