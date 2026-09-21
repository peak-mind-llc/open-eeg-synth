"""Artifact plug-ins. Importing this package registers the built-in kinds."""

from __future__ import annotations

# the built-in kinds; each module registers itself on import
from open_eeg_synth.artifacts import (
    blink,  # noqa: F401
    contact_noise,  # noqa: F401
    dead_channel,  # noqa: F401
    electrode_pop,  # noqa: F401
    eye_movement,  # noqa: F401
    jaw_emg,  # noqa: F401
    mains_hum,  # noqa: F401
    noisy_channel,  # noqa: F401
    sweat_drift,  # noqa: F401
)
from open_eeg_synth.artifacts.base import (  # noqa: F401
    ContinuousArtifact,
    Event,
    EventArtifact,
    Occupancy,
    Remedy,
    RenderContext,
    TransformArtifact,
    TruthRecord,
)
from open_eeg_synth.artifacts.registry import ARTIFACTS, make_artifact, register  # noqa: F401
