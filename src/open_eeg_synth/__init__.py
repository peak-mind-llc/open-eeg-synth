"""open-eeg-synth: realistic, seeded synthetic EEG for teaching, demos and testing.

``open_eeg_synth.classic`` holds the streaming engine that powered Coherence
Recorder's mock devices. A layered head-model engine is in development; see
``docs/DESIGN.md``.
"""

from __future__ import annotations

from open_eeg_synth.case import CaseSpec, make_case, make_engine, make_subject
from open_eeg_synth.casefile.writer import write_case
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.recipes import resting_brain, resting_case
from open_eeg_synth.version import SIGNAL_VERSION, __version__

__all__ = [
    "SIGNAL_VERSION",
    "CaseSpec",
    "__version__",
    "load_head_model",
    "make_case",
    "make_engine",
    "make_subject",
    "resting_brain",
    "resting_case",
    "write_case",
]
