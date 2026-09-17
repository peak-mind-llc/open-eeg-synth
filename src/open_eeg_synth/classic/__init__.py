"""The classic streaming engine, moved from Coherence Recorder.

These generators are kept sample-identical to the originals so that existing
mock devices behave exactly as before. They use a hand-set per-electrode alpha
weight and a shared common-mode noise trace instead of a head model, so their
spatial structure is not realistic once the signal is re-referenced; the layered
engine replaces that.
"""

from open_eeg_synth.classic.cardio import MockAccSource, MockEcgSource, MockRRSource
from open_eeg_synth.classic.oddball import (
    CONDITION_CODES,
    CONDITION_RATIOS,
    ERP_COMPONENTS,
    RESPONSE_CODE,
    ErpComponent,
    ErpInjector,
    OddballEvent,
    OddballParadigm,
    generate_trial_sequence,
    schedule,
)
from open_eeg_synth.classic.synth import MarkerSchedule, RealisticEEGSynthesizer, alpha_weight

__all__ = [
    "CONDITION_CODES",
    "CONDITION_RATIOS",
    "ERP_COMPONENTS",
    "RESPONSE_CODE",
    "ErpComponent",
    "ErpInjector",
    "MarkerSchedule",
    "MockAccSource",
    "MockEcgSource",
    "MockRRSource",
    "OddballEvent",
    "OddballParadigm",
    "RealisticEEGSynthesizer",
    "alpha_weight",
    "generate_trial_sequence",
    "schedule",
]
