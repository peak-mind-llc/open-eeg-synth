"""open-eeg-synth: realistic, seeded synthetic EEG for teaching, demos and testing.

The layered head-model engine makes streams (``StreamSource``) and whole practice cases with a
sealed truth file (``make_case``, ``write_case``); ``docs/DESIGN.md`` describes it.
``open_eeg_synth.classic`` keeps the earlier streaming synthesizer, unchanged.

The names below are imported on first use (a module ``__getattr__``), so importing
``open_eeg_synth.classic`` loads numpy only, not scipy or the engine (DESIGN §10).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from open_eeg_synth.version import SIGNAL_VERSION, __version__

if TYPE_CHECKING:
    from open_eeg_synth.case import CaseSpec, make_case, make_engine, make_subject
    from open_eeg_synth.casefile.writer import write_case
    from open_eeg_synth.headmodel import load_head_model
    from open_eeg_synth.markers import MarkerSchedule
    from open_eeg_synth.recipes import resting_brain, resting_case
    from open_eeg_synth.stream import StreamSource

_LAZY: dict[str, str] = {
    "CaseSpec": "open_eeg_synth.case",
    "MarkerSchedule": "open_eeg_synth.markers",
    "StreamSource": "open_eeg_synth.stream",
    "load_head_model": "open_eeg_synth.headmodel",
    "make_case": "open_eeg_synth.case",
    "make_engine": "open_eeg_synth.case",
    "make_subject": "open_eeg_synth.case",
    "resting_brain": "open_eeg_synth.recipes",
    "resting_case": "open_eeg_synth.recipes",
    "write_case": "open_eeg_synth.casefile.writer",
}

__all__ = [
    "SIGNAL_VERSION",
    "CaseSpec",
    "MarkerSchedule",
    "StreamSource",
    "__version__",
    "load_head_model",
    "make_case",
    "make_engine",
    "make_subject",
    "resting_brain",
    "resting_case",
    "write_case",
]


def __getattr__(name: str) -> Any:
    try:
        module = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # later lookups skip this function
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
