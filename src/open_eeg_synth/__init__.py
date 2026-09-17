"""open-eeg-synth: realistic, seeded synthetic EEG for teaching, demos and testing.

``open_eeg_synth.classic`` holds the streaming engine that powered Coherence
Recorder's mock devices. A layered head-model engine is in development; see
``docs/DESIGN.md``.
"""

from open_eeg_synth.version import SIGNAL_VERSION, __version__

__all__ = ["SIGNAL_VERSION", "__version__"]
