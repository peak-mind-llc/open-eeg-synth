"""Ready-made specifications: the tuned resting model and the artifact set (DESIGN §4.3)."""

from __future__ import annotations

from open_eeg_synth.brain.background import BackgroundSpec
from open_eeg_synth.brain.layer import BrainSpec
from open_eeg_synth.brain.network import NetworkSpec
from open_eeg_synth.brain.rhythm import RhythmSpec


def resting_brain() -> BrainSpec:
    """The feasibility test's tuned resting model (amplitudes re-derived in Task 12)."""
    return BrainSpec(
        background=BackgroundSpec(smoothing_mm=20.0, exponent=1.2, rms_uv=20.0, network_frac=0.5),
        network=NetworkSpec(),
        rhythms=(
            RhythmSpec(
                "alpha",
                f0_hz=10.0,
                amp_uv=20.0,
                region="posterior",
                n_patches=4,
                width_mm=10.0,
                lag_ms=30.0,
                indep=0.6,
                f0_jitter_hz=0.6,
                state_gain={"eyes_open": 0.3, "drowsy": 0.35},
                state_f0_shift_hz={"drowsy": -1.0},
            ),
            RhythmSpec(
                "theta",
                f0_hz=6.0,
                amp_uv=7.0,
                sites=("Fz", "F3", "F4", "Cz"),
                env_tau_s=0.8,
                env_sd=0.7,
                lag_ms=40.0,
                state_gain={"drowsy": 2.0},
            ),
            RhythmSpec(
                "beta",
                f0_hz=19.0,
                amp_uv=7.0,
                sites=("C3", "C4", "F3", "F4", "P3", "P4"),
                f_sd=1.2,
                f_tau_s=0.6,
                env_tau_s=0.3,
                env_sd=0.9,
                env_lo=0.1,
                env_hi=3.0,
                lag_ms=35.0,
            ),
            RhythmSpec(
                "smr",
                f0_hz=13.5,
                amp_uv=6.0,
                sites=("C3", "C4", "Cz"),
                f_sd=0.6,
                f_tau_s=1.0,
                env_tau_s=0.4,
                env_sd=0.9,
                env_lo=0.1,
                env_hi=3.0,
                lag_ms=20.0,
            ),
        ),
    )
