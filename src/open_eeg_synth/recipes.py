"""Ready-made specifications: the tuned resting model and the artifact set (DESIGN §4.3)."""

from __future__ import annotations

from open_eeg_synth.brain.background import BackgroundSpec
from open_eeg_synth.brain.layer import BrainSpec
from open_eeg_synth.brain.network import NetworkSpec
from open_eeg_synth.brain.rhythm import RhythmSpec
from open_eeg_synth.brain.state import StateSegment, StateTimeline
from open_eeg_synth.case import ArtifactSpec, CaseSpec, ConditionSpec
from open_eeg_synth.channels import CHANNELS_19

# Calibration (Task 12, DESIGN §4.3 "M2 re-derives amp_uv", §9.1 targets). The feasibility test's
# amplitudes were set in its empirical convention; under the analytic convention (§4.3, §11.6) the
# resting model was re-tuned on MEDIANS over 96 subjects (case seeds 100-195, `resting_case(seed,
# duration_s=60, artifacts=())`, brain + sensor noise, perturbed head), measured as DESIGN §9.1
# measures them: average reference, Cz RMS after MNE's default 1-45 Hz FIR band-pass, alpha share
# = 8-13 Hz / 1-40 Hz Welch power (tests.helpers.band_power), exponent = tests.helpers.psd_slope.
#
#   median over 96 seeds       before (shipped)     after              target
#   Cz RMS 1-45 Hz, EC / EO    13.9 / 13.7 uV       13.2 / 12.8 uV     real medians 13.7 / 12.7
#   O1 alpha share EC (p10-90) 0.37 (0.12-0.68)     0.60 (0.36-0.80)   real 0.58 (0.35-0.79)
#   O1 alpha share EO          0.14                 0.20               real band 0.05-0.30
#   Fz alpha share EC / EO     0.17 / 0.10          0.38 / 0.13        feasibility 0.40 / 0.14
#   exponent EC / EO           1.24 / 1.24          1.24 / 1.24        real band 0.8-1.4
#   O1-O2 8-13 Hz correlation  0.51 (5/96 below 0)  0.79 (0/96)        real: high
# ("after" re-measured once per-patch offsets/lags became subject draws and colliding mirror
# patches were redrawn; neither moved a median by more than 0.01 or 0.1 uV.)
#
# The Cz RMS must be read band-limited: the 1/f background runs down to 0.03 Hz, so more than half
# of the unfiltered variance lies below 1 Hz and an unfiltered std reads ~19 uV at these settings
# (an unfiltered reading is what once made the background look twice too loud). Knobs:
_BACKGROUND_RMS_UV = 18.0  # was 20: Cz 1-45 Hz RMS 13.9 -> 13.2 uV EC, 13.7 -> 12.8 uV EO
_ALPHA_AMP_UV = 31.0  # was 20: O1 alpha share EC 0.37 -> 0.60 (with the geometry below)
# 4 patches of 10 mm put the loudest alpha channel anywhere from P3/P4/Pz to T5/T6 and gave an O1
# share spread of 0.12-0.68 (p10-p90), far wider than real; 10 patches of 25 mm narrow it to
# 0.36-0.80 and lift Fz toward the feasibility test's 0.4 (0.33 at 20 mm). `indep` stays 0.6
# (DESIGN §11.2 wants it no lower); more and wider patches raise the O1-O2 correlation instead.
_ALPHA_N_PATCHES = 10  # was 4
_ALPHA_WIDTH_MM = 25.0  # was 10
# theta, beta and smr keep the feasibility test's values: the targets above were met without them.


def resting_brain() -> BrainSpec:
    """The resting model: the feasibility test's structure, amplitudes calibrated in Task 12."""
    return BrainSpec(
        background=BackgroundSpec(
            smoothing_mm=20.0, exponent=1.2, rms_uv=_BACKGROUND_RMS_UV, network_frac=0.5
        ),
        network=NetworkSpec(),
        rhythms=(
            RhythmSpec(
                "alpha",
                f0_hz=10.0,
                amp_uv=_ALPHA_AMP_UV,
                region="posterior",
                n_patches=_ALPHA_N_PATCHES,
                width_mm=_ALPHA_WIDTH_MM,
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


def ordinary_artifacts() -> tuple[ArtifactSpec, ...]:
    """Blinks, eye movements and occasional jaw tension — what every ordinary recording has
    (DESIGN §5.6)."""
    return (
        ArtifactSpec("blink"),
        ArtifactSpec("eye_movement"),
        ArtifactSpec("emg", {"side": "random"}),
    )


def resting_case(
    seed: int,
    *,
    duration_s: float = 240.0,
    plants=(),
    artifacts=None,
    drowsy_from_s: float | None = None,
    fs: float = 256.0,
    channels=CHANNELS_19,
) -> CaseSpec:
    """The common two-condition case: eyes closed then eyes open, ``duration_s`` each (§7.2).

    ``drowsy_from_s`` turns the rest of the eyes-closed recording drowsy; ``artifacts=None``
    means :func:`ordinary_artifacts`.
    """
    if drowsy_from_s is None:
        ec = [StateSegment(0.0, duration_s, "eyes_closed")]
    else:
        ec = [
            StateSegment(0.0, drowsy_from_s, "eyes_closed"),
            StateSegment(drowsy_from_s, duration_s, "drowsy"),
        ]
    conditions = (
        ConditionSpec("eyes_closed", duration_s, StateTimeline(ec)),
        ConditionSpec("eyes_open", duration_s, StateTimeline.constant("eyes_open", duration_s)),
    )
    return CaseSpec(
        seed=int(seed),
        fs=fs,
        channels=tuple(channels),
        plants=tuple(plants),
        artifacts=tuple(artifacts) if artifacts is not None else ordinary_artifacts(),
        conditions=conditions,
    )
