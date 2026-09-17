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
# (an unfiltered reading is what once made the background look twice too loud). Knobs (the table
# above describes the Task 12 recipe; the values below are the Task 32 ones, Task 12's in comments):
_BACKGROUND_RMS_UV = 19.0  # Task 12: 20 -> 18 (Cz 13.9 -> 13.2 uV EC); Task 32: 18 -> 19, see below
_ALPHA_AMP_UV = 37.0  # Task 12: 20 -> 31 (O1 alpha share EC 0.37 -> 0.60); Task 32: 31 -> 37
# 4 patches of 10 mm put the loudest alpha channel anywhere from P3/P4/Pz to T5/T6 and gave an O1
# share spread of 0.12-0.68 (p10-p90), far wider than real; 10 patches of 25 mm narrow it to
# 0.36-0.80 and lift Fz toward the feasibility test's 0.4 (0.33 at 20 mm). `indep` stays 0.6
# (DESIGN §11.2 wants it no lower); more and wider patches raise the O1-O2 correlation instead.
_ALPHA_N_PATCHES = 10  # was 4
_ALPHA_WIDTH_MM = 25.0  # was 10
# theta, beta and smr kept the feasibility test's values in Task 12.
#
# Realism tuning (Task 32, DESIGN §9.1, tests/realism/test_realism.py). The Task 12 recipe failed 12
# lines of the realism test (seeds 100-105, 60 s, artifacts=()): four too little lagged coupling
# (dwPLI: average theta EO 0.067 < 0.083, bipolar alpha EC 0.126 < 0.143, Laplacian alpha EC 0.088
# < 0.104, bipolar delta EO 0.001 < 0.002); four beta too coherent (bipolar mean EO 0.288 > 0.281,
# average nearest bin EO, Laplacian nearest and > 150 mm bins EC, > 150 mm bin EO); three Laplacian
# too coherent (theta nearest bin EC 0.567 > 0.513 and EO, alpha 120-150 mm bin EC); and the
# bipolar delta nearest bin EO too low (0.292 < 0.299). The zero-lag parts (background,
# in-phase rhythm patches) set coherence; only the network's delays and the rhythms' per-patch lags
# make dwPLI, so the tuning moved variance from the first to the second and spread the lags. Every
# change is below with its old value; "reverting it" is the measured effect of putting that one
# constant back in the final recipe (realism test, same seeds; calibration medians over 24 seeds,
# 100-123, measured as Task 12 measures them).
#
#   constant                  old -> new   reverting it
#   background rms_uv          18 -> 19     still passes; Cz 1-45 Hz RMS EC 12.8 -> 12.3 uV (the
#                                           quieter zero-lag background cost Cz amplitude)
#   background smoothing_mm    20 -> 16     2 lines: Laplacian beta and theta nearest bin EC
#   background network_frac   0.5 -> 0.65   4 lines: theta dwPLI (bipolar EC, average and bipolar
#                                           EO), bipolar delta nearest bin EO
#   network coupling          1.0 -> 2.0    3 lines: bipolar beta dwPLI EC, bipolar theta dwPLI EO,
#                                           bipolar delta nearest bin EO
#   network width_mm           15 -> 12     3 lines: Laplacian beta nearest bin EC, average theta
#                                           dwPLI EO, bipolar delta nearest bin EO
#   alpha lag_ms               30 -> 50     3 lines: bipolar and Laplacian alpha dwPLI EC,
#                                           Laplacian alpha 120-150 mm bin EC
#   alpha amp_uv               31 -> 37     still passes; O1 alpha share EC 0.59 -> 0.51 (wider lags
#                                           spread the patches' phases, so the in-phase amplitude
#                                           convention over-states the realised alpha; 37 restores
#                                           the calibrated share)
#   theta lag_ms               40 -> 100    2 lines: Laplacian theta nearest bin EC, average theta
#                                           dwPLI EO
#   theta width_mm             12 -> 40     3 lines: Laplacian theta nearest (and EC > 150 mm) bin
#                                           EC and EO, average theta dwPLI EO (a focal theta patch
#                                           under F3/F4 makes a Laplacian centre-surround pair with
#                                           F7/F8)
#   beta amp_uv                 7 -> 5.5    1 line: Laplacian beta nearest bin EC
#   beta lag_ms                35 -> 80     1 line: Laplacian beta > 150 mm bin EC
#
# Found one change at a time down to 2 failing lines (alpha lag 45, beta lag 70, coupling 1.5,
# theta lag 80, network width 12, alpha amp 34, theta width 30), then, since no single change
# reduced those two Laplacian lines, by a random search and a coordinate search over these same
# constants around that point. Result: 0 failing lines; the 24-seed medians (100-123) also pass
# every line; the closest line is 0.004 inside its band (Laplacian theta nearest bin EC, 0.509
# against 0.513), so the pass is narrow. Calibration after tuning, medians over 96 seeds (100-195):
# Cz 1-45 Hz RMS 13.15 uV EC (Task 12: 13.22), 12.49 EO; O1 alpha share 0.60 EC, 0.20 EO; exponent
# 1.25 EC, 1.24 EO. The O1-O2 8-13 Hz correlation fell from 0.79 to 0.70 (0/96 negative) with the
# longer alpha lags.
_BACKGROUND_SMOOTHING_MM = 16.0  # was 20
_NETWORK_FRAC = 0.65  # was 0.5
_NETWORK_COUPLING = 2.0  # was 1.0 (the NetworkSpec default)
_NETWORK_WIDTH_MM = 12.0  # was 15 (the NetworkSpec default)
_ALPHA_LAG_MS = 50.0  # was 30
_THETA_LAG_MS = 100.0  # was 40
_THETA_WIDTH_MM = 40.0  # was 12 (the RhythmSpec default)
_BETA_AMP_UV = 5.5  # was 7
_BETA_LAG_MS = 80.0  # was 35


def resting_brain() -> BrainSpec:
    """The resting model: the feasibility test's structure, calibrated (Task 12) and tuned (32)."""
    return BrainSpec(
        background=BackgroundSpec(
            smoothing_mm=_BACKGROUND_SMOOTHING_MM,
            exponent=1.2,
            rms_uv=_BACKGROUND_RMS_UV,
            network_frac=_NETWORK_FRAC,
        ),
        network=NetworkSpec(coupling=_NETWORK_COUPLING, width_mm=_NETWORK_WIDTH_MM),
        rhythms=(
            RhythmSpec(
                "alpha",
                f0_hz=10.0,
                amp_uv=_ALPHA_AMP_UV,
                region="posterior",
                n_patches=_ALPHA_N_PATCHES,
                width_mm=_ALPHA_WIDTH_MM,
                lag_ms=_ALPHA_LAG_MS,
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
                width_mm=_THETA_WIDTH_MM,
                env_tau_s=0.8,
                env_sd=0.7,
                lag_ms=_THETA_LAG_MS,
                state_gain={"drowsy": 2.0},
            ),
            RhythmSpec(
                "beta",
                f0_hz=19.0,
                amp_uv=_BETA_AMP_UV,
                sites=("C3", "C4", "F3", "F4", "P3", "P4"),
                f_sd=1.2,
                f_tau_s=0.6,
                env_tau_s=0.3,
                env_sd=0.9,
                env_lo=0.1,
                env_hi=3.0,
                lag_ms=_BETA_LAG_MS,
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
    """The default artifact set (DESIGN §5.6); empty until Task 20 fills it in."""
    return ()


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
