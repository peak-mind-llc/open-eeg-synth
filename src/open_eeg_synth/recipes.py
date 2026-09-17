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
_BACKGROUND_RMS_UV = 18.0  # Task 12: 20 -> 18 (Cz 13.9 -> 13.2 uV EC); Task 32 kept 18 (see below)
_ALPHA_AMP_UV = 37.0  # Task 12: 20 -> 31 (O1 alpha share EC 0.37 -> 0.60); Task 32: 31 -> 37
# 4 patches of 10 mm put the loudest alpha channel anywhere from P3/P4/Pz to T5/T6 and gave an O1
# share spread of 0.12-0.68 (p10-p90), far wider than real; 10 patches of 25 mm narrow it to
# 0.36-0.80 and lift Fz toward the feasibility test's 0.4 (0.33 at 20 mm). `indep` stays 0.6
# (DESIGN §11.2 wants it no lower); more and wider patches raise the O1-O2 correlation instead.
_ALPHA_N_PATCHES = 10  # was 4
_ALPHA_WIDTH_MM = 25.0  # was 10
# theta, beta and smr kept the feasibility test's values in Task 12.
#
# Realism tuning (Task 32, DESIGN §9.1, tests/realism/test_realism.py: 24 seeds, 100-123, 60 s,
# artifacts=()). The zero-lag parts of the model (background, in-phase rhythm patches) set
# coherence; only the network's delays and the rhythms' per-patch lags make dwPLI, so the tuning
# moved background variance into the delayed network, slowed its conduction and spread the alpha
# and beta lags. The Task 12 recipe failed 12 population-median lines (seeds 100-339): dwPLI too low
# (theta, delta), beta and Laplacian coherence too high.
#
# Theta is fixed by a ruling, not by the realism test: these cases are read by eye, so the
# frontal-midline theta generator stays focal (12 mm, lag <= 40 ms). A first pass widened it to
# 40 mm with 100 ms lags; that passed more connectivity lines but halved the Fz theta rhythm,
# moved the loudest eyes-open theta channel off Fz/Cz and made a LateralImbalance("theta", ...)
# plant hard to see. Measured for this recipe (eyes closed unless named; average reference):
#
#                                               Task 12     first pass    this recipe
#                                                           (theta 40 mm)
#   eyes-open loudest theta ch Fz or Cz (96)    80 %        23 %          65 %
#   Fz theta-rhythm power (rhythm alone, 96)    20.1 uV^2   10.3 uV^2     21.8 uV^2
#   theta left x0.4: F4/F3 theta shift (64)     +1.87 dB    +0.88 dB      +1.94 dB (p10 +1.31)
#   C3 beta-rhythm power (rhythm alone, 96)     33.7 uV^2   22.4 uV^2     22.4 uV^2
#
# (seeds 100-195, or 100-163 for the plant, 60 s. A theta plant is visible in the recording only
# if the network does not out-shout it: coupling 2.0 with a 0.75 network share and a 19 uV
# background left Fz/Cz loudest in only 56 %, so coupling is 1.5 and the background Task 12's 18.)
#
# The only line that fails is Laplacian theta, nearest distance bin, eyes closed (population 0.531
# against 0.513): the focal theta patches under Fz, F3, F4 and Cz share a driver, so neighbouring
# Laplacian channels stay coherent. It is a named known gap in the test, not a wider tolerance.
# Seeds 100-123 were used to choose these values, so their result is in-sample. Held out (200
# random 24-seed sets from seeds 124-339): 31 % pass every line outside the known gap, a mean of
# 1.00 other failing lines (max 4); the lines that fail most are eyes-open bipolar beta nearest bin
# (43 % of sets, population margin +0.003) and eyes-closed Laplacian beta > 150 mm and nearest
# bins (36 %, +0.003). Six-seed medians are too noisy to gate on: none of 36 disjoint six-seed
# sets from seeds 124-339 passes every line.
#
# Each change, reverted alone in this recipe: extra population lines (seeds 100-339) and the mean
# failing lines, known gap included, of 100 random held-out 24-seed sets (this recipe: 1.83).
#   constant                  was -> now   reverting it alone
#   background smoothing_mm    20 -> 16     no new line; the known gap misses by more; 2.08
#   background network_frac   0.5 -> 0.75   +2: theta dwPLI EO, bipolar beta nearest bin EO; 3.77
#   network coupling          1.0 -> 1.5    +1: bipolar beta nearest bin EO; 2.44
#   network width_mm           15 -> 12     +1: Laplacian beta nearest bin EC; 2.33
#   network velocity_m_s        6 -> 3.5    +1: theta dwPLI EO; 2.90
#   alpha lag_ms               30 -> 50     +1: Laplacian alpha 120-150 mm bin EC; 2.80
#   alpha amp_uv               31 -> 37     no new line; 1.71; O1 alpha share EC falls 0.61 -> 0.53
#                                           (measure.py's reading, seeds 100-123)
#   theta amp_uv                7 -> 7.3    no new line; 1.77; eyes-open Fz/Cz theta 65 % -> 61 %
#   beta amp_uv                 7 -> 5.5    +3: Laplacian beta bins EC, average and bipolar beta
#                                           mean coherence EO; 3.97
#   beta lag_ms                35 -> 80     +2: Laplacian beta > 150 mm bin EC, bipolar beta
#                                           nearest bin EO; 2.60
#
# Beta lags drawn from U(0, 80 ms) span more than a cycle at 19 Hz, so the per-patch lag acts as a
# random phase: the patches add in power, not in amplitude. Alpha lags up to 50 ms (half a cycle at
# 10 Hz) likewise lower the realised alpha under the in-phase amplitude convention, which is why
# the alpha amplitude went up. The O1-O2 8-13 Hz correlation (MNE FIR band-pass, average
# reference, eyes closed) is 0.68 over 240 seeds (100-339; 1/240 negative) against Task 12's 0.78
# (1/240 negative); a Butterworth band-pass reads 3/240 negative for the first-pass recipe.
#
# Calibration after tuning, medians over 96 seeds (100-195), measured as above: Cz 1-45 Hz RMS
# 13.02 uV EC, 12.31 EO (Task 12: 13.22, 12.77); O1 alpha share 0.62 EC, 0.21 EO; exponent 1.25 EC,
# 1.24 EO.
_BACKGROUND_SMOOTHING_MM = 16.0  # was 20
_NETWORK_FRAC = 0.75  # was 0.5
_NETWORK_COUPLING = 1.5  # was 1.0 (the NetworkSpec default)
_NETWORK_WIDTH_MM = 12.0  # was 15 (the NetworkSpec default)
_NETWORK_VELOCITY_M_S = 3.5  # was 6.0 (the NetworkSpec default)
_ALPHA_LAG_MS = 50.0  # was 30
_THETA_AMP_UV = 7.3  # was 7
_THETA_WIDTH_MM = 12.0  # the RhythmSpec default; keep it focal (see the ruling above)
_THETA_LAG_MS = 40.0  # unchanged; keep it <= 40 ms (see the ruling above)
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
        network=NetworkSpec(
            coupling=_NETWORK_COUPLING,
            velocity_m_s=_NETWORK_VELOCITY_M_S,
            width_mm=_NETWORK_WIDTH_MM,
        ),
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
                amp_uv=_THETA_AMP_UV,
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
