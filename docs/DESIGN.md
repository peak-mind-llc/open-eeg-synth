# open-eeg-synth — package design

**Status:** the design of the v0.2.0 layered engine, revised after implementation so that it
describes the code as released, including the decisions taken while building it and their reasons.
The v0.1.x line is the `classic` subpackage: a recording application's original mock-device
synthesizer, moved here verbatim and protected by a same-seed golden test. Everything below builds
beside `classic`, never on top of it.

**What the package is.** A deterministic generator of synthetic scalp EEG with known truth. It produces
either a continuous, phase-continuous stream of chunks (for a recording application's mock amplifiers)
or whole recordings written to disk with a sealed truth file (for a QEEG analysis application's
practice cases). The first two consumers are Coherence Recorder and Coherence Workstation; nothing in
the package knows about either.

**What it is not.** It is not a model of pathology and it does not claim clinical realism beyond the
measured properties in §9. Every file it writes is labelled synthetic. Consumers own their own
vocabulary (profile names, finding names, tool names); this package speaks only in signal terms.

---

## 1. Goals and constraints

| Goal | How the design meets it |
|---|---|
| One engine, two output modes | `Engine.render(t0, n)` is the only signal path; whole recordings are rendered through it in blocks, streams call it chunk by chunk (§7). |
| Realistic spatial structure | Cortical sources projected through a real-head lead field (§3, §4), which a feasibility test showed matches public resting EEG in coherence-by-distance under average, bipolar and Laplacian references; on its test seeds the tuned recipe matches the public reference on every measured line but one named distance bin, and a second named bin fails on fresh seeds (§9.1). |
| Truth for grading | The recording is the sum of named layers; every layer stays retrievable; every artifact and planted pattern writes a truth record (§5, §7.4). |
| Extensible artifacts | A plug-in contract with a registry; four plug-ins ship (blink, eye movement, jaw EMG, and a dead channel that exercises the transform hook), nine more are shown to fit (§5). |
| Determinism | One case seed derives every random stream by name; chunking does not change the samples beyond floating-point rounding (§8). |
| Light runtime | numpy + scipy only. MNE is a development extra used by three offline scripts and the realism suite (§10). |
| Public and attributable | Head-model data and empirical scalp patterns ship with their notices (§3.2, §6.4). |

Python `>=3.10,<3.13`, Apache-2.0, Hatchling, `src/` layout, GitHub Actions (pytest + ruff).

---

## 2. Architecture

### 2.1 The layer model

A recording is the sum of layers, each a `(n_channels, n_samples)` float32 array in microvolts:

```
recording = brain + Σ artifact_k + sensor + Σ transform_k
```

- **brain** — background cortical noise, a delayed cortical network, rhythm generators, planted
  patterns, all driven by a state timeline and projected through the head model.
- **artifact:<kind>[#i]** — one layer per artifact plug-in instance (blink, eye movement, jaw EMG, …).
  Additive: signal = time course × scalp pattern, or channel-local.
- **sensor** — amplifier/electrode white noise.
- **transform:<kind>** — the rare artifacts that are not additive (a dead channel, a bridged pair).
  They act on the running mix and their layer stores the *difference* they made, so the sum
  identity still holds exactly.

Every layer implements the same small protocol (§7.1). A consumer that wants to grade cleaning applies
its own tools to each layer separately — that is why the layers are kept, not just the mix.

### 2.2 Module layout

```
src/open_eeg_synth/
  __init__.py            version, public re-exports: CaseSpec, make_case, make_engine, make_subject,
                         resting_brain, resting_case, write_case, StreamSource, MarkerSchedule,
                         load_head_model, SIGNAL_VERSION, __version__
  version.py             __version__ = "0.2.0", SIGNAL_VERSION = 2
  _canon.py              plain-type normalisation of the spec dataclasses (equal specs → equal JSON)
  channels.py            CHANNELS_19, MIRROR, MIDLINE, HEART_LABELS, label aliases, canonical_label(),
                         UnknownChannelError
  seeds.py               stream_seed(), stream_rng(), fresh_case_seed()
  dsp.py                 OU processes, 1/f^β cascade, anti-alias decimation, envelopes
  headmodel/
    __init__.py          HeadModel, load_head_model(), head_model_channels(), subsets, perturbation,
                         patch maps
    data/colin27_19ch.npz
    data/NOTICE-colin27.txt
  brain/
    background.py        smoothed cortical noise (§4.1)
    network.py           delayed cortical network (§4.2)
    rhythm.py            RhythmSpec, BurstGate, Rhythm, per-patch subject draws (§4.3)
    placement.py         sources under electrodes, mirroring, regions (§4.3)
    plants.py            planted-pattern primitives, Modifier, PlantRecord, plant registry (§4.4)
    state.py             StateSegment, StateTimeline (§4.5)
    layer.py             BrainSpec, BrainLayer (composition)
  artifacts/
    base.py              Remedy, TruthRecord, Occupancy, RenderContext, Event, EventArtifact,
                         TransformArtifact
    registry.py          register(), make_artifact(), ARTIFACTS, discover()
    patterns.py          empirical and analytic scalp patterns
    data/eog_patterns.npz
    data/NOTICE-eegmmidb.txt
    blink.py  eye_movement.py  jaw_emg.py            (v0.2.0)
    dead_channel.py                                  (v0.2.0, proves the transform hook)
    …                                                (future, §5.7)
  sensor.py              SensorNoise layer
  heart.py               HeartSource: RR intervals + ECG waveform (the stream's heart rows today; future
                         heartbeat-bleed plug-ins)
  engine.py              Layer, Transform, Frame, Engine, Recording
  case.py                SensorSpec, ArtifactSpec, ConditionSpec, CaseSpec, Subject, compiled_rhythms(),
                         make_subject(), make_engine(), make_recording(), make_case(), case_id_for(), Case
  recipes.py             resting_brain(), ordinary_artifacts(), resting_case()
  stream.py              StreamSource (for a recording application's mock amplifiers), marker clock
  markers.py             re-exports classic.MarkerSchedule unchanged
  casefile/
    __init__.py          (empty)
    edf.py               write_edf() (optional `edfio` extra)
    truth.py             case_truth(), sealed write_truth()/read_truth(), render_layers(),
                         LayerMismatchError
    writer.py            write_case(), read_case_truth(), CasePaths; re-exports render_layers()
  __main__.py            `python -m open_eeg_synth make-case …`
  classic/               v0.1.x synthesizer, untouched: synth.py (RealisticEEGSynthesizer, MarkerSchedule),
                         cardio.py (MockRRSource, MockEcgSource, MockAccSource), oddball.py (OddballParadigm,
                         schedule, ErpInjector — seeded, carries responses across chunk boundaries)
scripts/
  export_head_model.py          dev-only (MNE): forward solution → colin27_19ch.npz
  derive_artifact_patterns.py   dev-only (MNE): PhysioNet ICA → eog_patterns.npz, plus the selection
                                log scripts/output/eog_patterns_selection.json
  build_realism_reference.py    dev-only (MNE): PhysioNet baselines → tests/realism/reference/*.json
  update_golden.py              regenerates the layered engine's fingerprint after a deliberate
                                signal change (§8.3)
tools/
  make_classic_golden.py        records the classic engine's fixture (tests/data/classic_golden.npz)
tests/
  test_*.py                     fast unit tests (about 65 s in total; the target is 60 s, §9.2)
  test_golden_engine.py         the layered engine's signal fingerprint, with golden/ (§8.3)
  classic/                      the classic engine's tests, including its bit-exact golden test
  realism/                      measure.py, gaps.py, reference/, test_realism.py (marked `realism`, opt-in)
```

A loose-lead plug-in would need a `ContactTimeline` (per-channel impedance over time, which a
recording application's impedance display could share); it is future work and has no module yet
(§5.7).

### 2.3 Conventions

- Units: microvolts for every signal; metres for positions; seconds for `_s` parameters; samples for
  `t0`/`n`. Head-model gain stays in V per A·m as exported; all cortical patch maps are normalised
  (unit RMS across channels) and amplitudes are specified in µV at a named channel. Scalp patterns of
  non-cortical signals (§5.5) are 1.0 at the full head's loudest channel, whichever channels a case
  records.
- Coordinates: the template head frame (x right, y anterior, z superior). Mirror = `x → −x`.
- Channel order of `CHANNELS_19` is the head-model row order:
  `Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 Fz Cz Pz`. Aliases `T7/T8/P7/P8 → T3/T4/T5/T6`;
  labels are matched case-insensitively, ignoring surrounding whitespace.
- Reference: the lead field's potentials are in the BEM solver's native reference (its columns do not
  sum to zero), so channels share a common component exactly as a referential amplifier recording does.
  Consumers re-reference as they would real data. The realism numbers in §9 are all measured after
  average referencing, where that common component cancels.

---

## 3. Head model

### 3.1 Data file: `headmodel/data/colin27_19ch.npz`

Exported once by `scripts/export_head_model.py` from a 3-layer BEM forward solution on the Colin27
template (conductivities scalp 0.3, skull 0.006, brain 0.3 S/m; cortical surface source space; sources
closer than 5 mm to the inner skull dropped). The script reads the forward solution with MNE
(development extra) and writes plain arrays:

| array | dtype | shape | meaning |
|---|---|---|---|
| `channel_names` | `<U4` | (19,) | `CHANNELS_19` order |
| `electrode_pos` | float32 | (19, 3) | electrode positions, metres, head frame |
| `source_pos` | float32 | (4871, 3) | cortical source positions, metres |
| `source_normal` | float32 | (4871, 3) | cortical-patch-statistics normals (unit vectors; the 3 sources whose patch normal is undefined in the source solution carry the outward radial direction instead, as the `attribution` string records) |
| `gain_free` | float32 | (19, 4871, 3) | free-orientation lead field, V/(A·m), x/y/z components |
| `hemisphere` | int8 | (4871,) | 0 = left (first 2472 rows), 1 = right |
| `bem_conductivity` | float32 | (3,) | `[0.3, 0.006, 0.3]` |
| `mindist_mm` | float32 | () | 5.0 |
| `attribution` | `<U…` | () | the notice text plus provenance (script name, MNE version, export date) |

Sizes: `gain_free` is 1.11 MB uncompressed; the whole file is ~1.13 MB compressed. The fixed-orientation
lead field is derived at load time as `gain[c, s] = Σ_k gain_free[c, s, k] · source_normal[s, k]`
(verified against MNE's own fixed-orientation conversion to a relative error of 3e-8), so it is not
stored. The free-orientation form is shipped instead of the 0.37 MB fixed form because the orientation
perturbation in §3.3 needs it.

Facts used by the placement code, measured on the shipped file: median source spacing 3.8 mm; the
nearest source to the mirror image of any source is 2.74 mm away on median (4.51 mm at the 90th
percentile, 10.7 mm at most), so mirrored patch placement is faithful.

### 3.2 Attribution

`headmodel/data/NOTICE-colin27.txt` travels with the array file, is included in the wheel, and is
embedded verbatim in the `attribution` array:

> Colin27 head model redistributed by EEGLAB (`head_modelColin27_5003_Standard-10-5-Cap339.mat`); the
> lead field bundled in open-eeg-synth was computed from its surfaces and electrode cap with MNE-Python
> and is redistributed under the notice below.
>
> Copyright (C) 1993–2009 Louis Collins, McConnell Brain Imaging Centre, Montreal Neurological
> Institute, McGill University. Permission to use, copy, modify, and distribute this software and its
> documentation for any purpose and without fee is hereby granted, provided that the above copyright
> notice appear in all copies. The authors and McGill University make no representations about the
> suitability of this software for any purpose. It is provided "as is" without express or implied
> warranty. The authors are not responsible for any data loss, equipment damage, property loss, or
> injury to subjects or patients resulting from the use or misuse of this software package.

The package README's "Data and attribution" section repeats the notice.

### 3.3 Runtime API

```python
@dataclass(frozen=True)
class HeadModel:
    name: str                       # "colin27_19ch"
    channels: tuple[str, ...]
    electrode_pos: np.ndarray       # (n_ch, 3) m
    source_pos: np.ndarray          # (n_src, 3) m
    source_normal: np.ndarray       # (n_src, 3)
    gain_free: np.ndarray           # (n_ch, n_src, 3)
    hemisphere: np.ndarray          # (n_src,) int8
    attribution: str
    perturbation: dict | None = None  # None for the nominal model; the draw record after perturbed()

    @cached_property
    def gain(self) -> np.ndarray:   # (n_ch, n_src), fixed orientation
    n_channels: int                 # property
    n_sources: int                  # property
    def index(self, label: str) -> int                                  # row of a canonicalised label
    def subset(self, labels: Sequence[str]) -> "HeadModel"
    def perturbed(self, rng: np.random.Generator, *, tilt_deg=8.0, gain_sd=0.06,
                  blur_range=(0.03, 0.10), corr_mm=15.0) -> "HeadModel"
    def patch_map(self, centre: int, width_mm: float) -> np.ndarray     # (n_ch,), unit RMS
    def sources_under(self, label: str, n: int = 40) -> np.ndarray      # the n sources nearest the electrode
    def mirror_source(self, i: int) -> int                              # the source nearest (−x, y, z)
    def smoothed_mixing(self, scale_mm: float) -> np.ndarray            # (n_ch, n_ch)

def load_head_model(name: str = "colin27_19ch") -> HeadModel             # cached per name
def head_model_channels(name: str = "colin27_19ch") -> tuple[str, ...]  # channel names only
def all_head_model_channels() -> tuple[str, ...]  # all shipped models' names (§4.4)
```

`load_head_model` and `head_model_channels` raise `ValueError` listing the head models that ship
when `name` is unknown. `head_model_channels` reads only the `channel_names` array (an `.npz` member
is decompressed only when it is indexed), so `CaseSpec` can canonicalise its channels against the
selected head model on every construction without loading the lead field (§7.2).

**Channel subsets.** `subset(labels)` canonicalises each label (`channels.canonical_label`) and selects
rows of `electrode_pos` and `gain_free`; the source side is untouched. Cortical patches are always placed
on the full model (a patch under Fz exists whether or not Fz is recorded), and a subset only chooses
which rows are rendered, so one seed gives one brain under every montage. Unknown labels raise
`UnknownChannelError` naming the available labels. Non-EEG labels (`HR`, `ECG`, `EKG`) are handled by
the stream layer, never by the head model. Larger channel sets are a later export of the same shape
(`colin27_31ch.npz`, `colin27_37ch.npz`); `load_head_model` picks by name, and a `channels` argument on
`CaseSpec` selects the subset.

**Perturbation (the "inverse crime" mitigation).** A QEEG application that localises sources with an
inverse built from the same BEM would otherwise see a synthetic case as an exact forward/inverse pair
and localise it implausibly well. The forward operator cannot be re-solved at runtime (no BEM, no MNE),
so the perturbation is applied to the exported operator in three deterministic steps, all drawn from
the case's `subject:head` stream (§8):

1. **Source-orientation tilt.** A random 3-vector field over sources is smoothed with a Gaussian kernel
   of `corr_mm` = 15 mm, scaled to unit standard deviation, multiplied by `tan(tilt_deg)` with
   `tilt_deg` = 8, added to each normal, and renormalised. The fixed lead field is then recomputed from
   `gain_free` with the tilted normals. Real cortex never has the template's orientations; this is the
   most physical of the three steps and the reason the free-orientation gain is shipped.
2. **Per-channel gain.** Each channel row is scaled by `1 + N(0, gain_sd)` with `gain_sd` = 0.06,
   clipped to `[0.85, 1.15]` — local skull thickness and electrode contact variation.
3. **Scalp blur.** `gain' = ((1−ε) I + ε A) gain`, where `A` is the row-normalised Gaussian neighbour
   matrix over electrode distances (σ = 50 mm, zero diagonal) and `ε ~ U(0.03, 0.10)` — a slightly
   more or less conductive skull.

Measured on the nominal model the combined relative change `‖gain' − gain‖_F / ‖gain‖_F` lands between
5 % and 25 % (0.20–0.22 over subject seeds 0–19, with a median source tilt of 8.5–10.2°); a unit test
pins that range and a median tilt between 5° and 15°. `perturbation` records `tilt_deg`, `corr_mm`,
the median tilt actually applied (`median_tilt_deg`), the per-channel gains (`channel_gain`) and `ε`
(`blur_eps`), so a truth file can state exactly what was done. A one-channel head has no neighbour to
blur into: the blur is skipped there, but its draw is still consumed. Whether this is *enough* to make
a consumer's localisation look like real data is an open question (§11); the knobs are exposed for
that experiment.

---

## 4. Brain layer

All time-series generation is stateful and block-wise so that streaming and whole-recording rendering
produce the same samples (§8). The dataclass defaults shown below are the feasibility test's values.
The package's resting model (`recipes.resting_brain()`, the brain of every default case and stream)
departs from them where §4.3's table says, after a calibration pass and a tuning pass against the
realism reference (§9.1). Amplitudes follow an analytic convention (§4.3), not the feasibility test's
empirical one.

### 4.1 Background: smoothed cortical noise

Independent 1/f^β noise on every cortical source, spatially smoothed with a Gaussian (`smoothing_mm`:
20 mm by default, 16 mm in the recipe), projected through the lead field. Computing 4871 source time series is wasteful when only their 19-channel
projection is ever observed, so the feasibility test used a shortcut: the sensor covariance
`C = (G K)(G K)ᵀ` of that construction (`K` = Gaussian smoothing over source distances) is a 19×19
matrix, and `M = C^{1/2}` applied to 19 independent noise series has exactly that covariance.
**Decision: keep the shortcut.** It is exact in second-order statistics, which is all the background
contributes; a rank-19 Gaussian background is indistinguishable from the full construction to spectra,
coherence, and ICA (Gaussian sources are not separable anyway). `C^{1/2}` is computed with
`numpy.linalg.eigh` (no scipy needed here); `M` is normalised so the mean channel variance is 1 and
then scaled by `rms_uv · √(1 − network_frac)`. `M` depends on the perturbed gain, so it is computed
once per subject (~0.3 s) and every condition shares it (`Subject.mixing`).

```python
@dataclass(frozen=True)
class BackgroundSpec:            # the recipe: 16 mm, 1.2, 18 µV, 0.75
    smoothing_mm: float = 20.0
    exponent: float = 1.2        # power ∝ 1/f^exponent
    rms_uv: float = 20.0         # mean channel RMS of background + network together
    network_frac: float = 0.5    # share of background variance carried by the delayed network
```

Time courses: `dsp.PinkCascade` — 1/f^β by a cascade of first-order pole–zero sections (two per decade
from 0.03 Hz to 0.45·fs; Corsini & Saletti's construction), bilinear-transformed, run with carried
filter state. It is scaled to unit variance at construction from the energy of the cascade's impulse
response over 40 s, which is exact and needs no random draw (an earlier calibration from a noise
probe was biased by 3–14 % depending on the sample rate and exponent). Every cascade (background and
network) is run for 20 s on discarded noise from its own time stream when it is built, so its filter
states start stationary. The FFT-shaped noise the feasibility test used is kept only as the unit
test's oracle (fitted PSD slope over 1–40 Hz must equal β ± 0.1).

### 4.2 Delayed cortical network

Forty cortical patches (`width_mm`) at random source positions; each node's time course is its own
1/f^β series plus delayed copies of three other nodes' series (weight `coupling / fan_in` each), the
delay being `distance / velocity_m_s + synaptic_ms`. This is what gives the recording non-zero lagged
coupling (debiased wPLI) — the feasibility test showed that delays between *identical* maps are
invisible, so the network needs distinct patches. The network carries `network_frac` of the
background variance. The recipe uses 12 mm patches, coupling 1.5 and 3.5 m/s + 5 ms, which for
subject seed 20260916 puts the longest delay at 14 samples (55 ms) at 256 Hz. The wiring (node
centres, the nodes each one listens to, the lags in samples) is drawn once per subject from
`subject:network` and written to the truth file (`subject.network`).

```python
@dataclass(frozen=True)
class NetworkSpec:               # the recipe: coupling 1.5, velocity 3.5 m/s, width 12 mm
    n_nodes: int = 40
    fan_in: int = 3
    coupling: float = 1.0
    velocity_m_s: float = 6.0
    synaptic_ms: float = 5.0
    width_mm: float = 15.0
```

Streaming: each node keeps a history of `max_lag` samples. The history starts at zero, so the first
`max_lag` samples of a recording (tens of milliseconds) lack the delayed term. Normalisation is
analytic: node series are scaled by `1/√(1 + coupling²/fan_in)` to unit variance and the sensor
projection by `1/√(mean_c Σ_j map[j, c]²)`, so no calibration pass over the recording is needed.

### 4.3 Rhythm generator

A rhythm is a set of cortical patches under target electrodes that share a delayed driver plus some
independent activity. Mirror-pair electrodes get mirror-image patches (the feasibility test found
unmirrored random placement fires spurious asymmetry findings).

```python
@dataclass(frozen=True)
class BurstGate:
    on_s: tuple[float, float] = (1.0, 3.0)    # burst length drawn from U(on_s)
    off_s: tuple[float, float] = (8.0, 25.0)  # gap drawn from U(off_s); the first burst starts after one
    ramp_s: float = 0.3                       # linear ramp at each edge

@dataclass(frozen=True)
class RhythmSpec:
    name: str                          # "alpha", "theta", "beta", "smr", "plant:…"
    f0_hz: float
    amp_uv: float                      # see amplitude convention below
    sites: tuple[str, ...] = ()        # placement A: one patch under each site (mirrored)
    region: str | None = None          # placement B: n_patches inside a named cortical region
    n_patches: int = 4
    width_mm: float = 12.0
    f_sd: float = 0.35                 # centre-frequency wander (OU, Hz)
    f_tau_s: float = 4.0
    env_tau_s: float = 1.5             # log-envelope OU
    env_sd: float = 0.5
    env_lo: float = 0.25
    env_hi: float = 1.8
    lag_ms: float = 25.0               # per-patch lag drawn from U(0, lag_ms)
    indep: float = 0.3                 # share of each patch's variance that is its own
    f0_jitter_hz: float = 0.0          # per-subject draw of the centre frequency
    state_gain: dict[str, float] = field(default_factory=dict)      # amplitude multiplier per state
    state_f0_shift_hz: dict[str, float] = field(default_factory=dict)
    hemisphere_gain: dict[str, float] = field(default_factory=dict)  # {"left": 0.6} (plants)
    burst: BurstGate | None = None                                   # intermittent on/off gating (plants)
```

A `RhythmSpec` needs exactly one of `sites` and `region`, and an even `n_patches` for a region.

**Placement** is drawn once per subject from `subject:rhythm:<name>`, always on the full head
(§3.3). Site placement: each site's candidates are the 25 sources nearest its electrode; a lateral
site takes one of them at random, and its mirror partner, if it is also a target, takes that source's
mirror image; a midline site takes the candidate nearest the midline. Region placement (used by
alpha): candidates are sources with `y` below its 12th percentile and `z` above its 30th percentile
(occipito-parietal); `n_patches/2` are drawn from the left half (x < −10 mm) and mirrored. Mirror
images are nearest-source matches, so two left draws can share one; a left draw whose image is already
taken is redrawn from the left candidates whose images are still free, so no source carries two
patches.

**Time course.** For the driver and for each patch's own component, instantaneous frequency is an OU
process around the centre frequency (σ `f_sd`, τ `f_tau_s`), the envelope is `exp` of an OU process
around 0 (σ `env_sd`, τ `env_tau_s`) clipped to `[env_lo, env_hi]`, and phase is the cumulative sum of
frequency. Both OU processes use the exact discretisation `x[n] = μ + a (x[n−1] − μ) + σ√(1−a²) w[n]`,
`a = exp(−1/(τ fs))`, and start from their stationary distribution, so there is no warm-up transient.
Patch `p` renders `√(1 − indep) · driver(t − lag_p) + √indep · own_p(t)`, where `own_p` is centred at
`f0 + δ_p`. The time streams (`<condition>:rhythm:<name>`) are split into the driver, one oscillator
per patch and the burst gate.

**Per-patch subject draws.** The centre-frequency offset `δ_p ~ N(0, 0.3 Hz)` (the small spread the
feasibility test used, so that patches are not perfectly locked to one frequency) and the lag
`lag_p ~ U(0, lag_ms)` describe the subject, not the passage of time. `make_subject` draws them from
`subject:rhythm:<name>` after the placement and the centre-frequency jitter, and stores them on
`Subject` (`patch_f0_offsets_hz`, `patch_lags_ms`), so every condition of a case uses the same values
and the truth file records them (§7.4). Drawn from the condition's stream instead, they differed
between a subject's eyes-closed and eyes-open recordings. `BrainLayer` refuses to build a rhythm whose
per-patch values are missing rather than silently drawing new ones; only a standalone `Rhythm` draws
its own.

**Orientation.** Each patch's map is signed so that it is positive at its reference channel: its own
site for site placement, and its own strongest channel for region placement (patches in one region
peak at very different channels, so one shared reference channel suits some of them badly). A mirror
patch then takes its partner's polarity: its sign is chosen so that its value at the mirror of the
partner's reference channel is positive, like the partner's. Orienting each patch independently
flipped mirror pairs inconsistently wherever their local geometry differed, and made O1–O2 alpha
anti-correlated for some subjects.

**Amplitude convention.** `amp_uv` is the peak amplitude (√2 × RMS) at the loudest target channel when
the envelope is 1 and all patches are in phase: `scale = amp_uv / max_c |Σ_p map_p[c]|` over the
target channels (for region placement: the one channel where the summed map is largest). The scale is
fixed from the un-gained maps, and `hemisphere_gain` is applied afterwards, keyed by each patch's
source side (left below x = −5 mm, right above +5 mm, otherwise midline); applied before the scale, the
scale partly undid it. The feasibility test scaled empirically over the finished recording, so its
numbers carry an envelope-mean factor of about 1.1–1.5; the recipe's amplitudes were re-derived under
this convention (below). Per-patch lags lower the realised amplitude below the in-phase figure: alpha
lags of up to 50 ms span half a cycle, and beta lags of up to 80 ms span more than a cycle at 19 Hz,
where they act as a random phase and the patches add in power rather than in amplitude.

**Tuned resting model** (`recipes.resting_brain()`):

| rhythm | placement | f0 | amp (µV) | other |
|---|---|---|---|---|
| alpha | region posterior, 10 patches, 25 mm | 10.0, jitter 0.6 | 37 | lag 50 ms, indep 0.6, `state_gain {eyes_open: 0.3, drowsy: 0.35}`, `state_f0_shift_hz {drowsy: −1.0}` |
| theta | Fz F3 F4 Cz, 12 mm | 6.0 | 7.3 | env τ 0.8 s, env sd 0.7, lag 40 ms, `state_gain {drowsy: 2.0}` |
| beta | C3 C4 F3 F4 P3 P4, 12 mm | 19.0 | 5.5 | f sd 1.2, f τ 0.6 s, env τ 0.3 s, env sd 0.9, env 0.1–3.0, lag 80 ms |
| smr | C3 C4 Cz, 12 mm | 13.5 | 6 | f sd 0.6, f τ 1.0 s, env τ 0.4 s, env sd 0.9, env 0.1–3.0, lag 20 ms |

Background 18 µV, exponent 1.2, smoothing 16 mm, network share 0.75; network of 40 nodes of 12 mm,
fan-in 3, coupling 1.5, 3.5 m/s plus 5 ms; sensor noise 1.5 µV. Anything not listed takes the
dataclass default.

**How the values were chosen.** Always on medians over many subjects, because one seed misleads:
under the first placement, nine subjects' eyes-closed O1 alpha shares ranged from 0.13 to 0.68. The
effect of each change is tabled beside the constants in `recipes.py`.

1. *Calibration* (96 subjects, seeds 100–195, 60 s, brain and sensor noise only, measured as §9.1
   measures). The background went from 20 to 18 µV and alpha from 20 to 31 µV, to put Cz RMS and the
   eyes-closed O1 alpha share on the real medians. Cz RMS is read after 1–45 Hz filtering: the 1/f
   background has more than half its variance below 1 Hz, and an unfiltered reading once made the
   background look twice too loud. Alpha moved from 4 patches of 10 mm to 10 patches of 25 mm: the
   smaller set put the loudest alpha channel anywhere from Pz to T5/T6 and spread the eyes-closed O1
   alpha share over 0.12–0.68 (p10–p90) between subjects; the larger set narrows that to 0.36–0.80.
2. *Realism tuning* (the §9.1 test seeds 100–123, and a pool of seeds 100–339). The zero-lag parts of
   the model (background, in-phase patches) set coherence; only the network's delays and the rhythms'
   per-patch lags produce lagged coupling (dwPLI). The calibrated recipe failed 12 population-median
   lines: dwPLI too low in theta and delta, beta and Laplacian coherence too high. The tuning moved
   background variance into the delayed network (share 0.5 → 0.75, coupling 1.0 → 1.5, node width
   15 → 12 mm), slowed its conduction (6 → 3.5 m/s), narrowed the background smoothing (20 → 16 mm),
   spread the alpha and beta lags (30 → 50 ms, 35 → 80 ms) and lowered beta to 5.5 µV. Reverting any
   one of these alone adds failing lines or raises the mean number of failing lines over random
   24-seed sets. Alpha went up to 37 µV because the wider lags lower the realised alpha: at 31 µV the
   eyes-closed O1 alpha share on the test seeds falls from 0.61 to 0.53.
3. **Decision: frontal-midline theta stays focal, at the cost of one realism line.** Theta keeps
   12 mm patches and lags of at most 40 ms, with its amplitude raised from 7 to 7.3 µV. A wider, later
   theta (40 mm, lags up to 100 ms) passed more connectivity lines, but it halved the Fz theta rhythm,
   moved the loudest eyes-open theta channel off Fz/Cz, and hid `LateralImbalance("theta", …)` plants
   inside the spread between subjects. These cases are read by eye: a reader expects frontal-midline
   theta, and a planted theta asymmetry must be visible. The cost is a known gap: eyes-closed
   Laplacian theta coherence in the nearest distance bin is too high (0.536 on the test seeds and
   0.531–0.536 at every population median measured, against a limit of 0.513), because the theta
   patches under Fz, F3, F4 and Cz share a driver and neighbouring Laplacian channels stay coherent
   (§9.1, §11).

The theta checks behind that decision (average reference; seeds 100–195, eyes closed unless named):

| check | calibrated recipe | 40 mm theta | this recipe |
|---|---|---|---|
| eyes-open loudest theta channel is Fz or Cz | 80 % | 23 % | 65 % |
| Fz theta-rhythm power, rhythm alone | 20.1 µV² | 10.3 µV² | 21.8 µV² |
| theta ×0.4 on the left: shift in F4/F3 theta balance (seeds 100–163) | +1.87 dB | +0.88 dB | +1.94 dB |

A louder or more focal network pushes temporal channels (mostly T3) to the top of the eyes-open
theta map: coupling 2.0 with a 0.75 network share and a 19 µV background left Fz or Cz loudest in only
56 % of subjects, which is why coupling stopped at 1.5 and the background at 18 µV. The 65 % sits close
to the 60 % floor that was set for this check, and the realism test does not gate it (§11).

After tuning (medians over seeds 100–195, 60 s; real medians from the §9.1 reference in brackets):
Cz RMS after 1–45 Hz filtering 13.0 µV eyes closed and 12.3 µV eyes open [13.7, 12.7]; O1 alpha share
0.62 and 0.21 [0.58, 0.13]; aperiodic exponent 1.25 and 1.24 [1.20, 1.12]. The eyes-closed O1–O2
8–13 Hz correlation (average reference, seeds 100–339) is 0.68; it is negative in 1 of 240 subjects
with MNE's FIR band-pass (3 of 240 with a 4th-order Butterworth band-pass).

The template head is not quite symmetric (C3 sits 8.6 mm from the mirror image of C4), and the recipe
carries small left/right biases whose origin in the template and the placement was not isolated
further. Eyes closed, average reference, medians over seeds 100–195: alpha power is 0.73 dB lower at
O2 than at O1 and 0.62 dB lower at P4 than at P3, and 12–15 Hz power is 0.40 dB lower at C4 than at
C3.

### 4.4 Planted-pattern primitives

Plants are generic signal terms. A consumer maps them to its own finding vocabulary; the package only
promises the signal and a truth record. Each plant compiles to extra `RhythmSpec`s and/or modifiers on
the base rhythms:

```python
class Plant(Protocol):
    kind: ClassVar[str]
    def rhythms(self) -> tuple[RhythmSpec, ...]
    def modifiers(self) -> tuple[Modifier, ...]
    def record(self) -> PlantRecord

@dataclass(frozen=True)
class Modifier:
    rhythm: str                              # the base rhythm it changes
    amp_scale: float = 1.0
    f0_shift_hz: float = 0.0
    hemisphere_gain: dict[str, float] = {}   # multiplied into the rhythm's own
    extra_sites: tuple[str, ...] = ()        # appended to the rhythm's sites

@dataclass(frozen=True)
class PlantRecord:
    kind: str
    sites: tuple[str, ...]                   # () for a modifier-only plant
    band_hz: tuple[float, float] | None      # f0 ± 1 Hz; None for a modifier-only plant
    amp_uv: float | None                     # None for a modifier-only plant
    onset_s: float                           # 0.0: plants are present for the whole recording
    offset_s: float | None                   # None
    params: dict                             # the plant's own fields, JSON-safe (a modifier's target rhythm is params["rhythm"])
                                             # plus "bursts_s" for a bursting plant
    description: str                         # generic prose
```

| primitive | signal | example (feasibility test) |
|---|---|---|
| `FocalSlow(site, f0_hz=2.5, amp_uv=45, width_mm=12, f_sd=0.4, env_tau_s=1.5, env_sd=0.6, indep=0.3, state_gain={})` | a slow rhythm from one patch under one site, own driver, no lag | focal delta under a left frontal-temporal site |
| `RhythmicBursts(sites, f0_hz=6.5, amp_uv=30, width_mm=15, burst_s=(1, 3), gap_s=(8, 25), env_tau_s=1.0, env_sd=0.8, indep=0.2, state_gain={})` | a rhythm gated on and off in bursts (`BurstGate`, 0.3 s linear ramps; envelope 0.1–3.0, no lag) | frontal midline theta bursts |
| `LateralImbalance(rhythm, side, factor)` | multiplies the patches of one hemisphere (`side` is `"left"` or `"right"`) of a base rhythm | alpha ×0.6 on the left |
| `WidespreadExcess(rhythm, factor, extra_sites=())` | scales a base rhythm, optionally adding patches under more sites | theta ×2.2 everywhere |
| `PeakShift(rhythm, shift_hz)` | moves a base rhythm's centre frequency | alpha −1.5 Hz |
| `ReducedRhythm(rhythm, factor)` | scales a base rhythm down | sensorimotor rhythm ×0.2 |

**Compilation** (`case.compiled_rhythms(spec)`): the base rhythms with every modifier applied in plant
order, followed by the plants' own rhythms. A rhythm plant's compiled name carries its site(s) and its
frequency — `plant:focal_slow:F7:2.5hz`, `plant:rhythmic_bursts:Fz+Cz:6.5hz` — so two plants of one
kind at one site but at different frequencies are two rhythms. Every per-rhythm structure (subject
draws, time streams, the brain layer's parts) is keyed by that name, so two compiled rhythms with the
same name raise `ValueError`; before the frequency was part of the name, two such plants collided and
rendered as one rhythm at the last plant's frequency while the truth listed both. Because the name
selects the subject stream, a plant's own draws, and the base rhythms' draws, do not depend on which
other plants are present. The exception is `extra_sites`: the added sites draw from their rhythm's
subject stream, so the rhythm's per-patch lags change for its original patches too, and when two
modifiers add sites to one rhythm the site order follows the plant order.

**Validation.** Every value must be finite, `f0_hz` must be positive (also after a `PeakShift`),
factors must be at least 0, and `LateralImbalance.side` must be `"left"` or `"right"` (any other
string, such as `"Left"`, used to do nothing).

**Site names.** `FocalSlow.site`, `RhythmicBursts.sites`, `WidespreadExcess.extra_sites`,
`Modifier.extra_sites` and `RhythmSpec.sites` are canonicalised on construction with
`channels.canonical_label` against the channel names of the shipped head models
(`headmodel.all_head_model_channels()`, today the 19 of `colin27_19ch`), as a case's channels are
(§7.2): `FocalSlow("f7")` equals `FocalSlow("F7")`, gives the same case id and records the site as
`F7`, and an unknown site raises `UnknownChannelError` when the plant is built rather than when the
subject is drawn. Two spellings of one site in one list raise `ValueError`, and so does
`placement.placed_centres` when handed such a list. Because both lists are canonical,
`apply_modifiers` recognises a site the rhythm already has: `WidespreadExcess("theta", 1.0,
extra_sites=("f3",))` adds no patch and leaves theta unchanged (it used to add a second patch under
F3). Canonical spellings are unchanged, so existing case ids do not move. Plant kinds register by
`kind` (`register_plant`; a second class with the same kind raises); a spec's plants round-trip
through JSON as `{kind, params}`, and an unknown kind raises `ValueError` naming the known ones.

**State confinement.** `state_gain` on the two rhythm-creating primitives is passed through to the
`RhythmSpec` they compile to, so a consumer can confine a plant to some states (`{"eyes_open": 0.0}`
plants it eyes-closed only). The `PlantRecord.params` carry it, and the description names every state
where it is zero (`"… present throughout (absent in eyes_open)"`). A plant whose own `state_gain` is
zero in every state that a condition's timeline contains renders nothing there, and its record is left
out of that condition's plant list (`case.make_recording`, §7.2), so a truth file never lists a plant
that cannot be seen. A modifier-only plant is never left out.

**Burst times.** A `RhythmicBursts` record carries the bursts it actually produced in that
condition, as `params["bursts_s"]`: a list of `[on, off]` pairs in seconds, sorted, within the
recording. The burst gate draws its on/off edges in order from the rhythm's time stream and keeps
every edge it draws; an edge's time is the sample where its 0.3 s ramp is half-way, and a burst
still on at the end of the recording is cut there. `case.plant_records(spec, engine, condition)`
builds a condition's plant records from what `engine` has rendered, so the intervals do not depend
on how the rendering was chunked (tested with random partitions), and `render_layers` reproduces
them. The intervals are the gate's; the plant's `state_gain` still scales the rhythm inside them.
Each condition draws its own bursts.

The feasibility test showed one plant can register as several findings in a consumer's catalogue,
sometimes at a neighbouring site — that mapping (plant → set of findings) lives in the consumer.

### 4.5 State timeline

```python
@dataclass(frozen=True)
class StateSegment: t0_s: float; t1_s: float; state: str      # "eyes_closed" | "eyes_open" | "drowsy" | any string

@dataclass(frozen=True)                                        # compares by value
class StateTimeline:
    segments: tuple[StateSegment, ...]                         # contiguous from 0, each t1_s > t0_s; the last may be inf
    ramp_s: float = 2.0
    def state_at(self, t_s: float) -> str
    def weights(self, t0: int, n: int, fs: float) -> dict[str, np.ndarray]     # per-state weights (n,), linear ramps across boundaries
    def gain(self, per_state: dict[str, float], t0: int, n: int, fs: float, default: float = 1.0) -> np.ndarray
    def rate(self, per_state: dict[str, float], t_s: float) -> float              # for artifact scheduling, no ramp; a missing state is 0
    @classmethod
    def constant(cls, state: str, duration_s: float = math.inf) -> "StateTimeline"
    def to_dict(self) -> dict                                  # {"ramp_s": …, "segments": [...]}; an infinite t1_s is null
    @classmethod
    def from_dict(cls, d: dict) -> "StateTimeline"
```

A state change ramps linearly over `ramp_s`, centred on the boundary, and the per-state weights sum to
one at every sample. The last segment's state continues past its `t1_s`. Rhythm amplitude is
multiplied by `gain(state_gain, …)` per block and centre frequency shifted by `state_f0_shift_hz`;
artifact rates read `rate(rate_by_state, t)`. A drowsiness onset is therefore one segment:
`[StateSegment(0, 120, "eyes_closed"), StateSegment(120, 240, "drowsy")]` → alpha fades and slows,
theta rises, blinks stop, slow roving eye movements start. Measured on the default recipe (seed 31,
120 s eyes closed, drowsy from 60 s, brain layer, the first 55 s against the last 55 s):
O1 alpha power falls to 0.14 of its alert value, Fz theta power rises 3.0-fold, the O1 alpha peak
moves down 1 Hz, and 7 slow roving eye movements follow the onset. The timeline is written to the
truth file so a consumer can grade a "drowsy stretch" decision.

---

## 5. Artifact plug-in framework

### 5.1 Contract

```python
class Remedy(str, Enum):
    REMOVE_COMPONENT = "remove-component"     # ICA-style component rejection
    MASK_SEGMENT = "mask-segment"             # exclude a time span
    MARK_BAD_CHANNEL = "mark-bad-channel"
    INTERPOLATE = "interpolate"
    NOTCH = "notch"
    RAISE_HIGH_PASS = "raise-high-pass"
    LOWER_LOW_PASS = "lower-low-pass"
    RE_REFERENCE = "re-reference"
    LEAVE = "leave"                           # do nothing; cleaning it would cost signal

@dataclass(frozen=True)
class TruthRecord:
    kind: str                     # "blink", "eye_movement", "emg", "dead_channel", …
    subtype: str | None           # "single"/"double", "saccade"/"slow_roving", "jaw", …
    side: str | None              # "left" | "right" | "both" | None
    channels: tuple[str, ...]     # where |pattern| ≥ 0.3 of the full-head maximum (§5.5)
    onset_s: float
    offset_s: float | None        # None: to the end of the recording (an open-ended dead channel)
    peak_uv: float                # on the full head, whichever channels are recorded
    remedies: tuple[Remedy, ...]  # preferred first
    layer: str                    # the layer name that holds this signal
    params: dict                  # everything drawn for this event, JSON-safe
    def to_dict(self) -> dict     # JSON-safe and unrounded, so from_dict(to_dict(r)) == r
    @classmethod
    def from_dict(cls, d: dict) -> "TruthRecord"

@dataclass
class RenderContext:
    channels: tuple[str, ...]
    fs: float
    electrode_pos: np.ndarray | None      # (n_ch, 3) for the EEG channels
    head: HeadModel | None
    timeline: StateTimeline
    rng: np.random.Generator              # this plug-in instance's time stream
    subject_rng: np.random.Generator      # this plug-in kind's subject stream (pattern jitter)
    occupancy: Occupancy                  # shared scheduling for exclusive artifacts (§5.2)
    layer_name: str                       # "artifact:<kind>[#i]" or "transform:<kind>[#i]" (§7.1)

class Artifact(Protocol):
    kind: ClassVar[str]
    mode: ClassVar[str] = "additive"      # or "transform"
    jitter_pattern: ClassVar[str | None] = None   # the empirical map this kind jitters per subject
    name: str                                     # property: the layer name given at bind()
    def bind(self, ctx: RenderContext) -> None
    def render(self, t0: int, n: int) -> np.ndarray               # additive: (n_ch, n) µV
    def render_transform(self, t0: int, n: int, mix: np.ndarray) -> np.ndarray   # transform: returns the delta
    def truth(self) -> list[TruthRecord]
    def params(self) -> dict                                      # constructor params, JSON-safe
```

`jitter_pattern` names the empirical map (§5.5) that a plug-in jitters per subject, or is None when
its map is fixed (jaw EMG) or it has none (dead channel). It decides whether a subject jitter for that
kind is drawn and recorded (§7.2), so a truth file never seals a jitter that nothing rendered. Blink
declares `"blink"` and eye movement `"heog"`; no other built-in kind declares one. `params()` reads
back the constructor arguments from the attributes of the same names (a parameter never stored falls
back to its declared default), so a spec rebuilt from it makes the same plug-in. An artifact has no
name until it is bound, so artifacts are bound before the engine is built.

The remedy vocabulary is deliberately generic. The mapping to any application's actual tools (its
filters, reference choices, bad-channel marking, component rejection, segment masks) lives in that
application, not here.

### 5.2 Event artifacts

Most artifacts are events: a finite waveform placed at an onset. `EventArtifact` owns scheduling and
buffering; a plug-in implements only `make_event`.

```python
@dataclass
class Event:
    onset: int                    # sample
    block: np.ndarray             # (n_ch, L) µV; must be 2-D
    truth: TruthRecord
    end: int                      # property: onset + L
    @classmethod
    def from_pattern(cls, onset, pattern: np.ndarray, waveform: np.ndarray, truth) -> "Event"  # outer product

class EventArtifact(ABC):
    def __init__(self, *, rate_by_state: dict[str, float], min_gap_s: float = 0.0,
                 exclusive: bool = False)
    rate_by_state: dict[str, float]       # events per second per state; missing state → 0
    min_gap_s: float                      # refractory gap after an event ends
    exclusive: bool                       # if True, never overlaps another exclusive artifact
    @abstractmethod
    def make_event(self, onset: int) -> Event
```

**The onset rule.** `make_event(onset)` must return an `Event` whose `onset` equals the onset it was
given: nothing may start before it, and a delayed start is made with leading zeros in the block, not
by moving the event. It draws everything from `self.ctx.rng` in a fixed order, so a rebuild is
reproducible. The scheduler checks the rule on every build, including the rebuild after an exclusive
push, and raises `ValueError` naming the plug-in's kind. Without the check, an event that did not
start where the scheduler believed could keep colliding after every push and rebuild forever.

**Scheduling** is a thinned Poisson process: candidate onsets are drawn at the maximum rate over
states and accepted with probability `rate(state at t) / rate_max`. An accepted onset that falls
inside the refractory gap of the previous event is pushed to the end of the gap, never dropped, so
the nominal rate holds as long as `rate × (event length + gap)` stays well below one (§5.6 gives a
busy case). Candidates are drawn in time order and each event's waveform is drawn at the moment it
is scheduled, so the sequence of random draws does not depend on how the timeline is chunked (§8).
`truth()` lists the events whose onset lies inside the span rendered so far (a pushed event may
start later than its candidate). A record keeps the event's whole span: an event that starts near
the end of a recording is cut there in the signal but not in its record, so its `offset_s` may lie
past the recording's end, and a consumer clips it. Overlap of non-exclusive artifacts (a blink
during a jaw clench) is allowed — that is realistic.

**Exclusive artifacts** share one scheduler, the case's `Occupancy`. Each exclusive plug-in scheduling
only up to its own block's end was not chunk-invariant: which of two plug-ins claimed a span first
depended on how the render calls were chunked and interleaved. Instead, whenever any exclusive
plug-in needs events up to sample `T`, every exclusive plug-in registered on the `Occupancy` advances
together to `T`, committing candidates in global onset order (ties go to the plug-in registered
first); each keeps its own random stream. An accepted onset that falls inside another exclusive
event's span is pushed to the end of that span and the event is rebuilt there, so `make_event` always
sees the onset the event really has. Registration happens in `bind`, and only before the first
render: a plug-in cannot join an `Occupancy` that has already advanced. Rebinding to the same
`Occupancy` (re-using an instance) drops that plug-in's own spans and replays it from sample 0 against
its peers' spans.

**Span pruning.** On a long stream the list of committed spans would grow without bound, so after
each advance the `Occupancy` drops every span that ends at or before a watermark: the smallest, over
the exclusive plug-ins that can still schedule (`rate_max > 0`), of the later of that plug-in's
earliest possible next onset and its already-drawn next candidate. The candidate term matters: a
plug-in that fires rarely, or only in a state the timeline never visits, would otherwise hold the
watermark at its long-past last event and stall pruning for everyone. Pruning never changes a
scheduling decision (tested against the same scenarios with pruning switched off), but it removes
the history that a same-occupancy rebind relies on, so a rebind is refused with `ValueError` once
any *other* participant's span has been pruned (a plug-in that only ever pruned its own spans still
rebinds). The `Occupancy` tells participants apart by identity, never by equality, both in its list
of registered participants and in its record of pruned ones, so two value-equal dataclass plug-ins
are two participants that both schedule and never overlap.

None of the built-in plug-ins is exclusive, so a default case or stream never commits a span (a
30-minute default stream ends with none). Two limits remain, neither reachable with the built-in
plug-ins: when `rate × (event length + gap)` approaches one the queue of pushed events grows without
bound, and an exclusive plug-in that is bound but never rendered still takes part in its `Occupancy`'s
schedule.

### 5.3 Transform artifacts

`TransformArtifact.render_transform(t0, n, mix)` receives a read-only view of the running sum of every
layer rendered so far and returns the delta to add. The engine stores that delta as the artifact's
layer. Transform artifacts run last (after sensor noise). A dead channel returns `−mix[ch] + tiny
noise`; a bridged pair would return the delta that moves both channels to their mean plus shared
noise.

### 5.4 Registry and discovery

```python
ARTIFACTS: dict[str, type[EventArtifact] | type[TransformArtifact]]
def register(cls: type[A]) -> type[A]       # decorator; key = cls.kind; duplicate kind raises
def make_artifact(kind: str, **params) -> EventArtifact | TransformArtifact   # unknown kind: ValueError
def discover() -> None                      # loads entry points in group "open_eeg_synth.artifacts"
```

Built-in plug-ins register on import of `open_eeg_synth.artifacts`. Third-party packages add
`[project.entry-points."open_eeg_synth.artifacts"] mykind = "mypkg.module:MyArtifact"`. A `CaseSpec`
refers to artifacts by kind and parameter dict (`ArtifactSpec(kind, params)`), which is what the
truth file stores. `register` refuses a class that does not define its own `kind` (inheriting one
would silently replace the parent kind). `discover` tries each entry point once; a broken one is
skipped without blocking the rest, warned about on every call, and named in the `ValueError` that
`make_artifact` raises for an unknown kind (the error lists the known kinds). `make_subject`
forwards each such warning once per process rather than once per case.

### 5.5 Where scalp patterns come from

`artifacts/patterns.py` offers two sources, and each plug-in states which it uses:

```python
def empirical(name: str, channels: Sequence[str], *, rng: np.random.Generator | None = None,
              jitter_sd: float = 0.6, electrode_pos: np.ndarray | None = None,
              z: float | None = None) -> np.ndarray                  # name: "blink" | "heog"
def analytic_focal(centre_m, electrode_pos_m, sigma_mm: float) -> np.ndarray      # exp(−d²/2σ²); 1 only at the centre
def analytic_dipole(pos_m, moment, electrode_pos_m) -> np.ndarray                  # (r·m)/|r|³ ÷ its max |value| on the template head
```

`empirical` reads `data/eog_patterns.npz` (§6) and selects the requested channels by canonical name.

**Reference.** The file stores average-referenced ICA topographies (§6.3). `empirical` re-references
them at load to the mean of the file's T9 and T10 (ear-adjacent 10-10 sites). The synthetic recording
is referential (§2.3), so its eye maps should be referential too: an average-reference view of the
result is unchanged, and a linked-ears view no longer shows the false posterior blink (about −0.25 of
the peak at O1/O2) that the average-referenced map carries. Measured at z = 0 on the 10-20 set, as a
fraction of the full 64-channel map's peak:

| map | Fp1 | Fp2 | F7 | F8 | F3 | Fz | F4 | C3 | T3 | T4 | O1 | O2 | Pz |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blink, as stored | 1.00 | 0.99 | 0.30 | 0.24 | 0.19 | 0.13 | 0.16 | −0.12 | −0.16 | −0.17 | −0.25 | −0.25 | −0.21 |
| blink, referenced | 1.00 | 0.99 | 0.43 | 0.37 | 0.33 | 0.29 | 0.31 | 0.08 | 0.05 | 0.04 | −0.03 | −0.03 | 0.01 |
| heog, referenced | −0.14 | 0.27 | −0.91 | 1.00 | −0.04 | 0.07 | 0.23 | −0.10 | −0.29 | 0.34 | 0.01 | 0.09 | 0.08 |

The reference shifts every heog value by +0.03 of the file's scale, which makes small heog values less
symmetric than in the stored map (O2 +0.09 against O1 +0.01, for example).

**Subject jitter.** One scalar per subject, `z ~ N(0, jitter_sd)` with `jitter_sd` = 0.6, clipped to
[−1, 1] (9 % of subject draws reach the clip), drawn from `subject_rng` — the kind's subject stream, so
one subject's blinks share one map across conditions — unless `z` is given. Each channel then moves
away from zero as z grows:

```
jittered[c] = mean[c] + z · sign(mean[c]) · min(sd[c], 0.9 · |mean[c]|)
```

**Decision: jitter that keeps signs.** The first formula, `mean + z · sd`, distorted the maps: on the
shipped file (after the reference) the between-subject sd exceeds |mean| on 30 of the 64 blink
channels and 54 of the 64 heog channels, so it flipped signs, grew the posterior blink, and at |z| ≈ 1 made one side of the heog
map vanish. The sign factor makes mirror channels of opposite sign grow and shrink together. The cap
at 0.9 of each channel's own |mean| keeps every channel's sign for any z in [−1, 1] (without it, heog
T3 flips at z = −1 while T4 does not); below the cap the formula is exact. Over z in [−1, 1], heog
|F7|/|F8| stays between 0.89 and 0.99 and blink |O1|, |O2| stay below 0.05. Only plug-ins that declare
`jitter_pattern` draw this jitter — blink (`"blink"`) and eye movement (`"heog"`) — and the subject's
`pattern_jitter` records exactly the clipped z each one used (§7.2).

**Amplitudes refer to the full head.** Every pattern is scaled against the full head, never against
the requested channels, so a four-channel case records the same numbers those four channels have in a
nineteen-channel case, and a drawn artifact amplitude always means the same thing:

- `empirical` divides the referenced, jittered map by its largest |value| over all 64 file channels;
  a requested file channel gets exactly its value there.
- `analytic_focal` returns the raw Gaussian, which is 1 only at the centre point; a plug-in divides it
  by its value at the full template head's loudest channel.
- `analytic_dipole` divides by its largest |value| over the 19-channel template head
  (`load_head_model()`).

Scaling to the requested channels made the loudest *recorded* channel 1.0 whatever it was: for one
seed, a blink on an O1/O2/T3/T4 case peaked at 187 µV at O1 instead of about 1 µV, and jaw EMG reached
its full RMS at C3.

**Channels absent from the file.** The file has 64 channels of the 10-10 system, including every 10-20
label, so the 19-channel head model never needs this path. For any other label, `empirical` requires
`electrode_pos` and uses an analytic stand-in: an equivalent dipole at the eyes (centre (0, 85, −20) mm
in the head frame; moment (0, 0.3, 1) for blink, (1, 0, 0) for heog), scaled to the file from the
requested channels that are in the file (the anchors):

1. An anchor is usable only if the dipole's sign there agrees with the map's sign. A disagreeing
   anchor is not weak evidence but wrong evidence (the model has the direction of the response wrong
   there, typically near its null), and fitting through it dragged the scale to a wrong compromise.
2. Each usable anchor gets a reliability `r` from the dipole's direction cosine there: 0 at
   |cos| ≤ 0.05, 1 from |cos| ≥ 0.15, and a smoothstep between. An anchor near the model's null says
   almost nothing about the scale; flooring its cosine instead (the earlier rule) turned it into a
   scale about 20 times too large on small montages.
3. The five nearest anchors with `r > 0` (exact distance ties broken by label, so request order never
   matters) give the weighted least-squares scale `s = argmin Σ w (value/envelope − s · cos)²` with
   `w = r / d²`, where `envelope = |m| / |r_e|²` is the dipole's distance falloff at the anchor.
   Dividing out the falloff stops the anchor nearest the eyes (Fp1 is about 60 times louder than O1 in
   the model) from outweighing every nearer anchor; five is the smallest count that kept mirror
   channels (A1/A2, TP9/TP10) within a factor of two of each other in at least 90 % of random ±5 mm
   electrode moves.
4. The fallback is `trust · s · envelope · cos + (1 − trust) · dipole`, with `trust` the largest `r`
   among the anchors used: exactly the fit whenever one anchor is fully reliable, sliding to the pure
   dipole as the last anchors approach the null. With a single anchor the weights cannot fade it
   (`s = value / cos` whatever its weight), so the blend is what keeps a small electrode move from
   switching the result.
5. With no usable anchor (none requested, none agreeing in sign, or all at the null) the fallback is
   the pure dipole on its full-head unit.
6. Every fallback is clipped to [−1, 1], the file map's peak. Next to the eyes a fit can exceed the
   peak: on standard_1005 positions the clip holds blink at Fp1h/Fp2h and along the AFp row, and heog
   at AF9/AF10, along the AFF row and, on small montages, at F9/F10, where heog would otherwise read
   about 1.1 of the peak.

**Decision: trust the file where it is informative and the geometry where it is not.** Measured on
3,000 random sparse requests per map (2–6 channels of the 10-20 set plus 1–2 of A1, A2, TP9, TP10, P9,
P10, F9, F10, on template-fitted positions; about 4,500 fallback values each): no fallback has the
wrong sign; no blink fallback is more than 3 times its full-montage value (worst 2.4 times); 5 heog
fallbacks are (worst 3.4 times), all fits whose only usable anchor is O2, which the T9/T10 reference
makes much larger than O1. The fast test repeats the sweep on another seed and allows at most 0 blink
and 6 heog values over 3 times. The reliability ramp ends at 3 times the
floor rather than 2 because real montage positions put anchors inside the ramp: on easycap-M1
positions a 2× ramp left 25 of 4,486 blink fallbacks more than 3 times too large (worst 5.3 times) and
the 3× ramp none (worst 2.5 times). The remaining limits are listed in §11: montages whose geometry
departs further from the template still get some blink fallbacks more than 3 times off, and a request
whose only anchors disagree in sign or sit at the null gets the pure dipole (heog F9 requested with
only O1 and Pz reads 0.30 of its full-montage value).

### 5.6 First plug-ins

`rate_by_state=None` (every plug-in's default) means the rates listed below; `{}` means never.

**Blink** (`kind = "blink"`, pattern `empirical("blink")`, `jitter_pattern = "blink"`)
- `rate_by_state = {"eyes_open": 0.25, "eyes_closed": 0.0, "drowsy": 0.0}`, `min_gap_s = 0.4`.
- Waveform: duration `U(0.2, 0.4)` s; a raised-cosine rise over the first 35 %, then exponential decay
  (τ = 22 % of the duration), with the last 5 % of the duration tapered to zero by a raised cosine;
  normalised to a unit peak on the sampled grid. Peak `lognormal(median 120 µV, σ 0.3)` clipped to
  60–250 µV, at the full head's loudest channel (Fp1 or Fp2). With probability 0.15 a second blink of
  the same duration at 0.8 × the peak follows 150 ms after the first one ends (subtype `"double"`; the
  record's offset covers both).
- Truth: channels with |pattern| ≥ 0.3 of the full-head maximum; `params = {duration_s}`; remedies
  `(REMOVE_COMPONENT, MASK_SEGMENT)`. At z = 0 on the 10-20 set the channels are Fp1, Fp2, F7, F8, F3
  and F4 (Fz reads 0.29). Over 200 simulated subjects Fp1 and Fp2 are always listed and O1/O2 never;
  73 add Fz to those six, 51 have exactly those six, and 15 list only Fp1 and Fp2.

**Horizontal eye movement** (`kind = "eye_movement"`, pattern `empirical("heog")`,
`jitter_pattern = "heog"`)
- `rate_by_state = {"eyes_open": 0.10, "eyes_closed": 0.02, "drowsy": 0.08}`, `min_gap_s = 0.5`.
- The subtype follows the state at the event's onset. Eyes open → `"saccade"`: a step with a 30 ms
  raised-cosine rise, hold `U(0.3, 2.0)` s, 40 ms return; amplitude `U(30, 100)` µV at the map's peak
  (F8 on the shipped map). Otherwise → `"slow_roving"`: one to three cycles of a sinusoid at
  `U(0.2, 0.5)` Hz under 0.2 s raised-cosine edges, amplitude `U(20, 60)` µV.
- The sign is ±1 with equal probability. The map is positive at F8 and negative at F7, so a positive
  sign is gaze to the right.
- Truth: `params` records `sign` (±1), `direction` (`"right"` when the sign is positive, `"left"`
  otherwise), and `hold_s` for a saccade or `freq_hz` and `cycles` for slow roving; channels with
  |pattern| ≥ 0.3 of the full-head maximum; remedies `(REMOVE_COMPONENT, MASK_SEGMENT)`. At z = 0 the
  channels are F7, F8 and T4 (T3 reads 0.29): a threshold on a slightly asymmetric map lists one side's
  channel without its mirror, which is accepted as what a 0.3 threshold does. Over 200 subjects:
  F7/F8/T4 91, F7/F8/T3/T4 59, F7/F8 25, Fp2/F7/F8/T3/T4 25.
- **Load in drowsy stretches.** A slow roving movement lasts about 6 s on average, so at the drowsy
  rate the movements and their gaps fill about half of a drowsy stretch (measured over 20,000 s:
  48 % of the time, 0.081 events per second against the nominal 0.08). Because the scheduler pushes
  rather than drops, the rate holds, but 52 % of the movements start exactly at the end of the
  previous one's gap. A consumer grading "eyes drifting" should expect near-continuous slow roving
  once a subject is drowsy.

**Jaw-tension EMG** (`kind = "emg"`, subtype `"jaw"`, pattern analytic)
- `side` = `"left" | "right" | "both" | "random"` (random draws left/right/both with probabilities
  0.4/0.4/0.2 per event); `"both"` uses two independent time courses, one per side.
- `rate_by_state = {"eyes_open": 1/60, "eyes_closed": 1/60, "drowsy": 1/120}`, `min_gap_s = 5`.
- Pattern: `analytic_focal` with σ = 50 mm, centred 5 mm lateral, 5 mm anterior and 7.5 mm inferior of
  the template's T3 (left) or T4 (right), divided by its value at the full template head's loudest
  channel (T3 or T4). **Decision: an anatomical centre with a broad falloff.** The design's first
  placement (5 mm lateral, 10 mm anterior, 10 mm inferior, σ 35 mm) gave F7 0.62, T5 0.12 and C3 below
  0.1 instead of the intended ≈0.6/0.5/0.3, much steeper than a temporalis burst looks. The shipped placement gives (fractions of the loudest
  channel):

  | side | T3/T4 | F7/F8 | T5/T6 | F3/F4 | C3/C4 | opposite temporal |
  |---|---|---|---|---|---|---|
  | left | 1.00 | 0.72 | 0.40 | 0.23 | 0.21 | 0.003 |
  | right | 1.00 | 0.59 | 0.38 | 0.22 | 0.29 | 0.003 |

  The two sides differ because the template's T3 and T4 are not mirror images of each other.
- Waveform, generated at `oversample = 8` × fs like a real signal hitting an amplifier: white noise
  shaped in the frequency domain by `H(f) = (f/f_p) / (1 + (f/f_p)²)` with `f_p` = 80 Hz (broadband,
  rising above ~20 Hz, broad peak near 80 Hz, slow roll-off), multiplied by a raised-cosine envelope
  (100 ms rise and fall) over a duration `lognormal(median 1.2 s, σ 0.5)` clipped to 0.5–3 s, then
  low-passed and decimated by 8 (`dsp.lowpass_decimate`: a Kaiser-windowed FIR with 401 taps, β 8,
  passband to 0.9 of the new Nyquist, applied through `scipy.signal.resample_poly`), then scaled so the
  RMS over the middle 70 % of the burst is `lognormal(median 50 µV, σ 0.6)` clipped to 20–200 µV at
  the loudest channel. At a 256 Hz device the passband ends at 115 Hz and aliased images are at least
  60 dB down, so nothing folds into the beta band that was not there physically: the burst's
  13–30 Hz power relative to its 30–100 Hz power lands at 0.73–1.19 times the |H|² prediction over 40
  seeds, where taking every 8th sample gives 1.59–2.81 times.
- Truth: the resolved `side`; channels where the pattern (the larger of the two sides for `"both"`) is
  at least 0.3 — T3, F7, T5 on the left, T4, F8, T6 on the right; `peak_uv` is the event's peak on the
  full template head; `params = {duration_s, rms_uv}`; remedies `(MASK_SEGMENT, REMOVE_COMPONENT)`.

**Dead channel** (`kind = "dead_channel"`, mode transform) — ships in v0.2.0 only to prove the transform
hook. `DeadChannel(channel, onset_s=0.0, offset_s=None, noise_uv=0.3)` replaces one channel with
white noise for the samples `[round(onset_s · fs), round(offset_s · fs))`, or to the end when
`offset_s` is None. Both times must be finite, `onset_s ≥ 0` and `offset_s > onset_s`; a span that
rounds to no whole sample raises at bind. Truth: the channel; onset and offset are the rendered sample
boundaries divided by fs (offset None when open-ended), not the requested seconds; `peak_uv` 0;
`params = {noise_uv}`; remedies `(MARK_BAD_CHANNEL, INTERPOLATE)`. The record is listed once rendering
has reached the first dead sample.

### 5.7 Future plug-ins and how they fit

| plug-in | base | pattern source | signal | truth | remedies |
|---|---|---|---|---|---|
| electrode pop | `EventArtifact` | channel-local | step of 200–2000 µV with exponential recovery (τ 0.2–2 s), high-pass-filter-shaped tail | channel, onset, peak | `MASK_SEGMENT`, `MARK_BAD_CHANNEL` |
| loose lead | `ContinuousArtifact` (stateful `render`) | channel-local | low-frequency drift + pop bursts whose rate scales with `ContactTimeline.impedance(ch, t)`; the same timeline feeds a recording app's impedance display | channel, spans where impedance > threshold | `MARK_BAD_CHANNEL`, `INTERPOLATE`, `RAISE_HIGH_PASS` |
| bridged pair | `TransformArtifact` | two channels | both channels → their mean + shared 1 µV noise | pair | `MARK_BAD_CHANNEL` |
| dead channel (ships) | `TransformArtifact` | one channel | see §5.6 | channel | `MARK_BAD_CHANNEL`, `INTERPOLATE` |
| mains hum | `ContinuousArtifact` | per-channel gains (loose channels louder) | `freq_hz` **required** (50 or 60; no default), harmonics 2–3 at −20 dB, slow AM | channels, freq | `NOTCH` |
| sweat drift | `ContinuousArtifact` | frontal/temporal focal | 0.05–0.3 Hz OU wander, 50–500 µV | channels, spans | `RAISE_HIGH_PASS`, `MASK_SEGMENT` |
| movement | `EventArtifact` | analytic focal, random centre | 0.5–3 s low-frequency swing on many channels + EMG mix | channels, span | `MASK_SEGMENT` |
| heartbeat bleed | `ContinuousArtifact` | empirical or analytic left-lateral | `HeartSource` beat times → QRS-shaped 5–30 µV at each beat | channels | `REMOVE_COMPONENT` |
| pulse wave | `ContinuousArtifact` | channel-local (an electrode over a vessel) | `HeartSource` beats → smooth 20–100 µV wave lagging the beat ~250 ms | channel | `MARK_BAD_CHANNEL`, `REMOVE_COMPONENT` |
| forehead / neck muscle | `EventArtifact` | analytic focal at Fp1/Fp2 (frontalis) or O1/O2/T5/T6 (neck) | the jaw EMG generator with a different centre and spectrum peak (frontalis ~40 Hz) | side, channels | `MASK_SEGMENT`, `REMOVE_COMPONENT` |

`EventArtifact`, `TransformArtifact`, the pattern sources and `HeartSource` exist in v0.2.0.
`ContinuousArtifact` and `ContactTimeline` do not yet: a continuous plug-in is an additive layer with a
stateful `render`, and `make_engine` already accepts any registered class with `mode = "additive"`,
`bind`, `render`, `truth`, `params` and a `name`, so adding that base class needs no change to the
engine or the truth schema.

---

## 6. Empirical eye and muscle scalp patterns

### 6.1 The choice

The head model has no eyes and no scalp muscles, so eye and muscle artifacts cannot be projected
through it. Two ways to get their scalp maps:

| | empirical (ICA of public recordings) | analytic (dipole / Gaussian on electrode positions) |
|---|---|---|
| blink topography | the real thing: Fp maximum, a broad frontal spread (0.29–0.43 of the peak at F3/Fz/F4/F7/F8 once referenced to T9/T10, §5.5), near zero posteriorly; classified as "eye" by ICA-label tools trained on real data | plausible shape, but a point dipole near the eyes over-weights F7/F8 and misses the skull's smearing; ICA-label tools may not call it "eye" with confidence |
| horizontal eye movement | opposite signs at F7/F8 with the real AF/Fp gradient | a lateral dipole reproduces the sign flip; magnitude gradient approximate |
| muscle | subject-specific and numerous components; a subject-average is a blur | spatially local by nature; Gaussian on scalp distance is close to what a real temporalis burst looks like |
| channel sets | 64 channels in the source data cover every 10-10 label in the 19-, 31- and 37-channel sets | any label with a position |
| cost | a derivation script (dev-only MNE), a 4 kB data file, an attribution notice | none |

**Recommendation: empirical for the two eye patterns, analytic for muscle**, with the analytic dipole
as the fallback for any channel the empirical file lacks. Blink and eye-movement maps are the ones
where a wrong shape would mis-train a reader and mislead component-labelling tools; muscle maps are
local enough that geometry does the job.

The design first expected the blink map at about 40 % of its peak at F3/F4/Fz. The average-referenced
file reads 0.13–0.19 there, because the average reference subtracts the blink's own mean and leaves a
negative posterior field of about −0.25; after the T9/T10 reference of §5.5 the same channels read
0.29–0.33 and the posterior field is gone, so the expectation roughly holds for the map the package
actually applies.

### 6.2 Source data

PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0 (Schalk 2009, doi 10.13026/C28G6P), 109 subjects,
64 electrodes of the 10-10 system (all 10-20 sites present under 10-10 names), 160 Hz; runs 1 and 2
are one-minute eyes-open and eyes-closed baselines. Licence: Open Data Commons Attribution License
v1.0 (https://physionet.org/content/eegmmidb/view-license/1.0.0/). Downloaded on demand with
`mne.datasets.eegbci.load_data`.

### 6.3 Derivation script: `scripts/derive_artifact_patterns.py`

For subjects 1–30 (`--subjects`), runs 1 and 2: read and concatenate the two runs,
`eegbci.standardize`, rename `T7/T8/P7/P8 → T3/T4/T5/T6`, `standard_1020` montage, 1–45 Hz
band-pass, average reference, fit `ICA(n_components=30, method="fastica", random_state=0,
max_iter="auto")`. For each map, the five components whose topographies correlate best (in absolute
value) with the map's prior are tried in that order:

- **blink**: prior `(Fp1, Fp2, AF3, AF4, AF7, AF8 = 1; F3, Fz, F4 = 0.4; rest 0)`. The first of the
  five whose time course has excess kurtosis > 5 (blinks are sparse) is taken, signed by its
  correlation with the prior (so it is positive at the front).
- **heog**: prior `(F7, AF7 = −1; F8, AF8 = +1; Fp1 = −0.5; Fp2 = +0.5)`. The first of the five with
  less than 40 % of its power above 5 Hz is taken (after the 1 Hz high-pass a 20 % bound keeps only 4
  of the first 20 subjects; 40 % keeps 25 of 30), provided its |correlation| with the prior is at
  least 0.25; it is sign-aligned so `F8 − F7 > 0`. **Decision: a minimum match of 0.25.** A weak match
  is not evidence of an eye-movement map, and 0.25 is the smallest value (tried from 0.20 to 0.50 in
  steps of 0.05) at which every kept component's correlation sign agrees with `sign(F8 − F7)`; it
  drops subject 21 (|r| = 0.24). A bound of 0.50 would keep only 12 subjects.
- A subject with no qualifying component for a map is skipped for that map and logged. If a subject
  cannot be downloaded the script stops at the subjects already on disk. Subject 4's ICA does not
  converge at any iteration count tried (up to 50,000), but its picks match the priors (heog
  r = −0.49) and it is kept.

Normalise each map to max |1|, average across subjects, keep the per-channel standard deviation. The
shipped file holds blink maps from all 30 subjects and heog maps from 24 (5 subjects have no slow
candidate, 1 has too weak a match).

Outputs: `src/open_eeg_synth/artifacts/data/eog_patterns.npz` with `channel_names (64,)`, `blink_mean`,
`blink_sd`, `heog_mean`, `heog_sd` (all `(64,)` float32), `n_subjects` (the smaller of the two map
counts, 24), `subjects_used` (every subject that contributed a map, 1–30) and `attribution`;
`scripts/output/eog_patterns_selection.json` listing the criteria and, per subject, the chosen
component and its scores or the reason it was skipped (kept in the repo for reproducibility). The
stored maps are average-referenced; the T9/T10 reference is applied at load (§5.5).

Fast tests on the shipped file: Fp1 and Fp2 are the two largest entries of `blink_mean`; F3, Fz, F4, F7
and F8 are positive; |O1|, |O2| < 0.25; `heog_mean` has `sign(F7) = −sign(F8)` with F8 positive, and its
two largest magnitudes are among F7, F8, AF7 and AF8; `blink_sd` is below 0.35 everywhere and `heog_sd`
below 0.45 (measured 0.22 over 30 subjects and 0.42 over 24; the heog map varies more between people
than the blink map does); `n_subjects` is at least 15; the attribution names the dataset's DOI.

### 6.4 Attribution

`artifacts/data/NOTICE-eegmmidb.txt`, included in the wheel:

> The scalp patterns in eog_patterns.npz are statistical summaries (subject-averaged, unit-normalised
> independent-component topographies) derived from the EEG Motor Movement/Imagery Dataset v1.0.0,
> made available on PhysioNet under the Open Data Commons Attribution License v1.0
> (https://physionet.org/content/eegmmidb/view-license/1.0.0/). No recordings are redistributed.
> The same notice covers the percentile summaries in tests/realism/reference/, which are statistical
> summaries of the same recordings and likewise contain no recordings.
>
> Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet.
> https://doi.org/10.13026/C28G6P
> Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. BCI2000: A General-Purpose
> Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034-1043,
> 2004.
> Pollard, T., et al. PhysioNet as a global platform for biomedical research. Nature Health
> 1(8):792-795, 2026. https://doi.org/10.1038/s44360-026-00096-z

The realism reference (`tests/realism/reference/*.json`, §9.1) embeds this text verbatim. The pattern
file's `attribution` array embeds the notice as it stood when the maps were derived, followed by a
line naming the script, the subjects, the MNE version and the date; that copy predates the sentence
about the realism reference, which does not concern the patterns, so the file was not re-derived for
it.

---

## 7. Output modes and case files

### 7.1 Engine

```python
class Layer(Protocol):
    name: str
    def render(self, t0: int, n: int) -> np.ndarray      # (n_ch, n) float32 µV; t0 must continue the previous call
    def truth(self) -> list[TruthRecord]

class Transform(Protocol):
    name: str
    def render_transform(self, t0: int, n: int, mix: np.ndarray) -> np.ndarray   # the delta
    def truth(self) -> list[TruthRecord]

@dataclass
class Frame:
    t0: int; n: int
    layers: dict[str, np.ndarray]                        # name → (n_ch, n)
    @property
    def mixed(self) -> np.ndarray                        # sum of layers, float32

class Engine:
    def __init__(self, channels: Sequence[str], fs: float, layers: Sequence[Layer], transforms: Sequence[Transform] = ())
    def render(self, t0: int, n: int) -> Frame           # contiguous calls only; raises otherwise
    def render_all(self, n_samples: int, block: int = 4096) -> Recording
    def truth(self) -> list[TruthRecord]
    @property
    def position(self) -> int                            # next sample to render

@dataclass
class Recording:
    fs: float
    channels: tuple[str, ...]
    layers: dict[str, np.ndarray]                        # each (n_ch, n_samples) float32 µV
    truth: list[TruthRecord] = []
    plants: list[PlantRecord] = []
    timeline: StateTimeline | None = None
    @property
    def mixed(self) -> np.ndarray
    @property
    def n_samples(self) -> int
    @property
    def duration_s(self) -> float
```

Layer names: `"brain"`, `"artifact:<kind>"` (`"artifact:<kind>#<i>"` when a kind appears more than
once), `"sensor"`, `"transform:<kind>"` (`"transform:<kind>#<i>"` likewise; the engine keys every
artifact layer, additive or transform, by the instance's own layer name). The engine requires at least
one layer and unique layer names, checks every block's shape, copies every block it stores (a layer
may reuse its own buffer) and passes transforms a read-only view of the running mix. The identity
`mixed == Σ layers` is a tested invariant.

### 7.2 Case specification and subject

A case is one synthetic subject recorded under one or more conditions. Everything that belongs to the
subject (perturbed head model, patch placements, centre-frequency and per-patch draws, network wiring,
pattern jitters) is drawn once and shared; everything that belongs to the passage of time (noise,
envelopes, artifact events) is drawn per condition.

```python
@dataclass(frozen=True)
class SensorSpec: white_uv: float = 1.5

@dataclass(frozen=True)
class ArtifactSpec: kind: str; params: dict = field(default_factory=dict)   # params kept as JSON reads them back
    def identity(self) -> dict           # non-bool numbers as floats; equality, hash, digest

@dataclass(frozen=True)
class ConditionSpec:
    name: str                          # "eyes_closed", "eyes_open", … (also the truth/file key)
    duration_s: float                  # whole seconds; inf only for a stream (serialised as null)
    timeline: StateTimeline

@dataclass(frozen=True)
class CaseSpec:
    seed: int
    fs: float = 256.0
    channels: tuple[str, ...] = CHANNELS_19
    head_model: str = "colin27_19ch"
    perturb_head: bool = True
    brain: BrainSpec = field(default_factory=recipes.resting_brain)
    plants: tuple[Plant, ...] = ()
    artifacts: tuple[ArtifactSpec, ...] = field(default_factory=recipes.ordinary_artifacts)
    sensor: SensorSpec = SensorSpec()                               # 1.5 µV
    conditions: tuple[ConditionSpec, ...] = (eyes closed 240 s, eyes open 240 s)   # constant timelines
    label: str = "synthetic"
    def to_dict(self) -> dict            # JSON-safe, round-trips through from_dict
    @classmethod
    def from_dict(cls, d) -> "CaseSpec"
    def digest(self) -> str              # 8-byte blake2b of the canonical JSON: 16 hex chars

@dataclass
class Subject:
    head: HeadModel                                  # perturbed (or nominal), full head
    mixing: np.ndarray                               # background smoothing matrix (§4.1)
    placements: dict[str, list[int]]                 # rhythm name → patch centres (source indices)
    f0_hz: dict[str, float]                          # rhythm name → this subject's centre frequency
    network: NetworkWiring                           # §4.2
    pattern_jitter: dict[str, float]                 # artifact kind → clipped z; kinds with a jitter_pattern only
    rows: list[int]                                  # head rows the case records, in CaseSpec.channels order
    patch_f0_offsets_hz: dict[str, list[float]]      # rhythm name → per-patch centre-frequency offsets (§4.3)
    patch_lags_ms: dict[str, list[float]]            # rhythm name → per-patch driver lags (§4.3)
    def to_dict(self) -> dict                        # the truth file's "subject" (§7.4)

def compiled_rhythms(spec: CaseSpec) -> tuple[RhythmSpec, ...]                     # §4.4
def make_subject(spec: CaseSpec) -> Subject
def make_engine(spec: CaseSpec, subject: Subject, condition: ConditionSpec) -> Engine
def make_recording(spec: CaseSpec, subject: Subject, condition: ConditionSpec) -> Recording
def plant_records(spec: CaseSpec, engine: Engine, condition: ConditionSpec) -> list[PlantRecord]
def make_case(spec: CaseSpec) -> Case
def case_id_for(spec: CaseSpec) -> str

@dataclass
class Case:
    spec: CaseSpec
    case_id: str                         # case_id_for(spec)
    subject: Subject
    recordings: dict[str, Recording]     # by condition name
```

**Spec identity.** A spec's identity is its canonical JSON (sorted keys, compact separators, no
NaN), so specs that compare equal must serialise identically. On construction every spec dataclass
normalises its scalar fields to plain Python types (`_canon.canonical_fields`): an int given for a
float field becomes a float (`duration_s=8` and `duration_s=8.0` give one digest), numpy scalars are
accepted, and wrong types raise `TypeError` — a string for a number (`fs="256"`), anything but a
real `bool` for a bool field (`perturb_head="false"` is truthy), a bool for an int (`seed=True`).
`CaseSpec` also canonicalises its channels against the selected head model's own channel list
(`head_model_channels`), so aliases and letter case give the same spec, digest and case id (`T7` and
`t3` are both `T3`); it rejects channels that collapse to one name (`("T3", "t7")`) and duplicate
condition names. A bare `CaseSpec(seed=s)` equals `recipes.resting_case(s)`, digest included. An
`ArtifactSpec`'s `params` are the plug-in's constructor arguments and are kept exactly as JSON reads
them back, so a plug-in receives what was given (the built-ins coerce their numbers themselves). The
spec's identity reads them through `ArtifactSpec.identity()`, where every number that is not a bool
is a float: equality, hash and `CaseSpec.digest` all use that form, so `{"rms_median_uv": 50}` and
`{"rms_median_uv": 50.0}` (or a numpy scalar) are one spec with one case id, while `{"flag": True}`
and `{"flag": 1}` stay two specs with two ids (a bool stays a bool). Specs whose artifact numbers
were already floats keep their digests. `case_id_for(spec)` is `"synth-"` followed by the 8 hex
characters of a 4-byte blake2b digest of `"<seed>:<spec digest>"`.

**Subject.** `make_subject` works on the full head model and draws, each from its own named stream
(§8.1): the head perturbation (when `perturb_head`); for each compiled rhythm, its placement, its
centre-frequency jitter and its per-patch offsets and lags; the network wiring; and, for each artifact
kind whose plug-in declares a `jitter_pattern` (§5.1), one `z ~ N(0, 0.6)` clipped to [−1, 1], exactly
as §5.5 applies it. It also computes the background mixing matrix once. Only jittering kinds get a
`pattern_jitter` entry: jaw EMG's map is fixed and a dead channel has none, so recording a jitter for
them described nothing that was rendered, and not drawing from their streams moves no other draw.
Patches are placed under every 10-20 site whether or not the case records it, so a four-channel
stream and a nineteen-channel case of the same seed share one brain.

**One condition.** `make_engine` builds the brain layer on the full head (it returns only the
recorded rows); every artifact from its `ArtifactSpec` through `make_artifact`, bound to the
recorded subset of the head, with its time stream `<condition>:artifact:<kind>:<i>` (`i` counts
instances of that kind only, so adding another kind never moves these draws), its kind's subject
stream and one `Occupancy` shared by the condition; and the sensor layer. Additive artifacts become
layers, transform artifacts transforms. `make_recording` renders `round(duration_s · fs)` samples
and attaches the condition's timeline and its plant records (`plant_records`: the plants its
timeline does not silence, a bursting plant with its burst times, §4.4). A case condition must last
a positive whole number of seconds, because EDF records are whole seconds: anything else, including
a non-finite `duration_s` (a whole recording needs an end, and an unbounded one is a stream, §7.3),
raises `ValueError` naming the condition (`case.check_case_duration`). `make_case` checks every
condition before it draws the subject; `make_recording` checks its own before rendering. `make_case`
calls `make_subject` once and `make_recording` for each condition; `casefile.truth.render_layers`
calls the same `make_recording`, so a re-rendered condition cannot drift from what `make_case`
wrote.

`recipes.resting_case(seed, *, duration_s=240, plants=(), artifacts=None, drowsy_from_s=None,
fs=256.0, channels=CHANNELS_19)` builds the common two-condition spec: eyes closed then eyes open,
`duration_s` each. `artifacts=None` means `recipes.ordinary_artifacts()` (blink, eye movement, and jaw
EMG with `side="random"`), and `drowsy_from_s` turns the rest of the eyes-closed recording drowsy.

### 7.3 Streaming mode

```python
class StreamSource:
    def __init__(self, channel_labels: Sequence[str], srate: float, *, seed: int | None = None,
                 markers: MarkerSchedule | None = None, timeline: StateTimeline | None = None,
                 brain: BrainSpec | None = None, artifacts: Sequence[ArtifactSpec] | None = None,
                 sensor: SensorSpec | None = None, perturb_head: bool = True)
    def next_chunk(self, n_samples: int) -> np.ndarray             # (n_labels, n) float32 µV; any n ≥ 1
    def due_markers(self, n_samples: int) -> list[tuple[float, str]]   # same contract as classic
    @property
    def truth(self) -> list[TruthRecord]                           # events whose onset has been rendered
    @property
    def seed(self) -> int
    @property
    def unmodelled_labels(self) -> tuple[str, ...]                 # labels with sensor noise only
```

The constructor signature mirrors `classic.RealisticEEGSynthesizer(channel_labels, srate, *, seed,
markers)` so a recording application swaps one import. Labels are split into heart labels (`HR`,
`ECG`, `EKG`, in any letter case) and EEG labels. Every heart row carries the same `HeartSource`
ECG, seeded from `stream:heart`: an R-wave of about 200 µV and RR intervals around a mean of 72 bpm
with a 2 bpm spread (the classic constants), plus respiratory sinus arrhythmia of ±40 ms at 5.5
breaths a minute and a 2 % wave at 0.1 Hz. Heart labels may repeat. Every other label is
canonicalised against the head model's channels (`head_model_channels()`, the 19 of the 10-20
system), and two labels that collapse to one channel raise `ValueError`, as they do in a `CaseSpec`.
A label the head model lacks does not raise, as it does not in `classic`: 10-10 sites such as `Fpz`
and `Oz` are not projected in 0.2.0, and neither are ear references such as `A1` or any other name.
Each such label becomes an unmodelled row that carries sensor noise only (white noise at
`sensor.white_uv`, drawn time-major from its own stream `stream:unmodelled`, so the modelled rows
are exactly those of a source built without it); the source warns once at construction naming those
labels and lists them, as given, in `unmodelled_labels`. Unmodelled labels may repeat, and each such
row gets its own noise. The label list must not be empty, and `srate` must be finite and positive (a
NaN or infinite rate used to fail deep inside the network wiring).

Internally the source is one `CaseSpec` with a single condition named `"stream"` of unbounded duration
(`duration_s = inf`, serialised as null), with the resting brain, the ordinary artifacts and the
default sensor noise unless overridden, built through the same `make_subject` and `make_engine` as a
case; its time streams are therefore `stream:background`, `stream:artifact:blink:0` and so on.
`seed=None` draws a fresh seed and exposes it as `.seed`, and a second source built with that seed
reproduces the signal, the truth and the markers. The default timeline is a constant `"eyes_open"`; a
recording application may pass its own scenario. Chunk size is unlimited, chunks are phase-continuous,
and — unlike `classic`, which drew blinks per chunk — the samples do not depend on chunk size (§8.2).

`truth` grows with the stream: it lists every artifact event whose onset lies inside the span
rendered so far (an event pushed later, past a refractory gap, appears once its onset is rendered),
and nothing discards it (a default 19-channel stream at 256 Hz holds 196–261 events after 10 minutes
and 650–708 after 30 over seeds 0–9, as the rates predict: about 220 and 660). A recording
application that streams for hours should read it incrementally. The default artifacts are not
exclusive, so the scheduler keeps no spans at all (§5.2).

`due_markers` follows `classic.MarkerSchedule` (kinds `"none"`, `"periodic"` and `"oddball"`; the
default is `"none"`) and is called in lockstep with `next_chunk`. It reimplements the classic marker
clock rather than calling it, because `classic` stays byte-identical; oddball targets are drawn from
`stream:markers`. Evoked responses to markers are not part of v0.2.0 (§11).

The raw stream is referential (§2.3): every channel carries the lead field's common component, so
alpha looks far less posterior-dominant in a raw view than in `classic`'s output; a view that
re-references (average or linked ears) shows the gradient, as it would for a real referential
amplifier.

**Cost.** Construction takes about half a second (0.54–0.58 s measured; the head perturbation and the
smoothing matrix on the full head), so a mock device builds one source when it starts, not one per
test. A chunk then costs about 0.55 ms (mean over 600 calls for 20 labels at 256 Hz in 32-sample
chunks and for 19 labels at 500 Hz in 50-sample chunks; 99th percentile under 1 ms), far inside the
100–125 ms between the calls of a mock amplifier that asks 8–10 times a second. A fast test requires a
mean below 25 ms; a 30-minute default stream renders in about 2 s with no drift in the per-call cost.

### 7.4 Case files

`casefile.writer.write_case(directory, case, *, embed_layers=False, physical_range_uv=(-2000, 2000))
-> CasePaths` writes:

```
<case_id>_<condition>.edf          one EDF per condition, no annotations
<case_id>.truth                    sealed truth container
<case_id>.layers.npz               only with embed_layers=True
```

and returns their paths (`CasePaths(truth, recordings, layers)`, `layers` None unless embedded).
`casefile.writer.read_case_truth(path)` reads a truth file back as a dict.

**EDF** (`casefile.edf.write_edf`, requires the `edf` extra, i.e. `edfio>=0.4`):
- one `EdfSignal` per channel, `physical_dimension="uV"`, `physical_range` in µV (default ±2000 µV,
  0.06 µV per bit at 16 bits — below the sensor noise floor, wide enough for pops), data clipped to
  the range, `transducer_type="synthetic"`, label = canonical channel name;
- whole seconds only (a case condition always lasts whole seconds, §7.2; a trailing partial second
  of any other recording is dropped); a 240 s, 19-channel, 256 Hz condition is 2,339,840 bytes;
- `Patient(code="SYNTHETIC", name="X", additional=("synthetic",))`, `Recording(startdate=None,
  equipment_code="open-eeg-synth")`. `startdate=None` writes the EDF+ "unknown" value; readers that
  need a date substitute their own convention (MNE reads it as 1985-01-01) and, because the file
  carries no annotations, that substitute date has no side effects;
- **no annotations of any kind** — no condition markers, no artifact spans, no annotation signal (the
  header's reserved field stays blank, so it is a plain EDF). Condition is carried by the file name
  and the truth file.

**Sealed truth** (`casefile.truth`): a binary container, magic `b"OESTRUTH\x01"` followed by
zlib-compressed (level 9) JSON with sorted keys; `read_truth` raises `ValueError` on anything without
the magic. Sealing here means the truth is a separate, non-text file that a consumer's user will not
read by accident and that never rides inside the EDF; it is obfuscation, not security (§11). The
default 2 × 240 s case writes about 6 kB. `case_truth(case, files)` builds the dict and raises
`ValueError` naming any condition missing from `files`. Contents (shown pretty-printed):

```json
{
  "format": "open-eeg-synth/truth", "format_version": 1,
  "generator": {"package": "open-eeg-synth", "version": "0.2.0", "signal_version": 2, "numpy": "…", "scipy": "…"},
  "case_id": "synth-3f9a1c2b", "seed": 1234567, "label": "synthetic",
  "spec": { …CaseSpec.to_dict()… },
  "subject": {
    "head_perturbation": {"tilt_deg": 8.0, "corr_mm": 15.0, "median_tilt_deg": …, "channel_gain": [19 values], "blur_eps": …},
    "placements": {"alpha": [10 source indices], "theta": [4 source indices], …},
    "f0_hz": {"alpha": …, "theta": …, …},
    "patch_f0_offsets_hz": {"alpha": [10 values], …},
    "patch_lags_ms": {"alpha": [10 values], …},
    "network": {"centres": [40 source indices], "sources": [[3 node indices], …], "lags": [[3 lags in samples], …]},
    "pattern_jitter": {"blink": …, "eye_movement": …}
  },
  "recordings": {
    "eyes_closed": {
      "file": "synth-3f9a1c2b_eyes_closed.edf", "fs": 256.0, "duration_s": 240.0, "channels": […],
      "layers": ["brain", "artifact:blink", "artifact:eye_movement", "artifact:emg", "sensor"],
      "layer_rms_uv": {"brain": [19 values, 4 decimals], …},
      "timeline": {"ramp_s": 2.0, "segments": [{"t0_s": 0.0, "t1_s": 240.0, "state": "eyes_closed"}]},
      "truth": [ {TruthRecord…}, … ],
      "plants": [ {PlantRecord…}, … ]
    },
    "eyes_open": { … }
  }
}
```

An event's `offset_s` in `truth` may exceed the recording's `duration_s` (an event that starts near
the end keeps its whole span, §5.2), and so may a dead channel's requested end; consumers clip both
to the recording. `bursts_s` is already clipped. `pattern_jitter` is keyed by artifact kind
(`"eye_movement"`), not by map name. A condition's `plants`
list leaves out plants that its timeline silences, and a `RhythmicBursts` record lists that
condition's bursts in `params["bursts_s"]` (§4.4).

**Re-rendering the layers.** Layers are not embedded by default: they are re-rendered on demand by
`casefile.writer.render_layers(truth_dict, condition, *, rtol=1e-3) -> Recording` (defined in
`casefile.truth`), which rebuilds the subject from `spec` and renders the condition through
`make_recording` (§7.2). If the truth file's `SIGNAL_VERSION` differs from this package's, it warns,
naming both package versions; another package version with the same `SIGNAL_VERSION` makes the same
samples and does not warn. It raises `ValueError` naming the condition when the spec has no such
condition or the truth file has no recording for it. It raises `LayerMismatchError` when the
re-rendered layer names differ from the truth file's `layers` list, or when a layer's per-channel
RMS differs from `layer_rms_uv` beyond `rtol` (with an absolute allowance of 1e-3 µV), so a consumer
never grades against the wrong layers. `embed_layers=True` writes `<case_id>.layers.npz` (keys
`<condition>/<layer>`, float32 `(n_ch, n_samples)`, plus `label` = `"synthetic"` like every other
file the package writes) for archival or for consumers that cannot install the package version that
made the case. A noise-like layer (brain, sensor) of a 240 s, 19-channel, 256 Hz condition
compresses to about 4.3 MB, an artifact layer to much less; the default two-condition case writes
18.6 MB.

### 7.5 Command line

`python -m open_eeg_synth make-case --seed 42 --out ./cases [--duration 240] [--spec spec.json]
[--embed-layers] [--print-truth]` builds `recipes.resting_case(seed, duration_s=duration)`, or the
`CaseSpec` in the given JSON file with its seed replaced by `--seed` (`--duration` is then ignored),
and writes the files with `write_case`. It prints the truth file's path, or with `--print-truth` the
truth itself as JSON. Requires the `edf` extra. Before building anything it checks that `edfio` can
be imported and that `--duration` (or every condition of the `--spec` file) is a positive whole
number of seconds; otherwise it exits with status 2 and says why.

---

## 8. Determinism, seeding, versioning, performance

### 8.1 One seed, named streams

`seeds.stream_rng(case_seed, name)` returns a `numpy.random.Generator(PCG64)` seeded from
`SeedSequence([case_seed, blake2b(name)])`. Every random draw in the engine comes from a named stream:

| stream name | owner |
|---|---|
| `subject:head` | head-model perturbation |
| `subject:rhythm:<name>` | patch placement, centre-frequency jitter, per-patch centre-frequency offsets and lags (`<name>` is the compiled rhythm name, §4.4) |
| `subject:network` | network node placement and wiring |
| `subject:artifact:<kind>` | pattern jitter (only for kinds that declare a `jitter_pattern`) |
| `<condition>:background`, `<condition>:network` | time courses, including each cascade's 20 s warm-up |
| `<condition>:rhythm:<name>` | the rhythm's driver, each patch's own oscillator and the burst gate |
| `<condition>:artifact:<kind>:<i>` | scheduling, event waveforms and a transform's noise; `i` counts instances of that kind only |
| `<condition>:sensor` | sensor noise |
| `stream:heart`, `stream:markers` | a `StreamSource`'s ECG (RR intervals and waveform noise) and oddball markers |
| `stream:unmodelled` | a `StreamSource`'s sensor noise on labels the head model lacks (§7.3) |

A `StreamSource`'s single condition is named `stream`, so its time streams read `stream:background`
and so on. Adding a rhythm, a plant or an artifact of another kind therefore never changes the draws
of any other component, and two conditions of one case share a subject by construction. Two
exceptions are deliberate consequences of the naming: a modifier's `extra_sites` changes its target
rhythm's subject draws (§4.4), and an artifact inserted before others of its own kind renumbers them.

### 8.2 Chunk invariance

`Engine.render_all(N)` equals the concatenation of `render(t0, n)` calls for any partition of
`[0, N)`. The tests render engines and whole cases (brain with plants and a drowsy segment, artifacts,
sensor noise, transforms) in random partitions and require every layer to agree within 1e-4 µV; the OU
processes and the artifact layers agree bit for bit, and the remaining differences are floating-point
rounding in carried filter states. The heart source is tested at 1e-3 µV (measured 2.4e-4 µV over 60 s
in 32-sample chunks): each beat's waves are evaluated only within a window around the beat, so a
chunk edge can cut a Gaussian tail of that order. The `StreamSource` test, which includes a heart row,
holds 1e-4 µV on its seed. Four rules make this hold:

1. Noise is always drawn time-major — `rng.standard_normal((n, n_series)).T` — so a longer block
   consumes exactly the draws that two shorter blocks would.
2. Every filter carries state across calls (`scipy.signal.lfilter` with `zi`, node histories, driver
   histories, OU states, burst-gate edges); nothing is computed over "the whole recording". The one
   exception, the jaw EMG's decimation, runs over a whole event drawn at once.
3. Event artifacts draw candidate onsets and waveforms in onset order, at the moment the block
   containing the onset is requested (§5.2).
4. Exclusive event artifacts schedule together, in global onset order, through their shared
   `Occupancy`, so neither chunk size nor the order of render calls decides which of two claims a
   span (§5.2).

### 8.3 Versioning

`version.__version__` is the package version. `version.SIGNAL_VERSION` (integer) is bumped whenever the
same seed would produce different samples. `SIGNAL_VERSION = 2` identifies the first released layered
signal, v0.2.0; the `classic` signal came before it. The layered signal changed many times during
development without a bump, because no layered signal had been released that anything could differ
from. From v0.2.0 on, every change to generated samples bumps it.

A fingerprint test (`tests/test_golden_engine.py`) renders `recipes.resting_case(seed=20260916)` for
20 s, long enough for the eyes-open recording's first jaw-EMG burst (at 17.6 s), and compares these
values with `tests/golden/resting_seed20260916.json` (µV values rounded to 0.01 and compared within
0.05; counts and onsets exactly):

- eyes closed: every channel's RMS and the first 32 samples of Fz;
- eyes open: the first 32 samples of Fz; the RMS of the blink layer at Fp1, the eye-movement layer
  at F8 and the jaw-EMG layer at T3 and T4; and, per artifact kind, the event count and the first
  event's onset and peak;
- the eyes-open recording again with a dead O1 from 4 to 9 s: the RMS of the dead channel's
  transform delta, and of O1 inside the dead span;
- the eyes-closed F7 of the same case with `FocalSlow("F7")` planted.

Halving the blink waveform or the jaw-EMG burst, zeroing any artifact kind's eyes-open rate, or
changing the dead channel's delta or noise fails it. The 20 s eyes-closed recording holds no eye
movement or jaw-EMG event, so those kinds' eyes-closed rates are not covered. The test fails when
the signal changes while `SIGNAL_VERSION` stays the same, and when `SIGNAL_VERSION` differs from the
stored one it fails too, before rendering, with "SIGNAL_VERSION changed: regenerate with
scripts/update_golden.py", so a bump is never a silent pass. After a deliberate change, bump
`SIGNAL_VERSION` and run `scripts/update_golden.py` in the same commit to regenerate the file; run
it also when the test changes what it records (the test then reports that the fingerprint's contents
changed). Both numbers are written into every truth file, and `render_layers` warns when
`SIGNAL_VERSION` differs (§7.4). The `classic` subpackage keeps its own bit-exact golden test
against the recording application's original output (`tools/make_classic_golden.py` records that
fixture).

### 8.4 Performance budget

Target: a two-condition, 2 × 240 s, 19-channel, 256 Hz case with the default artifacts renders in
≤ 6 s on a 2020 laptop, excluding file writes; `tests/test_perf.py` (marked `slow`) fails above 15 s
and checks that the timed case carries the artifact layers. Measured on an Apple Silicon Mac:
`make_case(resting_case(20260916))` takes 0.86 s, of which `make_subject` (head perturbation and
smoothing matrix) takes 0.53 s. Cost centres and why they fit: the smoothing matrix and the
perturbation (once per subject); ~80 filtered noise series × 123 k samples over both conditions (well
under a second in `lfilter`); a 19×40 network projection per block; about 60 blinks, 30 eye movements
and 8 EMG bursts at 8× oversampling (milliseconds; seed 20260916 draws 62, 31 and 8). The feasibility
test's per-sample Python loops for OU processes are replaced by `lfilter` from the start. A
`StreamSource` costs about half a second to build and 0.55 ms per chunk (§7.3).

---

## 9. Tests

### 9.1 Realism suite (`tests/realism/`, marker `realism`, opt-in)

**What is measured** (`measure.py`, using MNE and mne-connectivity, both development extras — no
consumer code):
- per band (Delta 1–4, Theta 4–8, Alpha 8–13, Beta 13–30 Hz), per reference (average; longitudinal
  bipolar double-banana plus the Fz–Cz and Cz–Pz midline pairs; Laplacian via MNE's
  current-source-density): mean coherence and mean debiased wPLI over all pairs
  (`spectral_connectivity_epochs`, methods `coh` and `wpli2_debiased`, band-averaged), and mean
  coherence in five inter-electrode distance bins (< 60, 60–90, 90–120, 120–150 and 150–250 mm; pair
  positions from the `standard_1020` montage, the midpoint for a bipolar pair). Coherence here is
  mne-connectivity's coherence magnitude, `|S_xy| / √(S_xx S_yy)`, not its square;
- aperiodic exponent: the slope of the median Welch spectrum (4 s segments) over 2–40 Hz excluding
  7–14 Hz; alpha share: 8–13 Hz over 1–40 Hz power at O1, Fz and Cz; per-channel RMS after 1–45 Hz
  filtering and average reference.
- Preparation: 1–45 Hz band-pass, `standard_1020` montage, average reference, 4 s epochs, epochs with
  peak-to-peak > 800 µV dropped, at least 8 epochs.
- Method: `spectral_connectivity_epochs(epochs, method=["wpli2_debiased", "coh"], fmin=[1, 4, 8, 13],
  fmax=[4, 8, 13, 30], faverage=True)` on the 4 s fixed-length epochs, pair values read as they are from
  the lower triangle of the dense output (mne-connectivity fills that triangle and leaves the other zero).
  The feasibility test measured through a consumer function that averaged the dense matrix with its
  transpose and so halved every pair value; the package does not reproduce that defect. Every coherence
  and dwPLI figure in this section, and every value in the reference file, is the true pair value —
  twice the figure the feasibility test reported. Synthetic and real recordings are measured
  identically, so relative comparisons are unchanged.

**Reference**: PhysioNet eegmmidb subjects 1–20, runs 1 (eyes open) and 2 (eyes closed), blink
components removed by ICA (fastica, 15 components, `find_bads_eog` on Fp1/Fp2 at threshold 3).
`scripts/build_realism_reference.py` downloads on demand (`mne.datasets.eegbci`), measures, and writes
`tests/realism/reference/eegmmidb_baselines_s01-20.json` with p10/p50/p90 of every metric per condition
plus the attribution of §6.4. The JSON is committed, so the suite itself needs no download; only
regenerating the reference does. Built with MNE 1.13.2, every recording kept at least 14 epochs, and
the result reproduces twice the feasibility test's connectivity figures within 0.003 and its exponent
and alpha shares within 0.002; two eyes-open subjects (10 and 14) move by up to 0.004 under the newer
MNE and scikit-learn versions.

**Synthetic side**: `recipes.resting_case(seed, duration_s=60, artifacts=())` for the 24 seeds
100–123 (brain and sensor noise only), each seed's case rendered once for both conditions and measured
identically (without ICA); the medians over the seeds are compared with the reference. The suite
takes about 20 s. **These seeds are in-sample:** they were part of the objective the recipe was tuned
on (§4.3), so passing on them shows that the recipe has not drifted, not that any set of subjects
passes; the out-of-sample figures are below. The design first used six seeds, but a six-seed median is
too noisy to gate on: only 10–12 % of disjoint six-seed sets drawn from fresh seeds pass every line
outside the known gaps.

**Pass bands** (10th–90th percentile of the 20 real subjects, true pair values; in parentheses the
recipe's median over the 24 test seeds, measured for this document; Laplacian means may exceed p90 by
up to 0.08, see the acceptance rule):

| reference / band | mean coherence EC | EO | dwPLI EC | EO |
|---|---|---|---|---|
| average / Delta | 0.260–0.424 (0.351) | 0.236–0.732 (0.351) | 0.009–0.135 (0.034) | 0.008–0.529 (0.040) |
| average / Theta | 0.328–0.407 (0.375) | 0.295–0.558 (0.366) | 0.051–0.205 (0.101) | 0.083–0.214 (0.083) |
| average / Alpha | 0.365–0.570 (0.505) | 0.291–0.451 (0.397) | 0.134–0.383 (0.176) | 0.060–0.257 (0.115) |
| average / Beta | 0.300–0.398 (0.367) | 0.266–0.376 (0.371) | 0.119–0.240 (0.191) | 0.098–0.233 (0.193) |
| bipolar / Delta | 0.180–0.340 (0.242) | 0.170–0.513 (0.244) | 0.007–0.081 (0.031) | 0.002–0.370 (0.031) |
| bipolar / Theta | 0.226–0.305 (0.272) | 0.202–0.378 (0.262) | 0.043–0.151 (0.098) | 0.051–0.217 (0.100) |
| bipolar / Alpha | 0.272–0.472 (0.387) | 0.210–0.323 (0.300) | 0.143–0.347 (0.180) | 0.040–0.259 (0.112) |
| bipolar / Beta | 0.220–0.288 (0.268) | 0.206–0.281 (0.268) | 0.126–0.243 (0.161) | 0.064–0.261 (0.168) |
| laplacian / Delta | 0.211–0.288 (0.278) | 0.223–0.439 (0.279) | 0.005–0.092 (0.030) | 0.000–0.485 (0.029) |
| laplacian / Theta | 0.225–0.284 (0.312) | 0.234–0.324 (0.312) | 0.046–0.158 (0.131) | 0.034–0.324 (0.131) |
| laplacian / Alpha | 0.256–0.387 (0.399) | 0.244–0.317 (0.334) | 0.144–0.386 (0.167) | 0.061–0.270 (0.135) |
| laplacian / Beta | 0.274–0.330 (0.366) | 0.265–0.357 (0.378) | 0.141–0.266 (0.227) | 0.070–0.265 (0.238) |

Other metrics: aperiodic exponent within 0.8–1.4 (real p10–p90 0.62–1.37 eyes closed, 0.43–1.39 eyes
open; recipe 1.29 / 1.29); O1 alpha share within 0.35–0.79 eyes closed and 0.05–0.30 eyes open (recipe
0.61 / 0.21; the reference's own p10–p90 is 0.17–0.79 and 0.05–0.27, so the eyes-closed lower bound is
stricter than the real data); RMS at Cz within 8–20 µV (real medians 13.7 / 12.7 µV; recipe 12.6 /
12.4 µV).

**Acceptance.** A condition passes when every average-reference and bipolar mean coherence and dwPLI
lies inside its band; every Laplacian mean lies inside `[p10 − 0.04, p90 + 0.08]` (twice the
feasibility test's `[−0.02, +0.04]`, for the same reason as the pair values); every distance bin, under
all three references, lies inside `[p10 − 0.06, p90 + 0.06]` (`BIN_TOL`: twice the feasibility test's
±0.03, and the same for the Laplacian bins as for the others — the asymmetric Laplacian allowance
applies to the means only); and the exponent, the O1 alpha share and the Cz RMS lie inside their
ranges — except for the lines named in `KNOWN_GAPS`.

**Known gaps.** `KNOWN_GAPS` maps a line (condition, reference, band, metric) to a reason and, for a
binned metric, the distance bins it excuses (`tests/realism/gaps.py`, unit-tested without MNE). The
test fails on any failing line that is not listed, and on any failing bin that a listed line does not
name; the tolerances are not widened for the listed lines. When a listed line passes, the test warns,
so a gap that closes gets noticed. Two lines are listed:

- eyes closed, Laplacian, theta, nearest distance bin: 0.536 on the test seeds against a limit of
  0.513, and 0.531–0.536 at every population median measured (seeds 100–339, 340–579 and 580–819).
  This is the cost of keeping frontal-midline theta focal (§4.3).
- eyes open, bipolar, beta, nearest distance bin: a population-level gap. It passes on the test seeds
  (0.328 against a lower limit of 0.317, so every run carries its "known gap passes" warning) but fails
  at the population median of fresh seeds (0.310 on 340–579, 0.308 on 580–819).

With those two excused, the test seeds pass every other line. The tightest remaining margins on them
are eyes-open average-reference theta dwPLI (+0.0004 above its p10) and eyes-closed Laplacian beta in
the farthest distance bin (+0.002).

**Out of sample.** On fresh seeds that the tuning never saw, 200 random 24-seed sets per pool: with
both gaps excused, 26 % of the sets from seeds 340–579 and 49 % from 580–819 pass every other line (the
other failing lines average 1.18 and 0.63 per set, at most 4 and 2); excusing only the Laplacian theta
gap, 9 % of each pool pass. The lines that fail most often beyond the two gaps are eyes-closed
Laplacian beta in the farthest or the nearest bin (47 % and 44 % of the sets in the two pools; on
340–579 the population median clears the farthest bin by 0.001 and the nearest by 0.007), eyes-closed
Laplacian alpha at 120–150 mm (33 % and 4 %; margin 0.006) and eyes-open average-reference theta dwPLI
(20 % and 7 %). The gate is therefore weak out of sample
(§11): it catches a recipe that drifts on its own seeds, and a change that reshuffles the random
streams may flip one of these edge lines.

**Known limits of the reference**: 160 Hz source data (nothing above 80 Hz is compared), one minute per
condition per subject, healthy young adults, 2009-era amplifiers.

### 9.2 Fast tests (`pytest`, run in CI)

`pytest` runs the fast set: 344 tests in 64–69 s on the development machine under load (two runs,
alternating with the 335 tests of the previous revision, which took 64–66 s), against a target of
60 s that the suite does not meet. The slowest tests render whole cases over several seeds, because
a statistical check on a single seed can pass by luck; they are the candidates for shared fixtures or
the `slow` marker if the suite grows. `pytest -m slow` runs three more (the performance budget, the
realism smoke test and a FocalSlow spatial-specificity check), and `pytest -m realism` runs the
realism suite of §9.1 together with its smoke test.

CI (GitHub Actions) runs ruff and the fast set on Python 3.10, 3.11 and 3.12 for pushes and pull
requests to `main`, the `slow` set on 3.11, and builds the wheel and checks that the two data files and
their two notices are inside it. On every Python version it then installs that wheel with its `edf`
extra into a bare virtual environment (no development extras) and, outside the checkout, makes the
default case with `python -m open_eeg_synth make-case --seed 1` and re-renders its eyes-open
condition with `render_layers`. The realism suite runs weekly and on demand.

What the fast set covers, per module: seeds (determinism, name independence, stability across
processes); dsp (PSD slope and unit variance of the 1/f cascade at two sample rates and exponents,
OU stationary sd, mean and lag-1 autocorrelation, chunk invariance of every filter, a tone near the
new Nyquist passes decimation within 1 dB and images at and above it are 60 dB down); head model
(file contents, fixed = free·normal, subsets, aliases, unknown head models, mirror sources,
smoothing matrix, perturbation magnitude, determinism and one-channel heads); background and network
(RMS, slope, measurable lag between coupled nodes, chunk invariance); rhythm (spectral peak at f0 at
the target sites, mirror symmetry and polarity of maps, a cross-spectrum phase lag between mirror
sites, the drowsy alpha peak shift, placement without shared sources, per-patch draws, burst times
under random chunking); brain layer (calibration pins as medians over nine seeds, a visible theta
imbalance, eyes-open alpha suppression, missing per-patch draws refused); timeline (weights sum to
one, ramps, value equality, JSON with infinite ends); plants (each primitive changes band power at
its sites by the expected factor, state confinement, names, validation, records, canonical sites);
artifact framework (scheduling rates by state, the thinned process, refractory gap, exclusivity and
the shared scheduler under random partitions and call orders, the onset rule, rebinding, span
pruning against an unpruned control, registry and discovery); patterns (the T9/T10 reference and
pinned map values, sign-keeping capped jitter, full-head amplitudes, the fallback's sign rule,
reliability, blend and clip, mirror montages, continuity under small moves, a 3,000-request sparse
sweep per map); plug-ins (each one's waveform, rates, truth contents and channels, subsets, jaw EMG
placement and anti-aliasing, dead-channel spans); engine (sum identity, contiguity check, chunk
invariance, buffer copies, read-only mix); case (defaults, identity and type strictness, artifact
parameters as one identity, alias canonicalisation, duplicate channels and conditions, whole-second
durations, subject shared across conditions, pattern jitter only for jittering kinds, JSON round
trip, chunk invariance of layers, truth and plant records with artifacts, plants and drowsiness,
drowsiness end to end); case files (EDF round trip in µV, no annotations, MNE read-back, anonymised
date, clipping; truth container round trip and contents; `render_layers` verification and errors,
burst times; `write_case` layout; the command line and its checks before building); stream (the
recording application's contract: shape, dtype, RMS range, determinism, fresh seeds, 1/f slope,
posterior-dominant alpha, ECG only on heart labels, duplicate labels, labels outside the head model,
non-finite rates, other label sets at 250 and 500 Hz, chunk invariance, markers, seed replay,
per-call cost); the realism gap matcher; the golden fingerprint; the `classic` golden test, and that
importing `classic` loads numpy only.

---

## 10. Packaging and consumption

`pyproject.toml` (excerpt):

```toml
[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "open-eeg-synth"
dynamic = ["version"]
requires-python = ">=3.10,<3.13"
license = "Apache-2.0"
dependencies = ["numpy>=1.24", "scipy>=1.10"]

[project.optional-dependencies]
edf = ["edfio>=0.4"]
realism = ["mne>=1.6,<1.14", "mne-connectivity>=0.6"]   # <1.14: the standard_1020 montage name goes away
dev = ["pytest>=7.0", "ruff>=0.6", "edfio>=0.4", "mne>=1.6,<1.14", "mne-connectivity>=0.6"]

[tool.hatch.version]
path = "src/open_eeg_synth/version.py"

[tool.hatch.build.targets.wheel]
packages = ["src/open_eeg_synth"]          # data/*.npz and NOTICE files inside the package are included

[tool.hatch.build.targets.sdist]
include = ["src", "tests", "tools", "scripts", "docs", "README.md", "CHANGELOG.md", "LICENSE", "NOTICE"]

[tool.pytest.ini_options]
addopts = "-ra -m 'not realism and not slow'"   # markers: realism, slow
```

- **scipy at runtime** is a deliberate deviation from the earlier "numpy only" proposal: streaming
  IIR filters with carried state and polyphase anti-alias decimation are exactly what `scipy.signal`
  does robustly, both consumers already ship scipy, and the call sites (`lfilter`, `firwin`,
  `resample_poly`) are isolated in `dsp.py`; the truth file also records scipy's version. The
  package root imports its engine exports (`StreamSource`, `make_case`, `write_case`, …) on first
  use through a module `__getattr__`, so `import open_eeg_synth.classic` loads numpy only (tested in
  a fresh interpreter: 0.02 s, against 0.35 s when the root imported the engine eagerly), and the
  first engine name costs the scipy import instead.
- **Install by git tag**: `open-eeg-synth @ git+https://github.com/peak-mind-llc/open-eeg-synth.git@v0.2.0`
  (with `[tool.hatch.metadata] allow-direct-references = true` in the consumer if it builds with
  Hatchling). Optional extras are selected as `open-eeg-synth[edf] @ git+…`.
- **Data in the wheel**: `headmodel/data/colin27_19ch.npz`, `headmodel/data/NOTICE-colin27.txt`,
  `artifacts/data/eog_patterns.npz`, `artifacts/data/NOTICE-eegmmidb.txt`. They are read with
  `importlib.resources.files("open_eeg_synth.headmodel") / "data" / …`, which works from a wheel, an
  editable install and a frozen bundle. CI checks that the built wheel contains all four.
- **PyInstaller in a consuming app**: add `open_eeg_synth` to the spec's hidden-import collection
  (`collect_submodules("open_eeg_synth")`) and to the set of first-party packages whose absence fails
  the build, and add `collect_data_files("open_eeg_synth")` to `datas` so the two `.npz` files and the
  notices ship. Nothing else: no MNE, no download at runtime.
- **Local editable development**: `pip install -e ".[dev]"` in this repo; a consumer developing against
  a local checkout runs `pip install -e /path/to/open-eeg-synth` after its own install, which pip then
  treats as satisfying the git requirement.
- **Releases**: tag `vX.Y.Z` on `main` after CI is green; `CHANGELOG.md` entry; consumers bump the tag.
  v0.2.0 = this design; v0.1.x = `classic` only.

---

## 11. Open questions

Each with the current recommendation; the questions the implementation answered say so.

1. **Is the head-model perturbation strong enough to stop a consumer's inverse from localising planted
   sources "too well"?** Unknown until measured in a consuming application. Recommendation: ship the
   three-step perturbation with the defaults above, expose the knobs, and have the consumer compare
   residual variance of dipole fits on synthetic versus real cases; raise `tilt_deg`/`gain_sd` if
   synthetic fits are still implausibly good.
2. **Laplacian coherence against the real band — partly closed.** The feasibility model's Laplacian
   alpha and theta coherence was slightly too high at long distance. After the tuning of §4.3 every
   long-distance bin is inside the real band ±0.06; the eyes-closed Laplacian alpha and theta means
   still sit above p90 (0.40 against 0.39, 0.31 against 0.28) and pass only through the looser
   Laplacian allowance, which was not tightened. The first proposals were only partly used: network
   patches went from 15 to 12 mm, but alpha kept `indep` 0.6 and went to 10 patches of 25 mm instead of
   6. One line is left, at short distance: eyes-closed Laplacian theta in the nearest bin (0.536
   against 0.513), kept as a named gap because frontal-midline theta stays focal (§4.3). Fresh seeds
   added a population-level gap, eyes-open bipolar beta in the nearest bin (§9.1). Recommendation:
   keep both named and listed in the README; close the theta line only with a change that keeps the
   theta checks of §4.3; remove a gap only when it passes on fresh seeds, not just on the test seeds.
3. **Store layers or re-render them?** Decided: re-render from the seed with the RMS check (§7.4) as
   the default; `embed_layers=True` for archival. Revisit if consumers cannot keep package versions
   aligned with stored cases.
4. **How sealed is "sealed"?** Decided: compressed binary container with a magic header, no
   encryption. It stops accidental reading, not a determined one, which matches the consumers' own
   answer-key precedent. Anything stronger needs a key-management story that does not belong here.
5. **Physical range of the EDF.** ±2000 µV keeps electrode pops and large EMG unclipped at 0.06 µV
   resolution. Consumers that expect ±500 µV files should say so; the parameter is exposed.
6. **Amplitude convention — closed.** The analytic convention (§4.3) is kept (it is the only one a
   streaming generator can honour), and the four resting amplitudes were re-derived under it: alpha
   37, theta 7.3, beta 5.5 and smr 6 µV.
7. **Evoked responses in streaming mode** (a recording application's oddball mock injects a P300 at
   markers). Recommendation: v0.2.0 ships markers only; an `EvokedResponse` brain primitive (marker
   time → patch response) is a v0.3 item, at which point the standalone oddball mock can move over.
   `classic.oddball.ErpInjector` (seeded, carries a response across chunk boundaries) is the starting
   point for that primitive.
8. **Drowsiness beyond rhythm gains** (vertex sharp transients, sleep spindles, K-complexes).
   Recommendation: out of scope for v0.2.0; they are `EventArtifact`-shaped brain primitives and can
   be added without engine changes.
9. **Larger channel sets.** 31- and 37-channel exports are the same script with a different input; the
   empirical eye maps already cover their labels, and `CaseSpec` already canonicalises channels
   against the selected head model's own list. Recommendation: export when a consumer needs them.
10. **Should the recording application keep `classic` selectable after switching?** Recommendation:
    yes for one release behind an environment variable, then delete the switch; `classic` itself
    stays as the golden-tested baseline and for anyone who wants the old look.
11. **Mains frequency default.** None — `MainsHum(freq_hz=…)` is required. Consumers pass their
    configured value.
12. **Python 3.13.** Nothing in the package prevents it; the ceiling `<3.13` follows the consumers'
    pins and lifts when they do.
13. **The realism gate is weak out of sample.** With both gaps excused, 26 % and 49 % of fresh 24-seed
    sets pass (§9.1). The lines closest to failing are eyes-closed Laplacian beta (0.001 inside the
    band at one fresh population median) and eyes-open average-reference theta dwPLI (0.0004 inside on
    the test seeds). Recommendation: after a signal change, treat a new failure on the test seeds as a
    prompt to check the population medians before retuning; a stronger gate (more seeds, population
    medians) would cost suite time.
14. **Theta dominance has little headroom.** Fz or Cz is the loudest eyes-open theta channel in 65 % of
    subjects against a 60 % floor (61 % at a theta amplitude of 7.0 µV), and the realism test does not
    check it. Recommendation: re-measure it on seeds 100–195 whenever the network gets louder or more
    focal.
15. **Left/right biases.** Posterior alpha is 0.6–0.7 dB weaker on the right and 12–15 Hz power 0.4 dB
    weaker at C4 than at C3 (§4.3). Recommendation: a consumer that grades asymmetry findings should
    check these against its own thresholds before relying on resting cases being symmetric; correct in
    the placement if they matter.
16. **Scalp-pattern fallback away from the template.** Reachable only for labels outside the eye-pattern
    file and positions a caller supplies, never with the 19-channel head model: some blink fallbacks
    stay more than 3 times off on montages far from the template (27 of about 4,500 sweep values on
    brainproducts-RNP-BA-128, 146 on a spherical head), heog fits anchored only on O2 reach 3.4 times,
    and near-eye channels are held at the clip. Recommendation: revisit when a head model with such
    labels is exported; a wider reliability ramp (4 times the floor) removes most of the remaining
    cases at the cost of changing one tested reference value.
17. **Scheduler corner cases.** The queue of pushed events grows without bound when
    `rate × (length + gap)` reaches one, and an exclusive plug-in that is bound but never rendered
    still takes part in scheduling. Neither is reachable with the built-in plug-ins. (Registration
    now compares plug-ins by identity, so two value-equal dataclass plug-ins are no longer taken for
    one.) Recommendation: fix before the first exclusive plug-in ships.
18. **Artifact parameters in the spec identity — closed.** An int and a float spelling of one
    parameter used to give two case ids for specs that compared equal. The identity now reads every
    non-bool number as a float (§7.2), while the values a plug-in receives are left as given; only
    specs that spelled an artifact number as an int change id. Normalising the values themselves
    against each plug-in's constructor signature was not needed for the identity and would have
    changed what third-party plug-ins receive.
