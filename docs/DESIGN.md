# open-eeg-synth — package design

**Status:** design for the v0.2.0 layered engine. The v0.1.x line is the `classic` subpackage: a
recording application's original mock-device synthesizer, moved here verbatim and protected by a
same-seed golden test. Everything below builds beside `classic`, never on top of it.

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
| Realistic spatial structure | Cortical sources projected through a real-head lead field (§3, §4), which a feasibility test showed matches public resting EEG in coherence-by-distance under average, bipolar and Laplacian references (§9). |
| Truth for grading | The recording is the sum of named layers; every layer stays retrievable; every artifact and planted pattern writes a truth record (§5, §7.4). |
| Extensible artifacts | A plug-in contract with a registry; first three plug-ins specified in full, ten more shown to fit (§5). |
| Determinism | One case seed derives every random stream by name; chunking never changes the samples (§8). |
| Light runtime | numpy + scipy only. MNE is a development extra used by two offline scripts and the realism suite (§10). |
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
  __init__.py            version, public re-exports (make_case, StreamSource, load_head_model, …)
  version.py             __version__ = "0.2.0", SIGNAL_VERSION = 2
  channels.py            CHANNELS_19, label aliases, canonical_label()
  seeds.py               stream_seed(), stream_rng(), fresh_case_seed()
  dsp.py                 OU processes, 1/f^β cascade, decimation, envelopes
  headmodel/
    __init__.py          HeadModel, load_head_model(), subsets, perturbation, patch maps
    data/colin27_19ch.npz
    data/NOTICE-colin27.txt
  brain/
    background.py        smoothed cortical noise (§4.1)
    network.py           delayed cortical network (§4.2)
    rhythm.py            RhythmSpec, Rhythm (§4.3)
    placement.py         sources under electrodes, mirroring, regions (§4.3)
    plants.py            planted-pattern primitives (§4.4)
    state.py             StateTimeline (§4.5)
    layer.py             BrainSpec, BrainLayer (composition)
  artifacts/
    base.py              Remedy, TruthRecord, RenderContext, Artifact, EventArtifact, TransformArtifact
    registry.py          register(), ARTIFACTS, discovery
    patterns.py          empirical and analytic scalp patterns
    data/eog_patterns.npz
    data/NOTICE-eegmmidb.txt
    blink.py  eye_movement.py  jaw_emg.py            (v0.2.0)
    dead_channel.py                                  (v0.2.0, proves the transform hook)
    …                                                (future, §5.7)
  sensor.py              SensorNoise layer
  heart.py               HeartSource: RR intervals + ECG waveform (shared by the HR channel and future bleed plug-ins)
  contact.py             ContactTimeline: per-channel impedance over time (future loose-lead plug-in, recorder impedance mock)
  engine.py              Frame, Engine, Recording
  case.py                CaseSpec, ConditionSpec, Subject, make_subject(), make_engine(), make_case(), Case
  recipes.py             resting_brain(), ordinary_artifacts(), resting_case()
  stream.py              StreamSource (recorder-facing), marker clock
  markers.py             re-exports classic.MarkerSchedule; OddballSchedule
  casefile/
    edf.py               write_edf() (optional `edfio` extra)
    truth.py             truth dict, sealed container read/write, layer verification
    writer.py            write_case(), read_case_truth()
  __main__.py            `python -m open_eeg_synth make-case …`
  classic/               v0.1.x synthesizer, untouched: synth.py (RealisticEEGSynthesizer, MarkerSchedule),
                         cardio.py (MockRRSource, MockEcgSource, MockAccSource), oddball.py (OddballParadigm,
                         schedule, ErpInjector — seeded, carries responses across chunk boundaries)
scripts/
  export_head_model.py          dev-only (MNE): forward solution → colin27_19ch.npz
  derive_artifact_patterns.py   dev-only (MNE): PhysioNet ICA → eog_patterns.npz
  build_realism_reference.py    dev-only (MNE): PhysioNet baselines → tests/realism/reference/*.json
tests/
  test_*.py                     fast unit tests (< 60 s total)
  realism/                      measure.py, reference/, test_realism.py  (marked `realism`, opt-in)
```

### 2.3 Conventions

- Units: microvolts for every signal; metres for positions; seconds for `_s` parameters; samples for
  `t0`/`n`. Head-model gain stays in V per A·m as exported; all patch maps are normalised (unit RMS
  across channels) and amplitudes are specified in µV at a named channel.
- Coordinates: the template head frame (x right, y anterior, z superior). Mirror = `x → −x`.
- Channel order of `CHANNELS_19` is the head-model row order:
  `Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 Fz Cz Pz`. Aliases `T7/T8/P7/P8 → T3/T4/T5/T6`;
  labels are matched case-insensitively.
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

Facts used by the placement code: median source spacing 3.8 mm; the nearest source to the mirror
image of any source is 2.6 mm away on median (4.2 mm at the 90th percentile), so mirrored patch
placement is faithful.

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
    perturbation: dict | None       # None for the nominal model; the draw record after perturbed()

    @property
    def gain(self) -> np.ndarray:   # (n_ch, n_src), cached
    def subset(self, labels: Sequence[str]) -> "HeadModel"
    def perturbed(self, rng: np.random.Generator, *, tilt_deg=8.0, gain_sd=0.06,
                  blur_range=(0.03, 0.10), corr_mm=15.0) -> "HeadModel"
    def patch_map(self, centre: int, width_mm: float) -> np.ndarray     # (n_ch,), unit RMS
    def sources_under(self, label: str, n: int = 40) -> np.ndarray      # source indices
    def mirror_source(self, i: int) -> int
    def smoothed_mixing(self, scale_mm: float) -> np.ndarray            # (n_ch, n_ch)

def load_head_model(name: str = "colin27_19ch") -> HeadModel
```

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
5 % and 25 %; a unit test pins that range. `perturbation` records `tilt_deg`, the 19 channel gains and
`ε` so a truth file can state exactly what was done. Whether this is *enough* to make a consumer's
localisation look like real data is an open question (§11); the knobs are exposed for that experiment.

---

## 4. Brain layer

All time-series generation is stateful and block-wise so that streaming and whole-recording rendering
produce the same samples (§8). The numbers below are the feasibility test's tuned values; M2 of the
plan re-verifies them against the realism suite and may adjust amplitudes (the amplitude convention
here is analytic, the test's was empirical — see §4.3).

### 4.1 Background: smoothed cortical noise

Independent 1/f^β noise on every cortical source, spatially smoothed with a 20 mm Gaussian, projected
through the lead field. Computing 4871 source time series is wasteful when only their 19-channel
projection is ever observed, so the feasibility test used a shortcut: the sensor covariance
`C = (G K)(G K)ᵀ` of that construction (`K` = Gaussian smoothing over source distances) is a 19×19
matrix, and `M = C^{1/2}` applied to 19 independent noise series has exactly that covariance.
**Decision: keep the shortcut.** It is exact in second-order statistics, which is all the background
contributes; a rank-19 Gaussian background is indistinguishable from the full construction to spectra,
coherence, and ICA (Gaussian sources are not separable anyway). `C^{1/2}` is computed with
`numpy.linalg.eigh` (no scipy needed here); `M` is normalised so the mean channel variance is 1 and
then scaled by `rms_uv`. `M` depends on the perturbed gain, so it is computed once per subject (~0.3 s).

```python
@dataclass(frozen=True)
class BackgroundSpec:
    smoothing_mm: float = 20.0
    exponent: float = 1.2        # power ∝ 1/f^exponent
    rms_uv: float = 20.0         # mean channel RMS of background + network together
    network_frac: float = 0.5    # share of background variance carried by the delayed network
```

Time courses: `dsp.PinkCascade` — 1/f^β by a cascade of first-order pole–zero sections (two per decade
from 0.03 Hz to 0.45·fs; Corsini & Saletti's construction), bilinear-transformed, run with carried
filter state, gain-calibrated to unit variance at construction. The FFT-shaped noise the feasibility
test used is kept only as the unit test's oracle (fitted PSD slope over 1–40 Hz must equal β ± 0.1).

### 4.2 Delayed cortical network

Forty cortical patches (15 mm) at random source positions; each node's time course is its own 1/f^β
series plus delayed copies of three other nodes' series, the delay being `distance / 6 m/s + 5 ms`.
This is what gives the recording non-zero lagged coupling (debiased wPLI) — the feasibility test showed
that delays between *identical* maps are invisible, so the network needs distinct patches. The network
carries `network_frac` of the background variance.

```python
@dataclass(frozen=True)
class NetworkSpec:
    n_nodes: int = 40
    fan_in: int = 3
    coupling: float = 1.0
    velocity_m_s: float = 6.0
    synaptic_ms: float = 5.0
    width_mm: float = 15.0
```

Streaming: each node keeps a history of `max_lag` samples. Normalisation is analytic: node series are
scaled by `1/√(1 + coupling²/fan_in)` to unit variance and the sensor projection by
`1/√(mean_c Σ_j map[j, c]²)`, so no calibration pass over the recording is needed.

### 4.3 Rhythm generator

A rhythm is a set of cortical patches under target electrodes that share a delayed driver plus some
independent activity. Mirror-pair electrodes get mirror-image patches (the feasibility test found
unmirrored random placement fires spurious asymmetry findings). Midline sites take the source nearest
the midline among the candidates under the electrode.

```python
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

Time course of the driver and of each patch's own component: instantaneous frequency is an OU process
around `f0` (σ `f_sd`, τ `f_tau_s`), the envelope is `exp` of an OU process around 0 (σ `env_sd`, τ
`env_tau_s`) clipped to `[env_lo, env_hi]`, phase is the cumulative sum of frequency. Both OU processes
use the exact discretisation `x[n] = μ + a (x[n−1] − μ) + σ√(1−a²) w[n]`, `a = exp(−1/(τ fs))`, and
start from their stationary distribution, so there is no warm-up transient. Each patch's own component
is centred at `f0 + N(0, 0.3 Hz)`, drawn once per patch when the rhythm is built — the small spread the
feasibility test used so that patches are not perfectly locked to one frequency.

Region placement (used by alpha): candidates are sources with `y` below its 12th percentile and `z`
above its 30th percentile (occipito-parietal); `n_patches/2` are drawn from the left half and mirrored.

**Amplitude convention.** `amp_uv` is the peak amplitude (√2 × RMS) at the loudest target channel when
the envelope is 1 and all patches are in phase: `scale = amp_uv / max_c |Σ_p map_p[c]|` over target
channels (for region placement: over the channels where the summed map is largest). The feasibility
test scaled empirically over the finished recording, so its numbers carry an envelope-mean factor of
about 1.1–1.5; M2 re-derives `amp_uv` so per-channel RMS and the alpha share of the spectrum match the
test's values.

**Tuned resting model** (`recipes.resting_brain()`), from the feasibility test:

| rhythm | placement | f0 | amp | other |
|---|---|---|---|---|
| alpha | region posterior, 4 patches, 10 mm | 10.0, jitter 0.6 | 20 (eyes closed) | lag 30 ms, indep 0.6, `state_gain {eyes_open: 0.3, drowsy: 0.35}`, `state_f0_shift_hz {drowsy: −1.0}` |
| theta | Fz F3 F4 Cz | 6.0 | 7 | env τ 0.8 s, env sd 0.7, lag 40 ms, `state_gain {drowsy: 2.0}` |
| beta | C3 C4 F3 F4 P3 P4 | 19.0 | 7 | f sd 1.2, f τ 0.6 s, env τ 0.3 s, env sd 0.9, env 0.1–3.0, lag 35 ms |
| smr | C3 C4 Cz | 13.5 | 6 | f sd 0.6, f τ 1.0 s, env τ 0.4 s, env sd 0.9, env 0.1–3.0, lag 20 ms |

Background 20 µV, exponent 1.2, smoothing 20 mm, network share 0.5; sensor noise 1.5 µV.

### 4.4 Planted-pattern primitives

Plants are generic signal terms. A consumer maps them to its own finding vocabulary; the package only
promises the signal and a truth record. Each plant compiles to extra `RhythmSpec`s and/or modifiers on
the base rhythms:

```python
class Plant(Protocol):
    kind: ClassVar[str]
    def rhythms(self) -> tuple[RhythmSpec, ...]
    def modifiers(self) -> tuple[Modifier, ...]        # Modifier(target_rhythm, amp_scale, f0_shift_hz, hemisphere_gain, extra_sites)
    def record(self) -> PlantRecord

@dataclass(frozen=True)
class PlantRecord:
    kind: str; sites: tuple[str, ...]; band_hz: tuple[float, float]; amp_uv: float | None
    onset_s: float; offset_s: float | None; params: dict; description: str   # generic prose
```

| primitive | signal | example (feasibility test) |
|---|---|---|
| `FocalSlow(site, f0_hz=2.5, amp_uv=45, width_mm=12, f_sd=0.4, env_tau_s=1.5, env_sd=0.6, indep=0.3, state_gain={})` | a slow rhythm from one patch under one site, own driver | focal delta under a left frontal-temporal site |
| `RhythmicBursts(sites, f0_hz=6.5, amp_uv=30, width_mm=15, burst_s=(1, 3), gap_s=(8, 25), state_gain={})` | a rhythm gated on and off in bursts (`BurstGate`, 0.3 s ramps) | frontal midline theta bursts |
| `LateralImbalance(rhythm, side, factor)` | multiplies the patches of one hemisphere of a base rhythm | alpha ×0.6 on the left |
| `WidespreadExcess(rhythm, factor, extra_sites=())` | scales a base rhythm, optionally adding patches under more sites | theta ×2.2 everywhere |
| `PeakShift(rhythm, shift_hz)` | moves a base rhythm's centre frequency | alpha −1.5 Hz |
| `ReducedRhythm(rhythm, factor)` | scales a base rhythm down | sensorimotor rhythm ×0.2 |

`state_gain` on the two rhythm-creating primitives is passed through to the `RhythmSpec` they compile to,
so a consumer can confine a plant to some states (`{"eyes_open": 0.0}` plants it eyes-closed only); the
`PlantRecord.params` carry it.

The feasibility test showed one plant can register as several findings in a consumer's catalogue,
sometimes at a neighbouring site — that mapping (plant → set of findings) lives in the consumer.

### 4.5 State timeline

```python
@dataclass(frozen=True)
class StateSegment: t0_s: float; t1_s: float; state: str      # "eyes_closed" | "eyes_open" | "drowsy" | any string

class StateTimeline:
    def __init__(self, segments: Sequence[StateSegment], ramp_s: float = 2.0)   # contiguous from 0; last t1_s may be inf
    def state_at(self, t_s: float) -> str
    def weights(self, t0: int, n: int, fs: float) -> dict[str, np.ndarray]     # per-state weights (n,), linear ramps across boundaries
    def gain(self, per_state: dict[str, float], t0: int, n: int, fs: float, default: float = 1.0) -> np.ndarray
    def rate(self, per_state: dict[str, float], t_s: float) -> float              # for artifact scheduling, no ramp
    @classmethod
    def constant(cls, state: str) -> "StateTimeline"
```

Rhythm amplitude is multiplied by `gain(state_gain, …)` per block and centre frequency shifted by
`state_f0_shift_hz`; artifact rates read `rate(rate_by_state, t)`. A drowsiness onset is therefore one
segment: `[StateSegment(0, 120, "eyes_closed"), StateSegment(120, 240, "drowsy")]` → alpha fades and
slows, theta rises, blinks stop, slow roving eye movements start. The timeline is written to the truth
file so a consumer can grade a "drowsy stretch" decision.

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
    kind: str                     # "blink", "eye_movement", "emg", …
    subtype: str | None           # "single"/"double", "saccade"/"slow_roving", "jaw", …
    side: str | None              # "left" | "right" | "both" | None
    channels: tuple[str, ...]     # where |pattern| ≥ 0.3 · max
    onset_s: float
    offset_s: float
    peak_uv: float
    remedies: tuple[Remedy, ...]  # preferred first
    layer: str                    # the layer name that holds this signal
    params: dict                  # everything drawn for this event, JSON-safe

@dataclass
class RenderContext:
    channels: tuple[str, ...]
    fs: float
    electrode_pos: np.ndarray | None      # (n_ch, 3) for the EEG channels
    head: HeadModel | None
    timeline: StateTimeline
    rng: np.random.Generator              # this plug-in instance's time stream
    subject_rng: np.random.Generator      # this plug-in kind's subject stream (pattern jitter)
    occupancy: Occupancy                  # shared busy spans for overlap avoidance

class Artifact(Protocol):
    kind: ClassVar[str]
    mode: ClassVar[str] = "additive"      # or "transform"
    def bind(self, ctx: RenderContext) -> None
    def render(self, t0: int, n: int) -> np.ndarray               # additive: (n_ch, n) µV
    def render_transform(self, t0: int, n: int, mix: np.ndarray) -> np.ndarray   # transform: returns the delta
    def truth(self) -> list[TruthRecord]
    def params(self) -> dict                                      # constructor params, JSON-safe
```

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
    block: np.ndarray             # (n_ch, L) µV
    truth: TruthRecord
    @classmethod
    def from_pattern(cls, onset, pattern: np.ndarray, waveform: np.ndarray, truth) -> "Event"

class EventArtifact(ABC):
    rate_by_state: dict[str, float]       # events per second per state; missing state → 0
    min_gap_s: float                      # refractory gap after an event ends
    exclusive: bool = False               # if True, never overlaps another exclusive artifact
    @abstractmethod
    def make_event(self, onset: int) -> Event
```

Scheduling is a thinned Poisson process: candidate onsets are drawn at the maximum rate over states and
accepted with probability `rate(state at t) / rate_max`; an accepted onset is pushed later if it falls
inside the refractory gap or (for `exclusive`) inside another exclusive event's span on the shared
`Occupancy`. Candidates are drawn in time order and each event's waveform is drawn at the moment it is
scheduled, so the sequence of random draws does not depend on how the timeline is chunked (§8).
Overlap of non-exclusive artifacts (a blink during a jaw clench) is allowed — that is realistic.

### 5.3 Transform artifacts

`TransformArtifact.render_transform(t0, n, mix)` receives the running sum of every layer rendered so
far and returns the delta to add. The engine stores that delta as the artifact's layer. Transform
artifacts run last (after sensor noise). A dead channel returns `−mix[ch] + tiny noise`; a bridged pair
returns the delta that moves both channels to their mean plus shared noise.

### 5.4 Registry and discovery

```python
ARTIFACTS: dict[str, type[Artifact]]
def register(cls) -> cls                    # decorator; key = cls.kind; duplicate kind raises
def make_artifact(kind: str, **params) -> Artifact
def discover() -> None                      # loads entry points in group "open_eeg_synth.artifacts"
```

Built-in plug-ins register on import of `open_eeg_synth.artifacts`. Third-party packages add
`[project.entry-points."open_eeg_synth.artifacts"] mykind = "mypkg.module:MyArtifact"`. A `CaseSpec`
refers to artifacts by kind and parameter dict (`ArtifactSpec(kind, params)`), which is what the truth
file stores.

### 5.5 Where scalp patterns come from

`artifacts/patterns.py` offers two sources, and each plug-in states which it uses:

```python
def empirical(name: str, channels: Sequence[str], *, rng: np.random.Generator | None = None,
              jitter_sd: float = 0.6, electrode_pos: np.ndarray | None = None) -> np.ndarray
def analytic_focal(centre_m, electrode_pos_m, sigma_mm: float) -> np.ndarray      # Gaussian in scalp distance, max 1
def analytic_dipole(pos_m, moment, electrode_pos_m) -> np.ndarray                  # (r̂·m)/r², max |1|
```

`empirical` reads `data/eog_patterns.npz` (§6), selects the requested channels by canonical name, and
applies a per-subject shape jitter `mean + z · sd` with one scalar `z ~ N(0, jitter_sd)` drawn from
the subject stream (so the same subject's blinks share a map across conditions). A channel absent from
the empirical file falls back to the analytic dipole model for that name (`blink`: vertical dipole at
the eyes; `heog`: lateral dipole at the eyes), so any channel set works.

### 5.6 First plug-ins

**Blink** (`kind = "blink"`, pattern `empirical("blink")`)
- `rate_by_state = {"eyes_open": 0.25, "eyes_closed": 0.0, "drowsy": 0.0}`, `min_gap_s = 0.4`.
- Waveform: duration `U(0.2, 0.4)` s; rise over the first 35 % as a raised cosine, then exponential
  decay (τ = 22 % of duration) tapered to zero; peak `lognormal(median 120 µV, σ 0.3)` clipped to
  60–250 µV at the loudest channel; with probability 0.15 a second blink follows after 150 ms at 0.8×
  (subtype `"double"`).
- Truth: channels with |pattern| ≥ 0.3·max (Fp1, Fp2, F7, F8, F3, F4 typically), remedies
  `(REMOVE_COMPONENT, MASK_SEGMENT)`.

**Horizontal eye movement** (`kind = "eye_movement"`, pattern `empirical("heog")`)
- `rate_by_state = {"eyes_open": 0.10, "eyes_closed": 0.02, "drowsy": 0.08}`, `min_gap_s = 0.5`.
- Eyes open → subtype `"saccade"`: a step with a 30 ms raised-cosine rise, hold `U(0.3, 2.0)` s, 40 ms
  return; amplitude `U(30, 100)` µV at F7/F8, sign random.
- Otherwise → subtype `"slow_roving"`: one to three cycles of a sinusoid at `U(0.2, 0.5)` Hz,
  amplitude `U(20, 60)` µV, sign random.
- Truth remedies `(REMOVE_COMPONENT, MASK_SEGMENT)`.

**Jaw-tension EMG** (`kind = "emg"`, subtype `"jaw"`, pattern analytic)
- `side` = `"left" | "right" | "both" | "random"` (random: 0.4/0.4/0.2).
- `rate_by_state = {"eyes_open": 1/60, "eyes_closed": 1/60, "drowsy": 1/120}`, `min_gap_s = 5`.
- Pattern: `analytic_focal` with σ = 35 mm centred 5 mm lateral, 10 mm anterior and 10 mm inferior of
  T3 (left) / T4 (right), which puts ≈1.0 at T3, ≈0.6 at F7, ≈0.5 at T5, ≈0.3 at C3. `"both"` uses two
  independent time courses.
- Waveform, generated at `oversample = 8` × fs like a real signal hitting an amplifier: white noise
  shaped in the frequency domain by `H(f) = (f/f_p) / (1 + (f/f_p)²)` with `f_p` = 80 Hz (broadband,
  rising above ~20 Hz, broad peak near 80 Hz, slow roll-off), multiplied by a raised-cosine envelope
  (100 ms rise/fall, plateau so the burst lasts `lognormal(median 1.2 s, σ 0.5)` clipped to 0.5–3 s),
  then anti-alias low-passed and decimated by 8 (`scipy.signal.resample_poly`), then scaled so the
  plateau RMS at the loudest channel is `lognormal(median 50 µV, σ 0.6)` clipped to 20–200 µV. At a
  256 Hz device the content above ~100 Hz is removed exactly as an amplifier would remove it, so
  nothing aliases into the beta band that was not there physically.
- Truth: `side`, channels with pattern ≥ 0.3, remedies `(MASK_SEGMENT, REMOVE_COMPONENT)`.

**Dead channel** (`kind = "dead_channel"`, mode transform) — ships in v0.2.0 only to prove the transform
hook: replaces one channel with 0.3 µV white noise for `[onset, offset)`; truth remedies
`(MARK_BAD_CHANNEL, INTERPOLATE)`.

### 5.7 Future plug-ins and how they fit

| plug-in | base | pattern source | signal | truth | remedies |
|---|---|---|---|---|---|
| electrode pop | `EventArtifact` | channel-local | step of 200–2000 µV with exponential recovery (τ 0.2–2 s), high-pass-filter-shaped tail | channel, onset, peak | `MASK_SEGMENT`, `MARK_BAD_CHANNEL` |
| loose lead | `ContinuousArtifact` (stateful `render`) | channel-local | low-frequency drift + pop bursts whose rate scales with `ContactTimeline.impedance(ch, t)`; the same timeline feeds a recording app's impedance display | channel, spans where impedance > threshold | `MARK_BAD_CHANNEL`, `INTERPOLATE`, `RAISE_HIGH_PASS` |
| bridged pair | `TransformArtifact` | two channels | both channels → their mean + shared 1 µV noise | pair | `MARK_BAD_CHANNEL` |
| dead channel | `TransformArtifact` | one channel | see §5.6 | channel | `MARK_BAD_CHANNEL`, `INTERPOLATE` |
| mains hum | `ContinuousArtifact` | per-channel gains (loose channels louder) | `freq_hz` **required** (50 or 60; no default), harmonics 2–3 at −20 dB, slow AM | channels, freq | `NOTCH` |
| sweat drift | `ContinuousArtifact` | frontal/temporal focal | 0.05–0.3 Hz OU wander, 50–500 µV | channels, spans | `RAISE_HIGH_PASS`, `MASK_SEGMENT` |
| movement | `EventArtifact` | analytic focal, random centre | 0.5–3 s low-frequency swing on many channels + EMG mix | channels, span | `MASK_SEGMENT` |
| heartbeat bleed | `ContinuousArtifact` | empirical or analytic left-lateral | `HeartSource` beat times → QRS-shaped 5–30 µV at each beat | channels | `REMOVE_COMPONENT` |
| pulse wave | `ContinuousArtifact` | channel-local (an electrode over a vessel) | `HeartSource` beats → smooth 20–100 µV wave lagging the beat ~250 ms | channel | `MARK_BAD_CHANNEL`, `REMOVE_COMPONENT` |
| forehead / neck muscle | `EventArtifact` | analytic focal at Fp1/Fp2 (frontalis) or O1/O2/T5/T6 (neck) | the jaw EMG generator with a different centre and spectrum peak (frontalis ~40 Hz) | side, channels | `MASK_SEGMENT`, `REMOVE_COMPONENT` |

Every row uses a base class and a pattern source that exist in v0.2.0; none needs a change to the
engine or the truth schema.

---

## 6. Empirical eye and muscle scalp patterns

### 6.1 The choice

The head model has no eyes and no scalp muscles, so eye and muscle artifacts cannot be projected
through it. Two ways to get their scalp maps:

| | empirical (ICA of public recordings) | analytic (dipole / Gaussian on electrode positions) |
|---|---|---|
| blink topography | the real thing: Fp maximum, ~40 % at F3/F4/Fz, small negative posteriorly under average reference; classified as "eye" by ICA-label tools trained on real data | plausible shape, but a point dipole near the eyes over-weights F7/F8 and misses the skull's smearing; ICA-label tools may not call it "eye" with confidence |
| horizontal eye movement | opposite signs at F7/F8 with the real AF/Fp gradient | a lateral dipole reproduces the sign flip; magnitude gradient approximate |
| muscle | subject-specific and numerous components; a subject-average is a blur | spatially local by nature; Gaussian on scalp distance is close to what a real temporalis burst looks like |
| channel sets | 64 channels in the source data cover every 10-10 label in the 19-, 31- and 37-channel sets | any label with a position |
| cost | a derivation script (dev-only MNE), a ~20 kB data file, an attribution notice | none |

**Recommendation: empirical for the two eye patterns, analytic for muscle**, with the analytic dipole
as the fallback for any channel the empirical file lacks. Blink and eye-movement maps are the ones
where a wrong shape would mis-train a reader and mislead component-labelling tools; muscle maps are
local enough that geometry does the job.

### 6.2 Source data

PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0 (Schalk 2009, doi 10.13026/C28G6P), 109 subjects,
64 electrodes of the 10-10 system (all 10-20 sites present under 10-10 names), 160 Hz; runs 1 and 2
are one-minute eyes-open and eyes-closed baselines. Licence: Open Data Commons Attribution License
v1.0 (https://physionet.org/content/eegmmidb/view-license/1.0.0/). Downloaded on demand with
`mne.datasets.eegbci.load_data`.

### 6.3 Derivation script: `scripts/derive_artifact_patterns.py`

For subjects 1–30, runs 1 and 2: read, `eegbci.standardize`, rename `T7/T8/P7/P8 → T3/T4/T5/T6`,
1–45 Hz band-pass, `standard_1020` montage, average reference, concatenate the two runs, fit
`ICA(n_components=30, method="fastica", random_state=0)`. Select per subject:

- **blink**: the component whose average-referenced topography correlates best with the prior
  `(Fp1, Fp2, AF3, AF4, AF7, AF8 = 1; F3, Fz, F4 = 0.4; rest 0)` and whose time course has excess
  kurtosis > 5 (blinks are sparse). Sign-align so the mean at Fp1/Fp2 is positive.
- **heog**: the component whose topography correlates best with the prior `(F7, AF7 = −1; F8, AF8 = +1;
  Fp1 = −0.5; Fp2 = +0.5)` and whose time course has less than 40 % of its power above 5 Hz (after the
  1 Hz high-pass a 20 % bound keeps only 4 of the first 20 subjects; 40 % keeps 17, and 25 of 30).
  Sign-align so `F8 − F7 > 0`.
- A subject with no component clearing both criteria is skipped and logged. If a subject cannot be
  downloaded the script stops at the subjects already on disk.

Normalise each map to max |1|, average across subjects, keep the per-channel standard deviation.

Outputs: `src/open_eeg_synth/artifacts/data/eog_patterns.npz` with `channel_names (64,)`, `blink_mean`,
`blink_sd`, `heog_mean`, `heog_sd` (all `(64,)` float32), `n_subjects`, `subjects_used`, `attribution`;
`scripts/output/eog_patterns_selection.json` listing the chosen component index and its scores per
subject (kept in the repo for reproducibility).

Fast tests on the shipped file: Fp1 and Fp2 are the two largest entries of `blink_mean`, every frontal
entry is positive, |O1|, |O2| < 0.25; `heog_mean` has `sign(F7) = −sign(F8)` and those two are the
largest magnitudes; `blink_sd` is below 0.35 everywhere and `heog_sd` below 0.45 (measured 0.22 and 0.42
over 30 subjects; the heog map varies more between people than the blink map does).

### 6.4 Attribution

`artifacts/data/NOTICE-eegmmidb.txt`, included in the wheel and embedded in the file's `attribution`
array:

> The scalp patterns in `eog_patterns.npz` are statistical summaries (subject-averaged, unit-normalised
> independent-component topographies) derived from the EEG Motor Movement/Imagery Dataset v1.0.0,
> made available on PhysioNet under the Open Data Commons Attribution License v1.0. No recordings are
> redistributed.
> Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet.
> https://doi.org/10.13026/C28G6P
> Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. BCI2000: A General-Purpose
> Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034–1043,
> 2004.
> Pollard, T., et al. PhysioNet as a global platform for biomedical research. Nature Health
> 1(8):792–795, 2026. https://doi.org/10.1038/s44360-026-00096-z

The same notice covers `tests/realism/reference/*.json` (§9), which are percentile summaries of the same
recordings.

---

## 7. Output modes and case files

### 7.1 Engine

```python
class Layer(Protocol):
    name: str
    def render(self, t0: int, n: int) -> np.ndarray      # (n_ch, n) float32 µV; t0 must continue the previous call
    def truth(self) -> list[TruthRecord]

@dataclass
class Frame:
    t0: int; n: int
    layers: dict[str, np.ndarray]                        # name → (n_ch, n)
    @property
    def mixed(self) -> np.ndarray                        # sum of layers, float32

class Engine:
    def __init__(self, channels: Sequence[str], fs: float, layers: Sequence[Layer], transforms: Sequence[Artifact] = ())
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
    truth: list[TruthRecord]
    plants: list[PlantRecord]
    timeline: StateTimeline
    @property
    def mixed(self) -> np.ndarray
    @property
    def duration_s(self) -> float
```

Layer names: `"brain"`, `"artifact:<kind>"` (`"artifact:<kind>#<i>"` when a kind appears more than
once), `"sensor"`, `"transform:<kind>"` (`"transform:<kind>#<i>"` likewise; the engine keys every
artifact layer, additive or transform, by the instance's own layer name). The identity `mixed == Σ layers` is a tested invariant.

### 7.2 Case specification and subject

A case is one synthetic subject recorded under one or more conditions. Everything that belongs to the
subject (perturbed head model, patch placements, centre-frequency draws, pattern jitters) is drawn
once and shared; everything that belongs to the passage of time (noise, envelopes, artifact events) is
drawn per condition.

```python
@dataclass(frozen=True)
class ArtifactSpec: kind: str; params: dict = field(default_factory=dict)

@dataclass(frozen=True)
class ConditionSpec:
    name: str                          # "eyes_closed", "eyes_open", … (also the truth/file key)
    duration_s: float
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
    sensor: SensorSpec = SensorSpec(white_uv=1.5)
    conditions: tuple[ConditionSpec, ...] = (eyes closed 240 s, eyes open 240 s)
    label: str = "synthetic"
    def to_dict(self) -> dict            # JSON-safe, round-trips through from_dict
    @classmethod
    def from_dict(cls, d) -> "CaseSpec"
    def digest(self) -> str              # blake2b of canonical JSON, 16 hex chars

@dataclass
class Subject:
    head: HeadModel                      # perturbed (or nominal)
    placements: dict[str, list[int]]     # rhythm name → patch centres
    f0_hz: dict[str, float]
    pattern_jitter: dict[str, float]

def make_subject(spec: CaseSpec) -> Subject
def make_engine(spec: CaseSpec, subject: Subject, condition: ConditionSpec) -> Engine
def make_case(spec: CaseSpec) -> Case

@dataclass
class Case:
    spec: CaseSpec
    case_id: str                         # "synth-" + first 8 hex of blake2b(seed, spec digest)
    subject: Subject
    recordings: dict[str, Recording]     # by condition name
```

`recipes.resting_case(seed, *, duration_s=240, plants=(), artifacts=None, drowsy_from_s=None)` builds
the common two-condition spec; `drowsy_from_s` inserts the drowsy segment into the eyes-closed timeline.

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
    def truth(self) -> list[TruthRecord]                           # events scheduled so far
    @property
    def seed(self) -> int
```

The constructor signature mirrors `classic.RealisticEEGSynthesizer(channel_labels, srate, *, seed,
markers)` so a recording application swaps one import. Labels are split into EEG (canonicalised,
routed to the head model subset), heart (`HR`, `ECG`, `EKG` → `HeartSource` ECG, ~200 µV R-wave, RR
from a mean of 72 bpm with respiratory sinus arrhythmia and a 0.1 Hz wave, the classic constants) and
anything else (`UnknownChannelError`). `seed=None` draws a fresh seed and exposes it. The default
timeline is a constant `"eyes_open"`; a recording application may pass its own scenario. Chunk size is
unlimited, chunks are phase-continuous, and — unlike `classic`, which drew blinks per chunk — the
samples do not depend on chunk size (§8). `due_markers` reuses `classic.MarkerSchedule` unchanged;
evoked responses to markers are not part of v0.2.0 (§11).

### 7.4 Case files

`casefile.writer.write_case(directory, case, *, embed_layers=False, physical_range_uv=(-2000, 2000))`
writes:

```
<case_id>_<condition>.edf          one EDF per condition, no annotations
<case_id>.truth                    sealed truth container
<case_id>.layers.npz               only with embed_layers=True
```

**EDF** (`casefile.edf.write_edf`, requires the `edf` extra, i.e. `edfio>=0.4`):
- one `EdfSignal` per channel, `physical_dimension="uV"`, `physical_range` in µV (default ±2000 µV,
  0.06 µV per bit at 16 bits — below the sensor noise floor, wide enough for pops), data clipped to
  the range, `transducer_type="synthetic"`, label = canonical channel name;
- `Patient(code="SYNTHETIC", name="X", additional=("synthetic",))`, `Recording(startdate=None,
  equipment_code="open-eeg-synth")`. `startdate=None` writes the EDF+ "unknown" value; readers that
  need a date substitute their own convention (MNE reads it as 1985-01-01) and, because the file
  carries no annotations, that substitute date has no side effects;
- **no annotations of any kind** — no condition markers, no artifact spans. Condition is carried by the
  file name and the truth file.

**Sealed truth** (`casefile.truth`): a binary container, magic `b"OESTRUTH\x01"` followed by
zlib-compressed canonical JSON. Sealing here means the truth is a separate, non-text file that a
consumer's user will not read by accident and that never rides inside the EDF; it is obfuscation, not
security (§11). Contents:

```json
{
  "format": "open-eeg-synth/truth", "format_version": 1,
  "generator": {"package": "open-eeg-synth", "version": "0.2.0", "signal_version": 2, "numpy": "…", "scipy": "…"},
  "case_id": "synth-3f9a1c2b", "seed": 1234567, "label": "synthetic",
  "spec": { …CaseSpec.to_dict()… },
  "subject": {"head_perturbation": {…}, "placements": {…}, "f0_hz": {…}, "pattern_jitter": {…}},
  "recordings": {
    "eyes_closed": {
      "file": "synth-3f9a1c2b_eyes_closed.edf", "fs": 256, "duration_s": 240, "channels": […],
      "layers": ["brain", "artifact:blink", "artifact:eye_movement", "artifact:emg", "sensor"],
      "layer_rms_uv": {"brain": [19 values], …},
      "timeline": [{"t0_s": 0, "t1_s": 240, "state": "eyes_closed"}],
      "truth": [ {TruthRecord…}, … ],
      "plants": [ {PlantRecord…}, … ]
    },
    "eyes_open": { … }
  }
}
```

Layers are not embedded by default: they are re-rendered on demand by
`casefile.writer.render_layers(truth_dict, condition) -> Recording`, which rebuilds the case from
`spec` and `seed` and checks each layer's per-channel RMS against `layer_rms_uv` (relative tolerance
1e-3) before returning. If the package version differs from `generator.version`, it warns; if the
check fails, it raises `LayerMismatchError` so a consumer never grades against the wrong layers.
`embed_layers=True` writes `<case_id>.layers.npz` (float32, ~9 MB per layer per 240 s condition) for
archival or for consumers that cannot install the package version that made the case.

### 7.5 Command line

`python -m open_eeg_synth make-case --seed 42 --out ./cases [--duration 240] [--spec spec.json]
[--embed-layers]` — builds `recipes.resting_case` (or the given spec JSON), writes the files, prints
the truth as JSON to stdout with `--print-truth`. Requires the `edf` extra.

---

## 8. Determinism, seeding, versioning, performance

### 8.1 One seed, named streams

`seeds.stream_rng(case_seed, name)` returns a `numpy.random.Generator(PCG64)` seeded from
`SeedSequence([case_seed, blake2b(name)])`. Every random draw in the engine comes from a named stream:

| stream name | owner |
|---|---|
| `subject:head` | head-model perturbation |
| `subject:rhythm:<name>` | patch placement, centre-frequency jitter |
| `subject:network` | network node placement and wiring |
| `subject:artifact:<kind>` | pattern jitter |
| `<condition>:background`, `<condition>:network`, `<condition>:rhythm:<name>` | time courses |
| `<condition>:artifact:<kind>:<i>` | scheduling and event waveforms |
| `<condition>:sensor` | sensor noise |
| `<condition>:heart` | RR intervals |

Adding a plug-in or a rhythm therefore never changes the draws of any other component, and two
conditions of one case share a subject by construction.

### 8.2 Chunk invariance

`Engine.render_all(N)` equals the concatenation of `render(t0, n)` calls for any partition of `[0, N)`
(tested with random partitions, `allclose` at 1e-4 µV). Three rules make this hold:

1. Noise is always drawn time-major — `rng.standard_normal((n, n_series)).T` — so a longer block
   consumes exactly the draws that two shorter blocks would.
2. Every filter carries state across calls (`scipy.signal.lfilter` with `zi`, node histories, driver
   histories, OU states); nothing is computed over "the whole recording".
3. Event artifacts draw candidate onsets and waveforms in onset order, at the moment the block
   containing the onset is requested (§5.2).

### 8.3 Versioning

`version.__version__` is the package version. `version.SIGNAL_VERSION` (integer) is bumped whenever the
same seed would produce different samples; a fingerprint test (`tests/test_golden_engine.py`: per-channel
RMS and the first 32 samples of Fz for `recipes.resting_case(seed=20260916)`, 10 s, to 1e-2 µV) fails
when the signal changes without a bump. Both numbers are written into every truth file. The `classic`
subpackage keeps its own bit-exact golden test against the recording application's original output.

### 8.4 Performance budget

Target: a two-condition, 2 × 240 s, 19-channel, 256 Hz case with the default artifacts renders in
≤ 6 s on a 2020 laptop, excluding file writes; `tests/test_perf.py` (marked `slow`) fails above 15 s.
Cost centres and why they fit: the smoothing matrix (~0.3 s, once per subject); ~80 filtered noise
series × 123 k samples (well under a second in `lfilter`); a 19×40 network projection per block;
~60 blink events and ~8 EMG bursts at 8× oversampling (milliseconds). The feasibility test's
per-sample Python loops for OU processes are replaced by `lfilter` from the start.

---

## 9. Tests

### 9.1 Realism suite (`tests/realism/`, marker `realism`, opt-in)

**What is measured** (`measure.py`, using MNE and mne-connectivity, both development extras — no
consumer code):
- per band (Delta 1–4, Theta 4–8, Alpha 8–13, Beta 13–30 Hz), per reference (average; longitudinal
  bipolar double-banana plus midline pairs; Laplacian via MNE's current-source-density): mean
  magnitude-squared coherence and mean debiased wPLI over all pairs (`spectral_connectivity_epochs`,
  methods `coh` and `wpli2_debiased`, band-averaged), and mean coherence in five inter-electrode
  distance bins (< 60, 60–90, 90–120, 120–150, > 150 mm; pair position = midpoint for bipolar pairs);
- aperiodic exponent of the median spectrum over 2–40 Hz excluding 7–14 Hz; alpha share of 1–40 Hz
  power at O1, Fz, Cz; per-channel RMS after 1–45 Hz filtering and average reference.
- Preparation: 1–45 Hz band-pass, `standard_1020` montage, average reference, 4 s epochs, epochs with
  peak-to-peak > 800 µV dropped, at least 8 epochs.
- Method, exactly as the feasibility test ran it: `spectral_connectivity_epochs(epochs,
  method=["wpli2_debiased", "coh"], fmin=[1, 4, 8, 13], fmax=[4, 8, 13, 30], faverage=True)` on the 4 s
  fixed-length epochs, dense output read as `(M + Mᵀ)/2` over the upper triangle. mne-connectivity fills
  one triangle, so every coherence and dwPLI figure in this section is one half of the magnitude coherence
  / debiased wPLI — the same reading the consuming application's electrode-connectivity code makes. The
  reference file and the synthetic side share this convention (the rebuilt reference reproduces the
  feasibility test's percentiles to 0.002); do not square or re-read the triangle on one side only.

**Reference**: PhysioNet eegmmidb subjects 1–20, runs 1 (eyes open) and 2 (eyes closed), blink
components removed by ICA (fastica, 15 components, `find_bads_eog` on Fp1/Fp2 at threshold 3).
`scripts/build_realism_reference.py` downloads on demand (`mne.datasets.eegbci`), measures, and writes
`tests/realism/reference/eegmmidb_baselines_s01-20.json` with p10/p50/p90 of every metric per condition
plus the attribution of §6.4. The JSON is committed, so the suite itself needs no download; only
regenerating the reference does.

**Synthetic side**: `recipes.resting_case` at six seeds, 60 s per condition, measured identically
(without ICA), medians compared with the reference bands.

**Pass bands** (10th–90th percentile of the 20 real subjects; the feasibility test's tuned model in
parentheses for eyes closed / eyes open):

| reference / band | mean coherence EC | EO | dwPLI EC | EO |
|---|---|---|---|---|
| average / Delta | 0.13–0.21 (0.17) | 0.12–0.37 (0.17) | 0.005–0.067 (0.006) | 0.004–0.264 (0.006) |
| average / Theta | 0.16–0.20 (0.18) | 0.15–0.28 (0.17) | 0.026–0.102 (0.035) | 0.041–0.107 (0.029) |
| average / Alpha | 0.18–0.29 (0.26) | 0.15–0.23 (0.19) | 0.067–0.191 (0.125) | 0.029–0.128 (0.052) |
| average / Beta | 0.15–0.20 (0.17) | 0.13–0.19 (0.18) | 0.060–0.120 (0.062) | 0.049–0.117 (0.063) |
| bipolar / Delta | 0.09–0.17 (0.11) | 0.09–0.26 (0.11) | 0.004–0.040 (0.006) | 0.001–0.185 (0.005) |
| bipolar / Theta | 0.11–0.15 (0.12) | 0.10–0.19 (0.12) | 0.021–0.075 (0.026) | 0.026–0.108 (0.024) |
| bipolar / Alpha | 0.14–0.24 (0.20) | 0.11–0.16 (0.14) | 0.072–0.174 (0.090) | 0.020–0.129 (0.038) |
| bipolar / Beta | 0.11–0.14 (0.12) | 0.10–0.14 (0.12) | 0.063–0.121 (0.057) | 0.032–0.130 (0.059) |
| laplacian / Delta | 0.11–0.14 (0.13) | 0.11–0.22 (0.13) | 0.003–0.046 (0.007) | 0.000–0.243 (0.005) |
| laplacian / Theta | 0.11–0.14 (**0.155**) | 0.12–0.16 (0.15) | 0.023–0.079 (0.046) | 0.017–0.162 (0.042) |
| laplacian / Alpha | 0.13–0.19 (**0.223**) | 0.12–0.16 (0.16) | 0.072–0.193 (0.079) | 0.031–0.135 (0.036) |
| laplacian / Beta | 0.14–0.17 (0.16) | 0.13–0.18 (0.17) | 0.070–0.133 (0.087) | 0.035–0.133 (0.090) |

Other metrics: aperiodic exponent 0.8–1.4 (real p10–p90 0.62–1.37 EC, 0.43–1.39 EO; tuned 1.25 / 1.20);
alpha share at O1 0.35–0.79 EC (tuned 0.64) and 0.05–0.30 EO (0.21); RMS at Cz 8–20 µV (real medians
13.7 / 12.7; tuned 11.9 / 11.7). Coherence-by-distance bins: inside the real p10–p90 per bin with a
±0.03 allowance (the six-seed median is noisier than a twenty-subject one).

Acceptance for v0.2.0: every average and bipolar metric inside its band; every Laplacian metric inside
`[p10 − 0.02, p90 + 0.04]`. The two bold values are the feasibility test's known gap (Laplacian alpha and
theta slightly too coherent at long distance); M2 tries to close it (§11) and the tolerance is
tightened when it does.

**Known limits of the reference**: 160 Hz source data (nothing above 80 Hz is compared), one minute per
condition per subject, healthy young adults, 2009-era amplifiers.

### 9.2 Fast tests (`pytest`, < 60 s, run in CI on every push)

Per module: seeds (determinism, name independence), dsp (PSD slope, OU stationary sd and mean, chunk
invariance of every filter, decimation removes content above 0.45·fs), head model (file contents,
fixed = free·normal, subsets, aliases, mirror symmetry, perturbation magnitude and determinism),
background (RMS, slope), network (measurable lag between coupled nodes), rhythm (spectral peak at f0
at the target sites, mirror symmetry of maps, non-zero lag between sites), timeline (weights sum to
one, ramps), plants (each primitive changes band power at its sites by the expected factor), artifacts
(scheduling rates by state, refractory gap, exclusivity, chunk invariance, truth record contents, each
plug-in's waveform and spectral properties, no aliasing in jaw EMG), engine (sum identity, contiguity
check, chunk invariance), case (subject shared across conditions, JSON round trip), case files (EDF
round trip in µV, no annotations, anonymised date; truth container round trip; `render_layers`
verification), stream (the recording application's contract: shape, dtype, RMS range, determinism,
1/f slope, posterior-dominant alpha, ECG only on heart labels, arbitrary label sets at 250/256/500 Hz,
markers), golden fingerprint, `classic` golden test.

---

## 10. Packaging and consumption

`pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
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
```

- **scipy at runtime** is a deliberate deviation from the earlier "numpy only" proposal: streaming
  IIR filters with carried state and polyphase anti-alias decimation are exactly what `scipy.signal`
  does robustly, both consumers already ship scipy, and the three call sites are isolated in `dsp.py`.
  The `classic` subpackage stays numpy-only.
- **Install by git tag**: `open-eeg-synth @ git+https://github.com/peak-mind-llc/open-eeg-synth.git@v0.2.0`
  (with `[tool.hatch.metadata] allow-direct-references = true` in the consumer if it builds with
  Hatchling). Optional extras are selected as `open-eeg-synth[edf] @ git+…`.
- **Data in the wheel**: `headmodel/data/colin27_19ch.npz`, `headmodel/data/NOTICE-colin27.txt`,
  `artifacts/data/eog_patterns.npz`, `artifacts/data/NOTICE-eegmmidb.txt`. They are read with
  `importlib.resources.files("open_eeg_synth.headmodel") / "data" / …`, which works from a wheel, an
  editable install and a frozen bundle.
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

Each with the recommendation the plan adopts.

1. **Is the head-model perturbation strong enough to stop a consumer's inverse from localising planted
   sources "too well"?** Unknown until measured in a consuming application. Recommendation: ship the
   three-step perturbation with the defaults above, expose the knobs, and have the consumer compare
   residual variance of dipole fits on synthetic versus real cases; raise `tilt_deg`/`gain_sd` if
   synthetic fits are still implausibly good.
2. **Laplacian alpha and theta coherence at long distance are slightly above the real band.**
   Recommendation: in M2, try `indep` 0.6 → 0.75 for alpha and `n_patches` 4 → 6, and network patch
   width 15 → 12 mm; accept the looser Laplacian tolerance for v0.2.0 if two hours of tuning do not
   close it, and record the gap in the README.
3. **Store layers or re-render them?** Recommendation: re-render from the seed with the RMS check
   (§7.4) as the default; `embed_layers=True` for archival. Revisit if consumers cannot keep package
   versions aligned with stored cases.
4. **How sealed is "sealed"?** Recommendation: compressed binary container with a magic header, no
   encryption. It stops accidental reading, not a determined one, which matches the consumers' own
   answer-key precedent. Anything stronger needs a key-management story that does not belong here.
5. **Physical range of the EDF.** ±2000 µV keeps electrode pops and large EMG unclipped at 0.06 µV
   resolution. Consumers that expect ±500 µV files should say so; the parameter is exposed.
6. **Amplitude convention.** The analytic convention (§4.3) differs from the feasibility test's
   empirical one by an envelope-mean factor. Recommendation: keep the analytic definition (it is the
   only one a streaming generator can honour) and re-derive the four resting amplitudes in M2.
7. **Evoked responses in streaming mode** (a recording application's oddball mock injects a P300 at
   markers). Recommendation: v0.2.0 ships markers only; an `EvokedResponse` brain primitive (marker
   time → patch response) is a v0.3 item, at which point the standalone oddball mock can move over.
   `classic.oddball.ErpInjector` (seeded, carries a response across chunk boundaries) is the starting
   point for that primitive.
8. **Drowsiness beyond rhythm gains** (vertex sharp transients, sleep spindles, K-complexes).
   Recommendation: out of scope for v0.2.0; they are `EventArtifact`-shaped brain primitives and can
   be added without engine changes.
9. **Larger channel sets.** 31- and 37-channel exports are the same script with a different input; the
   empirical eye maps already cover their labels. Recommendation: export when a consumer needs them.
10. **Should the recording application keep `classic` selectable after switching?** Recommendation:
    yes for one release behind an environment variable, then delete the switch; `classic` itself
    stays as the golden-tested baseline and for anyone who wants the old look.
11. **Mains frequency default.** None — `MainsHum(freq_hz=…)` is required. Consumers pass their
    configured value.
12. **Python 3.13.** Nothing in the package prevents it; the ceiling `<3.13` follows the consumers'
    pins and lifts when they do.
