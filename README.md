# open-eeg-synth

Realistic, seeded synthetic EEG for teaching, demos and software testing.

## What it is

`open-eeg-synth` is a deterministic generator of synthetic scalp EEG with known truth. It produces
either a continuous stream of chunks, for a recording application's mock amplifiers, or whole
recordings written to disk with a sealed truth file, for a QEEG analysis application's practice
cases. It is not a model of pathology and does not claim clinical realism beyond the measured
properties listed under [Realism](#realism); every file it writes is labelled synthetic.

A recording is the sum of separate layers, and every layer can be rebuilt from the seed:

- **brain**: 1/f background noise, a network of cortical patches with conduction delays, and alpha,
  theta, beta and sensorimotor rhythms, all generated as cortical sources and projected through a
  real head model (Colin27, 19 channels), plus any patterns you plant (a focal slow rhythm, rhythmic
  bursts, a one-sided reduction, a shifted alpha peak, and so on);
- **artifacts**: blinks, horizontal eye movements, jaw-muscle bursts and a dead channel, as plug-ins;
  other packages can add more;
- **sensor**: amplifier noise.

The truth file lists every artifact event (when, where, how big, and how to clean it), every planted
pattern, the subject's random draws and the state timeline (eyes closed, eyes open, drowsy). The
design, with the reasons behind each choice, is in [`docs/DESIGN.md`](docs/DESIGN.md).

## Install

```bash
pip install "open-eeg-synth[edf] @ git+https://github.com/peak-mind-llc/open-eeg-synth.git@v0.2.0"
```

The runtime needs only numpy and scipy. The `edf` extra adds `edfio`, which writing case files
needs; leave it out if you only stream. Python 3.10 to 3.12.

## Quick start

Make a practice case and write it to disk:

```python
from open_eeg_synth import make_case, resting_case, write_case
from open_eeg_synth.brain.plants import FocalSlow

# One synthetic subject: 240 s eyes closed, then 240 s eyes open,
# with a 2.5 Hz rhythm planted under F7.
case = make_case(resting_case(42, plants=[FocalSlow("F7")]))
paths = write_case("cases", case)
print(paths.truth)
```

```text
cases/synth-6689f87b.truth
```

The `cases` directory now holds `synth-6689f87b_eyes_closed.edf`, `synth-6689f87b_eyes_open.edf`
and the sealed truth file. The EDF files carry no annotations, so nothing in them gives the answers
away. To grade, read the truth back and rebuild the layers:

```python
from open_eeg_synth.casefile.writer import read_case_truth, render_layers

truth = read_case_truth(paths.truth)
eyes_open = truth["recordings"]["eyes_open"]
first = eyes_open["truth"][0]
print(len(eyes_open["truth"]), "events; first:", first["kind"], first["onset_s"], first["channels"])
print(eyes_open["plants"][0]["description"])

recording = render_layers(truth, "eyes_open")  # re-rendered, checked against the truth file
print(sorted(recording.layers))
```

```text
101 events; first: blink 6.7265625 ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8']
a 2.5 Hz rhythm of about 45 uV from one cortical patch under F7, present throughout
['artifact:blink', 'artifact:emg', 'artifact:eye_movement', 'brain', 'sensor']
```

Rebuild layers with the package version that made the case: `render_layers` warns when the versions
differ, and raises `LayerMismatchError` if the rebuilt layers do not match the RMS values stored in
the truth file.

Stream chunks, for example from a mock amplifier:

```python
from open_eeg_synth import StreamSource

source = StreamSource(["Fp1", "Fp2", "C3", "C4", "O1", "O2", "HR"], 256.0, seed=7)
chunk = source.next_chunk(32)  # float32 microvolts, shape (7, 32); the HR row carries an ECG
```

Chunks join without seams, and the samples do not depend on the chunk size. Building a source takes
about half a second; each 32-sample chunk then takes about half a millisecond. Heart-rate labels
(`HR`, `ECG`, `EKG`) carry the ECG. A label outside the 19 channels of the 10-20 system, such as a
10-10 site (`Fpz`, `Oz`) or an ear reference (`A1`), is accepted but not modelled in 0.2.0: its row
carries sensor noise only, the source warns once, and `source.unmodelled_labels` lists those labels. `source.truth` lists
the artifact events so far, and `source.due_markers(n)` returns event markers in step with the
chunks.

Or write a resting case from the command line:

```bash
python -m open_eeg_synth make-case --seed 42 --out cases
```

It prints the path of the truth file. `--duration` sets the length of each condition in seconds
(default 240), `--spec` reads a full case specification from JSON, `--embed-layers` also writes every
layer to an `.npz` file, and `--print-truth` prints the truth as JSON instead of the path.

## Data and attribution

The package ships two data files. Their notices travel with them inside the package.

The head model (`headmodel/data/colin27_19ch.npz`), from `NOTICE-colin27.txt`:

```text
Colin27 head model redistributed by EEGLAB (head_modelColin27_5003_Standard-10-5-Cap339.mat); the
lead field bundled in open-eeg-synth was computed from its surfaces and electrode cap with MNE-Python
and is redistributed under the notice below.

Copyright (C) 1993-2009 Louis Collins, McConnell Brain Imaging Centre, Montreal Neurological
Institute, McGill University. Permission to use, copy, modify, and distribute this software and its
documentation for any purpose and without fee is hereby granted, provided that the above copyright
notice appear in all copies. The authors and McGill University make no representations about the
suitability of this software for any purpose. It is provided "as is" without express or implied
warranty. The authors are not responsible for any data loss, equipment damage, property loss, or
injury to subjects or patients resulting from the use or misuse of this software package.
```

The blink and eye-movement scalp patterns (`artifacts/data/eog_patterns.npz`) and the realism
reference (`tests/realism/reference/`), from `NOTICE-eegmmidb.txt`:

```text
The scalp patterns in eog_patterns.npz are statistical summaries (subject-averaged, unit-normalised
independent-component topographies) derived from the EEG Motor Movement/Imagery Dataset v1.0.0,
made available on PhysioNet under the Open Data Commons Attribution License v1.0
(https://physionet.org/content/eegmmidb/view-license/1.0.0/). No recordings are redistributed.
The same notice covers the percentile summaries in tests/realism/reference/, which are statistical
summaries of the same recordings and likewise contain no recordings.

Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet.
https://doi.org/10.13026/C28G6P
Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. BCI2000: A General-Purpose
Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034-1043,
2004.
Pollard, T., et al. PhysioNet as a global platform for biomedical research. Nature Health
1(8):792-795, 2026. https://doi.org/10.1038/s44360-026-00096-z
```

## Realism

The realism suite measures the resting recipe the same way it measured 20 real subjects from the
PhysioNet EEG Motor Movement/Imagery Dataset: coherence and debiased weighted phase-lag index (dwPLI)
per band under average, bipolar and Laplacian references, coherence by electrode distance, the 1/f
exponent, the alpha share at O1 and the RMS at Cz. The real recordings had their blink components
removed first; the synthetic side is the median of 24 subjects without artifacts (seeds 100–123,
60 s per condition).

Average reference, real 10th–90th percentile with the synthetic median in parentheses:

| band | coherence, eyes closed | eyes open | dwPLI, eyes closed | eyes open |
|---|---|---|---|---|
| Delta 1–4 Hz | 0.260–0.424 (0.351) | 0.236–0.732 (0.351) | 0.009–0.135 (0.034) | 0.008–0.529 (0.040) |
| Theta 4–8 Hz | 0.328–0.407 (0.375) | 0.295–0.558 (0.366) | 0.051–0.205 (0.101) | 0.083–0.214 (0.083) |
| Alpha 8–13 Hz | 0.365–0.570 (0.505) | 0.291–0.451 (0.397) | 0.134–0.383 (0.176) | 0.060–0.257 (0.115) |
| Beta 13–30 Hz | 0.300–0.398 (0.367) | 0.266–0.376 (0.371) | 0.119–0.240 (0.191) | 0.098–0.233 (0.193) |

The bipolar and Laplacian rows, the distance bins, the tolerances and the other metrics are in
[`docs/DESIGN.md` §9.1](docs/DESIGN.md#91-realism-suite-testsrealism-marker-realism-opt-in). Read
the next section before relying on these numbers.

## Known gaps

- **One Laplacian line fails.** With eyes closed, Laplacian theta coherence between neighbouring
  electrodes is too high: 0.536 against a limit of 0.513. The frontal theta sources are kept small and
  close in timing so that frontal-midline theta, and a planted theta asymmetry, stay visible to a
  person reading the recording. Broader, later theta passed this line but hid those features. The
  realism test names this line as a known gap.
- **The realism test passes on seeds the recipe was tuned on.** Seeds 100–123 were part of the
  tuning, so the pass is in-sample. On fresh seeds, only 26 % to 49 % of random 24-subject sets pass
  every line, even with both named gaps excused. The second named gap, eyes-open bipolar beta
  coherence between neighbouring electrodes, fails at the population median of fresh seeds (0.310
  against a limit of 0.317) although it passes on the test seeds, so every realism run warns that it
  passes. Eyes-closed Laplacian beta and eyes-open theta dwPLI sit at the edge of their bands.
- **Frontal-midline theta dominance has little headroom.** Fz or Cz is the loudest eyes-open theta
  channel in 65 % of subjects, against the 60 % the tuning required. A louder or more focal cortical
  network pushes the temporal channels to the top.
- **Small left/right biases.** With eyes closed, median alpha power is 0.7 dB lower at O2 than at O1
  and 0.6 dB lower at P4 than at P3, and 12–15 Hz power is 0.4 dB lower at C4 than at C3. Check these
  against your asymmetry thresholds before treating resting cases as symmetric.
- **Eye patterns away from the template.** Channels outside the 64-channel pattern file get their
  blink and eye-movement values from a dipole model fitted to nearby channels. On electrode layouts
  far from the template head, some of those values are off by more than a factor of three. The
  19-channel head model never needs this, because all its channels are in the file.
- **No evoked responses yet.** Streams can emit event markers but add no response to them.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest              # the fast tests: about 65 s today, against a 60 s target
pytest -m slow      # the performance budget and two slower checks
pytest -m realism   # the realism suite, about 20 s; needs the dev extra (MNE)
```

The realism suite reads committed reference numbers, so it needs no download. The development
scripts need MNE and are rerun only when their inputs change:

| script | rerun when |
|---|---|
| `scripts/export_head_model.py --forward <forward.fif> --name colin27_19ch` | the head model changes (another channel set or head) |
| `scripts/derive_artifact_patterns.py --subjects 30 --data-path <dir>` | the eye-pattern derivation changes; downloads PhysioNet data on first run |
| `scripts/build_realism_reference.py --subjects 20 --data-path <dir>` | `tests/realism/measure.py` changes, because real and synthetic data must be measured the same way |
| `scripts/update_golden.py` | you change generated samples on purpose, in the same commit as a `SIGNAL_VERSION` bump |
| `tools/make_classic_golden.py` | the classic engine's output is meant to change (rare) |

See [CONTRIBUTING.md](CONTRIBUTING.md) for the ground rules.

## Versioning

`open_eeg_synth.__version__` is the package version. `open_eeg_synth.SIGNAL_VERSION` is an integer
that changes whenever the same seed would produce different samples. Both are written into every
truth file, and `render_layers` warns when a file was made by a different version. A fingerprint test
fails if the samples change without a `SIGNAL_VERSION` bump. `SIGNAL_VERSION` 2 is the first released
layered signal (v0.2.0).

## The classic engine

`open_eeg_synth.classic` is the synthesizer a recording application originally used for its mock
devices, moved here unchanged in v0.1. A golden test keeps its output sample-identical. It has no head
model: its channels share one noise trace, it has one rhythm (alpha) and one artifact (blinks), and its
blinks are drawn per chunk. New code should use `StreamSource`, which takes the same constructor
arguments and more.

```python
from open_eeg_synth.classic import MarkerSchedule, RealisticEEGSynthesizer

labels = "Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 Fz Cz Pz HR".split()
synth = RealisticEEGSynthesizer(
    labels, 256.0, seed=42, markers=MarkerSchedule(kind="oddball", period_s=1.5)
)

chunk = synth.next_chunk(32)  # float32, shape (20, 32), microvolts
markers = synth.due_markers(32)  # [(onset_seconds, label), ...] in lockstep
```

## Not clinical data

Generated recordings are synthetic. They are meant for teaching, demonstrations and testing
software, and must never be presented as a recording of a real person or used to make decisions
about one.

## License and governance

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Contributions follow
[CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md); see
[GOVERNANCE.md](GOVERNANCE.md) for how decisions are made and [SECURITY.md](SECURITY.md) for
reporting vulnerabilities.
