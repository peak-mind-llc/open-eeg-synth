# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## 0.2.0 (unreleased)

### Added
- A layered engine. `Engine.render(t0, n)` is the one signal path; a recording is the sum of named
  layers (brain, one layer per artifact, sensor noise, transforms), and every layer stays
  retrievable. One case seed derives every random stream by name, and chunking does not change the
  samples.
- A head model file, `headmodel/data/colin27_19ch.npz`: the Colin27 template's free-orientation lead
  field for the 19 channels of the 10-20 system, with a per-case perturbation (source tilt, channel
  gains, scalp blur) so that a synthetic case is not an exact forward/inverse pair.
- A brain layer: smoothed 1/f background, a cortical network with conduction delays, and mirrored
  rhythm generators (alpha, theta, beta, sensorimotor rhythm) in a resting recipe tuned against public
  EEG; planted patterns (`FocalSlow`, `RhythmicBursts`, `LateralImbalance`, `WidespreadExcess`,
  `PeakShift`, `ReducedRhythm`) with truth records; and a state timeline (eyes closed, eyes open,
  drowsy) that drives rhythm gains and artifact rates.
- An artifact plug-in framework (event and transform artifacts, a registry with entry-point
  discovery, empirical and analytic scalp patterns) and four plug-ins: blink, horizontal eye
  movement, jaw EMG and dead channel. Every event writes a truth record with its channels, times,
  size and suggested remedies.
- Case files: `make_case` and `write_case` write one EDF per condition (microvolts, no annotations,
  anonymised date), a sealed truth file and, optionally, every layer; `render_layers` rebuilds a
  condition's layers from the truth file and checks them. `python -m open_eeg_synth make-case` does
  the same from the command line.
- `StreamSource`: chunked, seamless output for a recording application's mock amplifiers, with the
  classic synthesizer's constructor, an ECG on heart-rate labels, event markers and the artifact truth
  so far.
- A realism suite (`pytest -m realism`) that compares the resting recipe with committed percentiles
  from PhysioNet recordings measured the same way, with two named known gaps.
- `SIGNAL_VERSION` and a signal fingerprint test; both version numbers are written into every truth
  file.
- Data files and their notices: the head model (`NOTICE-colin27.txt`), and the eye-movement and blink
  patterns and realism reference derived from the PhysioNet EEG Motor Movement/Imagery Dataset
  (`NOTICE-eegmmidb.txt`).
- Development scripts (need MNE): `export_head_model.py`, `derive_artifact_patterns.py`,
  `build_realism_reference.py`, and `update_golden.py` for deliberate signal changes.

### Changed
- scipy is now a runtime dependency (`scipy>=1.10`), for streaming filters and anti-alias
  decimation. The `classic` engine's code is unchanged and still uses only numpy, but importing it
  through the package now also loads scipy.
- The version module is `open_eeg_synth.version` (was `_version`); `open_eeg_synth.__version__` is
  still exported, alongside the new public names.
- New `realism` extra (MNE, mne-connectivity); the `dev` extra now includes it and `edfio`.

### Known gaps
- Listed in the README: one Laplacian theta line, the out-of-sample realism pass rate, thin
  frontal-theta headroom, small left/right biases, and eye-pattern values on non-template electrode
  layouts.

## [0.1.0] - 2026-09-16

### Added
- `open_eeg_synth.classic`: the streaming EEG synthesizer, marker schedule and
  cardiac/respiration mocks moved from Coherence Recorder. Output is
  sample-identical to the originals; `tests/classic/test_golden.py` checks it
  against a recorded fixture.
- `open_eeg_synth.classic.oddball`: the oddball paradigm schedule and simulated
  event-related responses from Coherence Recorder's standalone mock ERP stream,
  now seeded, and with responses carried across chunk boundaries. The original
  script never added its responses because each one began after the end of the
  chunk that held its stimulus.
- Project scaffolding: Apache-2.0 license, notice, contribution, governance,
  security and conduct files, and CI.
