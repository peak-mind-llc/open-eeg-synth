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
  `PeakShift`, `ReducedRhythm`) with truth records (a `RhythmicBursts` record lists the bursts it
  produced, `params["bursts_s"]`); and a state timeline (eyes closed, eyes open, drowsy) that drives
  rhythm gains and artifact rates. Plant and rhythm sites are canonicalised
  against the head model's channel names (`FocalSlow("f7") == FocalSlow("F7")`), and a site named
  twice raises.
- An artifact plug-in framework (event, continuous and transform artifacts, a registry with
  entry-point discovery, empirical and analytic scalp patterns) and nine plug-ins: blink,
  horizontal eye movement, jaw EMG, electrode pop, contact noise, sweat drift, persistent noisy
  channel, mains hum and dead channel. Every artifact writes truth with its channels, times, size,
  mechanism parameters and suggested remedies.
- Case files: `make_case` and `write_case` write one EDF per condition (microvolts, no annotations,
  anonymised date), a sealed truth file and, optionally, every layer; `render_layers` rebuilds a
  condition's layers from the truth file, checks them, and warns when the file was made under
  another `SIGNAL_VERSION`. A case's id comes from its specification; artifact parameters count as
  one value whether written as an int or a float. A case condition lasts a positive whole
  number of seconds, as EDF records do. `python -m open_eeg_synth make-case` does the same from the
  command line, and checks for `edfio` and whole-second durations before building anything.
- `StreamSource`: chunked, seamless output for a recording application's mock amplifiers, with the
  classic synthesizer's constructor, an ECG on heart-rate labels, event markers and the artifact
  truth so far. Like the classic synthesizer, it accepts labels outside the head model; in 0.2.0
  those rows (10-10 sites such as Fpz and Oz, ear references, anything else) carry sensor noise
  only, the source warns once, and `StreamSource.unmodelled_labels` lists them.
- A realism suite (`pytest -m realism`) that compares the resting recipe with committed percentiles
  from PhysioNet recordings measured the same way, with two named known gaps.
- `SIGNAL_VERSION` and a signal fingerprint test that covers the brain, the artifact layers and
  their events, a dead channel and a planted pattern; after a `SIGNAL_VERSION` change it fails until
  `scripts/update_golden.py` has regenerated its reference. Both version numbers are written into
  every truth file.
- Data files and their notices: the head model (`NOTICE-colin27.txt`), and the eye-movement and blink
  patterns and realism reference derived from the PhysioNet EEG Motor Movement/Imagery Dataset
  (`NOTICE-eegmmidb.txt`).
- Development scripts (need MNE): `export_head_model.py`, `derive_artifact_patterns.py`,
  `build_realism_reference.py`, and `update_golden.py` for deliberate signal changes.

### Changed
- scipy is now a runtime dependency (`scipy>=1.10`), for streaming filters and anti-alias
  decimation. The `classic` engine's code is unchanged and still uses only numpy, and importing it
  still loads numpy only: the package root imports the engine's names on first use.
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
