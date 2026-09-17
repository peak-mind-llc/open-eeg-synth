# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

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
