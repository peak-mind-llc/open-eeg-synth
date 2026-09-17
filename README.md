# open-eeg-synth

Realistic, seeded synthetic EEG for teaching, demos and software testing.

`open-eeg-synth` makes EEG that behaves like the real thing when an analysis
program looks at it, without an amplifier or a person. It is the signal source
behind the hardware-free mode of Coherence Recorder and the practice cases in
Coherence Workstation.

> **Status: 0.1, early.** This release contains the *classic* streaming engine
> moved from Coherence Recorder. A layered engine is in development: brain
> activity projected through a real head model, planted patterns, and pluggable
> artifacts that record what they are and how to clean them. See
> [`docs/DESIGN.md`](docs/DESIGN.md) and [`docs/PLAN.md`](docs/PLAN.md).

## Install

```bash
pip install "open-eeg-synth @ git+https://github.com/peak-mind-llc/open-eeg-synth.git@v0.1.0"
```

For local development, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick start

```python
from open_eeg_synth.classic import MarkerSchedule, RealisticEEGSynthesizer

labels = "Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 Fz Cz Pz HR".split()
synth = RealisticEEGSynthesizer(
    labels, 256.0, seed=42, markers=MarkerSchedule(kind="oddball", period_s=1.5)
)

chunk = synth.next_chunk(32)  # float32, shape (20, 32), microvolts
markers = synth.due_markers(32)  # [(onset_seconds, label), ...] in lockstep
```

The same seed always gives the same samples, and successive chunks join without
seams.

### What the classic engine models

- 1/f background noise with a strong common-mode component, as in raw
  referenced recordings.
- Posterior-dominant alpha whose frequency wanders and whose amplitude waxes and
  wanes.
- Frontal eye blinks, and an ECG on channels named `HR`, `ECG` or `EKG`.
- Periodic or oddball event markers, an oddball paradigm schedule with simulated
  P3a/P3b responses, and mock RR-interval, ECG and chest-accelerometer sources.

### What it does not model

The classic engine has no head model. Its channels share one noise trace, so the
spatial structure disappears once the signal is re-referenced, and it carries no
delayed coupling between sites. It has one rhythm (alpha) and one artifact
(blinks). The layered engine addresses all of these.

## Not clinical data

Generated recordings are synthetic. They are meant for teaching, demonstrations
and testing software, and must never be presented as a recording of a real
person or used to make decisions about one.

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
