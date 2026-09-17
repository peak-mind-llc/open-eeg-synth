# Contributing

Thank you for considering a contribution to open-eeg-synth.

## License and DCO

This project is licensed under Apache 2.0. By submitting a pull request you
certify that:

1. You wrote the code yourself, **or** you have the right to submit it under the
   project's license, **and**
2. You sign off on each commit using the
   [Developer Certificate of Origin](https://developercertificate.org/).

No separate CLA is required. Add a sign-off with:

```bash
git commit -s -m "your message"
```

## Development setup

```bash
git clone https://github.com/peak-mind-llc/open-eeg-synth.git
cd open-eeg-synth
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest -m "not slow"
```

## Ground rules

- **Seeded and deterministic.** Every random draw must come from the generator
  passed in or derived from the case seed. The same seed must give the same
  samples.
- **The classic engine does not change.** `open_eeg_synth.classic` must stay
  sample-identical to the recorded fixture. If an output change is intended,
  regenerate the fixture with `tools/make_classic_golden.py` and say so in
  `CHANGELOG.md`.
- **Keep the runtime light.** numpy is the only runtime dependency. Heavier
  tools, such as MNE for exporting head models, belong in optional extras or
  developer scripts.
- **Credit your data.** Any bundled data derived from third-party sources must
  carry its notice in `THIRD_PARTY_NOTICES.md`.
- **Synthetic is synthetic.** Generated recordings are for teaching, demos and
  testing. Never present them as recordings of a real person.
- **Golden fingerprint.** Any change that alters the generated samples for a
  fixed seed bumps `SIGNAL_VERSION` in `version.py` and runs
  `scripts/update_golden.py` in the same commit.
