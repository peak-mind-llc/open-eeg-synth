"""Record reference output of the classic engine for the golden test.

The classic engine was moved verbatim from Coherence Recorder. This script
records what a given implementation produces for a fixed set of seeds so
``tests/classic/test_golden.py`` can prove the moved code is sample-identical.

Run it against the ORIGINAL implementation (inside a Coherence Recorder
checkout, with its virtualenv) to create the fixture:

    python tools/make_classic_golden.py --source recorder --out tests/data/classic_golden.npz

or against this package to compare by hand:

    python tools/make_classic_golden.py --source package --out /tmp/check.npz

Regenerate the fixture only when an output change is intended, and say so in
CHANGELOG.md.
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np

Q21 = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
       "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz", "HR"]  # fmt: skip
BRAINBIT = ["O1", "O2", "T3", "T4"]
DRAGON = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
          "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]  # fmt: skip

CASES = [
    # (name, labels, srate, chunk, n_chunks, seed, marker kwargs or None)
    ("q21_seed0", Q21, 256.0, 32, 40, 0, None),
    ("q21_seed7", Q21, 256.0, 32, 40, 7, None),
    ("brainbit_seed1", BRAINBIT, 250.0, 25, 40, 1, None),
    ("dragon_seed2", DRAGON, 500.0, 50, 40, 2, None),
    ("oddball_seed3", ["Cz"], 100.0, 100, 50, 3, dict(kind="oddball", period_s=0.5, p_target=0.3)),
]


def _modules(source: str):
    if source == "recorder":
        return (
            importlib.import_module("recorder.backend.synth"),
            importlib.import_module("recorder.backend.hrv.mock_rr"),
        )
    return (
        importlib.import_module("open_eeg_synth.classic.synth"),
        importlib.import_module("open_eeg_synth.classic.cardio"),
    )


def record(source: str) -> dict[str, np.ndarray]:
    synth, cardio = _modules(source)
    out: dict[str, np.ndarray] = {}
    for name, labels, srate, chunk, n_chunks, seed, mk in CASES:
        markers = synth.MarkerSchedule(**mk) if mk else None
        s = synth.RealisticEEGSynthesizer(labels, srate, seed=seed, markers=markers)
        chunks, marks = [], []
        for _ in range(n_chunks):
            chunks.append(s.next_chunk(chunk))
            marks.extend(s.due_markers(chunk))
        out[f"{name}__signal"] = np.concatenate(chunks, axis=1)
        out[f"{name}__marker_onsets"] = np.array([t for t, _ in marks], dtype=np.float64)
        out[f"{name}__marker_labels"] = np.array([lbl for _, lbl in marks], dtype="U16")
    rr = cardio.MockRRSource(seed=11)
    out["rr_seed11"] = np.array([rr.next_rr() for _ in range(300)], dtype=np.float64)
    ecg = cardio.MockEcgSource(seed=12)
    out["ecg_seed12"] = np.array(ecg.next_chunk(1300), dtype=np.float64)
    acc = cardio.MockAccSource(seed=13)
    out["acc_seed13"] = np.array(acc.next_chunk(500), dtype=np.float64)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=["recorder", "package"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = record(args.source)
    np.savez_compressed(args.out, **data)
    print(f"wrote {len(data)} arrays to {args.out} (numpy {np.__version__})")


if __name__ == "__main__":
    main()
