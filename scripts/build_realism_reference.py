"""Measure PhysioNet eegmmidb baselines (subjects 1-20, runs 1-2) and write percentile bands.

Development-only: needs MNE, mne-connectivity, a network connection on first run. Usage:
    python scripts/build_realism_reference.py --subjects 20 --data-path ~/mne_data
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from open_eeg_synth.channels import CHANNELS_19  # noqa: E402
from tests.realism.measure import measure  # noqa: E402

OUT = ROOT / "tests" / "realism" / "reference" / "eegmmidb_baselines_s01-20.json"
NOTICE = ROOT / "src" / "open_eeg_synth" / "artifacts" / "data" / "NOTICE-eegmmidb.txt"
RENAME = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def load_real(subject: int, run: int, path):
    f = eegbci.load_data(subject, runs=[run], path=path, update_path=False, verbose=False)[0]
    raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
    eegbci.standardize(raw)
    raw.rename_channels(RENAME)
    raw.pick(list(CHANNELS_19))
    raw.reorder_channels(list(CHANNELS_19))
    raw.set_annotations(None)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    return raw


def summarise(group: list[dict]) -> dict:
    keys = [k for k in group[0] if k != "n_epochs"]
    out = {}
    for k in keys:
        vals = [g[k] for g in group if k in g]
        arr = np.array(vals, dtype=float)
        out[k] = {
            "p10": np.nanpercentile(arr, 10, axis=0).tolist(),
            "p50": np.nanpercentile(arr, 50, axis=0).tolist(),
            "p90": np.nanpercentile(arr, 90, axis=0).tolist(),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=20)
    ap.add_argument("--data-path", default=None)
    args = ap.parse_args()
    groups = {"eyes_open": [], "eyes_closed": []}
    for s in range(1, args.subjects + 1):
        for cond, run in (("eyes_open", 1), ("eyes_closed", 2)):
            m = measure(load_real(s, run, args.data_path), remove_blinks=True)
            if m.get("n_epochs", 0) >= 8:
                groups[cond].append(m)
            print(s, cond, m.get("n_epochs"), flush=True)
    ref = {
        "source": (
            "PhysioNet EEG Motor Movement/Imagery Dataset 1.0.0, "
            f"subjects 1-{args.subjects}, runs 1 (eyes open) and 2 (eyes closed), "
            "blink components removed by ICA"
        ),
        "license": "Open Data Commons Attribution License v1.0",
        "attribution": NOTICE.read_text(),
        "built": dt.date.today().isoformat(),
        "mne": mne.__version__,
        "n_subjects": {k: len(v) for k, v in groups.items()},
        "groups": {k: summarise(v) for k, v in groups.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ref, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
