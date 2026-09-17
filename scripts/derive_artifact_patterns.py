"""Derive subject-averaged blink and horizontal-eye-movement scalp maps from PhysioNet eegmmidb.

Development-only: needs MNE and a network connection on first run. Usage:
    python scripts/derive_artifact_patterns.py --subjects 30 --data-path ~/mne_data
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "open_eeg_synth" / "artifacts" / "data"
OUT_LOG = ROOT / "scripts" / "output" / "eog_patterns_selection.json"
RENAME = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def _prior(names: list[str], weights: dict[str, float]) -> np.ndarray:
    return np.array([weights.get(n, 0.0) for n in names])


BLINK_PRIOR = {
    "Fp1": 1,
    "Fp2": 1,
    "AF3": 1,
    "AF4": 1,
    "AF7": 1,
    "AF8": 1,
    "F3": 0.4,
    "Fz": 0.4,
    "F4": 0.4,
}
HEOG_PRIOR = {"F7": -1, "AF7": -1, "F8": 1, "AF8": 1, "Fp1": -0.5, "Fp2": 0.5}


def _excess_kurtosis(x: np.ndarray) -> float:
    x = x - x.mean()
    return float(np.mean(x**4) / np.mean(x**2) ** 2 - 3.0)


def _hf_share(x: np.ndarray, fs: float) -> float:
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return float(p[f > 5.0].sum() / p[f > 0.5].sum())


def _load_subject(subject: int, path: str) -> mne.io.Raw:
    files = eegbci.load_data(subject, runs=[1, 2], path=path, update_path=False, verbose=False)
    raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in files]
    raw = mne.concatenate_raws(raws, verbose=False)
    eegbci.standardize(raw)
    raw.rename_channels({k: v for k, v in RENAME.items() if k in raw.ch_names})
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    raw.filter(1.0, 45.0, verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=30)
    ap.add_argument("--data-path", default=None)
    args = ap.parse_args()

    blink_maps, heog_maps, log = [], [], {}
    names: list[str] | None = None
    for subj in range(1, args.subjects + 1):
        try:
            raw = _load_subject(subj, args.data_path)
        except Exception as exc:  # noqa: BLE001 - no network: use the subjects already on disk
            print(f"subject {subj}: cannot load ({exc!s:.80}); stopping at {subj - 1}", flush=True)
            break
        names = names or list(raw.ch_names)
        ica = mne.preprocessing.ICA(
            n_components=30, method="fastica", random_state=0, max_iter="auto", verbose=False
        )
        ica.fit(raw, verbose=False)
        topo = ica.get_components()  # (n_ch, n_comp)
        src = ica.get_sources(raw).get_data()
        fs = raw.info["sfreq"]
        entry: dict = {}
        # blink: best topography match to the prior, sparse time course
        cb = [
            np.corrcoef(topo[:, k], _prior(names, BLINK_PRIOR))[0, 1] for k in range(topo.shape[1])
        ]
        order = np.argsort(-np.abs(cb))
        for k in order[:5]:
            if _excess_kurtosis(src[k]) > 5.0:
                m = topo[:, k] * np.sign(cb[k])
                blink_maps.append(m / np.abs(m).max())
                entry["blink"] = {"component": int(k), "corr": float(cb[k])}
                break
        # heog: lateral prior, slow time course (< 40 % of the power above 5 Hz after the
        # 1 Hz high-pass; 0.2 keeps only 4 of 20 subjects, 0.4 keeps 17)
        ch = [
            np.corrcoef(topo[:, k], _prior(names, HEOG_PRIOR))[0, 1] for k in range(topo.shape[1])
        ]
        order = np.argsort(-np.abs(ch))
        for k in order[:5]:
            if _hf_share(src[k], fs) < 0.4:
                m = topo[:, k] * np.sign(ch[k])
                heog_maps.append(m / np.abs(m).max())
                entry["heog"] = {"component": int(k), "corr": float(ch[k])}
                break
        log[subj] = entry or "skipped"
        print(subj, entry or "skipped", flush=True)

    notice = (DATA / "NOTICE-eegmmidb.txt").read_text()
    attribution = (
        notice.rstrip() + f"\n\nDerived by scripts/derive_artifact_patterns.py from "
        f"subjects 1-{args.subjects}, runs 1-2, with MNE {mne.__version__} on "
        f"{dt.date.today().isoformat()}."
    )
    B, H = np.array(blink_maps), np.array(heog_maps)
    np.savez_compressed(
        DATA / "eog_patterns.npz",
        channel_names=np.array(names),
        blink_mean=B.mean(0).astype(np.float32),
        blink_sd=B.std(0).astype(np.float32),
        heog_mean=H.mean(0).astype(np.float32),
        heog_sd=H.std(0).astype(np.float32),
        n_subjects=np.int64(min(len(B), len(H))),
        subjects_used=np.array([s for s, e in log.items() if e != "skipped"]),
        attribution=np.array(attribution),
    )
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.write_text(json.dumps(log, indent=2))
    print(f"blink maps: {len(B)}, heog maps: {len(H)} -> {DATA / 'eog_patterns.npz'}")


if __name__ == "__main__":
    main()
