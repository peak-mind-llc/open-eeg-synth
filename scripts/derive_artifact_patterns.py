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

# Selection criteria (DESIGN §6.3); all of them are written to the selection log.
N_CANDIDATES = 5  # components tried per subject, best prior match first
BLINK_MIN_KURTOSIS = 5.0  # blinks are sparse
# heog time course: less than 40 % of the power above 5 Hz (after the 1 Hz high-pass a 20 % bound
# keeps only 4 of the first 20 subjects; 40 % keeps 17)
HEOG_MAX_HF_SHARE = 0.4
# A slow component matching the heog prior more weakly than this is not taken as an eye-movement
# map. Measured on subjects 1-30 over 0.20-0.50 in steps of 0.05: 0.25 is the smallest value at
# which every kept component's prior-correlation sign agrees with sign(F8 - F7) (subject 21,
# |r| = 0.24, disagrees), keeping 24 subjects with heog_sd max 0.42; 0.50 keeps only 12.
HEOG_MIN_ABS_CORR = 0.25


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
        # max_iter="auto" is 1000 for FastICA. Subject 4 never converges (tried up to 50 000
        # iterations and tol 1e-3); more iterations only swap its heog pick for another
        # component, so the default stays.
        ica = mne.preprocessing.ICA(
            n_components=30, method="fastica", random_state=0, max_iter="auto", verbose=False
        )
        ica.fit(raw, verbose=False)
        topo = ica.get_components()  # (n_ch, n_comp)
        src = ica.get_sources(raw).get_data()
        fs = raw.info["sfreq"]
        f7, f8 = names.index("F7"), names.index("F8")
        entry: dict = {}
        # blink: best topography match to the prior, sparse time course; sign from the match
        cb = [
            np.corrcoef(topo[:, k], _prior(names, BLINK_PRIOR))[0, 1] for k in range(topo.shape[1])
        ]
        order = np.argsort(-np.abs(cb))[:N_CANDIDATES]
        sparse = [k for k in order if _excess_kurtosis(src[k]) > BLINK_MIN_KURTOSIS]
        if sparse:
            k = sparse[0]
            m = topo[:, k] * np.sign(cb[k])
            blink_maps.append(m / np.abs(m).max())
            entry["blink"] = {
                "component": int(k),
                "corr": float(cb[k]),
                "excess_kurtosis": _excess_kurtosis(src[k]),
            }
        else:
            entry["blink"] = {
                "skipped": f"none of the {N_CANDIDATES} best matches has excess kurtosis "
                f"> {BLINK_MIN_KURTOSIS}"
            }
        # heog: lateral prior, slow time course, a match of at least HEOG_MIN_ABS_CORR;
        # sign-aligned so F8 - F7 > 0
        ch = [
            np.corrcoef(topo[:, k], _prior(names, HEOG_PRIOR))[0, 1] for k in range(topo.shape[1])
        ]
        order = np.argsort(-np.abs(ch))[:N_CANDIDATES]
        slow = [k for k in order if _hf_share(src[k], fs) < HEOG_MAX_HF_SHARE]
        if not slow:
            entry["heog"] = {
                "skipped": f"none of the {N_CANDIDATES} best matches has less than "
                f"{HEOG_MAX_HF_SHARE:.0%} of its power above 5 Hz"
            }
        elif abs(ch[slow[0]]) < HEOG_MIN_ABS_CORR:
            k = slow[0]
            entry["heog"] = {
                "skipped": f"the best slow match (component {k}) has |r| = {abs(ch[k]):.3f} "
                f"< {HEOG_MIN_ABS_CORR}"
            }
        else:
            k = slow[0]
            m = topo[:, k] * (1.0 if topo[f8, k] > topo[f7, k] else -1.0)
            m = m / np.abs(m).max()
            heog_maps.append(m)
            entry["heog"] = {
                "component": int(k),
                "corr": float(ch[k]),
                "hf_share": _hf_share(src[k], fs),
                "f8_minus_f7": float(m[f8] - m[f7]),
            }
        log[subj] = entry
        print(subj, entry, flush=True)

    notice = (DATA / "NOTICE-eegmmidb.txt").read_text()
    attribution = (
        notice.rstrip() + f"\n\nDerived by scripts/derive_artifact_patterns.py from "
        f"subjects 1-{args.subjects}, runs 1-2, with MNE {mne.__version__} on "
        f"{dt.date.today().isoformat()}."
    )
    B, H = np.array(blink_maps), np.array(heog_maps)
    used = [s for s, e in log.items() if any("component" in v for v in e.values())]
    np.savez_compressed(
        DATA / "eog_patterns.npz",
        channel_names=np.array(names),
        blink_mean=B.mean(0).astype(np.float32),
        blink_sd=B.std(0).astype(np.float32),
        heog_mean=H.mean(0).astype(np.float32),
        heog_sd=H.std(0).astype(np.float32),
        n_subjects=np.int64(min(len(B), len(H))),
        subjects_used=np.array(used),
        attribution=np.array(attribution),
    )
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    criteria = {
        "n_candidates": N_CANDIDATES,
        "blink_min_excess_kurtosis": BLINK_MIN_KURTOSIS,
        "heog_max_hf_share": HEOG_MAX_HF_SHARE,
        "heog_min_abs_corr": HEOG_MIN_ABS_CORR,
        "blink_sign": "sign of the correlation with the blink prior",
        "heog_sign": "F8 - F7 > 0",
    }
    OUT_LOG.write_text(json.dumps({"criteria": criteria, "subjects": log}, indent=2) + "\n")
    print(f"blink maps: {len(B)}, heog maps: {len(H)} -> {DATA / 'eog_patterns.npz'}")


if __name__ == "__main__":
    main()
