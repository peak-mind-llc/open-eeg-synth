"""Realism metrics (DESIGN §9.1). MNE + mne-connectivity only; no consumer code."""

from __future__ import annotations

import warnings

import mne
import numpy as np
from mne_connectivity import spectral_connectivity_epochs
from scipy.signal import welch

mne.set_log_level("ERROR")

BANDS = {"Delta": (1.0, 4.0), "Theta": (4.0, 8.0), "Alpha": (8.0, 13.0), "Beta": (13.0, 30.0)}
DIST_EDGES = np.array([0.0, 60.0, 90.0, 120.0, 150.0, 250.0])  # mm
CHAINS = [("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
          ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
          ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
          ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
          ("Fz", "Cz"), ("Cz", "Pz")]  # fmt: skip
REJECT_UV = 800.0
_MONT = mne.channels.make_standard_montage("standard_1020")
_POS = _MONT.get_positions()["ch_pos"]


def to_raw(x_uv: np.ndarray, fs: float, channels) -> mne.io.Raw:
    info = mne.create_info(list(channels), fs, ch_types="eeg")
    raw = mne.io.RawArray(np.asarray(x_uv, float) * 1e-6, info, verbose=False)
    raw.set_montage(_MONT, on_missing="ignore")
    return raw


def prep(raw: mne.io.Raw, remove_blinks: bool = False) -> mne.io.Raw:
    raw = raw.copy().filter(1.0, 45.0, verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    if remove_blinks:
        ica = mne.preprocessing.ICA(
            n_components=15, method="fastica", random_state=0, max_iter="auto"
        )
        ica.fit(raw, verbose=False)
        idx, _ = ica.find_bads_eog(raw, ch_name=["Fp1", "Fp2"], threshold=3.0, verbose=False)
        ica.exclude = idx
        raw = ica.apply(raw, verbose=False)
    return raw


def bipolar(raw: mne.io.Raw) -> mne.io.Raw:
    data, names, rows = raw.get_data(), [], []
    for a, b in CHAINS:
        if a in raw.ch_names and b in raw.ch_names:
            rows.append(data[raw.ch_names.index(a)] - data[raw.ch_names.index(b)])
            names.append(f"{a}-{b}")
    info = mne.create_info(names, raw.info["sfreq"], ch_types="eeg")
    return mne.io.RawArray(np.array(rows), info, verbose=False)


def laplacian(raw: mne.io.Raw) -> mne.io.Raw:
    return mne.preprocessing.compute_current_source_density(raw.copy(), verbose=False)


def _pair_positions(names):
    out = []
    for nm in names:
        if "-" in nm:
            a, b = nm.split("-")
            out.append((_POS[a] + _POS[b]) / 2)
        else:
            out.append(_POS[nm])
    return np.array(out)


def _epochs(raw: mne.io.Raw) -> mne.Epochs:
    return mne.make_fixed_length_epochs(raw, duration=4.0, preload=True, verbose=False)


def measure(raw: mne.io.Raw, remove_blinks: bool = False) -> dict:
    avg = prep(raw, remove_blinks)
    ep_avg = _epochs(avg)
    ptp = np.ptp(ep_avg.get_data(), axis=2).max(axis=1) * 1e6
    keep = np.where(ptp < REJECT_UV)[0]
    res: dict = {"n_epochs": int(len(keep))}
    if len(keep) < 8:
        return res
    fmin = [BANDS[b][0] for b in BANDS]
    fmax = [BANDS[b][1] for b in BANDS]
    for ref, mraw in (("average", avg), ("bipolar", bipolar(avg)), ("laplacian", laplacian(avg))):
        ep = _epochs(mraw)[keep]
        names = ep.ch_names
        with warnings.catch_warnings():
            # 1 Hz in a 4 s epoch is 4 cycles, under mne-connectivity's 5-cycle advice, so it warns
            # on every call. The committed reference was measured with the same epochs and fmin,
            # so real and synthetic data stay comparable; only that warning is silenced.
            warnings.filterwarnings(
                "ignore",
                message=r"fmin=1\.000 Hz corresponds to 4\.000 < 5 cycles",
                category=RuntimeWarning,
            )
            con = spectral_connectivity_epochs(
                ep,
                method=["wpli2_debiased", "coh"],
                fmin=fmin,
                fmax=fmax,
                faverage=True,
                verbose=False,
            )
        dw = con[0].get_data(output="dense")  # (n, n, n_bands)
        coh = con[1].get_data(output="dense")
        pos = _pair_positions(names)
        dist = np.linalg.norm(pos[:, None] - pos[None], axis=2) * 1000.0
        # True pair values: mne-connectivity fills the lower triangle of the dense output and leaves
        # the upper one zero, so the pair values are read from the lower triangle as they are. (The
        # feasibility test measured through a consumer function that averaged M with M.T and so
        # halved every value; the numbers in DESIGN §9.1 and the reference file are the true ones.)
        il = np.tril_indices(len(names), -1)
        bins = np.digitize(dist[il], DIST_EDGES) - 1
        for bi, band in enumerate(BANDS):
            c, w = coh[..., bi][il], dw[..., bi][il]
            res[f"{ref}/{band}/coh_by_dist"] = [
                float(c[bins == k].mean()) if np.any(bins == k) else None
                for k in range(len(DIST_EDGES) - 1)
            ]
            res[f"{ref}/{band}/coh_mean"] = float(c.mean())
            res[f"{ref}/{band}/dwpli_mean"] = float(w.mean())
    data = avg.get_data() * 1e6
    f, p = welch(data, fs=avg.info["sfreq"], nperseg=int(4 * avg.info["sfreq"]))
    fit = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    slope = np.polyfit(np.log10(f[fit]), np.log10(np.median(p[:, fit], axis=0)), 1)[0]
    res["aperiodic_exponent"] = float(-slope)
    ab, tot = (f >= 8) & (f <= 13), (f >= 1) & (f <= 40)
    for ch in ("O1", "Fz", "Cz"):
        i = avg.ch_names.index(ch)
        res[f"alpha_share/{ch}"] = float(p[i, ab].sum() / p[i, tot].sum())
    for i, ch in enumerate(avg.ch_names):
        res[f"rms_uv/{ch}"] = float(data[i].std())
    return res
