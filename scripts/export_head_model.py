"""Export an MNE forward solution to the package's plain array head-model file (DESIGN §3.1).

Development-only: needs MNE. Usage:
    python scripts/export_head_model.py --forward /path/to/forward_19ch.fif --name colin27_19ch
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "open_eeg_synth" / "headmodel" / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", required=True)
    ap.add_argument("--name", default="colin27_19ch")
    args = ap.parse_args()

    fwd = mne.read_forward_solution(args.forward, verbose=False)
    if fwd["source_ori"] != mne.io.constants.FIFF.FIFFV_MNE_FREE_ORI:
        raise SystemExit("expected a free-orientation forward solution")
    ch = list(fwd["info"]["ch_names"])
    n_src = int(fwd["nsource"])
    gain_free = np.asarray(fwd["sol"]["data"], dtype=np.float64).reshape(len(ch), n_src, 3)

    fixed = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=False
    )
    nn = np.asarray(fixed["source_nn"], dtype=np.float64)
    rr = np.asarray(fwd["source_rr"], dtype=np.float64)
    # A few sources carry no cortical-patch normal (a zero vector, so their fixed gain is zero).
    # Give them the outward radial direction so every normal is a unit vector (DESIGN §3.1).
    undefined = np.linalg.norm(nn, axis=1) < 1e-6
    if undefined.any():
        radial = rr[undefined] - rr.mean(axis=0)
        nn[undefined] = radial / np.linalg.norm(radial, axis=1, keepdims=True)
    derived = np.einsum("csk,sk->cs", gain_free, nn)
    err = (
        np.abs(derived - fixed["sol"]["data"])[:, ~undefined].max()
        / np.abs(fixed["sol"]["data"]).max()
    )
    if err > 1e-5:
        raise SystemExit(f"fixed gain != free gain . normal (relative error {err:.2e})")

    hemi = np.concatenate(
        [np.zeros(fwd["src"][0]["nuse"], np.int8), np.ones(fwd["src"][1]["nuse"], np.int8)]
    )
    elec = np.array([c["loc"][:3] for c in fwd["info"]["chs"]], dtype=np.float64)
    notice = (DATA / "NOTICE-colin27.txt").read_text()
    attribution = (
        notice.rstrip()
        + f"\n\nExported by scripts/export_head_model.py from {Path(args.forward).name} "
        f"with MNE {mne.__version__} on {dt.date.today().isoformat()}. {int(undefined.sum())} of "
        f"{n_src} sources had no cortical-patch normal and were given the outward radial direction."
    )
    out = DATA / f"{args.name}.npz"
    np.savez_compressed(
        out,
        channel_names=np.array(ch),
        electrode_pos=elec.astype(np.float32),
        source_pos=rr.astype(np.float32),
        source_normal=nn.astype(np.float32),
        gain_free=gain_free.astype(np.float32),
        hemisphere=hemi,
        bem_conductivity=np.array([0.3, 0.006, 0.3], np.float32),
        mindist_mm=np.float32(5.0),
        attribution=np.array(attribution),
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB), {len(ch)} channels, {n_src} sources")


if __name__ == "__main__":
    main()
