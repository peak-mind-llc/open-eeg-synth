"""One call writes a whole case: EDF per condition, sealed truth, optional layers (DESIGN §7.4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from open_eeg_synth.case import Case
from open_eeg_synth.casefile.edf import write_edf
from open_eeg_synth.casefile.truth import case_truth, read_truth, render_layers, write_truth

__all__ = ["CasePaths", "read_case_truth", "render_layers", "write_case"]


@dataclass
class CasePaths:
    truth: Path
    recordings: dict[str, Path]
    layers: Path | None


def write_case(
    directory,
    case: Case,
    *,
    embed_layers: bool = False,
    physical_range_uv: tuple[float, float] = (-2000.0, 2000.0),
) -> CasePaths:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    recordings: dict[str, Path] = {}
    for name, rec in case.recordings.items():
        recordings[name] = write_edf(
            directory / f"{case.case_id}_{name}.edf", rec, physical_range_uv=physical_range_uv
        )
    truth = write_truth(
        directory / f"{case.case_id}.truth",
        case_truth(case, {k: v.name for k, v in recordings.items()}),
    )
    layers = None
    if embed_layers:
        layers = directory / f"{case.case_id}.layers.npz"
        payload = {
            f"{c}/{k}": a for c, rec in case.recordings.items() for k, a in rec.layers.items()
        }
        # Every file the package writes says "synthetic" (DESIGN §1); the truth file and the EDFs
        # carry it as a plain string, so the layers archive carries it the same way its own
        # payload is shaped: a numpy string array under its own key.
        payload["label"] = np.array("synthetic")
        np.savez_compressed(layers, **payload)
    return CasePaths(truth, recordings, layers)


def read_case_truth(path) -> dict:
    return read_truth(path)
