"""EDF output: microvolt physical range, anonymised date, no annotations (DESIGN §7.4)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from open_eeg_synth.engine import Recording


def write_edf(
    path,
    recording: Recording,
    *,
    physical_range_uv: tuple[float, float] = (-2000.0, 2000.0),
    patient_code: str = "SYNTHETIC",
    equipment: str = "open-eeg-synth",
    additional: Sequence[str] = ("synthetic",),
) -> Path:
    try:
        import edfio
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "write_edf needs the 'edf' extra: pip install 'open-eeg-synth[edf]'"
        ) from exc
    lo, hi = physical_range_uv
    fs = recording.fs
    n = int(recording.n_samples // fs * fs)  # whole seconds only
    data = recording.mixed[:, :n]
    signals = [
        edfio.EdfSignal(
            np.clip(data[i].astype(np.float64), lo, hi),
            sampling_frequency=fs,
            label=ch,
            transducer_type="synthetic",
            physical_dimension="uV",
            physical_range=(lo, hi),
        )
        for i, ch in enumerate(recording.channels)
    ]
    edf = edfio.Edf(
        signals,
        patient=edfio.Patient(code=patient_code, name="X", additional=tuple(additional)),
        recording=edfio.Recording(startdate=None, equipment_code=equipment),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    edf.write(str(path))
    return path
