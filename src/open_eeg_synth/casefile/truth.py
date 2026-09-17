"""The sealed truth file: what was planted, where every artifact is, how to re-render layers.

DESIGN §7.4.
"""

from __future__ import annotations

import json
import warnings
import zlib
from pathlib import Path

import numpy as np
import scipy

from open_eeg_synth import version
from open_eeg_synth.case import Case, CaseSpec, make_recording, make_subject
from open_eeg_synth.engine import Recording

MAGIC = b"OESTRUTH\x01"


class LayerMismatchError(RuntimeError):
    """Re-rendered layers do not match the recorded per-channel RMS."""


def _channel_rms_uv(layer: np.ndarray) -> np.ndarray:
    """Per-channel RMS of one rendered layer, full precision: the one computation both the truth
    file's recorded values and `render_layers`' re-rendered check must agree on."""
    return np.sqrt(np.mean(layer.astype(np.float64) ** 2, axis=1))


def _recording_truth(rec: Recording, file: str) -> dict:
    return {
        "file": file,
        "fs": rec.fs,
        "duration_s": rec.duration_s,
        "channels": list(rec.channels),
        "layers": list(rec.layers),
        "layer_rms_uv": {
            k: [round(float(v), 4) for v in _channel_rms_uv(a)] for k, a in rec.layers.items()
        },
        "timeline": rec.timeline.to_dict() if rec.timeline is not None else None,
        "truth": [t.to_dict() for t in rec.truth],
        "plants": [p.to_dict() for p in rec.plants],
    }


def case_truth(case: Case, files: dict[str, str]) -> dict:
    missing = [name for name in case.recordings if name not in files]
    if missing:
        raise ValueError(f"files is missing entries for condition(s): {', '.join(missing)}")
    return {
        "format": "open-eeg-synth/truth",
        "format_version": 1,
        "generator": {
            "package": "open-eeg-synth",
            "version": version.__version__,
            "signal_version": version.SIGNAL_VERSION,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "case_id": case.case_id,
        "seed": case.spec.seed,
        "label": case.spec.label,
        "spec": case.spec.to_dict(),
        "subject": case.subject.to_dict(),
        "recordings": {
            name: _recording_truth(rec, files[name]) for name, rec in case.recordings.items()
        },
    }


def write_truth(path, d: dict) -> Path:
    path = Path(path)
    blob = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(MAGIC + zlib.compress(blob, 9))
    return path


def read_truth(path) -> dict:
    raw = Path(path).read_bytes()
    if not raw.startswith(MAGIC):
        raise ValueError(f"{path} is not an open-eeg-synth truth file")
    return json.loads(zlib.decompress(raw[len(MAGIC) :]).decode("utf-8"))


def render_layers(truth: dict, condition: str, *, rtol: float = 1e-3) -> Recording:
    """Rebuild one condition's layers from the sealed spec and check them against the recorded RMS.

    Raises :class:`LayerMismatchError` when the re-rendered layer names differ from the truth
    file's ``layers`` list, or when a layer's per-channel RMS drifts past ``rtol``.
    """
    gen = truth["generator"]
    if gen["version"] != version.__version__ or gen["signal_version"] != version.SIGNAL_VERSION:
        warnings.warn(
            f"truth made by open-eeg-synth {gen['version']} (signal {gen['signal_version']}); "
            f"this is {version.__version__} (signal {version.SIGNAL_VERSION})",
            stacklevel=2,
        )
    spec = CaseSpec.from_dict(truth["spec"])
    cond = next((c for c in spec.conditions if c.name == condition), None)
    if cond is None:
        raise ValueError(
            f"condition {condition!r} is not in this truth file's spec; known conditions: "
            f"{[c.name for c in spec.conditions]}"
        )
    if condition not in truth["recordings"]:
        raise ValueError(
            f"condition {condition!r} has no recording in this truth file; known recordings: "
            f"{list(truth['recordings'])}"
        )
    subject = make_subject(spec)
    # Same per-condition path as make_case (R1): render + timeline + plant-silencing all live in
    # one place, so a truth file's re-rendered plants can never drift from what make_case wrote.
    rec = make_recording(spec, subject, cond)

    expected = truth["recordings"][condition]
    expected_layer_names = set(expected["layers"])
    if set(rec.layers) != expected_layer_names:
        raise LayerMismatchError(
            f"layer set of condition {condition!r} does not match the truth file: "
            f"re-rendered {sorted(rec.layers)}, truth lists {sorted(expected_layer_names)}"
        )
    expected_rms = expected["layer_rms_uv"]
    for name, arr in rec.layers.items():
        got = _channel_rms_uv(arr)
        want = np.asarray(expected_rms[name])
        if not np.allclose(got, want, rtol=rtol, atol=1e-3):
            raise LayerMismatchError(
                f"layer {name!r} of {condition!r} does not match the truth file"
            )
    return rec
