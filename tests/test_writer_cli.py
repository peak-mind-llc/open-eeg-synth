import json
import subprocess
import sys

import numpy as np
import pytest

from open_eeg_synth.case import CaseSpec, make_case
from open_eeg_synth.casefile.writer import read_case_truth, write_case
from open_eeg_synth.recipes import resting_case

pytest.importorskip("edfio")


def test_write_case_layout(tmp_path):
    case = make_case(resting_case(61, duration_s=3.0))
    paths = write_case(tmp_path, case)
    assert paths.truth.name == f"{case.case_id}.truth"
    assert sorted(p.name for p in paths.recordings.values()) == sorted(
        f"{case.case_id}_{c}.edf" for c in ("eyes_closed", "eyes_open")
    )
    assert paths.layers is None
    d = read_case_truth(paths.truth)
    assert d["recordings"]["eyes_open"]["file"] == paths.recordings["eyes_open"].name
    with_layers = write_case(tmp_path / "L", case, embed_layers=True)
    z = np.load(with_layers.layers)
    assert "eyes_open/brain" in z.files and z["eyes_open/brain"].shape == (19, 3 * 256)
    # The .layers.npz says "synthetic" like every other file the package writes.
    assert str(z["label"]) == "synthetic"


def test_cli_make_case(tmp_path):
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "open_eeg_synth",
            "make-case",
            "--seed",
            "7",
            "--out",
            str(tmp_path),
            "--duration",
            "2",
            "--print-truth",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    d = json.loads(out.stdout)
    assert d["seed"] == 7 and (tmp_path / f"{d['case_id']}.truth").exists()


def test_cli_make_case_with_spec_file_keeps_the_files_spec_apart_from_the_seed(tmp_path):
    """--spec overrides --duration entirely: the written case's spec must equal the file's own
    spec dict with only ``seed`` replaced, not a from_dict -> to_dict -> from_dict round trip of
    it (which risks losing anything canonicalisation changes between the two trips)."""
    spec_dict = resting_case(1, duration_s=2.0).to_dict()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec_dict))

    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "open_eeg_synth",
            "make-case",
            "--seed",
            "42",
            "--out",
            str(tmp_path / "out"),
            "--spec",
            str(spec_path),
            "--print-truth",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    d = json.loads(out.stdout)
    assert d["seed"] == 42
    want = CaseSpec.from_dict({**spec_dict, "seed": 42})
    assert CaseSpec.from_dict(d["spec"]) == want
