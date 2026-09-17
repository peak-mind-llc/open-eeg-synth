"""The classic engine is sample-identical to the Coherence Recorder original.

``tests/data/classic_golden.npz`` was recorded by running
``tools/make_classic_golden.py --source recorder`` inside Coherence Recorder
(commit 15876c6, numpy 2.2.6). This test runs the same cases against this
package and compares. A small tolerance absorbs floating-point differences
between numpy builds; any real change in behaviour is far larger.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "data" / "classic_golden.npz"


def _recorder_tool():
    spec = importlib.util.spec_from_file_location(
        "make_classic_golden", ROOT / "tools" / "make_classic_golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recorded():
    with np.load(FIXTURE) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="module")
def produced():
    return _recorder_tool().record("package")


def test_same_arrays(recorded, produced):
    assert sorted(recorded) == sorted(produced)


@pytest.mark.parametrize("name", sorted(np.load(FIXTURE).files))
def test_array_matches(name, recorded, produced):
    expected, actual = recorded[name], produced[name]
    assert actual.shape == expected.shape
    if expected.dtype.kind in "US":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-4)
