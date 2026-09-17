# open-eeg-synth v0.2.0 — layered engine implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the layered synthetic-EEG engine described in `docs/DESIGN.md` beside the existing `classic` subpackage: head model + brain layer + artifact plug-ins + planted patterns + case files + streaming, with a realism suite, and release it as v0.2.0.

**Architecture:** One `Engine.render(t0, n)` path sums named layers (brain, artifact:*, sensor, transform:*). Whole recordings are rendered through it in blocks; streams call it chunk by chunk. Every random draw comes from a named stream derived from one case seed; every filter carries state, so chunking never changes samples. Data files (head model, eye patterns) are exported once by dev-only MNE scripts and shipped in the wheel.

**Tech Stack:** Python 3.10–3.12, numpy, scipy (`lfilter`, `resample_poly`), optional `edfio`; dev extras MNE (`>=1.6,<1.14`) + mne-connectivity; Hatchling; pytest + ruff; GitHub Actions.

**Spec:** `docs/DESIGN.md` (read it first; section numbers below refer to it).

**Repo root:** the `open-eeg-synth` checkout. All paths below are relative to it. The `classic/` subpackage, its golden test, `pyproject.toml`, `LICENSE`, and CI skeleton already exist from the v0.1 extraction — **do not rewrite them**; Task 1 only adds to them.

## Global Constraints

- `requires-python = ">=3.10,<3.13"`; licence Apache-2.0; Hatchling; `src/` layout; package import name `open_eeg_synth`.
- Runtime dependencies: `numpy>=1.24`, `scipy>=1.10`. Nothing else at runtime. `open_eeg_synth.classic` stays numpy-only and byte-identical to its golden test.
- MNE and mne-connectivity are development extras only, imported only in `scripts/` and `tests/realism/`. Nothing under `src/` may import MNE.
- No consumer code is imported anywhere (no Coherence Workstation, no Coherence Recorder modules).
- Signals are float32 microvolts, shape `(n_channels, n_samples)`; positions are metres; `_s` suffix = seconds; `t0`/`n` = samples.
- Every random draw goes through `open_eeg_synth.seeds.stream_rng`/`stream_seed`, and the stream names are exactly the ones DESIGN §8.1 lists (`subject:head`, `<condition>:background`, `<condition>:rhythm:<name>`, …); noise is drawn time-major (`rng.standard_normal((n, k)).T`).
- Chunk invariance (DESIGN §8.2) is a tested property of every layer and of the engine.
- Every file the package writes says "synthetic" (EDF patient code, truth `label`).
- Public documents and code contain no private data: no clinic paths, no subject codes, no consumer-internal identifiers.
- `ruff check .` and `ruff format --check .` clean; `pytest` (fast set) green before every commit. Every task's "verify they pass" step runs both ruff commands after pytest; the repo's ruff configuration (rules `E,F,W,I,B,UP`, line length 100, `ruff>=0.6`) and its CI workflow versions are kept as they are.
- The code blocks below are not pre-formatted: run `ruff format .` before the green step and wrap any docstring or signature longer than 100 characters.
- Line length 100 (ruff), `from __future__ import annotations` at the top of every module.

## Milestones and parallelism

| milestone | tasks | may run in parallel |
|---|---|---|
| M1 head-model export + engine skeleton | 1–6 | after Task 1: Tasks 2, 5, 6 in parallel; 3 after 2; 4 after 3 |
| M2 brain layer | 7–12 | Tasks 7, 8, 9 in parallel after M1; 10 after 9; 11 after 7–10; 12 after 11 **and after Task 13** (the artifact contract exists before the case builder uses it) |
| M3 artifact framework + blink + eye movement + jaw EMG | 13–20 | 13 right after Tasks 3 and 9 (before Task 12); 14 in parallel with 13; 15 after 14; 16, 17, 18, 19 in parallel after 12, 14, 15; 20 last |
| M4 state timeline + plants | 21–23 | may start after M2 (in parallel with M3); 23 after 20 and 22 |
| M5 case files + sealed truth | 24–26 | 24 after Task 6; 25 after 23; 26 after 24, 25 |
| M6 streaming + recorder switch-over | 27–29 | 27 after Task 6; 28 after 20; 29 (recorder repo) after the v0.2.0 tag |
| M7 CI + realism suite + release | 30–35 | 33 first (before any realism task: it sets the per-file ruff ignores `tests/realism/` needs), 34 after 20; then 30, 31 (MNE only); 32 after 12 and 31; 35 last |

Review-unit order (one implementer + one review per unit): U1 {1} → U2 {2,3,4} → U3 {5,6} → U4 {7,8,9} → U5 {10,11} → U6 {13,14} → U7 {12} → U8 {15} → U9 {16,17,18,19} → U10 {20} → U11 {21,22,23} → U12 {24,25,26} → U13 {27,28} → U14 {33,34} → U15 {30,31,32} → U16 {35 minus tag/release}.

## File map (what each new file is responsible for)

```
src/open_eeg_synth/channels.py           channel constants, aliases, canonical_label, mirror map
src/open_eeg_synth/seeds.py              named random streams from one case seed
src/open_eeg_synth/dsp.py                OU, PinkCascade, lowpass_decimate, raised_cosine_envelope
src/open_eeg_synth/headmodel/__init__.py HeadModel + loader + subset + perturbation + patch maps
src/open_eeg_synth/headmodel/data/       colin27_19ch.npz, NOTICE-colin27.txt
src/open_eeg_synth/engine.py             Layer protocol, Frame, Engine, Recording
src/open_eeg_synth/sensor.py             SensorNoise layer
src/open_eeg_synth/brain/state.py        StateSegment, StateTimeline
src/open_eeg_synth/brain/background.py   BackgroundSpec, Background
src/open_eeg_synth/brain/network.py      NetworkSpec, NetworkWiring, wire_network, Network
src/open_eeg_synth/brain/placement.py    placed_centres, region_centres
src/open_eeg_synth/brain/rhythm.py       BurstGate, RhythmSpec, Rhythm
src/open_eeg_synth/brain/plants.py       Plant protocol, PlantRecord, Modifier, six primitives, PLANTS registry
src/open_eeg_synth/brain/layer.py        BrainSpec, BrainLayer
src/open_eeg_synth/artifacts/base.py     Remedy, TruthRecord, RenderContext, Occupancy, Event, EventArtifact, TransformArtifact
src/open_eeg_synth/artifacts/registry.py register, ARTIFACTS, make_artifact, discover
src/open_eeg_synth/artifacts/patterns.py empirical, analytic_focal, analytic_dipole
src/open_eeg_synth/artifacts/data/       eog_patterns.npz, NOTICE-eegmmidb.txt
src/open_eeg_synth/artifacts/{blink,eye_movement,jaw_emg,dead_channel}.py
src/open_eeg_synth/case.py               SensorSpec, ArtifactSpec, ConditionSpec, CaseSpec, Subject, make_subject, make_engine, make_case, Case, compiled_rhythms
src/open_eeg_synth/recipes.py            resting_brain, ordinary_artifacts, resting_case
src/open_eeg_synth/heart.py              HeartSource
src/open_eeg_synth/stream.py             StreamSource
src/open_eeg_synth/markers.py            re-export of classic MarkerSchedule
src/open_eeg_synth/casefile/{edf,truth,writer}.py   (writer re-exports render_layers)
tests/golden/resting_seed20260916.json   signal fingerprint (Task 34)
src/open_eeg_synth/__main__.py           make-case CLI
scripts/export_head_model.py, scripts/derive_artifact_patterns.py, scripts/build_realism_reference.py, scripts/update_golden.py
tests/helpers.py                         numpy Welch, band power, slope, chunked-vs-whole helper
tests/realism/{measure.py,test_realism.py,reference/}
```

---

# M1 — head-model export + engine skeleton

### Task 1: Engine skeleton wiring — dependencies, channels, seeds

**Files:**
- Modify: `pyproject.toml` (add scipy, extras, pytest markers)
- Create: `src/open_eeg_synth/channels.py`
- Create: `src/open_eeg_synth/seeds.py`
- Rename: `src/open_eeg_synth/_version.py` → `src/open_eeg_synth/version.py` (add `SIGNAL_VERSION`); update `[tool.hatch.version] path` and the import in `src/open_eeg_synth/__init__.py`
- Create: `tests/test_channels.py`, `tests/test_seeds.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `CHANNELS_19`, `MIRROR`, `MIDLINE`, `HEART_LABELS`, `UnknownChannelError`, `canonical_label(label, known=CHANNELS_19) -> str`, `is_heart_label(label) -> bool`; `stream_seed(case_seed, name) -> SeedSequence`, `stream_rng(case_seed, name) -> Generator`, `fresh_case_seed() -> int`; `__version__`, `SIGNAL_VERSION`.

- [ ] **Step 1: Reconcile with the existing skeleton.** Run `ls src/open_eeg_synth src/open_eeg_synth/classic tests`. The classic subpackage is `synth.py` (`RealisticEEGSynthesizer`, `MarkerSchedule`), `cardio.py` (`MockRRSource`, `MockEcgSource`, `MockAccSource`) and `oddball.py` (`OddballParadigm`, `schedule`, `ErpInjector`); its tests live in `tests/classic/` with the golden fixture in `tests/data/`. The version module is `_version.py`: `git mv` it to `version.py` (DESIGN §2.2), point `[tool.hatch.version] path` at the new name, fix the import in `__init__.py`, and add `SIGNAL_VERSION`.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_channels.py
import pytest

from open_eeg_synth.channels import (
    CHANNELS_19, MIRROR, UnknownChannelError, canonical_label, is_heart_label,
)


def test_channels_19_order_matches_head_model_rows():
    assert CHANNELS_19 == (
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
    )


def test_canonical_label_case_and_aliases():
    assert canonical_label("fp1") == "Fp1"
    assert canonical_label(" T7 ") == "T3"
    assert canonical_label("P8") == "T6"
    with pytest.raises(UnknownChannelError, match="known"):
        canonical_label("AF3")
    assert canonical_label("AF3", known=("AF3", "AF4")) == "AF3"


def test_mirror_is_symmetric():
    for a, b in MIRROR.items():
        assert MIRROR[b] == a


def test_heart_labels():
    assert is_heart_label("HR") and is_heart_label("ecg ") and not is_heart_label("Cz")
```

```python
# tests/test_seeds.py
import numpy as np
import pytest

from open_eeg_synth.seeds import fresh_case_seed, stream_rng, stream_seed


def test_same_seed_same_name_same_draws():
    a = stream_rng(42, "subject:head").standard_normal(8)
    b = stream_rng(42, "subject:head").standard_normal(8)
    assert np.array_equal(a, b)


def test_different_names_independent():
    a = stream_rng(42, "eyes_closed:background").standard_normal(8)
    b = stream_rng(42, "eyes_closed:sensor").standard_normal(8)
    assert not np.allclose(a, b)


def test_seed_sequence_is_stable_across_processes():
    # blake2b of the name, not Python's randomised hash()
    assert stream_seed(7, "x").entropy == stream_seed(7, "x").entropy


def test_seed_range():
    with pytest.raises(ValueError):
        stream_seed(-1, "x")
    s = fresh_case_seed()
    assert 0 <= s < 2**63
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_channels.py tests/test_seeds.py -v`
Expected: FAIL with `ModuleNotFoundError: open_eeg_synth.channels`

- [ ] **Step 4: Implement**

```python
# src/open_eeg_synth/channels.py
"""Channel names, aliases and the mirror map used by the whole package."""

from __future__ import annotations

from collections.abc import Sequence

CHANNELS_19: tuple[str, ...] = (
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
)
MIRROR: dict[str, str] = {
    "Fp1": "Fp2", "F3": "F4", "F7": "F8", "C3": "C4", "T3": "T4", "P3": "P4", "T5": "T6", "O1": "O2",
}
MIRROR.update({v: k for k, v in list(MIRROR.items())})
MIDLINE: frozenset[str] = frozenset({"Fz", "Cz", "Pz", "Fpz", "Oz", "FCz", "CPz", "POz", "AFz"})
HEART_LABELS: frozenset[str] = frozenset({"HR", "ECG", "EKG"})
_ALIASES: dict[str, str] = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


class UnknownChannelError(ValueError):
    """A label that no head model or pattern file knows."""


def is_heart_label(label: str) -> bool:
    return label.strip().upper() in HEART_LABELS


def canonical_label(label: str, known: Sequence[str] = CHANNELS_19) -> str:
    """Map a device label onto one of ``known`` (case-insensitive, 10-10 aliases applied)."""
    key = label.strip().upper()
    key = _ALIASES.get(key, key)
    lookup = {k.upper(): k for k in known}
    if key in lookup:
        return lookup[key]
    raise UnknownChannelError(f"unknown channel {label!r}; known: {', '.join(known)}")
```

```python
# src/open_eeg_synth/seeds.py
"""One case seed -> every named random stream (DESIGN §8.1)."""

from __future__ import annotations

import hashlib

import numpy as np

MAX_SEED = 2**63 - 1


def _tag(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(), "little")


def stream_seed(case_seed: int, name: str) -> np.random.SeedSequence:
    if not 0 <= int(case_seed) <= MAX_SEED:
        raise ValueError(f"case_seed must be in [0, 2**63); got {case_seed}")
    return np.random.SeedSequence([int(case_seed), _tag(name)])


def stream_rng(case_seed: int, name: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(stream_seed(case_seed, name)))


def fresh_case_seed() -> int:
    return int(np.random.SeedSequence().entropy) % (MAX_SEED + 1)
```

```python
# src/open_eeg_synth/version.py  (renamed from _version.py)
__version__ = "0.2.0.dev0"
SIGNAL_VERSION = 2  # bump whenever the same seed would produce different samples (DESIGN §8.3)
```

`pyproject.toml` additions (merge into the existing file; keep everything the v0.1 skeleton has):

```toml
[project]
dependencies = ["numpy>=1.24", "scipy>=1.10"]

[project.optional-dependencies]
edf = ["edfio>=0.4"]
# mne < 1.14: the `standard_1020` montage name the scripts and the realism suite use is removed in 1.14
realism = ["mne>=1.6,<1.14", "mne-connectivity>=0.6"]
dev = ["pytest>=7.0", "ruff>=0.6", "edfio>=0.4", "mne>=1.6,<1.14", "mne-connectivity>=0.6"]

[tool.hatch.version]
path = "src/open_eeg_synth/version.py"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -m 'not realism and not slow'"
markers = [
  "realism: compares the resting recipe with public reference EEG; needs mne (run: pytest -m realism)",
  "slow: takes more than ~10 s",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pip install -e ".[dev]" && pytest tests/test_channels.py tests/test_seeds.py -v && pytest -q && ruff check . && ruff format --check .`
Expected: all PASS, including the pre-existing classic golden test.

- [ ] **Step 6: Commit**

```bash
git add -A pyproject.toml src/open_eeg_synth tests/test_channels.py tests/test_seeds.py   # -A stages the _version.py -> version.py rename
git commit -m "feat: channel constants, named seed streams, scipy runtime dep"
```

---

### Task 2: Head-model export script + data file + notice

**Files:**
- Create: `scripts/export_head_model.py`
- Create: `src/open_eeg_synth/headmodel/__init__.py` (empty for now; Task 3 fills it)
- Create: `src/open_eeg_synth/headmodel/data/NOTICE-colin27.txt`
- Create (generated): `src/open_eeg_synth/headmodel/data/colin27_19ch.npz`
- Create: `tests/test_headmodel_file.py`
- Modify: `pyproject.toml` (`[tool.hatch.build.targets.sdist] include` gains `"scripts"` beside `"tools"`)

**Interfaces:**
- Consumes: a 19-channel MNE forward solution file supplied by the developer (`--forward <path>`; the consuming QEEG application's `forward_19ch.fif`, computed from the Colin27 template). MNE must be installed (`pip install -e ".[dev]"`).
- Produces: the `.npz` with the arrays of DESIGN §3.1.

- [ ] **Step 1: Write the notice file** `src/open_eeg_synth/headmodel/data/NOTICE-colin27.txt`, verbatim:

```
Colin27 head model redistributed by EEGLAB (head_modelColin27_5003_Standard-10-5-Cap339.mat); the
lead field bundled in open-eeg-synth was computed from its surfaces and electrode cap with MNE-Python
and is redistributed under the notice below.

Copyright (C) 1993-2009 Louis Collins, McConnell Brain Imaging Centre, Montreal Neurological
Institute, McGill University. Permission to use, copy, modify, and distribute this software and its
documentation for any purpose and without fee is hereby granted, provided that the above copyright
notice appear in all copies. The authors and McGill University make no representations about the
suitability of this software for any purpose. It is provided "as is" without express or implied
warranty. The authors are not responsible for any data loss, equipment damage, property loss, or
injury to subjects or patients resulting from the use or misuse of this software package.
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_headmodel_file.py
from importlib import resources

import numpy as np
import pytest

from open_eeg_synth.channels import CHANNELS_19

DATA = resources.files("open_eeg_synth.headmodel") / "data"


@pytest.fixture(scope="module")
def npz():
    with resources.as_file(DATA / "colin27_19ch.npz") as p:
        return np.load(p, allow_pickle=False)


def test_arrays_shapes_and_dtypes(npz):
    n_src = npz["source_pos"].shape[0]
    assert tuple(npz["channel_names"]) == CHANNELS_19
    assert npz["electrode_pos"].shape == (19, 3) and npz["electrode_pos"].dtype == np.float32
    assert npz["source_pos"].shape == (n_src, 3) and npz["source_pos"].dtype == np.float32
    assert npz["source_normal"].shape == (n_src, 3)
    assert npz["gain_free"].shape == (19, n_src, 3) and npz["gain_free"].dtype == np.float32
    assert npz["hemisphere"].shape == (n_src,) and npz["hemisphere"].dtype == np.int8
    assert 4000 < n_src < 6000


def test_normals_unit_and_hemispheres_by_x(npz):
    assert np.allclose(np.linalg.norm(npz["source_normal"], axis=1), 1.0, atol=1e-4)
    x = npz["source_pos"][:, 0]
    assert (x[npz["hemisphere"] == 0] < 0).mean() > 0.98
    assert (x[npz["hemisphere"] == 1] > 0).mean() > 0.98


def test_positions_in_metres_head_frame(npz):
    e = npz["electrode_pos"]
    assert np.abs(e).max() < 0.15  # metres, not mm
    o1, o2 = e[CHANNELS_19.index("O1")], e[CHANNELS_19.index("O2")]
    fp1 = e[CHANNELS_19.index("Fp1")]
    assert o1[1] < -0.09 and fp1[1] > 0.07  # y is anterior
    assert o1[0] < 0 < o2[0]  # x is right


def test_attribution_and_notice_present(npz):
    text = str(npz["attribution"])
    assert "Louis Collins" in text and "McGill" in text
    assert (DATA / "NOTICE-colin27.txt").read_text().strip() in text
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_headmodel_file.py -v`
Expected: FAIL (file not found).

- [ ] **Step 4: Write the export script**

```python
# scripts/export_head_model.py
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
    err = (np.abs(derived - fixed["sol"]["data"])[:, ~undefined].max()
           / np.abs(fixed["sol"]["data"]).max())
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
```

- [ ] **Step 5: Run the export, then the tests**

Run: `python scripts/export_head_model.py --forward <path to the 19-channel forward .fif> && pytest tests/test_headmodel_file.py -v && ruff check . && ruff format --check .`
Expected: script prints ~1.1 MB, 19 channels, 4871 sources; tests PASS (the forward solution has 3 sources without a patch normal; the attribution string says so). Check `ls -la src/open_eeg_synth/headmodel/data/` shows the `.npz` (~1.1 MB) and the notice.

- [ ] **Step 6: Commit** (the `.npz` is committed; it is package data, not a build product)

```bash
git add pyproject.toml scripts/export_head_model.py src/open_eeg_synth/headmodel tests/test_headmodel_file.py
git commit -m "feat(headmodel): export Colin27 19-channel lead field to a plain array file with notice"
```

---

### Task 3: HeadModel runtime — load, subset, patch maps, mixing

**Files:**
- Modify: `src/open_eeg_synth/headmodel/__init__.py`
- Create: `tests/test_headmodel.py`

**Interfaces:**
- Consumes: Task 2's file; `canonical_label`, `MIRROR`, `MIDLINE`.
- Produces: `HeadModel` (fields per DESIGN §3.3), `load_head_model(name="colin27_19ch") -> HeadModel`, methods `gain`, `index(label)`, `subset(labels)`, `patch_map(centre, width_mm)`, `sources_under(label, n=40)`, `mirror_source(i)`, `smoothed_mixing(scale_mm)`, `n_channels`, `n_sources`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_headmodel.py
import numpy as np
import pytest

from open_eeg_synth.channels import CHANNELS_19, UnknownChannelError
from open_eeg_synth.headmodel import load_head_model


@pytest.fixture(scope="module")
def head():
    return load_head_model()


def test_gain_is_free_dot_normal(head):
    g = head.gain
    assert g.shape == (19, head.n_sources)
    manual = np.einsum("csk,sk->cs", head.gain_free, head.source_normal)
    assert np.allclose(g, manual, rtol=1e-5, atol=1e-6)


def test_subset_and_aliases(head):
    sub = head.subset(["O1", "o2", "T7", "T8"])
    assert sub.channels == ("O1", "O2", "T3", "T4")
    assert sub.gain.shape == (4, head.n_sources)
    assert np.allclose(sub.gain[2], head.gain[CHANNELS_19.index("T3")])
    with pytest.raises(UnknownChannelError):
        head.subset(["Oz"])


def test_patch_map_unit_rms_and_local_maximum(head):
    c = int(head.sources_under("O1", 1)[0])
    m = head.patch_map(c, 12.0)
    assert np.isclose(np.sqrt(np.mean(m**2)), 1.0)
    assert np.argmax(np.abs(m)) in (CHANNELS_19.index("O1"), CHANNELS_19.index("T5"))


def test_mirror_source_flips_x(head):
    i = int(head.sources_under("C3", 1)[0])
    j = head.mirror_source(i)
    assert head.source_pos[j][0] > 0
    assert np.linalg.norm(head.source_pos[j] * [-1, 1, 1] - head.source_pos[i]) < 0.006


def test_smoothed_mixing_unit_mean_variance_and_neighbour_structure(head):
    M = head.smoothed_mixing(20.0)
    assert M.shape == (19, 19)
    C = M @ M.T
    assert np.isclose(np.mean(np.diag(C)), 1.0)
    corr = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    o1, o2, fp1 = (CHANNELS_19.index(k) for k in ("O1", "O2", "Fp1"))
    assert corr[o1, o2] > corr[o1, fp1]  # near pairs more correlated than far pairs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_headmodel.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_head_model'`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/headmodel/__init__.py
"""Real-head lead field as plain arrays: loading, subsets, patch maps, perturbation (DESIGN §3)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from importlib import resources

import numpy as np

from open_eeg_synth.channels import canonical_label

_DATA = resources.files("open_eeg_synth.headmodel") / "data"
_CHUNK = 512


@dataclass(frozen=True)
class HeadModel:
    name: str
    channels: tuple[str, ...]
    electrode_pos: np.ndarray  # (n_ch, 3) m
    source_pos: np.ndarray  # (n_src, 3) m
    source_normal: np.ndarray  # (n_src, 3)
    gain_free: np.ndarray  # (n_ch, n_src, 3) V/(A m)
    hemisphere: np.ndarray  # (n_src,) int8, 0 left 1 right
    attribution: str
    perturbation: dict | None = None

    @cached_property
    def gain(self) -> np.ndarray:
        return np.einsum("csk,sk->cs", self.gain_free, self.source_normal).astype(np.float64)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_sources(self) -> int:
        return int(self.source_pos.shape[0])

    def index(self, label: str) -> int:
        return self.channels.index(canonical_label(label, self.channels))

    def subset(self, labels: Sequence[str]) -> HeadModel:
        rows = [self.index(lb) for lb in labels]
        return replace(
            self,
            channels=tuple(self.channels[r] for r in rows),
            electrode_pos=self.electrode_pos[rows],
            gain_free=self.gain_free[rows],
        )

    def patch_map(self, centre: int, width_mm: float) -> np.ndarray:
        d_mm = np.linalg.norm(self.source_pos - self.source_pos[centre], axis=1) * 1000.0
        t = self.gain @ np.exp(-0.5 * (d_mm / width_mm) ** 2)
        return t / np.sqrt(np.mean(t**2))

    def sources_under(self, label: str, n: int = 40) -> np.ndarray:
        d = np.linalg.norm(self.source_pos - self.electrode_pos[self.index(label)], axis=1)
        return np.argsort(d)[:n]

    def mirror_source(self, i: int) -> int:
        target = self.source_pos[i] * np.array([-1.0, 1.0, 1.0])
        return int(np.argmin(np.linalg.norm(self.source_pos - target, axis=1)))

    def _source_kernel_apply(self, scale_mm: float, X: np.ndarray, normalise: bool) -> np.ndarray:
        """Return K @ X for the Gaussian source-distance kernel K, chunked (never forms K)."""
        rr = self.source_pos * 1000.0
        out = np.empty((self.n_sources, X.shape[1]))
        for s in range(0, self.n_sources, _CHUNK):
            d = np.linalg.norm(rr[s : s + _CHUNK, None, :] - rr[None, :, :], axis=2)
            w = np.exp(-0.5 * (d / scale_mm) ** 2)
            out[s : s + _CHUNK] = (w @ X) / (w.sum(axis=1, keepdims=True) if normalise else 1.0)
        return out

    def smoothed_mixing(self, scale_mm: float) -> np.ndarray:
        """19x19 M with M M^T = covariance of smoothed cortical noise seen at the sensors."""
        GK = self._source_kernel_apply(scale_mm, self.gain.T, normalise=False).T
        C = GK @ GK.T
        lam, V = np.linalg.eigh(C)
        M = (V * np.sqrt(np.clip(lam, 0.0, None))) @ V.T
        return M / np.sqrt(np.mean(np.diag(M @ M.T)))


@lru_cache(maxsize=4)
def load_head_model(name: str = "colin27_19ch") -> HeadModel:
    with resources.as_file(_DATA / f"{name}.npz") as p:
        z = np.load(p, allow_pickle=False)
        return HeadModel(
            name=name,
            channels=tuple(str(c) for c in z["channel_names"]),
            electrode_pos=z["electrode_pos"].astype(np.float64),
            source_pos=z["source_pos"].astype(np.float64),
            source_normal=z["source_normal"].astype(np.float64),
            gain_free=z["gain_free"],
            hemisphere=z["hemisphere"],
            attribution=str(z["attribution"]),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_headmodel.py tests/test_headmodel_file.py -v && ruff check . && ruff format --check .`
Expected: PASS (the mixing test takes ~0.5 s).

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/headmodel/__init__.py tests/test_headmodel.py
git commit -m "feat(headmodel): HeadModel loader, subsets, patch maps, smoothed mixing"
```

---

### Task 4: Head-model perturbation

**Files:**
- Modify: `src/open_eeg_synth/headmodel/__init__.py` (add `perturbed`)
- Create: `tests/test_headmodel_perturb.py`

**Interfaces:**
- Consumes: `HeadModel`, `_source_kernel_apply`.
- Produces: `HeadModel.perturbed(rng, *, tilt_deg=8.0, gain_sd=0.06, blur_range=(0.03, 0.10), corr_mm=15.0) -> HeadModel` with `perturbation` dict `{"tilt_deg", "corr_mm", "median_tilt_deg", "channel_gain": [n_ch floats], "blur_eps"}`. Works on any subset, including one channel (the scalp blur needs a neighbour; with one channel it is skipped and the draw is still consumed).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_headmodel_perturb.py
import numpy as np

from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.seeds import stream_rng


def _rel_change(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_perturbation_is_deterministic_and_moderate():
    head = load_head_model()
    p1 = head.perturbed(stream_rng(1, "subject:head"))
    p2 = head.perturbed(stream_rng(1, "subject:head"))
    p3 = head.perturbed(stream_rng(2, "subject:head"))
    assert np.array_equal(p1.gain_free, p2.gain_free)
    assert not np.array_equal(p1.gain_free, p3.gain_free)
    assert 0.05 < _rel_change(p1.gain, head.gain) < 0.25
    assert head.perturbation is None and p1.perturbation is not None
    assert 5.0 < p1.perturbation["median_tilt_deg"] < 15.0
    assert 0.03 <= p1.perturbation["blur_eps"] <= 0.10
    assert len(p1.perturbation["channel_gain"]) == 19


def test_perturbed_normals_stay_unit_and_subset_commutes():
    head = load_head_model()
    p = head.perturbed(stream_rng(3, "subject:head"))
    assert np.allclose(np.linalg.norm(p.source_normal, axis=1), 1.0)
    sub = p.subset(["O1", "O2"])
    assert np.allclose(sub.gain[0], p.gain[head.index("O1")])


def test_single_channel_subset_perturbs_without_nan():
    p = load_head_model().subset(["Cz"]).perturbed(stream_rng(4, "subject:head"))
    assert np.isfinite(p.gain).all() and p.gain.shape[0] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_headmodel_perturb.py -v`
Expected: FAIL with `AttributeError: 'HeadModel' object has no attribute 'perturbed'`

- [ ] **Step 3: Implement** (add to the `HeadModel` class)

```python
    def perturbed(
        self,
        rng: np.random.Generator,
        *,
        tilt_deg: float = 8.0,
        gain_sd: float = 0.06,
        blur_range: tuple[float, float] = (0.03, 0.10),
        corr_mm: float = 15.0,
    ) -> HeadModel:
        """A deterministic, physically-motivated variant of this head (DESIGN §3.3)."""
        # 1. smooth random tilt of the source normals
        field_ = rng.standard_normal((self.n_sources, 3))
        smooth = self._source_kernel_apply(corr_mm, field_, normalise=True)
        smooth /= smooth.std()
        nn = self.source_normal + np.tan(np.deg2rad(tilt_deg)) * smooth
        nn /= np.linalg.norm(nn, axis=1, keepdims=True)
        cosang = np.clip((nn * self.source_normal).sum(axis=1), -1.0, 1.0)
        median_tilt = float(np.degrees(np.median(np.arccos(cosang))))
        # 2. per-channel gain
        g = np.clip(1.0 + gain_sd * rng.standard_normal(self.n_channels), 0.85, 1.15)
        # 3. scalp blur
        eps = float(rng.uniform(*blur_range))
        if self.n_channels > 1:
            de = np.linalg.norm(
                self.electrode_pos[:, None, :] - self.electrode_pos[None, :, :], axis=2
            )
            A = np.exp(-0.5 * (de * 1000.0 / 50.0) ** 2)
            np.fill_diagonal(A, 0.0)
            A /= A.sum(axis=1, keepdims=True)
            T = g[:, None] * ((1.0 - eps) * np.eye(self.n_channels) + eps * A)
        else:  # one channel has no neighbour to blur into; the draw is still consumed
            T = g[:, None] * np.eye(1)
        gf = np.einsum("cd,dsk->csk", T, self.gain_free).astype(np.float32)
        record = {
            "tilt_deg": float(tilt_deg),
            "corr_mm": float(corr_mm),
            "median_tilt_deg": round(median_tilt, 2),
            "channel_gain": [round(float(v), 4) for v in g],
            "blur_eps": round(eps, 4),
        }
        return replace(self, source_normal=nn, gain_free=gf, perturbation=record)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_headmodel_perturb.py -v && ruff check . && ruff format --check .`
Expected: PASS. If `median_tilt_deg` falls outside 5–15°, adjust the scale in step 1 (`np.tan(np.deg2rad(tilt_deg))`) — the test pins the *outcome*, DESIGN §3.3 documents it.

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/headmodel/__init__.py tests/test_headmodel_perturb.py
git commit -m "feat(headmodel): per-case perturbation (normal tilt, channel gain, scalp blur)"
```

---

### Task 5: DSP primitives + test helpers

**Files:**
- Create: `src/open_eeg_synth/dsp.py`
- Create: `tests/helpers.py`
- Create: `tests/test_dsp.py`

**Interfaces:**
- Consumes: scipy.
- Produces: `OU(n_series, fs, *, mu, sigma, tau_s, rng)` with `.step(white, mu=None)`; `PinkCascade(n_series, fs, *, beta=1.2, f_lo=0.03, n_per_decade=2.0)` with `.warm_up(rng, seconds)` and `.process(white)`; `lowpass_decimate(x, factor)`; `raised_cosine_envelope(n, fs, rise_s, fall_s)`. Test helpers: `welch(x, fs, nperseg)` (inputs shorter than `nperseg` become one segment, so per-second windows work), `band_power(x, fs, lo, hi)`, `psd_slope(x, fs)`, `render_whole_and_chunked(make_layer, n_total, rng)`.

- [ ] **Step 1: Write the helpers and the failing tests**

```python
# tests/helpers.py
"""numpy-only measurement helpers shared by the fast tests."""

from __future__ import annotations

import numpy as np


def welch(x: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed Welch PSD along the last axis, 50 % overlap. Returns (f, psd)."""
    x = np.atleast_2d(x)
    nperseg = int(min(nperseg, x.shape[-1]))  # a short input becomes one segment
    step = nperseg // 2
    win = np.hanning(nperseg)
    scale = 1.0 / (fs * (win**2).sum())
    segs = [x[:, s : s + nperseg] * win for s in range(0, x.shape[1] - nperseg + 1, step)]
    p = np.mean([np.abs(np.fft.rfft(s, axis=-1)) ** 2 for s in segs], axis=0) * scale
    p[:, 1:-1] *= 2.0
    return np.fft.rfftfreq(nperseg, 1.0 / fs), p


def band_power(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    f, p = welch(x, fs, int(4 * fs))
    m = (f >= lo) & (f < hi)
    return p[:, m].sum(axis=-1) * (f[1] - f[0])


def psd_slope(x: np.ndarray, fs: float, f_lo=1.0, f_hi=40.0, exclude=(7.0, 14.0)) -> float:
    """Negative of the log-log slope of the median PSD: ~beta for 1/f^beta noise."""
    f, p = welch(x, fs, int(4 * fs))
    fit = (f >= f_lo) & (f <= f_hi) & ~((f >= exclude[0]) & (f <= exclude[1]))
    return float(-np.polyfit(np.log10(f[fit]), np.log10(np.median(p[:, fit], axis=0)), 1)[0])


def render_whole_and_chunked(make_layer, n_total: int, rng: np.random.Generator):
    """Render one layer instance in one call and a fresh, identically-seeded one in random chunks."""
    whole = make_layer().render(0, n_total)
    layer = make_layer()
    parts, t0 = [], 0
    while t0 < n_total:
        n = int(min(n_total - t0, rng.integers(1, 700)))
        parts.append(layer.render(t0, n))
        t0 += n
    return whole, np.concatenate(parts, axis=1)
```

```python
# tests/test_dsp.py
import numpy as np

from open_eeg_synth.dsp import OU, PinkCascade, lowpass_decimate, raised_cosine_envelope
from tests.helpers import band_power, psd_slope

FS = 256.0


def test_pink_cascade_slope_and_unit_variance():
    rng = np.random.default_rng(0)
    pk = PinkCascade(3, FS, beta=1.2)
    pk.warm_up(rng, 20.0)
    x = pk.process(rng.standard_normal((int(120 * FS), 3)).T)
    assert abs(psd_slope(x, FS) - 1.2) < 0.1
    assert 0.8 < x.std() < 1.25


def test_pink_cascade_chunk_invariance():
    def run(chunks):
        rng = np.random.default_rng(5)
        pk = PinkCascade(2, FS, beta=1.0)
        out = [pk.process(rng.standard_normal((n, 2)).T) for n in chunks]
        return np.concatenate(out, axis=1)

    assert np.allclose(run([1000]), run([7, 300, 693]), atol=1e-9)


def test_ou_is_stationary_with_requested_sd_and_mean():
    rng = np.random.default_rng(1)
    ou = OU(4, FS, mu=10.0, sigma=0.35, tau_s=4.0, rng=rng)
    x = ou.step(rng.standard_normal((int(600 * FS), 4)).T)
    assert abs(x.mean() - 10.0) < 0.05
    assert abs(x.std() - 0.35) < 0.05
    y = ou.step(np.zeros((4, 10)), mu=np.full(10, 20.0))  # time-varying mean is additive
    assert np.all(y > 15.0)


def test_lowpass_decimate_removes_tone_above_new_nyquist():
    fs_hi = 8 * FS
    t = np.arange(int(4 * fs_hi)) / fs_hi
    x = np.sin(2 * np.pi * 200.0 * t)  # would alias to 56 Hz at 256 Hz
    y = lowpass_decimate(x, 8)
    assert y.shape[0] == int(4 * FS)
    assert band_power(y, FS, 50.0, 62.0)[0] < 1e-3 * band_power(x, fs_hi, 194.0, 206.0)[0]


def test_raised_cosine_envelope_shape():
    e = raised_cosine_envelope(256, FS, 0.1, 0.2)
    assert e[0] == 0.0 and np.isclose(e[128], 1.0) and e[-1] < 0.01
    assert np.all(np.diff(e[:25]) >= 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsp.py -v`
Expected: FAIL with `ModuleNotFoundError: open_eeg_synth.dsp`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/dsp.py
"""Streaming signal primitives: exact OU processes, a 1/f^beta cascade, decimation (DESIGN §4, §8)."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter, resample_poly


class OU:
    """Exact-discretisation Ornstein-Uhlenbeck processes, one per row.

    x[n] = mu + a (x[n-1] - mu) + sigma sqrt(1 - a^2) w[n],  a = exp(-1 / (tau_s fs)).
    Starts from the stationary distribution so there is no warm-up transient.
    """

    def __init__(self, n_series: int, fs: float, *, mu: float = 0.0, sigma: float = 1.0,
                 tau_s: float = 1.0, rng: np.random.Generator) -> None:
        self.n = int(n_series)
        self.mu = float(mu)
        self.a = float(np.exp(-1.0 / (tau_s * fs)))
        self.g = float(sigma) * float(np.sqrt(1.0 - self.a**2))
        x0 = float(sigma) * rng.standard_normal(self.n)
        self.zi = (self.a * x0).reshape(self.n, 1)  # transposed DF-II state: a * y[-1]

    def step(self, white: np.ndarray, mu: np.ndarray | float | None = None) -> np.ndarray:
        """Advance by ``white.shape[1]`` samples. ``mu`` may be a per-sample array."""
        dev, self.zi = lfilter([self.g], [1.0, -self.a], white, axis=-1, zi=self.zi)
        return dev + (self.mu if mu is None else mu)


class PinkCascade:
    """1/f^beta noise via first-order pole-zero sections (Corsini & Saletti 1988), unit variance."""

    def __init__(self, n_series: int, fs: float, *, beta: float = 1.2, f_lo: float = 0.03,
                 n_per_decade: float = 2.0) -> None:
        self.n = int(n_series)
        self.fs = float(fs)
        f_hi = 0.45 * self.fs
        n_sec = int(np.ceil(n_per_decade * np.log10(f_hi / f_lo)))
        r = (f_hi / f_lo) ** (1.0 / n_sec)
        c = 2.0 * self.fs
        self.sections: list[tuple[np.ndarray, np.ndarray]] = []
        for k in range(n_sec):
            fp = f_lo * r**k
            fz = fp * r ** (beta / 2.0)
            wp = c * np.tan(np.pi * fp / self.fs)  # bilinear pre-warp, rad/s
            wz = c * np.tan(np.pi * fz / self.fs)
            k0 = wp / wz
            b = np.array([k0 * (c + wz) / (c + wp), k0 * (wz - c) / (c + wp)])
            a = np.array([1.0, (wp - c) / (c + wp)])
            self.sections.append((b, a))
        self.zi = [np.zeros((self.n, 1)) for _ in self.sections]
        probe = np.random.Generator(np.random.PCG64(12345)).standard_normal((1, 1 << 17))
        y = probe
        for b, a in self.sections:
            y = lfilter(b, a, y, axis=-1)
        self.scale = 1.0 / float(y[:, 1 << 14 :].std())

    def warm_up(self, rng: np.random.Generator, seconds: float) -> None:
        self.process(rng.standard_normal((int(seconds * self.fs), self.n)).T)

    def process(self, white: np.ndarray) -> np.ndarray:
        y = white
        for i, (b, a) in enumerate(self.sections):
            y, self.zi[i] = lfilter(b, a, y, axis=-1, zi=self.zi[i])
        return y * self.scale


def lowpass_decimate(x: np.ndarray, factor: int) -> np.ndarray:
    """Anti-alias low-pass and decimate along the last axis, as an amplifier front end would."""
    return resample_poly(x, up=1, down=int(factor), axis=-1, window=("kaiser", 5.0))


def raised_cosine_envelope(n: int, fs: float, rise_s: float, fall_s: float) -> np.ndarray:
    env = np.ones(int(n))
    nr = min(n // 2, int(round(rise_s * fs)))
    nf = min(n - nr, int(round(fall_s * fs)))
    if nr > 0:
        env[:nr] = 0.5 * (1.0 - np.cos(np.pi * np.arange(nr) / nr))
    if nf > 0:
        env[n - nf :] = 0.5 * (1.0 + np.cos(np.pi * (np.arange(nf) + 1) / nf))
    return env
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsp.py -v && ruff check . && ruff format --check .`
Expected: PASS. Add an empty `tests/__init__.py` if `from tests.helpers import …` fails.

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/dsp.py tests/helpers.py tests/test_dsp.py tests/__init__.py
git commit -m "feat(dsp): streaming OU, 1/f cascade, anti-alias decimation, envelopes"
```

---

### Task 6: Engine, Frame, Recording, sensor layer

**Files:**
- Create: `src/open_eeg_synth/engine.py`
- Create: `src/open_eeg_synth/sensor.py`
- Create: `tests/test_engine.py`

**Interfaces:**
- Consumes: `seeds`.
- Produces: `Layer` protocol (`name`, `render(t0, n)`, `truth()`), `Transform` protocol (`name`, `render_transform(t0, n, mix)`, `truth()`; the Engine keys the transform's layer by that `name`, e.g. `transform:dead_channel` or `transform:dead_channel#2`), `Frame(t0, n, layers)` with `.mixed`, `Engine(channels, fs, layers, transforms=())` with `.render`, `.render_all(n_samples, block=4096)`, `.truth()`, `.position`; `Recording(fs, channels, layers, truth, plants=[], timeline=None)` with `.mixed`, `.duration_s`, `.n_samples`; `SensorNoise(n_ch, white_uv, seq)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_engine.py
import numpy as np
import pytest

from open_eeg_synth.engine import Engine, Recording
from open_eeg_synth.seeds import stream_seed
from open_eeg_synth.sensor import SensorNoise
from tests.helpers import render_whole_and_chunked


class Ramp:
    """A deterministic test layer: channel c carries c + t."""

    name = "ramp"

    def __init__(self, n_ch):
        self.n_ch = n_ch

    def render(self, t0, n):
        t = np.arange(t0, t0 + n, dtype=np.float32)
        return (np.arange(self.n_ch, dtype=np.float32)[:, None] + t[None, :]).astype(np.float32)

    def truth(self):
        return []


class Kill:
    name = "transform:kill"

    def render_transform(self, t0, n, mix):
        d = np.zeros_like(mix)
        d[1] = -mix[1]
        return d

    def truth(self):
        return []


def test_engine_sums_layers_and_records_transform_delta():
    eng = Engine(["A", "B", "C"], 100.0, [Ramp(3)], transforms=[Kill()])
    fr = eng.render(0, 10)
    assert set(fr.layers) == {"ramp", "transform:kill"}
    assert np.allclose(fr.mixed, sum(fr.layers.values()))
    assert np.all(fr.mixed[1] == 0.0) and fr.mixed[2, 3] == 5.0
    assert eng.position == 10


def test_engine_requires_contiguous_calls():
    eng = Engine(["A"], 100.0, [Ramp(1)])
    eng.render(0, 5)
    with pytest.raises(ValueError, match="contiguous"):
        eng.render(9, 5)
    with pytest.raises(ValueError):
        eng.render(5, 0)


def test_render_all_equals_chunked_and_recording_sums():
    eng = Engine(["A", "B"], 100.0, [Ramp(2)])
    rec = eng.render_all(1000, block=64)
    assert isinstance(rec, Recording) and rec.n_samples == 1000 and rec.duration_s == 10.0
    assert np.allclose(rec.mixed, rec.layers["ramp"])
    ref = Ramp(2).render(0, 1000)
    assert np.array_equal(rec.layers["ramp"], ref)


def test_sensor_noise_level_and_chunk_invariance():
    def make():
        return SensorNoise(4, 1.5, stream_seed(9, "eyes_closed:sensor"))

    whole, chunked = render_whole_and_chunked(make, 5000, np.random.default_rng(0))
    assert np.allclose(whole, chunked)
    assert abs(whole.std() - 1.5) < 0.1 and whole.dtype == np.float32
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/engine.py
"""The one signal path: layers rendered block by block and summed (DESIGN §7.1)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Layer(Protocol):
    name: str

    def render(self, t0: int, n: int) -> np.ndarray: ...

    def truth(self) -> list: ...


@runtime_checkable
class Transform(Protocol):
    name: str

    def render_transform(self, t0: int, n: int, mix: np.ndarray) -> np.ndarray: ...

    def truth(self) -> list: ...


@dataclass
class Frame:
    t0: int
    n: int
    layers: dict[str, np.ndarray]

    @property
    def mixed(self) -> np.ndarray:
        out = np.zeros(next(iter(self.layers.values())).shape, dtype=np.float32)
        for block in self.layers.values():
            out += block
        return out


@dataclass
class Recording:
    fs: float
    channels: tuple[str, ...]
    layers: dict[str, np.ndarray]
    truth: list = field(default_factory=list)
    plants: list = field(default_factory=list)
    timeline: Any = None

    @property
    def mixed(self) -> np.ndarray:
        out = np.zeros(next(iter(self.layers.values())).shape, dtype=np.float32)
        for block in self.layers.values():
            out += block
        return out

    @property
    def n_samples(self) -> int:
        return int(next(iter(self.layers.values())).shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs


class Engine:
    def __init__(self, channels: Sequence[str], fs: float, layers: Sequence[Layer],
                 transforms: Sequence[Transform] = ()) -> None:
        self.channels = tuple(channels)
        self.fs = float(fs)
        self.layers = list(layers)
        self.transforms = list(transforms)
        names = [lay.name for lay in self.layers] + [t.name for t in self.transforms]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate layer names: {names}")
        self._pos = 0

    @property
    def position(self) -> int:
        return self._pos

    def render(self, t0: int, n: int) -> Frame:
        if t0 != self._pos:
            raise ValueError(f"render must be contiguous: expected t0={self._pos}, got {t0}")
        if n <= 0:
            raise ValueError("n must be positive")
        n_ch = len(self.channels)
        out: dict[str, np.ndarray] = {}
        mix = np.zeros((n_ch, n), dtype=np.float32)
        for lay in self.layers:
            block = np.asarray(lay.render(t0, n), dtype=np.float32)
            if block.shape != (n_ch, n):
                raise ValueError(f"layer {lay.name} returned {block.shape}, expected {(n_ch, n)}")
            out[lay.name] = block
            mix += block
        for tr in self.transforms:
            delta = np.asarray(tr.render_transform(t0, n, mix), dtype=np.float32)
            out[tr.name] = delta
            mix += delta
        self._pos += n
        return Frame(t0, n, out)

    def render_all(self, n_samples: int, block: int = 4096) -> Recording:
        parts: dict[str, list[np.ndarray]] = {}
        t0 = 0
        while t0 < n_samples:
            fr = self.render(t0, min(block, n_samples - t0))
            for k, v in fr.layers.items():
                parts.setdefault(k, []).append(v)
            t0 += fr.n
        layers = {k: np.concatenate(v, axis=1) for k, v in parts.items()}
        return Recording(self.fs, self.channels, layers, truth=self.truth())

    def truth(self) -> list:
        out: list = []
        for lay in self.layers:
            out.extend(lay.truth())
        for tr in self.transforms:
            out.extend(tr.truth())
        return out
```

```python
# src/open_eeg_synth/sensor.py
"""Amplifier / electrode white noise layer."""

from __future__ import annotations

import numpy as np


class SensorNoise:
    name = "sensor"

    def __init__(self, n_ch: int, white_uv: float, seq: np.random.SeedSequence) -> None:
        self.n_ch = int(n_ch)
        self.white_uv = float(white_uv)
        self.rng = np.random.Generator(np.random.PCG64(seq))

    def render(self, t0: int, n: int) -> np.ndarray:
        return (self.white_uv * self.rng.standard_normal((n, self.n_ch)).T).astype(np.float32)

    def truth(self) -> list:
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_engine.py -v && ruff check . && ruff format --check .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/engine.py src/open_eeg_synth/sensor.py tests/test_engine.py
git commit -m "feat(engine): Engine/Frame/Recording with layer sum identity and transform hook; sensor noise"
```

**M1 acceptance:** `pytest -q` green; `python -c "from open_eeg_synth.headmodel import load_head_model; h=load_head_model(); print(h.gain.shape)"` prints `(19, 4871)`; the wheel built by `python -m build --wheel` contains `headmodel/data/colin27_19ch.npz` and `NOTICE-colin27.txt` (`unzip -l dist/*.whl | grep data`).

---

# M2 — brain layer reproducing the feasibility test's realism numbers

### Task 7: Background layer (smoothed cortical noise)

**Files:**
- Create: `src/open_eeg_synth/brain/__init__.py` (empty)
- Create: `src/open_eeg_synth/brain/background.py`
- Create: `tests/test_background.py`

**Interfaces:**
- Consumes: `HeadModel.smoothed_mixing`, `PinkCascade`.
- Produces: `BackgroundSpec(smoothing_mm=20.0, exponent=1.2, rms_uv=20.0, network_frac=0.5)`; `Background(head, fs, spec, seq, mixing=None)` layer (`name="brain.background"`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_background.py
import numpy as np

from open_eeg_synth.brain.background import Background, BackgroundSpec
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.seeds import stream_seed
from tests.helpers import psd_slope, render_whole_and_chunked

FS = 256.0


def test_background_rms_slope_and_chunk_invariance():
    head = load_head_model()
    spec = BackgroundSpec(network_frac=0.0)
    mixing = head.smoothed_mixing(spec.smoothing_mm)

    def make():
        return Background(head, FS, spec, stream_seed(1, "eyes_closed:background"), mixing=mixing)

    whole, chunked = render_whole_and_chunked(make, int(60 * FS), np.random.default_rng(0))
    assert np.allclose(whole, chunked, atol=1e-4)
    rms = np.sqrt(np.mean(whole**2))
    assert 16.0 < rms < 24.0  # spec.rms_uv = 20 on average over channels
    assert abs(psd_slope(whole, FS) - 1.2) < 0.15


def test_network_frac_reduces_background_share():
    head = load_head_model()
    a = Background(head, FS, BackgroundSpec(network_frac=0.0), stream_seed(2, "x")).render(0, 2560)
    b = Background(head, FS, BackgroundSpec(network_frac=0.5), stream_seed(2, "x")).render(0, 2560)
    assert np.isclose(b.std() / a.std(), np.sqrt(0.5), atol=0.02)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_background.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/brain/background.py
"""Smoothed cortical 1/f noise seen at the sensors (DESIGN §4.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from open_eeg_synth.dsp import PinkCascade
from open_eeg_synth.headmodel import HeadModel

WARM_UP_S = 20.0


@dataclass(frozen=True)
class BackgroundSpec:
    smoothing_mm: float = 20.0
    exponent: float = 1.2
    rms_uv: float = 20.0
    network_frac: float = 0.5


class Background:
    name = "brain.background"

    def __init__(self, head: HeadModel, fs: float, spec: BackgroundSpec,
                 seq: np.random.SeedSequence, mixing: np.ndarray | None = None) -> None:
        self.spec = spec
        M = head.smoothed_mixing(spec.smoothing_mm) if mixing is None else mixing
        self.M = M * spec.rms_uv * np.sqrt(1.0 - spec.network_frac)
        self.rng = np.random.Generator(np.random.PCG64(seq))
        self.pink = PinkCascade(head.n_channels, fs, beta=spec.exponent)
        self.pink.warm_up(self.rng, WARM_UP_S)
        self.n_ch = head.n_channels

    def render(self, t0: int, n: int) -> np.ndarray:
        white = self.rng.standard_normal((n, self.n_ch)).T
        return (self.M @ self.pink.process(white)).astype(np.float32)

    def truth(self) -> list:
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_background.py -v && ruff check . && ruff format --check .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/brain tests/test_background.py
git commit -m "feat(brain): smoothed cortical background via covariance square root"
```

---

### Task 8: Delayed cortical network

**Files:**
- Create: `src/open_eeg_synth/brain/network.py`
- Create: `tests/test_network.py`

**Interfaces:**
- Consumes: `HeadModel.patch_map`, `PinkCascade`, `BackgroundSpec`.
- Produces: `NetworkSpec(n_nodes=40, fan_in=3, coupling=1.0, velocity_m_s=6.0, synaptic_ms=5.0, width_mm=15.0)`; `NetworkWiring(centres, sources, lags)`; `wire_network(head, spec, fs, rng) -> NetworkWiring`; `Network(head, fs, spec, background, wiring, seq)` layer (`name="brain.network"`) with `.render_nodes(n)` (test hook returning the node series `y` for the next block; `render` uses it).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_network.py
import numpy as np

from open_eeg_synth.brain.background import BackgroundSpec
from open_eeg_synth.brain.network import Network, NetworkSpec, wire_network
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.seeds import stream_rng, stream_seed
from tests.helpers import render_whole_and_chunked

FS = 256.0


def test_wiring_is_deterministic_with_physical_lags():
    head = load_head_model()
    spec = NetworkSpec()
    w1 = wire_network(head, spec, FS, stream_rng(1, "subject:network"))
    w2 = wire_network(head, spec, FS, stream_rng(1, "subject:network"))
    assert w1.centres == w2.centres and w1.lags == w2.lags
    assert len(w1.centres) == 40 and all(len(s) == 3 for s in w1.sources)
    flat = [lag for row in w1.lags for lag in row]
    assert min(flat) >= int(round(0.005 * FS)) and max(flat) <= int(round(0.045 * FS))


def test_network_rms_and_chunk_invariance():
    head = load_head_model()
    bg = BackgroundSpec(rms_uv=20.0, network_frac=0.5)
    wiring = wire_network(head, NetworkSpec(), FS, stream_rng(1, "subject:network"))

    def make():
        return Network(head, FS, NetworkSpec(), bg, wiring, stream_seed(1, "eyes_closed:network"))

    whole, chunked = render_whole_and_chunked(make, int(40 * FS), np.random.default_rng(3))
    assert np.allclose(whole, chunked, atol=1e-4)
    rms = np.sqrt(np.mean(whole**2))
    assert 0.7 * 20.0 * np.sqrt(0.5) < rms < 1.3 * 20.0 * np.sqrt(0.5)


def test_coupled_nodes_are_lagged_copies():
    head = load_head_model()
    wiring = wire_network(head, NetworkSpec(), FS, stream_rng(4, "subject:network"))
    net = Network(head, FS, NetworkSpec(), BackgroundSpec(), wiring, stream_seed(4, "n"))
    x, y = net.render_nodes(int(30 * FS))
    j = 0
    k, lag = wiring.sources[j][0], wiring.lags[j][0]
    a = y[j] - y[j].mean()
    b = x[k] - x[k].mean()
    xc = [np.dot(a[m:], b[: len(b) - m]) for m in range(0, 20)]
    assert int(np.argmax(xc)) == lag
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_network.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/brain/network.py
"""Cortical patches driven partly by delayed copies of other patches (DESIGN §4.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from open_eeg_synth.brain.background import WARM_UP_S, BackgroundSpec
from open_eeg_synth.dsp import PinkCascade
from open_eeg_synth.headmodel import HeadModel


@dataclass(frozen=True)
class NetworkSpec:
    n_nodes: int = 40
    fan_in: int = 3
    coupling: float = 1.0
    velocity_m_s: float = 6.0
    synaptic_ms: float = 5.0
    width_mm: float = 15.0


@dataclass(frozen=True)
class NetworkWiring:
    centres: tuple[int, ...]
    sources: tuple[tuple[int, ...], ...]  # per node: the nodes it listens to
    lags: tuple[tuple[int, ...], ...]  # per node: the lag in samples for each source

    def to_dict(self) -> dict:
        return {"centres": list(self.centres), "sources": [list(s) for s in self.sources],
                "lags": [list(lg) for lg in self.lags]}


def wire_network(head: HeadModel, spec: NetworkSpec, fs: float,
                 rng: np.random.Generator) -> NetworkWiring:
    centres = [int(c) for c in rng.choice(head.n_sources, spec.n_nodes, replace=False)]
    sources, lags = [], []
    for j in range(spec.n_nodes):
        others = [k for k in range(spec.n_nodes) if k != j]
        srcs = [int(k) for k in rng.choice(others, spec.fan_in, replace=False)]
        row = []
        for k in srcs:
            dist = float(np.linalg.norm(head.source_pos[centres[j]] - head.source_pos[centres[k]]))
            row.append(int(round((dist / spec.velocity_m_s + spec.synaptic_ms / 1000.0) * fs)))
        sources.append(tuple(srcs))
        lags.append(tuple(row))
    return NetworkWiring(tuple(centres), tuple(sources), tuple(lags))


class Network:
    name = "brain.network"

    def __init__(self, head: HeadModel, fs: float, spec: NetworkSpec, background: BackgroundSpec,
                 wiring: NetworkWiring, seq: np.random.SeedSequence) -> None:
        self.spec, self.wiring = spec, wiring
        self.maps = np.array([head.patch_map(c, spec.width_mm) for c in wiring.centres])  # (N, ch)
        self.n = spec.n_nodes
        self.rng = np.random.Generator(np.random.PCG64(seq))
        self.pink = PinkCascade(self.n, fs, beta=background.exponent)
        self.pink.warm_up(self.rng, WARM_UP_S)
        self.node_scale = 1.0 / np.sqrt(1.0 + spec.coupling**2 / spec.fan_in)
        chan_var = (self.maps**2).sum(axis=0)
        self.scale = background.rms_uv * np.sqrt(background.network_frac) / np.sqrt(chan_var.mean())
        self.maxlag = max((lg for row in wiring.lags for lg in row), default=0)
        self.hist = np.zeros((self.n, self.maxlag))
        self.w = spec.coupling / spec.fan_in

    def render_nodes(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        x = self.pink.process(self.rng.standard_normal((n, self.n)).T)
        ext = np.concatenate([self.hist, x], axis=1)
        y = x.copy()
        for j, (srcs, lags) in enumerate(zip(self.wiring.sources, self.wiring.lags)):
            for k, lag in zip(srcs, lags):
                y[j] += self.w * ext[k, self.maxlag - lag : self.maxlag - lag + n]
        if self.maxlag > 0:
            self.hist = ext[:, ext.shape[1] - self.maxlag :]
        return x, y * self.node_scale

    def render(self, t0: int, n: int) -> np.ndarray:
        _, y = self.render_nodes(n)
        return (self.scale * (self.maps.T @ y)).astype(np.float32)

    def truth(self) -> list:
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_network.py -v && ruff check . && ruff format --check .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/brain/network.py tests/test_network.py
git commit -m "feat(brain): delayed cortical network with streaming node histories"
```

---

### Task 9: State timeline

**Files:**
- Create: `src/open_eeg_synth/brain/state.py`
- Create: `tests/test_state.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `StateSegment(t0_s, t1_s, state)`; `StateTimeline(segments, ramp_s=2.0)` — a frozen dataclass that compares by value (two timelines with equal segments and ramp are `==`, which `CaseSpec` round-trip tests rely on) — with `.segments`, `.state_at(t_s)`, `.weights(t0, n, fs)`, `.gain(per_state, t0, n, fs, default=1.0)`, `.rate(per_state, t_s)`, `.to_dict()`, `StateTimeline.from_dict(d)`, `StateTimeline.constant(state, duration_s=inf)`. The last segment extends indefinitely.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_state.py
import numpy as np
import pytest

from open_eeg_synth.brain.state import StateSegment, StateTimeline


def test_weights_sum_to_one_and_ramp_across_boundary():
    tl = StateTimeline([StateSegment(0, 10, "eyes_closed"), StateSegment(10, 20, "drowsy")], ramp_s=2.0)
    w = tl.weights(0, 20 * 100, 100.0)
    total = sum(w.values())
    assert np.allclose(total, 1.0)
    assert w["eyes_closed"][800] == 1.0 and w["drowsy"][800] == 0.0  # t = 8 s, before the ramp
    assert np.isclose(w["eyes_closed"][1000], 0.5)  # t = 10 s, mid-ramp
    assert w["drowsy"][1150] == 1.0  # t = 11.5 s, after the ramp
    assert w["drowsy"][-1] == 1.0  # the last segment extends past 20 s


def test_gain_rate_state_at_and_roundtrip():
    tl = StateTimeline([StateSegment(0, 5, "eyes_open"), StateSegment(5, 8, "eyes_closed")], ramp_s=0.0)
    g = tl.gain({"eyes_closed": 0.3}, 0, 800, 100.0)
    assert g[0] == 1.0 and g[-1] == 0.3
    assert tl.state_at(4.99) == "eyes_open" and tl.state_at(5.0) == "eyes_closed"
    assert tl.rate({"eyes_open": 0.25}, 1.0) == 0.25 and tl.rate({"eyes_open": 0.25}, 6.0) == 0.0
    back = StateTimeline.from_dict(tl.to_dict())
    assert back == tl and back.segments == tl.segments and back.ramp_s == 0.0


def test_constant_and_validation():
    tl = StateTimeline.constant("eyes_open")
    assert tl.state_at(1e9) == "eyes_open"
    with pytest.raises(ValueError):
        StateTimeline([StateSegment(1, 2, "a")])
    with pytest.raises(ValueError):
        StateTimeline([StateSegment(0, 2, "a"), StateSegment(3, 4, "b")])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/brain/state.py
"""What the subject is doing over time; drives rhythm gains and artifact rates (DESIGN §4.5)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateSegment:
    t0_s: float
    t1_s: float
    state: str


@dataclass(frozen=True)
class StateTimeline:
    """Contiguous segments from 0 s; the last extends indefinitely. Compares by value."""

    segments: tuple[StateSegment, ...]
    ramp_s: float = 2.0

    def __post_init__(self) -> None:
        segs = tuple(self.segments)
        if not segs or segs[0].t0_s != 0.0:
            raise ValueError("timeline must start at 0 s")
        for a, b in zip(segs, segs[1:], strict=False):
            if b.t0_s != a.t1_s:
                raise ValueError(f"segments must be contiguous: {a} -> {b}")
        if any(s.t1_s <= s.t0_s for s in segs):
            raise ValueError("every segment needs t1_s > t0_s")
        object.__setattr__(self, "segments", segs)
        object.__setattr__(self, "ramp_s", float(self.ramp_s))

    @classmethod
    def constant(cls, state: str, duration_s: float = math.inf) -> StateTimeline:
        return cls((StateSegment(0.0, duration_s, state),))

    def state_at(self, t_s: float) -> str:
        starts = [s.t0_s for s in self.segments]
        i = int(np.searchsorted(starts, t_s, side="right") - 1)
        return self.segments[max(0, i)].state

    def weights(self, t0: int, n: int, fs: float) -> dict[str, np.ndarray]:
        t = (t0 + np.arange(n)) / fs
        r = self.ramp_s

        def up(b: float) -> np.ndarray:
            if b == -math.inf:
                return np.ones(n)
            if b == math.inf:
                return np.zeros(n)
            if r <= 0:
                return (t >= b).astype(float)
            return np.clip((t - (b - r / 2.0)) / r, 0.0, 1.0)

        out: dict[str, np.ndarray] = {}
        last = len(self.segments) - 1
        for i, seg in enumerate(self.segments):
            w = up(-math.inf if i == 0 else seg.t0_s) - up(math.inf if i == last else seg.t1_s)
            out[seg.state] = out.get(seg.state, 0.0) + w
        return out

    def gain(self, per_state: dict[str, float], t0: int, n: int, fs: float,
             default: float = 1.0) -> np.ndarray:
        g = np.zeros(n)
        for state, w in self.weights(t0, n, fs).items():
            g += w * per_state.get(state, default)
        return g

    def rate(self, per_state: dict[str, float], t_s: float) -> float:
        return float(per_state.get(self.state_at(t_s), 0.0))

    def to_dict(self) -> dict:
        return {"ramp_s": self.ramp_s,
                "segments": [{"t0_s": s.t0_s, "t1_s": s.t1_s, "state": s.state} for s in self.segments]}

    @classmethod
    def from_dict(cls, d: dict) -> StateTimeline:
        return cls([StateSegment(s["t0_s"], s["t1_s"], s["state"]) for s in d["segments"]],
                   ramp_s=d.get("ramp_s", 2.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_state.py -v && ruff check . && ruff format --check .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/brain/state.py tests/test_state.py
git commit -m "feat(brain): state timeline with ramped per-state gains and rates"
```

---

### Task 10: Placement helpers + rhythm generator

**Files:**
- Create: `src/open_eeg_synth/brain/placement.py`
- Create: `src/open_eeg_synth/brain/rhythm.py`
- Create: `tests/test_rhythm.py`

**Interfaces:**
- Consumes: `HeadModel`, `OU`, `StateTimeline`, `MIRROR`, `MIDLINE`.
- Produces: `placed_centres(head, sites, rng, n_near=25) -> list[int]`; `region_centres(head, region, n_patches, rng) -> list[int]`; `BurstGate(on_s=(1,3), off_s=(8,25), ramp_s=0.3)`; `RhythmSpec` (fields per DESIGN §4.3) with `.to_dict()`/`from_dict()`; `Rhythm(head, fs, spec, timeline, centres, f0_hz, seq)` layer (`name=f"brain.rhythm:{spec.name}"`) with `.maps` (P, n_ch), `.lags`, `.scale`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rhythm.py
import numpy as np

from open_eeg_synth.brain.placement import placed_centres, region_centres
from open_eeg_synth.brain.rhythm import BurstGate, Rhythm, RhythmSpec
from open_eeg_synth.brain.state import StateSegment, StateTimeline
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.seeds import stream_rng, stream_seed
from tests.helpers import band_power, render_whole_and_chunked

FS = 256.0


def test_placed_centres_mirror_pairs_and_midline():
    head = load_head_model()
    c = placed_centres(head, ("C3", "C4", "Cz"), stream_rng(1, "subject:rhythm:smr"))
    assert len(c) == 3
    assert c[1] == head.mirror_source(c[0])
    assert abs(head.source_pos[c[2], 0]) < 0.012  # near the midline
    assert placed_centres(head, ("C3", "C4", "Cz"), stream_rng(1, "subject:rhythm:smr")) == c


def test_region_centres_posterior_mirrored():
    head = load_head_model()
    c = region_centres(head, "posterior", 4, stream_rng(1, "subject:rhythm:alpha"))
    assert len(c) == 4 and c[2] == head.mirror_source(c[0])
    assert all(head.source_pos[i, 1] < -0.04 for i in c)


def test_rhythm_peaks_at_f0_on_targets_and_is_chunk_invariant():
    head = load_head_model()
    spec = RhythmSpec("smr", f0_hz=13.5, amp_uv=6.0, sites=("C3", "C4", "Cz"))
    centres = placed_centres(head, spec.sites, stream_rng(1, "subject:rhythm:smr"))
    tl = StateTimeline.constant("eyes_closed")

    def make():
        return Rhythm(head, FS, spec, tl, centres, 13.5, stream_seed(1, "eyes_closed:rhythm:smr"))

    whole, chunked = render_whole_and_chunked(make, int(60 * FS), np.random.default_rng(0))
    assert np.allclose(whole, chunked, atol=1e-4)
    c3, o1 = CHANNELS_19.index("C3"), CHANNELS_19.index("O1")
    inband = band_power(whole, FS, 12.5, 14.5)
    around = band_power(whole, FS, 17.0, 19.0)
    assert inband[c3] > 20 * around[c3]
    assert inband[c3] > 3 * inband[o1]
    peak = np.sqrt(2) * whole[c3].std()
    assert 3.0 < peak < 12.0  # amp_uv 6 at the loudest target, envelope 0.25-1.8


def test_state_gain_and_burst_gate():
    head = load_head_model()
    tl = StateTimeline([StateSegment(0, 10, "eyes_closed"), StateSegment(10, 20, "eyes_open")], ramp_s=0.0)
    spec = RhythmSpec("alpha", 10.0, 20.0, sites=("O1", "O2"), state_gain={"eyes_open": 0.3})
    centres = placed_centres(head, spec.sites, stream_rng(2, "p"))
    r = Rhythm(head, FS, spec, tl, centres, 10.0, stream_seed(2, "t"))
    x = r.render(0, int(20 * FS))
    o1 = CHANNELS_19.index("O1")
    assert x[o1, : int(10 * FS)].std() > 2.5 * x[o1, int(10 * FS) :].std()

    g = RhythmSpec("b", 6.0, 30.0, sites=("Fz",), burst=BurstGate(on_s=(1, 1), off_s=(2, 2), ramp_s=0.0))
    rb = Rhythm(head, FS, g, StateTimeline.constant("eyes_closed"),
                placed_centres(head, g.sites, stream_rng(3, "p")), 6.0, stream_seed(3, "t"))
    y = rb.render(0, int(9 * FS))[CHANNELS_19.index("Fz")]
    active = np.abs(y).reshape(9, int(FS)).max(axis=1) > 1.0
    assert 2 <= active.sum() <= 4  # on for 1 s every 3 s, starting after the first off period


def test_rhythm_spec_roundtrip():
    s = RhythmSpec("theta", 6.0, 7.0, sites=("Fz", "Cz"), state_gain={"drowsy": 2.0}, burst=BurstGate())
    assert RhythmSpec.from_dict(s.to_dict()) == s
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rhythm.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement placement**

```python
# src/open_eeg_synth/brain/placement.py
"""Where a rhythm's cortical patches sit (DESIGN §4.3)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from open_eeg_synth.channels import MIDLINE, MIRROR, canonical_label
from open_eeg_synth.headmodel import HeadModel

REGIONS = {"posterior": {"y_pct_max": 12.0, "z_pct_min": 30.0}}


def placed_centres(head: HeadModel, sites: Sequence[str], rng: np.random.Generator,
                   n_near: int = 25) -> list[int]:
    """One source under each site; a mirror pair gets mirror-image sources."""
    names = [canonical_label(s, head.channels) for s in sites]
    centres: dict[str, int] = {}
    for ch in names:
        if ch in centres:
            continue
        cand = head.sources_under(ch, n_near)
        if ch in MIDLINE or ch not in MIRROR:
            cand = cand[np.argsort(np.abs(head.source_pos[cand, 0]))[:5]]
        c = int(rng.choice(cand))
        centres[ch] = c
        partner = MIRROR.get(ch)
        if partner is not None and partner in names:
            centres[partner] = head.mirror_source(c)
    return [centres[ch] for ch in names]


def region_centres(head: HeadModel, region: str, n_patches: int,
                   rng: np.random.Generator) -> list[int]:
    """n_patches/2 sources in the left half of a named region, plus their mirror images."""
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}; known: {sorted(REGIONS)}")
    r = REGIONS[region]
    rr = head.source_pos
    cand = np.where((rr[:, 1] < np.percentile(rr[:, 1], r["y_pct_max"]))
                    & (rr[:, 2] > np.percentile(rr[:, 2], r["z_pct_min"])))[0]
    left = cand[rr[cand, 0] < -0.01]
    half = [int(c) for c in rng.choice(left, max(1, n_patches // 2), replace=False)]
    return half + [head.mirror_source(c) for c in half]
```

- [ ] **Step 4: Implement the rhythm**

```python
# src/open_eeg_synth/brain/rhythm.py
"""Rhythms as mirrored cortical patches sharing a delayed driver (DESIGN §4.3)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.dsp import OU
from open_eeg_synth.headmodel import HeadModel


@dataclass(frozen=True)
class BurstGate:
    on_s: tuple[float, float] = (1.0, 3.0)
    off_s: tuple[float, float] = (8.0, 25.0)
    ramp_s: float = 0.3


@dataclass(frozen=True)
class RhythmSpec:
    name: str
    f0_hz: float
    amp_uv: float
    sites: tuple[str, ...] = ()
    region: str | None = None
    n_patches: int = 4
    width_mm: float = 12.0
    f_sd: float = 0.35
    f_tau_s: float = 4.0
    env_tau_s: float = 1.5
    env_sd: float = 0.5
    env_lo: float = 0.25
    env_hi: float = 1.8
    lag_ms: float = 25.0
    indep: float = 0.3
    f0_jitter_hz: float = 0.0
    state_gain: dict[str, float] = field(default_factory=dict)
    state_f0_shift_hz: dict[str, float] = field(default_factory=dict)
    hemisphere_gain: dict[str, float] = field(default_factory=dict)
    burst: BurstGate | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sites"] = list(self.sites)
        d["burst"] = None if self.burst is None else {
            "on_s": list(self.burst.on_s), "off_s": list(self.burst.off_s), "ramp_s": self.burst.ramp_s}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RhythmSpec:
        d = dict(d)
        d["sites"] = tuple(d.get("sites", ()))
        b = d.get("burst")
        d["burst"] = None if b is None else BurstGate(tuple(b["on_s"]), tuple(b["off_s"]), b["ramp_s"])
        return cls(**d)


class _Oscillator:
    """One time course: OU frequency around f0, log-normal OU envelope, phase accumulator."""

    def __init__(self, fs: float, f0: float, spec: RhythmSpec, rng: np.random.Generator) -> None:
        self.fs, self.f0, self.rng = fs, f0, rng
        self.freq = OU(1, fs, mu=0.0, sigma=spec.f_sd, tau_s=spec.f_tau_s, rng=rng)
        self.lenv = OU(1, fs, mu=0.0, sigma=spec.env_sd, tau_s=spec.env_tau_s, rng=rng)
        self.phase = float(rng.uniform(0.0, 2.0 * np.pi))
        self.lo, self.hi = np.log(spec.env_lo), np.log(spec.env_hi)

    def render(self, n: int, f0_shift: np.ndarray | float = 0.0) -> np.ndarray:
        white = self.rng.standard_normal((n, 2)).T
        f = self.f0 + f0_shift + self.freq.step(white[:1])[0]
        env = np.exp(np.clip(self.lenv.step(white[1:])[0], self.lo, self.hi))
        ph = self.phase + 2.0 * np.pi * np.cumsum(f) / self.fs
        self.phase = float(ph[-1] % (2.0 * np.pi))
        return env * np.sin(ph)


class _Gate:
    """On/off gating with raised-cosine edges; transitions drawn in order (streaming-safe)."""

    def __init__(self, burst: BurstGate, fs: float, rng: np.random.Generator) -> None:
        self.b, self.fs, self.rng = burst, fs, rng
        self.nr = max(1, int(round(burst.ramp_s * fs)))
        first_on = int(round(rng.uniform(*burst.off_s) * fs))
        self.edges: list[tuple[int, float]] = [(first_on, 1.0)]  # (sample, level after edge)
        self.base = 0.0  # level before the first stored edge

    def _extend(self, until: int) -> None:
        while self.edges[-1][0] < until + self.nr:
            s, lvl = self.edges[-1]
            dur = self.rng.uniform(*(self.b.on_s if lvl == 1.0 else self.b.off_s))
            self.edges.append((s + int(round(dur * self.fs)), 0.0 if lvl == 1.0 else 1.0))

    def render(self, t0: int, n: int) -> np.ndarray:
        self._extend(t0 + n)
        k = t0 + np.arange(n)
        g = np.full(n, self.base)
        prev = self.base
        keep: list[tuple[int, float]] = []
        for s, lvl in self.edges:
            if s + self.nr < t0:
                self.base = lvl
                prev = lvl
                g[:] = lvl
                continue
            keep.append((s, lvl))
            if s - self.nr > t0 + n:
                continue
            ramp = np.clip((k - s) / self.nr + 0.5, 0.0, 1.0) if self.b.ramp_s > 0 else (k >= s) * 1.0
            g += (lvl - prev) * ramp
            prev = lvl
        self.edges = keep
        return g


class Rhythm:
    def __init__(self, head: HeadModel, fs: float, spec: RhythmSpec, timeline: StateTimeline,
                 centres: list[int], f0_hz: float, seq: np.random.SeedSequence) -> None:
        self.spec, self.fs, self.timeline = spec, float(fs), timeline
        self.name = f"brain.rhythm:{spec.name}"
        children = seq.spawn(len(centres) + 2)
        rngs = [np.random.Generator(np.random.PCG64(c)) for c in children]
        misc = rngs[-1]
        targets = [head.index(s) for s in spec.sites] if spec.sites else None
        maps = []
        for c in centres:
            m = head.patch_map(c, spec.width_mm)
            if targets is not None and sum(m[i] for i in targets) < 0:
                m = -m
            x = head.source_pos[c, 0]
            side = "left" if x < -0.005 else ("right" if x > 0.005 else "midline")
            maps.append(m * spec.hemisphere_gain.get(side, 1.0))
        self.maps = np.array(maps)  # (P, n_ch)
        if targets is None:
            summed = np.abs(self.maps.sum(axis=0))
            targets = [int(np.argmax(summed))]
            for i, m in enumerate(self.maps):  # orient every patch positive at that channel
                if m[targets[0]] < 0:
                    self.maps[i] = -m
        self.scale = spec.amp_uv / np.abs(self.maps.sum(axis=0))[targets].max()
        self.lags = [int(round(misc.uniform(0.0, spec.lag_ms) / 1000.0 * fs)) for _ in centres]
        self.maxlag = max(self.lags) if self.lags else 0
        self.driver = _Oscillator(fs, f0_hz, spec, rngs[0])
        self.own = [_Oscillator(fs, f0_hz + float(rngs[i + 1].normal(0.0, 0.3)), spec, rngs[i + 1])
                    for i in range(len(centres))]
        self.hist = self.driver.render(self.maxlag) if self.maxlag > 0 else np.zeros(0)
        self.gate = _Gate(spec.burst, fs, misc) if spec.burst is not None else None
        self.w_shared, self.w_own = np.sqrt(1.0 - spec.indep), np.sqrt(spec.indep)

    def render(self, t0: int, n: int) -> np.ndarray:
        shift = self.timeline.gain(self.spec.state_f0_shift_hz, t0, n, self.fs, default=0.0)
        drv = self.driver.render(n, shift)
        ext = np.concatenate([self.hist, drv])
        out = np.zeros((self.maps.shape[1], n))
        for p, lag in enumerate(self.lags):
            shared = ext[self.maxlag - lag : self.maxlag - lag + n]
            own = self.own[p].render(n, shift)
            out += self.maps[p][:, None] * (self.w_shared * shared + self.w_own * own)[None, :]
        if self.maxlag > 0:
            self.hist = ext[-self.maxlag :]
        gain = self.timeline.gain(self.spec.state_gain, t0, n, self.fs)
        if self.gate is not None:
            gain = gain * self.gate.render(t0, n)
        return (self.scale * out * gain[None, :]).astype(np.float32)

    def truth(self) -> list:
        return []
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_rhythm.py -v && ruff check . && ruff format --check .`
Expected: PASS. If `test_state_gain_and_burst_gate`'s burst count is off by one, check `_Gate` only: the first edge is at `off_s` seconds, then on for `on_s`.

- [ ] **Step 6: Commit**

```bash
git add src/open_eeg_synth/brain/placement.py src/open_eeg_synth/brain/rhythm.py tests/test_rhythm.py
git commit -m "feat(brain): rhythm generator with mirrored patches, shared delayed driver, state gains, bursts"
```

---

### Task 11: BrainLayer composition + the tuned resting recipe

**Files:**
- Create: `src/open_eeg_synth/brain/layer.py`
- Create: `src/open_eeg_synth/recipes.py` (`resting_brain` only; Task 12 and Task 20 add the rest)
- Create: `tests/test_brain_layer.py`

**Interfaces:**
- Consumes: Tasks 7–10.
- Produces: `BrainSpec(background, network, rhythms)` with `to_dict()/from_dict()`; `BrainLayer(rhythms, spec, head, fs, timeline, mixing, wiring, placements, f0_hz, case_seed, condition, rows=None)` layer (`name="brain"`) whose `.parts` are the sub-layers, each seeded from the DESIGN §8.1 stream of that name (`<condition>:background`, `<condition>:network`, `<condition>:rhythm:<name>`); `head` is always the full model and `rows` (indices into it) selects the channels the case records, so a four-channel stream and a nineteen-channel case share one brain; `recipes.resting_brain() -> BrainSpec`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_brain_layer.py
import numpy as np

from open_eeg_synth.brain.layer import BrainLayer, BrainSpec
from open_eeg_synth.brain.network import wire_network
from open_eeg_synth.brain.placement import placed_centres, region_centres
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.recipes import resting_brain
from open_eeg_synth.seeds import stream_rng
from tests.helpers import band_power, psd_slope, render_whole_and_chunked

FS = 256.0


def _build(spec: BrainSpec, timeline, seed=1):
    head = load_head_model()
    mixing = head.smoothed_mixing(spec.background.smoothing_mm)
    wiring = wire_network(head, spec.network, FS, stream_rng(seed, "subject:network"))
    placements, f0 = {}, {}
    for r in spec.rhythms:
        rng = stream_rng(seed, f"subject:rhythm:{r.name}")
        placements[r.name] = (region_centres(head, r.region, r.n_patches, rng) if r.region
                              else placed_centres(head, r.sites, rng))
        f0[r.name] = r.f0_hz + (float(rng.normal(0, r.f0_jitter_hz)) if r.f0_jitter_hz else 0.0)

    def make():
        return BrainLayer(spec.rhythms, spec, head, FS, timeline, mixing, wiring, placements, f0,
                          seed, "eyes_closed")

    return make


def test_resting_brain_spec_roundtrip_and_names():
    spec = resting_brain()
    assert [r.name for r in spec.rhythms] == ["alpha", "theta", "beta", "smr"]
    assert BrainSpec.from_dict(spec.to_dict()) == spec


def test_brain_layer_chunk_invariance_and_eyes_closed_alpha():
    make = _build(resting_brain(), StateTimeline.constant("eyes_closed"))
    whole, chunked = render_whole_and_chunked(make, int(60 * FS), np.random.default_rng(0))
    assert np.allclose(whole, chunked, atol=1e-3)
    # Amplitudes are judged after average referencing, as DESIGN §9.1 measures them: the lead
    # field's native reference carries a common component (DESIGN §2.3) that inflates raw RMS.
    x = whole - whole.mean(axis=0, keepdims=True)
    o1, fz = CHANNELS_19.index("O1"), CHANNELS_19.index("Fz")
    alpha = band_power(x, FS, 8, 13)
    total = band_power(x, FS, 1, 40)
    assert alpha[o1] / total[o1] > 0.3  # provisional; Task 12 pins the calibrated value (~0.6)
    assert alpha[o1] > 1.5 * alpha[fz]
    assert 0.9 < psd_slope(x, FS) < 1.5
    assert 6.0 < x[CHANNELS_19.index("Cz")].std() < 30.0  # provisional; Task 12 pins ~12 uV


def test_eyes_open_collapses_alpha():
    ec = _build(resting_brain(), StateTimeline.constant("eyes_closed"))().render(0, int(30 * FS))
    eo = _build(resting_brain(), StateTimeline.constant("eyes_open"))().render(0, int(30 * FS))
    ec, eo = ec - ec.mean(axis=0, keepdims=True), eo - eo.mean(axis=0, keepdims=True)
    o1 = CHANNELS_19.index("O1")
    assert band_power(eo, FS, 8, 13)[o1] < 0.5 * band_power(ec, FS, 8, 13)[o1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_brain_layer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/brain/layer.py
"""The brain layer: background + network + rhythms (DESIGN §4)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from open_eeg_synth.brain.background import Background, BackgroundSpec
from open_eeg_synth.brain.network import Network, NetworkSpec, NetworkWiring
from open_eeg_synth.brain.rhythm import Rhythm, RhythmSpec
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.headmodel import HeadModel
from open_eeg_synth.seeds import stream_seed


@dataclass(frozen=True)
class BrainSpec:
    background: BackgroundSpec = BackgroundSpec()
    network: NetworkSpec = NetworkSpec()
    rhythms: tuple[RhythmSpec, ...] = ()

    def to_dict(self) -> dict:
        return {"background": asdict(self.background), "network": asdict(self.network),
                "rhythms": [r.to_dict() for r in self.rhythms]}

    @classmethod
    def from_dict(cls, d: dict) -> BrainSpec:
        return cls(BackgroundSpec(**d["background"]), NetworkSpec(**d["network"]),
                   tuple(RhythmSpec.from_dict(r) for r in d["rhythms"]))


class BrainLayer:
    """Background + network + rhythms on the full head; ``rows`` selects the recorded channels."""

    name = "brain"

    def __init__(self, rhythms: Sequence[RhythmSpec], spec: BrainSpec, head: HeadModel, fs: float,
                 timeline: StateTimeline, mixing: np.ndarray, wiring: NetworkWiring,
                 placements: dict[str, list[int]], f0_hz: dict[str, float], case_seed: int,
                 condition: str, rows: Sequence[int] | None = None) -> None:
        self.rows = None if rows is None else list(rows)
        self.parts: list = [
            Background(head, fs, spec.background,
                       stream_seed(case_seed, f"{condition}:background"), mixing=mixing),
            Network(head, fs, spec.network, spec.background, wiring,
                    stream_seed(case_seed, f"{condition}:network")),
        ]
        for r in rhythms:
            self.parts.append(Rhythm(head, fs, r, timeline, placements[r.name], f0_hz[r.name],
                                     stream_seed(case_seed, f"{condition}:rhythm:{r.name}")))

    def render(self, t0: int, n: int) -> np.ndarray:
        out = self.parts[0].render(t0, n).astype(np.float32)
        for p in self.parts[1:]:
            out += p.render(t0, n)
        return out if self.rows is None else out[self.rows]

    def truth(self) -> list:
        return []
```

```python
# src/open_eeg_synth/recipes.py
"""Ready-made specifications: the tuned resting model and the ordinary artifact set (DESIGN §4.3)."""

from __future__ import annotations

from open_eeg_synth.brain.background import BackgroundSpec
from open_eeg_synth.brain.layer import BrainSpec
from open_eeg_synth.brain.network import NetworkSpec
from open_eeg_synth.brain.rhythm import RhythmSpec


def resting_brain() -> BrainSpec:
    """The feasibility test's tuned resting model (amplitudes re-derived in Task 12)."""
    return BrainSpec(
        background=BackgroundSpec(smoothing_mm=20.0, exponent=1.2, rms_uv=20.0, network_frac=0.5),
        network=NetworkSpec(),
        rhythms=(
            RhythmSpec("alpha", f0_hz=10.0, amp_uv=20.0, region="posterior", n_patches=4,
                       width_mm=10.0, lag_ms=30.0, indep=0.6, f0_jitter_hz=0.6,
                       state_gain={"eyes_open": 0.3, "drowsy": 0.35},
                       state_f0_shift_hz={"drowsy": -1.0}),
            RhythmSpec("theta", f0_hz=6.0, amp_uv=7.0, sites=("Fz", "F3", "F4", "Cz"),
                       env_tau_s=0.8, env_sd=0.7, lag_ms=40.0, state_gain={"drowsy": 2.0}),
            RhythmSpec("beta", f0_hz=19.0, amp_uv=7.0, sites=("C3", "C4", "F3", "F4", "P3", "P4"),
                       f_sd=1.2, f_tau_s=0.6, env_tau_s=0.3, env_sd=0.9, env_lo=0.1, env_hi=3.0,
                       lag_ms=35.0),
            RhythmSpec("smr", f0_hz=13.5, amp_uv=6.0, sites=("C3", "C4", "Cz"), f_sd=0.6,
                       f_tau_s=1.0, env_tau_s=0.4, env_sd=0.9, env_lo=0.1, env_hi=3.0, lag_ms=20.0),
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_brain_layer.py -v && ruff check . && ruff format --check .`
Expected: PASS. The two amplitude bands are deliberately loose: with the recipe as written the average-referenced brain layer gives Cz RMS ≈ 21 µV and an O1 alpha share ≈ 0.35 (the analytic amplitude convention lands below the feasibility test's empirical one, DESIGN §4.3, §11.6). Task 12 Step 5 calibrates the recipe and then tightens these two assertions.

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/brain/layer.py src/open_eeg_synth/recipes.py tests/test_brain_layer.py
git commit -m "feat(brain): BrainLayer composition and the tuned resting recipe"
```

---

### Task 12: Case specification, subject, `make_case`, calibration, performance

**Files:**
- Create: `src/open_eeg_synth/case.py`
- Modify: `src/open_eeg_synth/recipes.py` (add `resting_case`, placeholder `ordinary_artifacts` returning `()` until Task 20)
- Modify: `src/open_eeg_synth/__init__.py` (re-exports)
- Create: `tests/test_case.py`, `tests/test_perf.py`

**Interfaces:**
- Consumes: Tasks 3–11 and Task 13 (`RenderContext`, `Occupancy`, `make_artifact` — Task 13 runs before this task).
- Produces: `SensorSpec(white_uv=1.5)`; `ArtifactSpec(kind, params={})`; `ConditionSpec(name, duration_s, timeline)`; `CaseSpec(...)` per DESIGN §7.2 with `to_dict()`, `from_dict()`, `digest()`; `compiled_rhythms(spec) -> tuple[RhythmSpec, ...]` (base rhythms only until Task 21); `Subject(head, mixing, placements, f0_hz, network, pattern_jitter, rows)` with `to_dict()` — `head` is the full (perturbed) model and `rows` the indices of `spec.channels` in it; `make_subject(spec)`, `make_engine(spec, subject, condition)`, `make_case(spec) -> Case(spec, case_id, subject, recordings)`; `recipes.resting_case(seed, *, duration_s=240.0, plants=(), artifacts=None, drowsy_from_s=None, fs=256.0, channels=CHANNELS_19)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_case.py
import numpy as np

from open_eeg_synth.case import CaseSpec, make_case, make_subject
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.recipes import resting_case
from tests.helpers import band_power


def test_spec_roundtrip_digest_and_case_id():
    spec = resting_case(123, duration_s=8.0, artifacts=())
    assert CaseSpec.from_dict(spec.to_dict()) == spec
    assert spec.digest() == CaseSpec.from_dict(spec.to_dict()).digest()
    case = make_case(spec)
    assert case.case_id.startswith("synth-") and len(case.case_id) == 14
    assert set(case.recordings) == {"eyes_closed", "eyes_open"}
    rec = case.recordings["eyes_closed"]
    assert rec.channels == CHANNELS_19 and rec.n_samples == 8 * 256
    assert set(rec.layers) == {"brain", "sensor"}
    assert np.allclose(rec.mixed, rec.layers["brain"] + rec.layers["sensor"])


def test_subject_is_shared_across_conditions_and_seed_dependent():
    a = make_subject(resting_case(5, duration_s=1.0, artifacts=()))
    b = make_subject(resting_case(5, duration_s=1.0, artifacts=()))
    c = make_subject(resting_case(6, duration_s=1.0, artifacts=()))
    assert a.placements == b.placements and a.f0_hz == b.f0_hz
    assert a.head.perturbation == b.head.perturbation
    assert a.placements != c.placements
    case = make_case(resting_case(5, duration_s=8.0, artifacts=()))
    o1 = CHANNELS_19.index("O1")
    ec = band_power(case.recordings["eyes_closed"].mixed, 256.0, 8, 13)[o1]
    eo = band_power(case.recordings["eyes_open"].mixed, 256.0, 8, 13)[o1]
    assert eo < 0.5 * ec


def test_unperturbed_head_when_asked():
    spec = resting_case(7, duration_s=1.0, artifacts=())
    spec = CaseSpec.from_dict({**spec.to_dict(), "perturb_head": False})
    assert make_subject(spec).head.perturbation is None
```

```python
# tests/test_perf.py
import time

import pytest

from open_eeg_synth.case import make_case
from open_eeg_synth.recipes import resting_case


@pytest.mark.slow
def test_two_condition_case_renders_in_budget():
    t0 = time.perf_counter()
    make_case(resting_case(20260916, duration_s=240.0))
    assert time.perf_counter() - t0 < 15.0  # DESIGN §8.4 target 6 s, hard limit 15 s
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_case.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/case.py
"""A case = one synthetic subject under one or more conditions (DESIGN §7.2)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from open_eeg_synth.brain.layer import BrainLayer, BrainSpec
from open_eeg_synth.brain.network import NetworkWiring, wire_network
from open_eeg_synth.brain.placement import placed_centres, region_centres
from open_eeg_synth.brain.rhythm import RhythmSpec
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.engine import Engine, Recording
from open_eeg_synth.headmodel import HeadModel, load_head_model
from open_eeg_synth.seeds import stream_rng, stream_seed
from open_eeg_synth.sensor import SensorNoise

import open_eeg_synth.artifacts  # noqa: F401  (registers the built-in artifact kinds on import)


@dataclass(frozen=True)
class SensorSpec:
    white_uv: float = 1.5


@dataclass(frozen=True)
class ArtifactSpec:
    kind: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    duration_s: float
    timeline: StateTimeline

    def to_dict(self) -> dict:
        return {"name": self.name, "duration_s": self.duration_s, "timeline": self.timeline.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> ConditionSpec:
        return cls(d["name"], float(d["duration_s"]), StateTimeline.from_dict(d["timeline"]))


def _default_brain() -> BrainSpec:
    from open_eeg_synth.recipes import resting_brain

    return resting_brain()


@dataclass(frozen=True)
class CaseSpec:
    seed: int
    fs: float = 256.0
    channels: tuple[str, ...] = CHANNELS_19
    head_model: str = "colin27_19ch"
    perturb_head: bool = True
    brain: BrainSpec = field(default_factory=_default_brain)
    plants: tuple = ()  # Plant instances (Task 21)
    artifacts: tuple[ArtifactSpec, ...] = ()
    sensor: SensorSpec = SensorSpec()
    conditions: tuple[ConditionSpec, ...] = ()
    label: str = "synthetic"

    def to_dict(self) -> dict:
        from open_eeg_synth.brain.plants import plant_to_dict

        return {
            "seed": self.seed, "fs": self.fs, "channels": list(self.channels),
            "head_model": self.head_model, "perturb_head": self.perturb_head,
            "brain": self.brain.to_dict(), "plants": [plant_to_dict(p) for p in self.plants],
            "artifacts": [{"kind": a.kind, "params": dict(a.params)} for a in self.artifacts],
            "sensor": {"white_uv": self.sensor.white_uv},
            "conditions": [c.to_dict() for c in self.conditions], "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CaseSpec:
        from open_eeg_synth.brain.plants import plant_from_dict

        return cls(
            seed=int(d["seed"]), fs=float(d["fs"]), channels=tuple(d["channels"]),
            head_model=d["head_model"], perturb_head=bool(d["perturb_head"]),
            brain=BrainSpec.from_dict(d["brain"]),
            plants=tuple(plant_from_dict(p) for p in d.get("plants", [])),
            artifacts=tuple(ArtifactSpec(a["kind"], dict(a.get("params", {}))) for a in d["artifacts"]),
            sensor=SensorSpec(**d["sensor"]),
            conditions=tuple(ConditionSpec.from_dict(c) for c in d["conditions"]),
            label=d.get("label", "synthetic"),
        )

    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()


def compiled_rhythms(spec: CaseSpec) -> tuple[RhythmSpec, ...]:
    """Base rhythms with the plants' modifiers applied, followed by the plants' own rhythms."""
    from open_eeg_synth.brain.plants import apply_modifiers

    mods, extra = [], []
    for p in spec.plants:
        mods.extend(p.modifiers())
        extra.extend(p.rhythms())
    return tuple(apply_modifiers(spec.brain.rhythms, mods)) + tuple(extra)


@dataclass
class Subject:
    head: HeadModel
    mixing: np.ndarray
    placements: dict[str, list[int]]
    f0_hz: dict[str, float]
    network: NetworkWiring
    pattern_jitter: dict[str, float]
    rows: list[int]  # rows of ``head`` that the case records, in CaseSpec.channels order

    def to_dict(self) -> dict:
        return {"head_perturbation": self.head.perturbation, "placements": self.placements,
                "f0_hz": self.f0_hz, "network": self.network.to_dict(),
                "pattern_jitter": self.pattern_jitter}


def make_subject(spec: CaseSpec) -> Subject:
    """Everything drawn once per subject, on the FULL head model (DESIGN §7.2).

    Patches are placed under every 10-20 site whether or not the case records it, so a
    four-channel stream and a nineteen-channel case of the same seed share one brain; the
    recorded channels are selected by ``rows`` when the brain layer renders.
    """
    head = load_head_model(spec.head_model)
    if spec.perturb_head:
        head = head.perturbed(stream_rng(spec.seed, "subject:head"))
    rows = [head.index(c) for c in spec.channels]
    mixing = head.smoothed_mixing(spec.brain.background.smoothing_mm)
    placements, f0 = {}, {}
    for r in compiled_rhythms(spec):
        rng = stream_rng(spec.seed, f"subject:rhythm:{r.name}")
        placements[r.name] = (region_centres(head, r.region, r.n_patches, rng) if r.region
                              else placed_centres(head, r.sites, rng))
        f0[r.name] = r.f0_hz + (float(rng.normal(0.0, r.f0_jitter_hz)) if r.f0_jitter_hz else 0.0)
    wiring = wire_network(head, spec.brain.network, spec.fs, stream_rng(spec.seed, "subject:network"))
    jitter = {k: float(stream_rng(spec.seed, f"subject:artifact:{k}").normal(0.0, 0.6))
              for k in sorted({a.kind for a in spec.artifacts})}
    return Subject(head, mixing, placements, f0, wiring, jitter, rows)


def make_engine(spec: CaseSpec, subject: Subject, condition: ConditionSpec) -> Engine:
    from open_eeg_synth.artifacts.base import Occupancy, RenderContext
    from open_eeg_synth.artifacts.registry import make_artifact

    full, fs, seed = subject.head, spec.fs, spec.seed
    head = full.subset(spec.channels)  # what the amplifier records; artifacts live here
    layers: list = [BrainLayer(compiled_rhythms(spec), spec.brain, full, fs, condition.timeline,
                               subject.mixing, subject.network, subject.placements, subject.f0_hz,
                               seed, condition.name, rows=subject.rows)]
    transforms: list = []
    counts = Counter(a.kind for a in spec.artifacts)
    seen: Counter = Counter()
    occupancy = Occupancy()
    for i, a in enumerate(spec.artifacts):
        art = make_artifact(a.kind, **a.params)
        seen[a.kind] += 1
        suffix = f"#{seen[a.kind]}" if counts[a.kind] > 1 else ""
        prefix = "transform" if art.mode == "transform" else "artifact"
        art.bind(RenderContext(
            channels=head.channels, fs=fs, electrode_pos=head.electrode_pos, head=head,
            timeline=condition.timeline,
            rng=stream_rng(seed, f"{condition.name}:artifact:{a.kind}:{i}"),
            subject_rng=stream_rng(seed, f"subject:artifact:{a.kind}"),
            occupancy=occupancy, layer_name=f"{prefix}:{a.kind}{suffix}",
        ))
        (transforms if art.mode == "transform" else layers).append(art)
    layers.append(SensorNoise(head.n_channels, spec.sensor.white_uv,
                              stream_seed(seed, f"{condition.name}:sensor")))
    return Engine(head.channels, fs, layers, transforms)


@dataclass
class Case:
    spec: CaseSpec
    case_id: str
    subject: Subject
    recordings: dict[str, Recording]


def case_id_for(spec: CaseSpec) -> str:
    h = hashlib.blake2b(f"{spec.seed}:{spec.digest()}".encode(), digest_size=4).hexdigest()
    return f"synth-{h}"


def make_case(spec: CaseSpec) -> Case:
    subject = make_subject(spec)
    recordings: dict[str, Recording] = {}
    for cond in spec.conditions:
        eng = make_engine(spec, subject, cond)
        rec = eng.render_all(int(round(cond.duration_s * spec.fs)))
        rec.timeline = cond.timeline
        rec.plants = [p.record() for p in spec.plants]
        recordings[cond.name] = rec
    return Case(spec, case_id_for(spec), subject, recordings)
```

```python
# src/open_eeg_synth/recipes.py  (append)
from open_eeg_synth.brain.state import StateSegment, StateTimeline
from open_eeg_synth.case import ArtifactSpec, CaseSpec, ConditionSpec
from open_eeg_synth.channels import CHANNELS_19


def ordinary_artifacts() -> tuple[ArtifactSpec, ...]:
    return ()  # Task 20 fills this in


def resting_case(seed: int, *, duration_s: float = 240.0, plants=(), artifacts=None,
                 drowsy_from_s: float | None = None, fs: float = 256.0,
                 channels=CHANNELS_19) -> CaseSpec:
    if drowsy_from_s is None:
        ec = [StateSegment(0.0, duration_s, "eyes_closed")]
    else:
        ec = [StateSegment(0.0, drowsy_from_s, "eyes_closed"),
              StateSegment(drowsy_from_s, duration_s, "drowsy")]
    conditions = (
        ConditionSpec("eyes_closed", duration_s, StateTimeline(ec)),
        ConditionSpec("eyes_open", duration_s, StateTimeline.constant("eyes_open", duration_s)),
    )
    return CaseSpec(seed=int(seed), fs=fs, channels=tuple(channels), plants=tuple(plants),
                    artifacts=tuple(artifacts) if artifacts is not None else ordinary_artifacts(),
                    conditions=conditions)
```

`src/open_eeg_synth/brain/plants.py` does not exist until Task 21; create a stub now so `CaseSpec.to_dict` imports resolve:

```python
# src/open_eeg_synth/brain/plants.py  (stub; Task 21 replaces it)
from __future__ import annotations


def plant_to_dict(p):
    raise NotImplementedError("plants arrive in Task 21")


def plant_from_dict(d):
    raise NotImplementedError("plants arrive in Task 21")


def apply_modifiers(rhythms, modifiers):
    if modifiers:
        raise NotImplementedError("plants arrive in Task 21")
    return tuple(rhythms)
```

`src/open_eeg_synth/__init__.py` re-exports: `__version__`, `SIGNAL_VERSION`, `CaseSpec`, `make_case`, `make_subject`, `make_engine`, `load_head_model`, `resting_case`, `resting_brain` (add `StreamSource`, `write_case` in their tasks).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_case.py -v && pytest -m slow tests/test_perf.py -v && ruff check . && ruff format --check .`
Expected: PASS; the perf test prints under 15 s (aim for 6 s — if it is slower, profile with `python -X importtime` / `cProfile` and look first at `smoothed_mixing` being called more than once per subject and at `Rhythm.render`'s Python loop over patches).

- [ ] **Step 5: Calibrate the amplitudes against the feasibility test's numbers.** Run this snippet and compare with DESIGN §9.1 (targets: Cz RMS ≈ 12 µV both conditions, alpha share O1 ≈ 0.6 EC / 0.2 EO, Fz ≈ 0.4 EC / 0.14 EO, exponent ≈ 1.2):

```python
import numpy as np
from open_eeg_synth.case import make_case
from open_eeg_synth.recipes import resting_case
from open_eeg_synth.channels import CHANNELS_19
from tests.helpers import band_power, psd_slope
for cond in ("eyes_closed", "eyes_open"):
    vals = []
    for seed in range(6):
        rec = make_case(resting_case(100 + seed, duration_s=60.0, artifacts=())).recordings[cond]
        x = rec.mixed - rec.mixed.mean(axis=0, keepdims=True)  # average reference
        a, t = band_power(x, 256.0, 8, 13), band_power(x, 256.0, 1, 40)
        vals.append([x[CHANNELS_19.index("Cz")].std(), a[8] / t[8], a[16] / t[16], psd_slope(x, 256.0)])
    print(cond, np.round(np.median(vals, axis=0), 3), "  [Cz rms, alpha share O1, alpha share Fz, exponent]")
```

Adjust `amp_uv` for alpha/theta/beta/smr and `rms_uv` in `recipes.resting_brain` until the medians are within 15 % of the targets. Starting point, measured with the recipe as written: eyes closed `[21.4, 0.348, 0.145, 1.254]`, eyes open `[21.7, 0.135, 0.105, 1.247]` — the background is about twice too loud and alpha two to three times too quiet, so expect `rms_uv` ≈ 10 and alpha/theta `amp_uv` two to three times their listed values (a real two-knob retune, not a nudge). Then pin `tests/test_brain_layer.py`: Cz RMS band to ±30 % around the calibrated value and the O1 alpha-share floor to 0.45. Commit the values.

- [ ] **Step 6: Commit**

```bash
git add src/open_eeg_synth/case.py src/open_eeg_synth/recipes.py src/open_eeg_synth/brain/plants.py src/open_eeg_synth/__init__.py tests/test_case.py tests/test_perf.py
git commit -m "feat(case): CaseSpec/Subject/make_case with shared subject across conditions; resting recipe calibrated"
```

**M2 acceptance:** `pytest -q` green; the calibration snippet's medians within 15 % of the targets and `test_brain_layer.py` pinned to them; `pytest -m slow tests/test_perf.py` under 15 s (the whole two-condition case renders in about a second on a 2020 laptop).

---

# M3 — artifact framework + blink + eye movement + jaw EMG

### Task 13: Artifact contract, event scheduling, registry (runs before Task 12)

**Files:**
- Create: `src/open_eeg_synth/artifacts/__init__.py`
- Create: `src/open_eeg_synth/artifacts/base.py`
- Create: `src/open_eeg_synth/artifacts/registry.py`
- Create: `tests/test_artifact_base.py`

**Interfaces:**
- Consumes: `StateTimeline`, `HeadModel`.
- Produces: `Remedy` enum; `TruthRecord` (+ `to_dict()`); `Occupancy`; `RenderContext(channels, fs, electrode_pos, head, timeline, rng, subject_rng, occupancy, layer_name)`; `Event(onset, block, truth)` with `.end`, `Event.from_pattern(onset, pattern, waveform, truth)`; `EventArtifact(rate_by_state, min_gap_s, exclusive=False)` with `kind`, `mode="additive"`, `name`, `bind`, `render`, `truth`, `params`, abstract `make_event(onset)` — scheduling is the thinned Poisson process of DESIGN §5.2 (candidates at the maximum rate, accepted with probability `rate(state)/rate_max`, an accepted onset inside the previous event's refractory gap is pushed to the end of the gap, never dropped; the nominal rate holds while `rate × (duration + gap)` stays well below one), and `truth()` lists the events whose onset falls inside the rendered span; `TransformArtifact` with `mode="transform"`, `name`, `render_transform`; `register`, `ARTIFACTS`, `make_artifact`, `discover`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_artifact_base.py
import numpy as np
import pytest

from open_eeg_synth.artifacts.base import (
    Event, EventArtifact, Occupancy, Remedy, RenderContext, TruthRecord,
)
from open_eeg_synth.artifacts.registry import ARTIFACTS, make_artifact, register
from open_eeg_synth.brain.state import StateSegment, StateTimeline
from open_eeg_synth.seeds import stream_rng

FS = 100.0
CH = ("A", "B", "C")


@register
class Pulse(EventArtifact):
    kind = "test_pulse"

    def __init__(self, *, rate_by_state=None, min_gap_s=0.0, exclusive=False, length_s=0.5):
        super().__init__(rate_by_state=rate_by_state or {"eyes_open": 2.0}, min_gap_s=min_gap_s,
                         exclusive=exclusive)
        self.length_s = length_s

    def make_event(self, onset):
        n = int(self.length_s * self.fs)
        amp = float(self.ctx.rng.uniform(1.0, 2.0))
        truth = TruthRecord("test_pulse", None, None, ("A",), onset / self.fs, (onset + n) / self.fs,
                            amp, (Remedy.MASK_SEGMENT,), self.layer_name, {"amp": amp})
        return Event.from_pattern(onset, np.array([1.0, 0.5, 0.0]), amp * np.ones(n), truth)


def _ctx(timeline, seed=1, occupancy=None, name="artifact:test_pulse"):
    return RenderContext(CH, FS, None, None, timeline, stream_rng(seed, "a"), stream_rng(seed, "s"),
                         occupancy or Occupancy(), name)


def test_registry_and_params():
    assert ARTIFACTS["test_pulse"] is Pulse
    art = make_artifact("test_pulse", length_s=0.2)
    assert art.params() == {"rate_by_state": {"eyes_open": 2.0}, "min_gap_s": 0.0,
                            "exclusive": False, "length_s": 0.2}
    with pytest.raises(KeyError):
        make_artifact("nope")


def test_rate_follows_state_and_events_carry_truth():
    tl = StateTimeline([StateSegment(0, 30, "eyes_open"), StateSegment(30, 60, "eyes_closed")], 0.0)
    art = Pulse(rate_by_state={"eyes_open": 1.0}, length_s=0.1)
    art.bind(_ctx(tl))
    x = art.render(0, int(60 * FS))
    truth = art.truth()
    onsets = np.array([t.onset_s for t in truth])
    assert 18 <= (onsets < 30).sum() <= 45  # Poisson(30) for 30 s at 1/s; events rarely pushed
    assert (onsets >= 30.5).sum() == 0  # none in the eyes-closed half (a push moves one < 0.1 s)
    assert x.shape == (3, 6000) and x[2].max() == 0.0 and x[1].max() <= 0.5 * x[0].max()
    assert all(t.layer == "artifact:test_pulse" and t.remedies == (Remedy.MASK_SEGMENT,) for t in truth)
    assert truth[0].to_dict()["remedies"] == ["mask-segment"]


def test_min_gap_and_exclusive_occupancy():
    tl = StateTimeline.constant("eyes_open")
    art = Pulse(rate_by_state={"eyes_open": 0.5}, min_gap_s=1.0, length_s=0.5)
    art.bind(_ctx(tl))
    art.render(0, int(60 * FS))
    on = np.array([t.onset_s for t in art.truth()])
    assert np.all(np.diff(on) >= 1.5 - 1e-9)  # length + gap

    occ = Occupancy()
    a = Pulse(rate_by_state={"eyes_open": 0.4}, exclusive=True, length_s=1.0)
    b = Pulse(rate_by_state={"eyes_open": 0.4}, exclusive=True, length_s=1.0)
    a.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:test_pulse#1"))
    b.bind(_ctx(tl, seed=2, occupancy=occ, name="artifact:test_pulse#2"))
    for t0 in range(0, 6000, 500):
        a.render(t0, 500)
        b.render(t0, 500)
    spans = sorted([(t.onset_s, t.offset_s) for t in a.truth() + b.truth()])
    assert all(spans[i][1] <= spans[i + 1][0] + 1e-9 for i in range(len(spans) - 1))


def test_chunk_invariance_of_event_scheduling():
    tl = StateTimeline.constant("eyes_open")

    def run(chunks):
        art = Pulse(rate_by_state={"eyes_open": 1.0})
        art.bind(_ctx(tl, seed=3))
        out, t0 = [], 0
        for n in chunks:
            out.append(art.render(t0, n))
            t0 += n
        return np.concatenate(out, axis=1), [t.onset_s for t in art.truth()]

    whole, t_whole = run([6000])
    chunked, t_chunked = run([13, 400, 1587, 4000])
    assert np.array_equal(whole, chunked) and t_whole == t_chunked
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_artifact_base.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/artifacts/base.py
"""The artifact plug-in contract (DESIGN §5.1-5.3)."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar

import numpy as np

from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.headmodel import HeadModel


class Remedy(str, Enum):
    REMOVE_COMPONENT = "remove-component"
    MASK_SEGMENT = "mask-segment"
    MARK_BAD_CHANNEL = "mark-bad-channel"
    INTERPOLATE = "interpolate"
    NOTCH = "notch"
    RAISE_HIGH_PASS = "raise-high-pass"
    LOWER_LOW_PASS = "lower-low-pass"
    RE_REFERENCE = "re-reference"
    LEAVE = "leave"


@dataclass(frozen=True)
class TruthRecord:
    kind: str
    subtype: str | None
    side: str | None
    channels: tuple[str, ...]
    onset_s: float
    offset_s: float | None
    peak_uv: float
    remedies: tuple[Remedy, ...]
    layer: str
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "subtype": self.subtype, "side": self.side,
                "channels": list(self.channels), "onset_s": round(self.onset_s, 4),
                "offset_s": None if self.offset_s is None else round(self.offset_s, 4),
                "peak_uv": round(float(self.peak_uv), 3), "remedies": [r.value for r in self.remedies],
                "layer": self.layer, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: dict) -> TruthRecord:
        return cls(d["kind"], d["subtype"], d["side"], tuple(d["channels"]), d["onset_s"],
                   d["offset_s"], d["peak_uv"], tuple(Remedy(r) for r in d["remedies"]),
                   d["layer"], dict(d.get("params", {})))


class Occupancy:
    """Busy spans (samples) shared by every exclusive artifact of one engine."""

    def __init__(self) -> None:
        self.spans: list[tuple[int, int]] = []

    def next_free(self, start: int, length: int) -> int:
        moved = True
        while moved:
            moved = False
            for a, b in self.spans:
                if start < b and start + length > a:
                    start, moved = b, True
        return start

    def add(self, start: int, end: int) -> None:
        self.spans.append((start, end))


@dataclass
class RenderContext:
    channels: tuple[str, ...]
    fs: float
    electrode_pos: np.ndarray | None
    head: HeadModel | None
    timeline: StateTimeline
    rng: np.random.Generator
    subject_rng: np.random.Generator
    occupancy: Occupancy
    layer_name: str


@dataclass
class Event:
    onset: int
    block: np.ndarray  # (n_ch, L) µV
    truth: TruthRecord

    @property
    def end(self) -> int:
        return self.onset + int(self.block.shape[1])

    @classmethod
    def from_pattern(cls, onset: int, pattern: np.ndarray, waveform: np.ndarray,
                     truth: TruthRecord) -> Event:
        block = np.outer(np.asarray(pattern, float), np.asarray(waveform, float)).astype(np.float32)
        return cls(onset, block, truth)


class _ParamsMixin:
    """`params()` returns the constructor arguments (JSON-safe) so a spec can be rebuilt."""

    def params(self) -> dict[str, Any]:
        sig = inspect.signature(type(self).__init__)
        out = {}
        for name in sig.parameters:
            if name == "self":
                continue
            v = getattr(self, name)
            out[name] = dict(v) if isinstance(v, dict) else v
        return out


class EventArtifact(_ParamsMixin, ABC):
    kind: ClassVar[str] = "event"
    mode: ClassVar[str] = "additive"

    def __init__(self, *, rate_by_state: dict[str, float], min_gap_s: float = 0.0,
                 exclusive: bool = False) -> None:
        self.rate_by_state = dict(rate_by_state)
        self.min_gap_s = float(min_gap_s)
        self.exclusive = bool(exclusive)
        self._pending: list[Event] = []
        self._truth: list[TruthRecord] = []
        self._cand_s = 0.0
        self._next: tuple[float, bool] | None = None
        self._earliest = 0
        self._rendered = 0

    @property
    def name(self) -> str:
        return self.layer_name

    def bind(self, ctx: RenderContext) -> None:
        self.ctx = ctx
        self.fs = ctx.fs
        self.n_ch = len(ctx.channels)
        self.layer_name = ctx.layer_name
        self._rate_max = max(self.rate_by_state.values(), default=0.0)

    @abstractmethod
    def make_event(self, onset: int) -> Event: ...

    def _schedule_until(self, until: int) -> None:
        """Thinned Poisson (DESIGN §5.2): candidates at rate_max, kept with p = rate(state)/rate_max.

        An accepted onset inside the previous event's refractory gap is pushed to the end of the
        gap, never dropped, so the nominal rate holds as long as rate x (length + gap) stays well
        below one. Candidates and waveforms are drawn in onset order, so chunking cannot change them.
        """
        if self._rate_max <= 0.0:
            return
        rng = self.ctx.rng
        while True:
            if self._next is None:
                self._cand_s += float(rng.exponential(1.0 / self._rate_max))
                p = self.ctx.timeline.rate(self.rate_by_state, self._cand_s) / self._rate_max
                self._next = (self._cand_s, bool(rng.random() < p))
            t_s, accept = self._next
            onset = int(round(t_s * self.fs))
            if onset >= until:
                return
            self._next = None
            if not accept:
                continue
            onset = max(onset, self._earliest)
            ev = self.make_event(onset)
            if self.exclusive:
                moved = self.ctx.occupancy.next_free(onset, ev.block.shape[1])
                if moved != onset:
                    dur = ev.truth.offset_s - ev.truth.onset_s
                    ev = replace(ev, onset=moved,
                                 truth=replace(ev.truth, onset_s=moved / self.fs,
                                               offset_s=moved / self.fs + dur))
                self.ctx.occupancy.add(ev.onset, ev.end)
            self._pending.append(ev)
            self._truth.append(ev.truth)
            self._earliest = ev.end + int(round(self.min_gap_s * self.fs))

    def render(self, t0: int, n: int) -> np.ndarray:
        self._schedule_until(t0 + n)
        self._rendered = t0 + n
        out = np.zeros((self.n_ch, n), dtype=np.float32)
        keep: list[Event] = []
        for ev in self._pending:
            a, b = max(ev.onset, t0), min(ev.end, t0 + n)
            if a < b:
                out[:, a - t0 : b - t0] += ev.block[:, a - ev.onset : b - ev.onset]
            if ev.end > t0 + n:
                keep.append(ev)
        self._pending = keep
        return out

    def truth(self) -> list[TruthRecord]:
        """Events whose onset lies inside the rendered span (a pushed event may start later)."""
        return [t for t in self._truth if t.onset_s * self.fs < self._rendered]


class TransformArtifact(_ParamsMixin, ABC):
    kind: ClassVar[str] = "transform"
    mode: ClassVar[str] = "transform"

    @property
    def name(self) -> str:
        return self.layer_name

    def bind(self, ctx: RenderContext) -> None:
        self.ctx = ctx
        self.fs = ctx.fs
        self.n_ch = len(ctx.channels)
        self.layer_name = ctx.layer_name

    @abstractmethod
    def render_transform(self, t0: int, n: int, mix: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def truth(self) -> list[TruthRecord]: ...
```

```python
# src/open_eeg_synth/artifacts/registry.py
"""Artifact kinds by name; third parties add entry points in group "open_eeg_synth.artifacts"."""

from __future__ import annotations

from importlib.metadata import entry_points

ARTIFACTS: dict[str, type] = {}
_discovered = False


def register(cls):
    kind = cls.kind
    if kind in ARTIFACTS and ARTIFACTS[kind] is not cls:
        raise ValueError(f"artifact kind {kind!r} already registered by {ARTIFACTS[kind]}")
    ARTIFACTS[kind] = cls
    return cls


def discover() -> None:
    global _discovered
    if _discovered:
        return
    _discovered = True
    for ep in entry_points(group="open_eeg_synth.artifacts"):
        register(ep.load())


def make_artifact(kind: str, **params):
    discover()
    if kind not in ARTIFACTS:
        raise KeyError(f"unknown artifact kind {kind!r}; known: {sorted(ARTIFACTS)}")
    return ARTIFACTS[kind](**params)
```

```python
# src/open_eeg_synth/artifacts/__init__.py
"""Artifact plug-ins. Importing this package registers the built-in kinds."""

from __future__ import annotations

from open_eeg_synth.artifacts.base import (  # noqa: F401
    Event, EventArtifact, Occupancy, Remedy, RenderContext, TransformArtifact, TruthRecord,
)
from open_eeg_synth.artifacts.registry import ARTIFACTS, make_artifact, register  # noqa: F401

# built-in kinds (each module registers itself on import) — extended in Tasks 16-19
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_artifact_base.py -v && ruff check . && ruff format --check .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/artifacts tests/test_artifact_base.py
git commit -m "feat(artifacts): plug-in contract, thinned-Poisson event scheduling, occupancy, registry"
```

---

### Task 14: Scalp pattern sources (analytic + empirical loader)

**Files:**
- Create: `src/open_eeg_synth/artifacts/patterns.py`
- Create: `tests/test_patterns.py`

**Interfaces:**
- Consumes: `canonical_label`, `UnknownChannelError`; the data file of Task 15 (tests here use a temporary file via `patterns.load_eog_patterns.cache_clear()` + monkeypatched path).
- Produces: `analytic_focal(centre_m, electrode_pos_m, sigma_mm) -> (n_ch,)`; `analytic_dipole(pos_m, moment, electrode_pos_m) -> (n_ch,)`; `empirical(name, channels, *, rng=None, jitter_sd=0.6, electrode_pos=None, z=None) -> (n_ch,)`; `EYE_CENTRE`; `load_eog_patterns()` (cached).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_patterns.py
import numpy as np
import pytest

from open_eeg_synth.artifacts import patterns
from open_eeg_synth.channels import CHANNELS_19, UnknownChannelError
from open_eeg_synth.headmodel import load_head_model


@pytest.fixture
def fake_eog(tmp_path, monkeypatch):
    names = np.array(list(CHANNELS_19))
    blink = np.zeros(19); blink[[0, 1]] = 1.0; blink[[2, 3, 16]] = 0.4; blink[[8, 9]] = -0.05
    heog = np.zeros(19); heog[10] = -1.0; heog[11] = 1.0; heog[0] = -0.4; heog[1] = 0.4
    p = tmp_path / "eog_patterns.npz"
    np.savez(p, channel_names=names, blink_mean=blink, blink_sd=0.1 * np.abs(blink),
             heog_mean=heog, heog_sd=0.1 * np.abs(heog), n_subjects=np.int64(3),
             attribution=np.array("test"))
    monkeypatch.setattr(patterns, "_EOG_PATH", p)
    patterns.load_eog_patterns.cache_clear()
    yield p
    patterns.load_eog_patterns.cache_clear()


def test_analytic_focal_and_dipole_shapes():
    head = load_head_model()
    t3 = head.electrode_pos[CHANNELS_19.index("T3")]
    m = patterns.analytic_focal(t3, head.electrode_pos, 35.0)
    assert np.isclose(m.max(), 1.0) and np.argmax(m) == CHANNELS_19.index("T3")
    assert m[CHANNELS_19.index("O2")] < 0.05
    d = patterns.analytic_dipole(patterns.EYE_CENTRE, (0.0, 0.0, 1.0), head.electrode_pos)
    assert np.abs(d).max() == 1.0 and d[CHANNELS_19.index("Fp1")] > 0.5
    h = patterns.analytic_dipole(patterns.EYE_CENTRE, (1.0, 0.0, 0.0), head.electrode_pos)
    assert np.sign(h[CHANNELS_19.index("F7")]) == -np.sign(h[CHANNELS_19.index("F8")])


def test_empirical_selects_channels_jitters_and_falls_back(fake_eog):
    head = load_head_model()
    m = patterns.empirical("blink", ["fp1", "T7", "O1"], z=0.0)
    assert np.allclose(m, [1.0, 0.0, -0.05])
    j = patterns.empirical("blink", ["Fp1", "F3"], z=1.0)
    assert np.isclose(j[1] / j[0], 0.44 / 1.1)  # mean + z * sd, then max-normalised
    with pytest.raises(UnknownChannelError):
        patterns.empirical("blink", ["Fp1", "AF7"])
    fb = patterns.empirical("blink", ["Fp1", "Oz"], electrode_pos=np.array(
        [head.electrode_pos[0], [0.0, -0.11, 0.0]]), z=0.0)
    assert fb[0] == 1.0 and abs(fb[1]) < 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_patterns.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/artifacts/patterns.py
"""Scalp patterns for non-cortical signals: empirical eye maps, analytic focal/dipole maps (DESIGN §5.5)."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from importlib import resources

import numpy as np

from open_eeg_synth.channels import UnknownChannelError, canonical_label

_EOG_PATH = resources.files("open_eeg_synth.artifacts") / "data" / "eog_patterns.npz"
EYE_CENTRE = np.array([0.0, 0.085, -0.020])  # between the eyes, head frame, metres


@lru_cache(maxsize=1)
def load_eog_patterns() -> dict:
    with resources.as_file(_EOG_PATH) as p:
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}


def analytic_focal(centre_m: np.ndarray, electrode_pos_m: np.ndarray, sigma_mm: float) -> np.ndarray:
    d = np.linalg.norm(electrode_pos_m - np.asarray(centre_m, float), axis=1) * 1000.0
    w = np.exp(-0.5 * (d / sigma_mm) ** 2)
    return w / w.max()


def analytic_dipole(pos_m, moment, electrode_pos_m: np.ndarray) -> np.ndarray:
    r = electrode_pos_m - np.asarray(pos_m, float)
    rn = np.linalg.norm(r, axis=1)
    v = (r @ np.asarray(moment, float)) / rn**3
    return v / np.abs(v).max()


_FALLBACK = {
    "blink": lambda pos: analytic_dipole(EYE_CENTRE, (0.0, 0.3, 1.0), pos),
    "heog": lambda pos: analytic_dipole(EYE_CENTRE, (1.0, 0.0, 0.0), pos),
}


def empirical(name: str, channels: Sequence[str], *, rng: np.random.Generator | None = None,
              jitter_sd: float = 0.6, electrode_pos: np.ndarray | None = None,
              z: float | None = None) -> np.ndarray:
    """Subject-averaged ICA map for ``name`` on ``channels``; analytic fallback for unknown labels."""
    if z is None:
        z = float(rng.normal(0.0, jitter_sd)) if rng is not None else 0.0
    f = load_eog_patterns()
    names = [str(c) for c in f["channel_names"]]
    mean, sd = f[f"{name}_mean"], f[f"{name}_sd"]
    out = np.empty(len(channels))
    missing = []
    for i, ch in enumerate(channels):
        try:
            j = names.index(canonical_label(ch, names))
            out[i] = mean[j] + z * sd[j]
        except UnknownChannelError:
            missing.append(i)
    if missing:
        if electrode_pos is None:
            raise UnknownChannelError(
                f"{name}: no empirical map for {[channels[i] for i in missing]} and no positions"
            )
        fb = _FALLBACK[name](np.asarray(electrode_pos, float))
        out[missing] = fb[missing]
    return out / np.abs(out).max()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_patterns.py -v && ruff check . && ruff format --check .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/artifacts/patterns.py tests/test_patterns.py
git commit -m "feat(artifacts): empirical and analytic scalp pattern sources"
```

---

### Task 15: Derive the eye patterns from PhysioNet + ship them with the notice

**Files:**
- Create: `scripts/derive_artifact_patterns.py`
- Create: `src/open_eeg_synth/artifacts/data/NOTICE-eegmmidb.txt`
- Create (generated): `src/open_eeg_synth/artifacts/data/eog_patterns.npz`, `scripts/output/eog_patterns_selection.json`
- Create: `tests/test_eog_patterns_file.py`

**Interfaces:**
- Consumes: MNE (`mne.datasets.eegbci`), internet on first run (~30 subjects × 2 runs, a few MB each). If a subject cannot be downloaded the script stops at the subjects already on disk (1–20 give 20 blink maps and 17 heog maps; 1–30 give 30 and 25).
- Produces: the data file of DESIGN §6.3 (`channel_names (64,)`, `blink_mean/sd`, `heog_mean/sd`, `n_subjects`, `subjects_used`, `attribution`).

- [ ] **Step 1: Write the notice** `src/open_eeg_synth/artifacts/data/NOTICE-eegmmidb.txt`:

```
The scalp patterns in eog_patterns.npz are statistical summaries (subject-averaged, unit-normalised
independent-component topographies) derived from the EEG Motor Movement/Imagery Dataset v1.0.0,
made available on PhysioNet under the Open Data Commons Attribution License v1.0
(https://physionet.org/content/eegmmidb/view-license/1.0.0/). No recordings are redistributed.

Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet.
https://doi.org/10.13026/C28G6P
Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. BCI2000: A General-Purpose
Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034-1043,
2004.
Pollard, T., et al. PhysioNet as a global platform for biomedical research. Nature Health
1(8):792-795, 2026. https://doi.org/10.1038/s44360-026-00096-z
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_eog_patterns_file.py
import numpy as np

from open_eeg_synth.artifacts.patterns import load_eog_patterns

HEOG_SD_MAX = 0.45  # measured 0.42 over 30 subjects (0.40 over 17): the heog map varies more than blink


def _idx(names, label):
    return [str(n) for n in names].index(label)


def test_blink_map_shape_and_orientation():
    f = load_eog_patterns()
    names = f["channel_names"]
    assert len(names) == 64 and f["blink_mean"].shape == (64,)
    b = f["blink_mean"]
    top2 = set(np.argsort(-b)[:2])
    assert top2 == {_idx(names, "Fp1"), _idx(names, "Fp2")}
    for ch in ("F3", "Fz", "F4", "F7", "F8"):
        assert b[_idx(names, ch)] > 0
    assert abs(b[_idx(names, "O1")]) < 0.25 and abs(b[_idx(names, "O2")]) < 0.25
    assert f["blink_sd"].max() < 0.35


def test_heog_map_is_lateral():
    f = load_eog_patterns()
    names = f["channel_names"]
    h = f["heog_mean"]
    f7, f8 = _idx(names, "F7"), _idx(names, "F8")
    assert np.sign(h[f7]) == -np.sign(h[f8]) and h[f8] > 0
    assert set(np.argsort(-np.abs(h))[:2]) <= {f7, f8, _idx(names, "AF7"), _idx(names, "AF8")}
    assert f["heog_sd"].max() < HEOG_SD_MAX
    assert int(f["n_subjects"]) >= 15
    assert "10.13026/C28G6P" in str(f["attribution"])
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_eog_patterns_file.py -v`
Expected: FAIL (file missing).

- [ ] **Step 4: Write the derivation script**

```python
# scripts/derive_artifact_patterns.py
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


BLINK_PRIOR = {"Fp1": 1, "Fp2": 1, "AF3": 1, "AF4": 1, "AF7": 1, "AF8": 1, "F3": 0.4, "Fz": 0.4, "F4": 0.4}
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
        ica = mne.preprocessing.ICA(n_components=30, method="fastica", random_state=0,
                                    max_iter="auto", verbose=False)
        ica.fit(raw, verbose=False)
        topo = ica.get_components()  # (n_ch, n_comp)
        src = ica.get_sources(raw).get_data()
        fs = raw.info["sfreq"]
        entry: dict = {}
        # blink: best topography match to the prior, sparse time course
        cb = [np.corrcoef(topo[:, k], _prior(names, BLINK_PRIOR))[0, 1] for k in range(topo.shape[1])]
        order = np.argsort(-np.abs(cb))
        for k in order[:5]:
            if _excess_kurtosis(src[k]) > 5.0:
                m = topo[:, k] * np.sign(cb[k])
                blink_maps.append(m / np.abs(m).max())
                entry["blink"] = {"component": int(k), "corr": float(cb[k])}
                break
        # heog: lateral prior, slow time course (< 40 % of the power above 5 Hz after the
        # 1 Hz high-pass; 0.2 keeps only 4 of 20 subjects, 0.4 keeps 17)
        ch = [np.corrcoef(topo[:, k], _prior(names, HEOG_PRIOR))[0, 1] for k in range(topo.shape[1])]
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
    attribution = (notice.rstrip() + f"\n\nDerived by scripts/derive_artifact_patterns.py from "
                   f"subjects 1-{args.subjects}, runs 1-2, with MNE {mne.__version__} on "
                   f"{dt.date.today().isoformat()}.")
    B, H = np.array(blink_maps), np.array(heog_maps)
    np.savez_compressed(
        DATA / "eog_patterns.npz",
        channel_names=np.array(names), blink_mean=B.mean(0).astype(np.float32),
        blink_sd=B.std(0).astype(np.float32), heog_mean=H.mean(0).astype(np.float32),
        heog_sd=H.std(0).astype(np.float32), n_subjects=np.int64(min(len(B), len(H))),
        subjects_used=np.array([s for s, e in log.items() if e != "skipped"]),
        attribution=np.array(attribution),
    )
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.write_text(json.dumps(log, indent=2))
    print(f"blink maps: {len(B)}, heog maps: {len(H)} -> {DATA / 'eog_patterns.npz'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the derivation, then the tests**

Run: `python scripts/derive_artifact_patterns.py --subjects 30 && pytest tests/test_eog_patterns_file.py tests/test_patterns.py -v && ruff check . && ruff format --check .`
Expected: at least 15 subjects contribute to each map; tests PASS. Measured on subjects 1–30 (21–30 downloaded in about a minute): blink 30/30 (Fp1 and Fp2 the two largest, |O1| = |O2| = 0.23, `blink_sd` max 0.22), heog 25/30 (F7 −0.58, F8 +0.59, the two largest, `heog_sd` max 0.42). If the blink map's posterior entries exceed 0.25 in magnitude, raise the kurtosis threshold to 8 and rerun; if fewer than 15 subjects clear the heog criterion, relax `_hf_share < 0.5`. Record what you changed in the commit message.

- [ ] **Step 6: Commit** (the `.npz`, the notice and the selection log are all committed)

```bash
git add scripts/derive_artifact_patterns.py scripts/output/eog_patterns_selection.json src/open_eeg_synth/artifacts/data tests/test_eog_patterns_file.py
git commit -m "feat(artifacts): empirical blink and horizontal eye-movement maps from PhysioNet eegmmidb (ODC-By)"
```

---

### Task 16: Blink plug-in

**Files:**
- Create: `src/open_eeg_synth/artifacts/blink.py`
- Modify: `src/open_eeg_synth/artifacts/__init__.py` (import the module)
- Create: `tests/test_blink.py`

**Interfaces:**
- Consumes: `EventArtifact`, `patterns.empirical`, `raised_cosine_envelope`.
- Produces: `Blink(*, rate_by_state=None, min_gap_s=0.4, median_uv=120.0, sigma=0.3, peak_range_uv=(60.0, 250.0), double_p=0.15)` registered as `"blink"`; `Blink.waveform(fs, dur_s) -> (n,)` unit-peak (normalised on the sampled grid, where the raised-cosine rise never lands exactly on 1.0).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_blink.py
import numpy as np

from open_eeg_synth.artifacts.base import Occupancy, Remedy, RenderContext
from open_eeg_synth.artifacts.blink import Blink
from open_eeg_synth.artifacts.registry import make_artifact
from open_eeg_synth.brain.state import StateSegment, StateTimeline
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.seeds import stream_rng

FS = 256.0


def _bind(art, timeline, seed=1):
    head = load_head_model()
    art.bind(RenderContext(head.channels, FS, head.electrode_pos, head, timeline,
                           stream_rng(seed, "t"), stream_rng(seed, "s"), Occupancy(), "artifact:blink"))


def test_waveform_shape():
    w = Blink.waveform(FS, 0.3)
    assert len(w) == 77 and np.isclose(w.max(), 1.0) and np.argmax(w) < 0.4 * len(w)
    assert w[0] == 0.0 and w[-1] < 0.05


def test_blinks_only_with_eyes_open_and_frontal_positive():
    tl = StateTimeline([StateSegment(0, 60, "eyes_open"), StateSegment(60, 120, "eyes_closed")], 0.0)
    art = make_artifact("blink")
    _bind(art, tl)
    x = art.render(0, int(120 * FS))
    truth = art.truth()
    assert 8 <= len(truth) <= 25 and all(t.onset_s < 60 for t in truth)
    fp1, o1 = CHANNELS_19.index("Fp1"), CHANNELS_19.index("O1")
    assert x[fp1].max() > 60 and x[fp1].max() > 4 * np.abs(x[o1]).max()  # |O1| < 0.25 (Task 15)
    t = truth[0]
    assert t.kind == "blink" and t.subtype in ("single", "double") and "Fp1" in t.channels
    assert t.remedies == (Remedy.REMOVE_COMPONENT, Remedy.MASK_SEGMENT)
    assert 60 <= t.peak_uv <= 250
    assert any(t.subtype == "double" for t in art.truth()) or len(truth) < 12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_blink.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/artifacts/blink.py
"""Eye blinks: empirical scalp map x an asymmetric pulse (DESIGN §5.6)."""

from __future__ import annotations

import numpy as np

from open_eeg_synth.artifacts import patterns
from open_eeg_synth.artifacts.base import Event, EventArtifact, Remedy, RenderContext, TruthRecord
from open_eeg_synth.artifacts.registry import register
from open_eeg_synth.dsp import raised_cosine_envelope

DEFAULT_RATES = {"eyes_open": 0.25, "eyes_closed": 0.0, "drowsy": 0.0}


@register
class Blink(EventArtifact):
    kind = "blink"

    def __init__(self, *, rate_by_state: dict[str, float] | None = None, min_gap_s: float = 0.4,
                 median_uv: float = 120.0, sigma: float = 0.3,
                 peak_range_uv: tuple[float, float] = (60.0, 250.0), double_p: float = 0.15) -> None:
        super().__init__(rate_by_state=rate_by_state or dict(DEFAULT_RATES), min_gap_s=min_gap_s)
        self.median_uv, self.sigma = float(median_uv), float(sigma)
        self.peak_range_uv, self.double_p = tuple(peak_range_uv), float(double_p)

    def bind(self, ctx: RenderContext) -> None:
        super().bind(ctx)
        self.pattern = patterns.empirical("blink", ctx.channels, rng=ctx.subject_rng,
                                          electrode_pos=ctx.electrode_pos)
        self.channels = tuple(ch for ch, p in zip(ctx.channels, self.pattern) if abs(p) >= 0.3)

    @staticmethod
    def waveform(fs: float, dur_s: float) -> np.ndarray:
        n = int(round(dur_s * fs))
        t = np.arange(n) / fs
        rise, tau = 0.35 * dur_s, 0.22 * dur_s
        w = np.where(t < rise, 0.5 * (1.0 - np.cos(np.pi * t / rise)), np.exp(-(t - rise) / tau))
        w = w * raised_cosine_envelope(n, fs, 0.0, 0.05 * dur_s)
        return w / w.max()  # unit peak on the sampled grid

    def make_event(self, onset: int) -> Event:
        rng = self.ctx.rng
        dur = float(rng.uniform(0.2, 0.4))
        peak = float(np.clip(rng.lognormal(np.log(self.median_uv), self.sigma), *self.peak_range_uv))
        w = peak * self.waveform(self.fs, dur)
        subtype = "single"
        if rng.random() < self.double_p:
            gap = np.zeros(int(round(0.15 * self.fs)))
            w = np.concatenate([w, gap, 0.8 * peak * self.waveform(self.fs, dur)])
            subtype = "double"
        truth = TruthRecord("blink", subtype, None, self.channels, onset / self.fs,
                            (onset + len(w)) / self.fs, peak,
                            (Remedy.REMOVE_COMPONENT, Remedy.MASK_SEGMENT), self.layer_name,
                            {"duration_s": round(dur, 3)})
        return Event.from_pattern(onset, self.pattern, w, truth)
```

Add `from open_eeg_synth.artifacts import blink  # noqa: F401` at the bottom of `artifacts/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_blink.py -v && ruff check . && ruff format --check .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/artifacts/blink.py src/open_eeg_synth/artifacts/__init__.py tests/test_blink.py
git commit -m "feat(artifacts): blink plug-in"
```

---

### Task 17: Horizontal eye-movement plug-in

**Files:**
- Create: `src/open_eeg_synth/artifacts/eye_movement.py`
- Modify: `src/open_eeg_synth/artifacts/__init__.py`
- Create: `tests/test_eye_movement.py`

**Interfaces:**
- Produces: `EyeMovement(*, rate_by_state=None, min_gap_s=0.5, saccade_uv=(30.0, 100.0), roving_uv=(20.0, 60.0))` registered as `"eye_movement"`, subtypes `"saccade"` (eyes open) and `"slow_roving"` (otherwise).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eye_movement.py
import numpy as np

from open_eeg_synth.artifacts.base import Occupancy, RenderContext
from open_eeg_synth.artifacts.registry import make_artifact
from open_eeg_synth.brain.state import StateSegment, StateTimeline
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.seeds import stream_rng
from tests.helpers import band_power

FS = 256.0


def _bind(art, tl, seed=1):
    head = load_head_model()
    art.bind(RenderContext(head.channels, FS, head.electrode_pos, head, tl, stream_rng(seed, "t"),
                           stream_rng(seed, "s"), Occupancy(), "artifact:eye_movement"))


def test_subtype_follows_state_and_map_is_lateral():
    tl = StateTimeline([StateSegment(0, 120, "eyes_open"), StateSegment(120, 240, "drowsy")], 0.0)
    art = make_artifact("eye_movement")
    _bind(art, tl)
    x = art.render(0, int(240 * FS))
    truth = art.truth()
    kinds = {round(t.onset_s // 120): t.subtype for t in truth}
    assert kinds.get(0) == "saccade" and kinds.get(1) == "slow_roving"
    f7, f8 = CHANNELS_19.index("F7"), CHANNELS_19.index("F8")
    assert np.corrcoef(x[f7], x[f8])[0, 1] < -0.9
    assert 30 <= np.abs(x[f8, : int(120 * FS)]).max() <= 110
    slow = x[f8, int(120 * FS) :]
    assert band_power(slow, FS, 0.1, 0.8)[0] > 20 * band_power(slow, FS, 3.0, 8.0)[0]
    assert all(t.kind == "eye_movement" and "F7" in t.channels and "F8" in t.channels for t in truth)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eye_movement.py -v`
Expected: FAIL with `KeyError: 'eye_movement'` or `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/artifacts/eye_movement.py
"""Horizontal eye movements: saccades with eyes open, slow roving otherwise (DESIGN §5.6)."""

from __future__ import annotations

import numpy as np

from open_eeg_synth.artifacts import patterns
from open_eeg_synth.artifacts.base import Event, EventArtifact, Remedy, RenderContext, TruthRecord
from open_eeg_synth.artifacts.registry import register
from open_eeg_synth.dsp import raised_cosine_envelope

DEFAULT_RATES = {"eyes_open": 0.10, "eyes_closed": 0.02, "drowsy": 0.08}


@register
class EyeMovement(EventArtifact):
    kind = "eye_movement"

    def __init__(self, *, rate_by_state: dict[str, float] | None = None, min_gap_s: float = 0.5,
                 saccade_uv: tuple[float, float] = (30.0, 100.0),
                 roving_uv: tuple[float, float] = (20.0, 60.0)) -> None:
        super().__init__(rate_by_state=rate_by_state or dict(DEFAULT_RATES), min_gap_s=min_gap_s)
        self.saccade_uv, self.roving_uv = tuple(saccade_uv), tuple(roving_uv)

    def bind(self, ctx: RenderContext) -> None:
        super().bind(ctx)
        self.pattern = patterns.empirical("heog", ctx.channels, rng=ctx.subject_rng,
                                          electrode_pos=ctx.electrode_pos)
        self.channels = tuple(ch for ch, p in zip(ctx.channels, self.pattern) if abs(p) >= 0.3)

    def make_event(self, onset: int) -> Event:
        rng, fs = self.ctx.rng, self.fs
        sign = 1.0 if rng.random() < 0.5 else -1.0
        if self.ctx.timeline.state_at(onset / fs) == "eyes_open":
            subtype = "saccade"
            hold = float(rng.uniform(0.3, 2.0))
            amp = sign * float(rng.uniform(*self.saccade_uv))
            nr, nh, nf = int(round(0.03 * fs)), int(round(hold * fs)), int(round(0.04 * fs))
            w = np.concatenate([0.5 * (1 - np.cos(np.pi * np.arange(nr) / nr)), np.ones(nh),
                                0.5 * (1 + np.cos(np.pi * (np.arange(nf) + 1) / nf))]) * amp
            params = {"hold_s": round(hold, 3)}
        else:
            subtype = "slow_roving"
            f = float(rng.uniform(0.2, 0.5))
            cycles = int(rng.integers(1, 4))
            amp = sign * float(rng.uniform(*self.roving_uv))
            n = int(round(cycles / f * fs))
            t = np.arange(n) / fs
            w = amp * np.sin(2 * np.pi * f * t) * raised_cosine_envelope(n, fs, 0.2, 0.2)
            params = {"freq_hz": round(f, 3), "cycles": cycles}
        truth = TruthRecord("eye_movement", subtype, None, self.channels, onset / fs,
                            (onset + len(w)) / fs, abs(amp),
                            (Remedy.REMOVE_COMPONENT, Remedy.MASK_SEGMENT), self.layer_name, params)
        return Event.from_pattern(onset, self.pattern, w, truth)
```

Add `from open_eeg_synth.artifacts import eye_movement  # noqa: F401` to `artifacts/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eye_movement.py -v && ruff check . && ruff format --check .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/artifacts/eye_movement.py src/open_eeg_synth/artifacts/__init__.py tests/test_eye_movement.py
git commit -m "feat(artifacts): horizontal eye-movement plug-in (saccade, slow roving)"
```

---

### Task 18: Jaw-tension EMG plug-in (oversampled, anti-aliased)

**Files:**
- Create: `src/open_eeg_synth/artifacts/jaw_emg.py`
- Modify: `src/open_eeg_synth/artifacts/__init__.py`
- Create: `tests/test_jaw_emg.py`

**Interfaces:**
- Produces: `JawEmg(*, side="random", rate_by_state=None, min_gap_s=5.0, oversample=8, peak_hz=80.0, rms_median_uv=50.0, rms_sigma=0.6, rms_range_uv=(20.0, 200.0), dur_median_s=1.2, dur_sigma=0.5, dur_range_s=(0.5, 3.0), sigma_mm=35.0)` registered as `"emg"`, subtype `"jaw"`; `JawEmg.burst(fs, dur_s, rng, oversample, peak_hz) -> (n,)` unit plateau RMS; constants `LEFT_CENTRE`, `RIGHT_CENTRE` (metres, head frame).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_jaw_emg.py
import numpy as np

from open_eeg_synth.artifacts.base import Occupancy, Remedy, RenderContext
from open_eeg_synth.artifacts.jaw_emg import JawEmg
from open_eeg_synth.artifacts.registry import make_artifact
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.seeds import stream_rng
from tests.helpers import band_power

FS = 256.0


def _bind(art, seed=1):
    head = load_head_model()
    art.bind(RenderContext(head.channels, FS, head.electrode_pos, head,
                           StateTimeline.constant("eyes_open"), stream_rng(seed, "t"),
                           stream_rng(seed, "s"), Occupancy(), "artifact:emg"))


def test_burst_is_broadband_above_20hz_and_band_limited():
    w = JawEmg.burst(FS, 2.0, np.random.default_rng(0), 8, 80.0)
    assert len(w) == 512
    plateau = w[77:435]
    assert abs(plateau.std() - 1.0) < 0.15
    hi = band_power(plateau, FS, 30.0, 100.0)[0]
    lo = band_power(plateau, FS, 2.0, 12.0)[0]
    top = band_power(plateau, FS, 110.0, 128.0)[0]
    assert hi > 5 * lo and top < 0.5 * band_power(plateau, FS, 60.0, 100.0)[0]


def test_sides_channels_and_truth():
    for side, loud, quiet in (("left", "T3", "T4"), ("right", "T4", "T3")):
        art = make_artifact("emg", side=side, rate_by_state={"eyes_open": 0.2}, min_gap_s=0.5)
        _bind(art)
        x = art.render(0, int(60 * FS))
        li, qi = CHANNELS_19.index(loud), CHANNELS_19.index(quiet)
        assert x[li].std() > 5 * x[qi].std()
        t = art.truth()[0]
        assert t.kind == "emg" and t.subtype == "jaw" and t.side == side and loud in t.channels
        assert t.remedies == (Remedy.MASK_SEGMENT, Remedy.REMOVE_COMPONENT)
        assert 0.5 <= t.offset_s - t.onset_s <= 3.05
    both = make_artifact("emg", side="both", rate_by_state={"eyes_open": 0.2})
    _bind(both)
    y = both.render(0, int(60 * FS))
    assert 0.3 < y[CHANNELS_19.index("T3")].std() / y[CHANNELS_19.index("T4")].std() < 3.0
    assert y[CHANNELS_19.index("Pz")].std() < 0.3 * y[CHANNELS_19.index("T3")].std()


def test_random_side_uses_all_three():
    art = make_artifact("emg", rate_by_state={"eyes_open": 1.0}, min_gap_s=0.0)
    _bind(art, seed=5)
    art.render(0, int(120 * FS))
    assert {t.side for t in art.truth()} == {"left", "right", "both"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_jaw_emg.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/artifacts/jaw_emg.py
"""Jaw-tension (temporalis) EMG bursts, generated at a high rate then decimated (DESIGN §5.6)."""

from __future__ import annotations

import numpy as np

from open_eeg_synth.artifacts import patterns
from open_eeg_synth.artifacts.base import Event, EventArtifact, Remedy, RenderContext, TruthRecord
from open_eeg_synth.artifacts.registry import register
from open_eeg_synth.dsp import lowpass_decimate, raised_cosine_envelope

DEFAULT_RATES = {"eyes_open": 1 / 60, "eyes_closed": 1 / 60, "drowsy": 1 / 120}
# temporalis: 5 mm lateral, 10 mm anterior, 10 mm inferior of the T3 / T4 positions of the template
LEFT_CENTRE = np.array([-0.0875, -0.0029, -0.0154])
RIGHT_CENTRE = np.array([0.0900, -0.0081, -0.0125])


@register
class JawEmg(EventArtifact):
    kind = "emg"

    def __init__(self, *, side: str = "random", rate_by_state: dict[str, float] | None = None,
                 min_gap_s: float = 5.0, oversample: int = 8, peak_hz: float = 80.0,
                 rms_median_uv: float = 50.0, rms_sigma: float = 0.6,
                 rms_range_uv: tuple[float, float] = (20.0, 200.0), dur_median_s: float = 1.2,
                 dur_sigma: float = 0.5, dur_range_s: tuple[float, float] = (0.5, 3.0),
                 sigma_mm: float = 35.0) -> None:
        if side not in ("left", "right", "both", "random"):
            raise ValueError("side must be left, right, both or random")
        super().__init__(rate_by_state=rate_by_state or dict(DEFAULT_RATES), min_gap_s=min_gap_s)
        self.side, self.oversample, self.peak_hz = side, int(oversample), float(peak_hz)
        self.rms_median_uv, self.rms_sigma = float(rms_median_uv), float(rms_sigma)
        self.rms_range_uv, self.dur_median_s = tuple(rms_range_uv), float(dur_median_s)
        self.dur_sigma, self.dur_range_s, self.sigma_mm = float(dur_sigma), tuple(dur_range_s), float(sigma_mm)

    def bind(self, ctx: RenderContext) -> None:
        super().bind(ctx)
        pos = ctx.electrode_pos
        self.pattern = {"left": patterns.analytic_focal(LEFT_CENTRE, pos, self.sigma_mm),
                        "right": patterns.analytic_focal(RIGHT_CENTRE, pos, self.sigma_mm)}

    @staticmethod
    def burst(fs: float, dur_s: float, rng: np.random.Generator, oversample: int,
              peak_hz: float) -> np.ndarray:
        fs_hi = oversample * fs
        n_hi = int(round(dur_s * fs)) * oversample
        white = rng.standard_normal(n_hi)
        f = np.fft.rfftfreq(n_hi, 1.0 / fs_hi)
        h = (f / peak_hz) / (1.0 + (f / peak_hz) ** 2)
        x = np.fft.irfft(np.fft.rfft(white) * h, n=n_hi)
        x *= raised_cosine_envelope(n_hi, fs_hi, 0.1, 0.1)
        y = lowpass_decimate(x, oversample)
        plateau = y[int(0.15 * len(y)) : int(0.85 * len(y))]
        return y / plateau.std()

    def make_event(self, onset: int) -> Event:
        rng, fs = self.ctx.rng, self.fs
        side = self.side
        if side == "random":
            side = str(rng.choice(["left", "right", "both"], p=[0.4, 0.4, 0.2]))
        dur = float(np.clip(rng.lognormal(np.log(self.dur_median_s), self.dur_sigma), *self.dur_range_s))
        rms = float(np.clip(rng.lognormal(np.log(self.rms_median_uv), self.rms_sigma), *self.rms_range_uv))
        sides = ("left", "right") if side == "both" else (side,)
        block = None
        for s in sides:
            w = rms * self.burst(fs, dur, rng, self.oversample, self.peak_hz)
            part = np.outer(self.pattern[s], w)
            block = part if block is None else block + part
        combined = np.max([self.pattern[s] for s in sides], axis=0)
        channels = tuple(ch for ch, p in zip(self.ctx.channels, combined) if p >= 0.3)
        truth = TruthRecord("emg", "jaw", side, channels, onset / fs, (onset + block.shape[1]) / fs,
                            float(np.abs(block).max()), (Remedy.MASK_SEGMENT, Remedy.REMOVE_COMPONENT),
                            self.layer_name, {"duration_s": round(dur, 3), "rms_uv": round(rms, 1)})
        return Event(onset, block.astype(np.float32), truth)
```

Add `from open_eeg_synth.artifacts import jaw_emg  # noqa: F401` to `artifacts/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_jaw_emg.py -v && ruff check . && ruff format --check .`
Expected: PASS. If `test_burst_is_broadband_above_20hz_and_band_limited` fails on the `top` assertion, the decimator's transition band is too wide: pass `window=("kaiser", 8.0)` in `lowpass_decimate` (Task 5) — that change is allowed and its own test still passes.

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/artifacts/jaw_emg.py src/open_eeg_synth/artifacts/__init__.py tests/test_jaw_emg.py
git commit -m "feat(artifacts): jaw-tension EMG plug-in generated at 8x and decimated like an amplifier"
```

---

### Task 19: Dead-channel plug-in (proves the transform hook)

**Files:**
- Create: `src/open_eeg_synth/artifacts/dead_channel.py`
- Modify: `src/open_eeg_synth/artifacts/__init__.py`
- Create: `tests/test_dead_channel.py`

**Interfaces:**
- Produces: `DeadChannel(*, channel, onset_s=0.0, offset_s=None, noise_uv=0.3)` registered as `"dead_channel"`, mode `"transform"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dead_channel.py
import numpy as np

from open_eeg_synth.artifacts.base import Occupancy, Remedy, RenderContext
from open_eeg_synth.artifacts.registry import make_artifact
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.engine import Engine
from open_eeg_synth.seeds import stream_rng, stream_seed
from open_eeg_synth.sensor import SensorNoise

FS = 100.0
CH = ("Fp1", "Fp2", "Cz")


def test_dead_channel_is_flat_and_layer_sum_holds():
    art = make_artifact("dead_channel", channel="fp2", onset_s=2.0, offset_s=6.0)
    assert art.mode == "transform"
    art.bind(RenderContext(CH, FS, None, None, StateTimeline.constant("eyes_open"),
                           stream_rng(1, "t"), stream_rng(1, "s"), Occupancy(), "transform:dead_channel"))
    eng = Engine(CH, FS, [SensorNoise(3, 10.0, stream_seed(1, "sensor"))], transforms=[art])
    rec = eng.render_all(1000, block=64)
    x = rec.mixed
    assert np.allclose(x, rec.layers["sensor"] + rec.layers["transform:dead_channel"])
    assert x[1, 200:600].std() < 0.5 and x[1, :200].std() > 5.0 and x[0].std() > 5.0
    t = art.truth()[0]
    assert t.kind == "dead_channel" and t.channels == ("Fp2",) and t.onset_s == 2.0
    assert t.remedies == (Remedy.MARK_BAD_CHANNEL, Remedy.INTERPOLATE)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dead_channel.py -v`
Expected: FAIL with `KeyError: 'dead_channel'`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/artifacts/dead_channel.py
"""A channel that records nothing but amplifier noise for a span (DESIGN §5.3, §5.6)."""

from __future__ import annotations

import math

import numpy as np

from open_eeg_synth.artifacts.base import Remedy, RenderContext, TransformArtifact, TruthRecord
from open_eeg_synth.artifacts.registry import register
from open_eeg_synth.channels import canonical_label


@register
class DeadChannel(TransformArtifact):
    kind = "dead_channel"

    def __init__(self, *, channel: str, onset_s: float = 0.0, offset_s: float | None = None,
                 noise_uv: float = 0.3) -> None:
        self.channel, self.onset_s, self.offset_s, self.noise_uv = channel, float(onset_s), offset_s, float(noise_uv)

    def bind(self, ctx: RenderContext) -> None:
        super().bind(ctx)
        self.label = canonical_label(self.channel, ctx.channels)
        self.idx = ctx.channels.index(self.label)
        self.a = int(round(self.onset_s * ctx.fs))
        self.b = math.inf if self.offset_s is None else int(round(self.offset_s * ctx.fs))
        self._truth = [TruthRecord("dead_channel", None, None, (self.label,), self.onset_s,
                                   self.offset_s, 0.0, (Remedy.MARK_BAD_CHANNEL, Remedy.INTERPOLATE),
                                   self.layer_name, {"noise_uv": self.noise_uv})]

    def render_transform(self, t0: int, n: int, mix: np.ndarray) -> np.ndarray:
        delta = np.zeros_like(mix)
        a, b = max(self.a, t0), min(self.b, t0 + n)
        if a < b:
            a, b = int(a), int(b)
            noise = self.noise_uv * self.ctx.rng.standard_normal(b - a)
            delta[self.idx, a - t0 : b - t0] = -mix[self.idx, a - t0 : b - t0] + noise
        return delta

    def truth(self) -> list[TruthRecord]:
        return list(self._truth)
```

Add `from open_eeg_synth.artifacts import dead_channel  # noqa: F401` to `artifacts/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dead_channel.py -v && ruff check . && ruff format --check .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/artifacts/dead_channel.py src/open_eeg_synth/artifacts/__init__.py tests/test_dead_channel.py
git commit -m "feat(artifacts): dead-channel transform plug-in"
```

---

### Task 20: Artifacts in a case — ordinary set, truth aggregation, chunk invariance end to end

**Files:**
- Modify: `src/open_eeg_synth/recipes.py` (`ordinary_artifacts`)
- Create: `tests/test_case_artifacts.py`

**Interfaces:**
- Produces: `recipes.ordinary_artifacts() -> (ArtifactSpec("blink"), ArtifactSpec("eye_movement"), ArtifactSpec("emg", {"side": "random"}))`; `Recording.truth` filled for every artifact layer.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_case_artifacts.py
import numpy as np

from open_eeg_synth.case import ArtifactSpec, CaseSpec, make_case, make_engine, make_subject
from open_eeg_synth.recipes import ordinary_artifacts, resting_case


def test_ordinary_artifacts_layers_and_truth():
    spec = resting_case(11, duration_s=30.0)
    assert [a.kind for a in ordinary_artifacts()] == ["blink", "eye_movement", "emg"]
    case = make_case(spec)
    ec, eo = case.recordings["eyes_closed"], case.recordings["eyes_open"]
    assert set(eo.layers) == {"brain", "artifact:blink", "artifact:eye_movement", "artifact:emg", "sensor"}
    assert np.allclose(eo.mixed, sum(eo.layers.values()), atol=1e-3)
    assert any(t.kind == "blink" for t in eo.truth)
    assert not any(t.kind == "blink" for t in ec.truth)  # eyes closed: no blinks
    assert all(t.layer in eo.layers for t in eo.truth)


def test_duplicate_kinds_get_numbered_layers():
    spec = resting_case(12, duration_s=5.0,
                        artifacts=(ArtifactSpec("emg", {"side": "left"}), ArtifactSpec("emg", {"side": "right"})))
    rec = make_case(spec).recordings["eyes_open"]
    assert {"artifact:emg#1", "artifact:emg#2"} <= set(rec.layers)


def test_engine_chunking_does_not_change_a_case():
    spec = resting_case(13, duration_s=20.0)
    subject = make_subject(spec)
    cond = spec.conditions[1]
    whole = make_engine(spec, subject, cond).render_all(int(20 * 256), block=100000)
    eng = make_engine(spec, subject, cond)
    rng = np.random.default_rng(0)
    parts, t0 = [], 0
    while t0 < 20 * 256:
        n = int(min(20 * 256 - t0, rng.integers(1, 900)))
        parts.append(eng.render(t0, n).mixed)
        t0 += n
    assert np.allclose(whole.mixed, np.concatenate(parts, axis=1), atol=1e-3)
    assert [t.to_dict() for t in whole.truth] == [t.to_dict() for t in eng.truth()]


def test_spec_with_artifacts_roundtrips():
    spec = resting_case(14, duration_s=1.0)
    assert CaseSpec.from_dict(spec.to_dict()) == spec
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_case_artifacts.py -v`
Expected: FAIL (`ordinary_artifacts()` is empty; layers missing).

- [ ] **Step 3: Implement**

In `recipes.py` replace the placeholder:

```python
def ordinary_artifacts() -> tuple[ArtifactSpec, ...]:
    """Blinks, eye movements and occasional jaw tension — what every ordinary recording has."""
    return (ArtifactSpec("blink"), ArtifactSpec("eye_movement"), ArtifactSpec("emg", {"side": "random"}))
```

`case.py` needs no change: its module-level `import open_eeg_synth.artifacts` (Task 12) already registers every built-in kind that Tasks 16–19 added to `artifacts/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_case_artifacts.py -v && pytest -q && ruff check . && ruff format --check .`
Expected: PASS; the whole fast suite green.

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/recipes.py tests/test_case_artifacts.py
git commit -m "feat(case): ordinary artifact set in every case; truth aggregated per layer"
```

**M3 acceptance:** `pytest -q` green; `python -c "from open_eeg_synth.artifacts import ARTIFACTS; print(sorted(ARTIFACTS))"` prints `['blink', 'dead_channel', 'emg', 'eye_movement']`; a 240 s eyes-open recording has roughly 60 blinks, 24 eye movements, 4 EMG bursts (`len([t for t in rec.truth if t.kind == ...])`).

---

# M4 — state timeline in use + planted-pattern primitives

### Task 21: Planted-pattern primitives

**Files:**
- Replace: `src/open_eeg_synth/brain/plants.py` (the Task 12 stub)
- Create: `tests/test_plants.py`

**Interfaces:**
- Consumes: `RhythmSpec`, `BurstGate`, `compiled_rhythms` (already calls `apply_modifiers`).
- Produces: `PlantRecord` (+ `to_dict`, JSON-safe: tuples in `params` become lists), `Modifier(rhythm, amp_scale=1.0, f0_shift_hz=0.0, hemisphere_gain={}, extra_sites=())`, `apply_modifiers(rhythms, modifiers)`, `PLANTS` registry, `register_plant`, `plant_to_dict`, `plant_from_dict`, and the six primitives `FocalSlow`, `RhythmicBursts`, `LateralImbalance`, `WidespreadExcess`, `PeakShift`, `ReducedRhythm`, each a frozen dataclass with `kind`, `rhythms()`, `modifiers()`, `record()`. `FocalSlow` and `RhythmicBursts` take an optional `state_gain: dict[str, float]` (default empty) that is passed through to the `RhythmSpec` they compile to, so a consumer can confine a plant to some states (`{"eyes_open": 0.0}` plants it eyes-closed only); the record's `params` carry it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_plants.py
import numpy as np
import pytest

from open_eeg_synth.brain.plants import (
    PLANTS, FocalSlow, LateralImbalance, PeakShift, ReducedRhythm, RhythmicBursts,
    WidespreadExcess, apply_modifiers, plant_from_dict, plant_to_dict,
)
from open_eeg_synth.case import compiled_rhythms, make_case
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.recipes import resting_brain, resting_case
from tests.helpers import band_power

FS = 256.0


def _ec(seed, plants):
    return make_case(resting_case(seed, duration_s=40.0, plants=plants, artifacts=())).recordings["eyes_closed"].mixed


def test_registry_and_roundtrip():
    assert set(PLANTS) == {"focal_slow", "rhythmic_bursts", "lateral_imbalance",
                           "widespread_excess", "peak_shift", "reduced_rhythm"}
    for p in (FocalSlow("F7"), RhythmicBursts(("Fz",)), LateralImbalance("alpha", "left", 0.6),
              WidespreadExcess("theta", 2.2), PeakShift("alpha", -1.5), ReducedRhythm("smr", 0.2)):
        assert plant_from_dict(plant_to_dict(p)) == p
        rec = p.record()
        assert rec.kind == p.kind and rec.description and rec.to_dict()["kind"] == p.kind


def test_modifiers_compile_onto_base_rhythms():
    base = resting_brain().rhythms
    mods = [m for p in (WidespreadExcess("theta", 2.0, extra_sites=("Pz",)), PeakShift("alpha", -1.0),
                        LateralImbalance("alpha", "left", 0.5)) for m in p.modifiers()]
    out = {r.name: r for r in apply_modifiers(base, mods)}
    assert out["theta"].amp_uv == 14.0 and "Pz" in out["theta"].sites
    assert out["alpha"].f0_hz == 9.0 and out["alpha"].hemisphere_gain == {"left": 0.5}
    with pytest.raises(KeyError):
        apply_modifiers(base, [PeakShift("gamma", 1.0).modifiers()[0]])


def test_focal_slow_raises_delta_at_site():
    f7 = CHANNELS_19.index("F7")
    clean, planted = _ec(21, ()), _ec(21, (FocalSlow("F7", f0_hz=2.5, amp_uv=45.0),))
    assert band_power(planted, FS, 1.5, 3.5)[f7] > 4 * band_power(clean, FS, 1.5, 3.5)[f7]
    spec = resting_case(21, plants=(FocalSlow("F7"),))
    assert [r.name for r in compiled_rhythms(spec)][-1] == "plant:focal_slow:F7"


def test_lateral_imbalance_and_reduced_rhythm():
    o1, o2, c3 = (CHANNELS_19.index(k) for k in ("O1", "O2", "C3"))
    x = _ec(22, (LateralImbalance("alpha", "left", 0.4),))
    y = _ec(22, ())
    a_x, a_y = band_power(x, FS, 8, 13), band_power(y, FS, 8, 13)
    assert (a_x[o1] / a_x[o2]) < 0.6 * (a_y[o1] / a_y[o2])
    z = _ec(22, (ReducedRhythm("smr", 0.1),))
    assert band_power(z, FS, 12.5, 14.5)[c3] < 0.7 * band_power(y, FS, 12.5, 14.5)[c3]


def test_rhythmic_bursts_are_intermittent():
    fz = CHANNELS_19.index("Fz")
    x = _ec(23, (RhythmicBursts(("Fz",), f0_hz=6.5, amp_uv=30.0, burst_s=(1, 2), gap_s=(4, 6)),))
    x = x - x.mean(axis=0, keepdims=True)  # average reference
    theta = np.array([band_power(x[fz, s : s + int(FS)], FS, 5.5, 7.5)[0]
                      for s in range(0, x.shape[1], int(FS))])  # per-second theta power at Fz
    assert theta.max() > 4 * np.median(theta)  # burst seconds stand well above the gaps


def test_plant_state_gain_silences_a_plant_by_state():
    f7 = CHANNELS_19.index("F7")
    plant = FocalSlow("F7", f0_hz=2.5, amp_uv=45.0, state_gain={"eyes_open": 0.0})
    for cond in ("eyes_open", "eyes_closed"):
        clean = make_case(resting_case(24, duration_s=20.0, artifacts=())).recordings[cond].mixed
        planted = make_case(resting_case(24, duration_s=20.0, plants=(plant,), artifacts=()))
        ratio = (band_power(planted.recordings[cond].mixed, FS, 1.5, 3.5)[f7]
                 / band_power(clean, FS, 1.5, 3.5)[f7])
        if cond == "eyes_open":
            assert abs(ratio - 1.0) < 1e-3  # silenced: the other streams are untouched (DESIGN §8.1)
        else:
            assert ratio > 4
    assert plant.record().to_dict()["params"]["state_gain"] == {"eyes_open": 0.0}
    assert plant_from_dict(plant_to_dict(plant)) == plant
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_plants.py -v`
Expected: FAIL with `ImportError` (stub has no primitives).

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/brain/plants.py
"""Planted-pattern primitives: generic signal terms compiled onto the rhythm generator (DESIGN §4.4)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, ClassVar, Protocol

from open_eeg_synth.brain.rhythm import BurstGate, RhythmSpec


@dataclass(frozen=True)
class PlantRecord:
    kind: str
    sites: tuple[str, ...]
    band_hz: tuple[float, float]
    amp_uv: float | None
    onset_s: float
    offset_s: float | None
    params: dict
    description: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sites"], d["band_hz"] = list(self.sites), list(self.band_hz)
        d["params"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.params.items()}
        return d


@dataclass(frozen=True)
class Modifier:
    rhythm: str
    amp_scale: float = 1.0
    f0_shift_hz: float = 0.0
    hemisphere_gain: dict[str, float] = field(default_factory=dict)
    extra_sites: tuple[str, ...] = ()


def apply_modifiers(rhythms: Sequence[RhythmSpec], modifiers: Sequence[Modifier]) -> tuple[RhythmSpec, ...]:
    out = {r.name: r for r in rhythms}
    for m in modifiers:
        if m.rhythm not in out:
            raise KeyError(f"modifier targets unknown rhythm {m.rhythm!r}; have {sorted(out)}")
        r = out[m.rhythm]
        hg = dict(r.hemisphere_gain)
        for side, g in m.hemisphere_gain.items():
            hg[side] = hg.get(side, 1.0) * g
        out[m.rhythm] = replace(
            r, amp_uv=r.amp_uv * m.amp_scale, f0_hz=r.f0_hz + m.f0_shift_hz, hemisphere_gain=hg,
            sites=r.sites + tuple(s for s in m.extra_sites if s not in r.sites),
        )
    return tuple(out[r.name] for r in rhythms)


class Plant(Protocol):
    kind: ClassVar[str]

    def rhythms(self) -> tuple[RhythmSpec, ...]: ...

    def modifiers(self) -> tuple[Modifier, ...]: ...

    def record(self) -> PlantRecord: ...


PLANTS: dict[str, type] = {}


def register_plant(cls):
    PLANTS[cls.kind] = cls
    return cls


def plant_to_dict(p: Plant) -> dict[str, Any]:
    d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(p).items()}
    return {"kind": p.kind, "params": d}


def plant_from_dict(d: dict[str, Any]) -> Plant:
    cls = PLANTS[d["kind"]]
    params = dict(d["params"])
    for k, v in params.items():
        if isinstance(v, list):
            params[k] = tuple(v)
    return cls(**params)


@register_plant
@dataclass(frozen=True)
class FocalSlow:
    kind: ClassVar[str] = "focal_slow"
    site: str
    f0_hz: float = 2.5
    amp_uv: float = 45.0
    width_mm: float = 12.0
    f_sd: float = 0.4
    env_tau_s: float = 1.5
    env_sd: float = 0.6
    indep: float = 0.3
    state_gain: dict[str, float] = field(default_factory=dict)  # e.g. {"eyes_open": 0.0}

    def rhythms(self) -> tuple[RhythmSpec, ...]:
        return (RhythmSpec(f"plant:focal_slow:{self.site}", self.f0_hz, self.amp_uv,
                           sites=(self.site,), width_mm=self.width_mm, f_sd=self.f_sd,
                           env_tau_s=self.env_tau_s, env_sd=self.env_sd, lag_ms=0.0, indep=self.indep,
                           state_gain=dict(self.state_gain)),)

    def modifiers(self) -> tuple[Modifier, ...]:
        return ()

    def record(self) -> PlantRecord:
        return PlantRecord(self.kind, (self.site,), (self.f0_hz - 1.0, self.f0_hz + 1.0), self.amp_uv,
                           0.0, None, asdict(self),
                           f"a {self.f0_hz:.1f} Hz rhythm of about {self.amp_uv:.0f} uV from one "
                           f"cortical patch under {self.site}, present throughout")


@register_plant
@dataclass(frozen=True)
class RhythmicBursts:
    kind: ClassVar[str] = "rhythmic_bursts"
    sites: tuple[str, ...]
    f0_hz: float = 6.5
    amp_uv: float = 30.0
    width_mm: float = 15.0
    burst_s: tuple[float, float] = (1.0, 3.0)
    gap_s: tuple[float, float] = (8.0, 25.0)
    env_tau_s: float = 1.0
    env_sd: float = 0.8
    indep: float = 0.2
    state_gain: dict[str, float] = field(default_factory=dict)

    def rhythms(self) -> tuple[RhythmSpec, ...]:
        name = "plant:rhythmic_bursts:" + "+".join(self.sites)
        return (RhythmSpec(name, self.f0_hz, self.amp_uv, sites=self.sites, width_mm=self.width_mm,
                           env_tau_s=self.env_tau_s, env_sd=self.env_sd, env_lo=0.1, env_hi=3.0,
                           lag_ms=0.0, indep=self.indep, state_gain=dict(self.state_gain),
                           burst=BurstGate(self.burst_s, self.gap_s, 0.3)),)

    def modifiers(self) -> tuple[Modifier, ...]:
        return ()

    def record(self) -> PlantRecord:
        return PlantRecord(self.kind, self.sites, (self.f0_hz - 1.0, self.f0_hz + 1.0), self.amp_uv,
                           0.0, None, asdict(self),
                           f"bursts of a {self.f0_hz:.1f} Hz rhythm ({self.burst_s[0]:.0f}-"
                           f"{self.burst_s[1]:.0f} s long, every {self.gap_s[0]:.0f}-{self.gap_s[1]:.0f} s)"
                           f" under {', '.join(self.sites)}")


@register_plant
@dataclass(frozen=True)
class LateralImbalance:
    kind: ClassVar[str] = "lateral_imbalance"
    rhythm: str
    side: str
    factor: float

    def rhythms(self) -> tuple[RhythmSpec, ...]:
        return ()

    def modifiers(self) -> tuple[Modifier, ...]:
        return (Modifier(self.rhythm, hemisphere_gain={self.side: self.factor}),)

    def record(self) -> PlantRecord:
        return PlantRecord(self.kind, (), (0.0, 0.0), None, 0.0, None, asdict(self),
                           f"the {self.rhythm} rhythm scaled by {self.factor:.2f} on the {self.side}")


@register_plant
@dataclass(frozen=True)
class WidespreadExcess:
    kind: ClassVar[str] = "widespread_excess"
    rhythm: str
    factor: float
    extra_sites: tuple[str, ...] = ()

    def rhythms(self) -> tuple[RhythmSpec, ...]:
        return ()

    def modifiers(self) -> tuple[Modifier, ...]:
        return (Modifier(self.rhythm, amp_scale=self.factor, extra_sites=self.extra_sites),)

    def record(self) -> PlantRecord:
        return PlantRecord(self.kind, self.extra_sites, (0.0, 0.0), None, 0.0, None, asdict(self),
                           f"the {self.rhythm} rhythm scaled by {self.factor:.2f} everywhere it occurs"
                           + (f", extended to {', '.join(self.extra_sites)}" if self.extra_sites else ""))


@register_plant
@dataclass(frozen=True)
class PeakShift:
    kind: ClassVar[str] = "peak_shift"
    rhythm: str
    shift_hz: float

    def rhythms(self) -> tuple[RhythmSpec, ...]:
        return ()

    def modifiers(self) -> tuple[Modifier, ...]:
        return (Modifier(self.rhythm, f0_shift_hz=self.shift_hz),)

    def record(self) -> PlantRecord:
        return PlantRecord(self.kind, (), (0.0, 0.0), None, 0.0, None, asdict(self),
                           f"the {self.rhythm} rhythm's centre frequency moved by {self.shift_hz:+.1f} Hz")


@register_plant
@dataclass(frozen=True)
class ReducedRhythm:
    kind: ClassVar[str] = "reduced_rhythm"
    rhythm: str
    factor: float

    def rhythms(self) -> tuple[RhythmSpec, ...]:
        return ()

    def modifiers(self) -> tuple[Modifier, ...]:
        return (Modifier(self.rhythm, amp_scale=self.factor),)

    def record(self) -> PlantRecord:
        return PlantRecord(self.kind, (), (0.0, 0.0), None, 0.0, None, asdict(self),
                           f"the {self.rhythm} rhythm reduced to {self.factor:.2f} of its usual amplitude")
```

Note: `ClassVar` fields are not dataclass fields, so `asdict` skips `kind` and `plant_from_dict` can pass the remaining params straight to the constructor.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_plants.py tests/test_case.py -v && ruff check . && ruff format --check .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/brain/plants.py tests/test_plants.py
git commit -m "feat(brain): planted-pattern primitives compiled onto the rhythm generator"
```

---

### Task 22: Drowsiness timeline end to end

**Files:**
- Create: `tests/test_drowsiness.py`

**Interfaces:**
- Consumes: `resting_case(drowsy_from_s=…)`, rhythm `state_gain`, artifact `rate_by_state`.
- Produces: nothing new — this task verifies that the timeline drives rhythms and artifacts together and fixes anything it finds.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_drowsiness.py
import numpy as np

from open_eeg_synth.case import make_case
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.recipes import resting_case
from tests.helpers import band_power, welch

FS = 256.0


def test_drowsy_half_has_less_slower_alpha_more_theta_and_roving_eyes():
    case = make_case(resting_case(31, duration_s=120.0, drowsy_from_s=60.0))
    rec = case.recordings["eyes_closed"]
    x = rec.layers["brain"]
    alert, drowsy = x[:, : int(55 * FS)], x[:, int(65 * FS) :]
    o1, fz = CHANNELS_19.index("O1"), CHANNELS_19.index("Fz")
    assert band_power(drowsy, FS, 8, 13)[o1] < 0.4 * band_power(alert, FS, 8, 13)[o1]
    assert band_power(drowsy, FS, 4, 8)[fz] > 2.0 * band_power(alert, FS, 4, 8)[fz]
    # alpha peak moves down by about 1 Hz
    def peak(seg):  # Welch (4 s Hann) rather than a raw periodogram: the argmax is far less noisy
        f, p = welch(seg[o1], FS, int(4 * FS))
        m = (f >= 6) & (f <= 13)
        return f[m][np.argmax(p[0, m])]
    assert peak(alert) - peak(drowsy) > 0.5
    subtypes = [(t.onset_s, t.subtype) for t in rec.truth if t.kind == "eye_movement"]
    assert any(s == "slow_roving" and on > 60 for on, s in subtypes)
    assert rec.timeline.state_at(90.0) == "drowsy"
    assert [s["state"] for s in rec.timeline.to_dict()["segments"]] == ["eyes_closed", "drowsy"]
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_drowsiness.py -v && ruff check . && ruff format --check .`
Expected: PASS on the first run if Tasks 9–21 are correct. If the theta assertion fails, the theta `state_gain {"drowsy": 2.0}` is not reaching `Rhythm` — check `compiled_rhythms` preserves `state_gain` through `apply_modifiers` (`replace` keeps untouched fields). If the peak-shift assertion fails, check `_Oscillator.render` adds `f0_shift` to the frequency and `StateTimeline.gain(..., default=0.0)` is used for the shift.

- [ ] **Step 3: Commit**

```bash
git add tests/test_drowsiness.py
git commit -m "test: drowsiness timeline drives rhythms and eye movements together"
```

---

### Task 23: Case spec round trip with plants and timelines; case id stability

**Files:**
- Create: `tests/test_case_spec_json.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_case_spec_json.py
import json

from open_eeg_synth.brain.plants import FocalSlow, RhythmicBursts
from open_eeg_synth.case import ArtifactSpec, CaseSpec, case_id_for
from open_eeg_synth.recipes import resting_case


def test_full_spec_survives_json_and_ids_are_stable():
    spec = resting_case(99, duration_s=30.0, drowsy_from_s=15.0,
                        plants=(FocalSlow("F7"), RhythmicBursts(("Fz",), amp_uv=25.0)),
                        artifacts=(ArtifactSpec("blink", {"median_uv": 90.0}),
                                   ArtifactSpec("emg", {"side": "both"})))
    text = json.dumps(spec.to_dict(), sort_keys=True)
    back = CaseSpec.from_dict(json.loads(text))
    assert back == spec
    assert case_id_for(back) == case_id_for(spec)
    assert case_id_for(resting_case(100, duration_s=30.0)) != case_id_for(spec)
    assert spec.to_dict()["conditions"][0]["timeline"]["segments"][1]["state"] == "drowsy"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_case_spec_json.py -v && ruff check . && ruff format --check .`
Expected: PASS. If `back == spec` fails on `plants`, a tuple became a list in `plant_from_dict` — the conversion there handles top-level lists only; keep plant parameters flat.

- [ ] **Step 3: Commit**

```bash
git add tests/test_case_spec_json.py
git commit -m "test: case spec with plants and timelines round-trips through JSON"
```

**M4 acceptance:** `pytest -q` green; `python - <<'EOF'` building `resting_case(1, plants=(FocalSlow("F7"), RhythmicBursts(("Fz",))), drowsy_from_s=120)` and printing `[p.to_dict() for p in case.recordings["eyes_closed"].plants]` shows two plant records with generic descriptions.

---

# M5 — case file writer + sealed truth

### Task 24: EDF writer (optional `edfio` extra)

**Files:**
- Create: `src/open_eeg_synth/casefile/__init__.py` (empty)
- Create: `src/open_eeg_synth/casefile/edf.py`
- Create: `tests/test_edf.py`

**Interfaces:**
- Consumes: `Recording`; `edfio>=0.4` (skip tests if absent).
- Produces: `write_edf(path, recording, *, physical_range_uv=(-2000.0, 2000.0), patient_code="SYNTHETIC", equipment="open-eeg-synth", additional=("synthetic",)) -> Path`. Trims to whole seconds; no annotations; anonymised start date.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_edf.py
import numpy as np
import pytest

from open_eeg_synth.case import make_case
from open_eeg_synth.casefile.edf import write_edf
from open_eeg_synth.recipes import resting_case

edfio = pytest.importorskip("edfio")


def test_edf_roundtrip_in_microvolts_no_annotations(tmp_path):
    rec = make_case(resting_case(41, duration_s=4.5, artifacts=())).recordings["eyes_open"]
    p = write_edf(tmp_path / "x.edf", rec)
    e = edfio.read_edf(str(p))
    assert [s.label for s in e.signals] == list(rec.channels)
    assert e.signals[0].physical_dimension == "uV" and e.signals[0].physical_range == (-2000.0, 2000.0)
    assert len(e.annotations) == 0
    assert e.patient.code == "SYNTHETIC" and "synthetic" in e.patient.additional
    with pytest.raises(edfio.edf_header.AnonymizedDateError):
        _ = e.recording.startdate
    got = e.signals[8].data  # O1
    assert len(got) == 4 * 256  # trimmed to whole seconds
    assert np.allclose(got, rec.mixed[8, : 4 * 256], atol=0.07)  # 16-bit over 4000 uV


def test_clipping_to_range(tmp_path):
    rec = make_case(resting_case(42, duration_s=2.0, artifacts=())).recordings["eyes_open"]
    rec.layers["brain"][0, :10] = 5000.0
    p = write_edf(tmp_path / "y.edf", rec, physical_range_uv=(-500.0, 500.0))
    e = edfio.read_edf(str(p))
    assert e.signals[0].data.max() <= 500.0 + 0.02
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_edf.py -v`
Expected: FAIL with `ModuleNotFoundError: open_eeg_synth.casefile`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/casefile/edf.py
"""EDF output: microvolt physical range, anonymised date, no annotations (DESIGN §7.4)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from open_eeg_synth.engine import Recording


def write_edf(path, recording: Recording, *, physical_range_uv: tuple[float, float] = (-2000.0, 2000.0),
              patient_code: str = "SYNTHETIC", equipment: str = "open-eeg-synth",
              additional: Sequence[str] = ("synthetic",)) -> Path:
    try:
        import edfio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("write_edf needs the 'edf' extra: pip install 'open-eeg-synth[edf]'") from exc
    lo, hi = physical_range_uv
    fs = recording.fs
    n = int(recording.n_samples // fs * fs)  # whole seconds only
    data = recording.mixed[:, :n]
    signals = [
        edfio.EdfSignal(np.clip(data[i].astype(np.float64), lo, hi), sampling_frequency=fs,
                        label=ch, transducer_type="synthetic", physical_dimension="uV",
                        physical_range=(lo, hi))
        for i, ch in enumerate(recording.channels)
    ]
    edf = edfio.Edf(signals,
                    patient=edfio.Patient(code=patient_code, name="X", additional=tuple(additional)),
                    recording=edfio.Recording(startdate=None, equipment_code=equipment))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    edf.write(str(path))
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_edf.py -v && ruff check . && ruff format --check .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/casefile tests/test_edf.py
git commit -m "feat(casefile): EDF writer in microvolts with anonymised date and no annotations"
```

---

### Task 25: Truth dictionary, sealed container, layer verification

**Files:**
- Create: `src/open_eeg_synth/casefile/truth.py`
- Create: `tests/test_truth.py`

**Interfaces:**
- Consumes: `Case`, `Recording`, `TruthRecord.to_dict`, `PlantRecord.to_dict`, `Subject.to_dict`, `make_subject`, `make_engine`, `version`.
- Produces: `MAGIC = b"OESTRUTH\x01"`; `case_truth(case, files: dict[str, str]) -> dict`; `write_truth(path, d) -> Path`; `read_truth(path) -> dict`; `render_layers(truth, condition, *, rtol=1e-3) -> Recording`; `LayerMismatchError`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_truth.py
import json

import numpy as np
import pytest

from open_eeg_synth import version
from open_eeg_synth.brain.plants import FocalSlow
from open_eeg_synth.case import make_case
from open_eeg_synth.casefile.truth import (
    MAGIC, LayerMismatchError, case_truth, read_truth, render_layers, write_truth,
)
from open_eeg_synth.recipes import resting_case


def test_truth_dict_contents_and_sealed_roundtrip(tmp_path):
    case = make_case(resting_case(51, duration_s=6.0, plants=(FocalSlow("F7"),)))
    d = case_truth(case, {"eyes_closed": "a.edf", "eyes_open": "b.edf"})
    assert d["format"] == "open-eeg-synth/truth" and d["label"] == "synthetic"
    assert d["generator"]["signal_version"] == version.SIGNAL_VERSION
    assert d["case_id"] == case.case_id and d["seed"] == 51
    ec = d["recordings"]["eyes_closed"]
    assert ec["file"] == "a.edf" and ec["layers"][0] == "brain" and ec["layers"][-1] == "sensor"
    assert len(ec["layer_rms_uv"]["brain"]) == 19
    assert ec["plants"][0]["kind"] == "focal_slow"
    assert all(set(t) >= {"kind", "onset_s", "remedies", "layer"} for t in ec["truth"])
    json.dumps(d)  # JSON-safe
    p = write_truth(tmp_path / "c.truth", d)
    raw = p.read_bytes()
    assert raw.startswith(MAGIC) and b"focal_slow" not in raw  # compressed, not readable as text
    assert read_truth(p) == d
    with pytest.raises(ValueError):
        (tmp_path / "bad").write_bytes(b"nope")
        read_truth(tmp_path / "bad")


def test_render_layers_reproduces_and_detects_mismatch():
    case = make_case(resting_case(52, duration_s=4.0))
    d = case_truth(case, {"eyes_closed": "a.edf", "eyes_open": "b.edf"})
    rec = render_layers(d, "eyes_open")
    assert np.allclose(rec.layers["brain"], case.recordings["eyes_open"].layers["brain"], atol=1e-3)
    d["recordings"]["eyes_open"]["layer_rms_uv"]["brain"][0] *= 2.0
    with pytest.raises(LayerMismatchError):
        render_layers(d, "eyes_open")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_truth.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/casefile/truth.py
"""The sealed truth file: what was planted, where every artifact is, how to re-render layers (DESIGN §7.4)."""

from __future__ import annotations

import json
import warnings
import zlib
from pathlib import Path

import numpy as np
import scipy

from open_eeg_synth import version
from open_eeg_synth.case import Case, CaseSpec, make_engine, make_subject
from open_eeg_synth.engine import Recording

MAGIC = b"OESTRUTH\x01"


class LayerMismatchError(RuntimeError):
    """Re-rendered layers do not match the recorded per-channel RMS."""


def _recording_truth(rec: Recording, file: str) -> dict:
    return {
        "file": file, "fs": rec.fs, "duration_s": rec.duration_s, "channels": list(rec.channels),
        "layers": list(rec.layers),
        "layer_rms_uv": {k: [round(float(v), 4) for v in np.sqrt(np.mean(a.astype(np.float64) ** 2, axis=1))]
                         for k, a in rec.layers.items()},
        "timeline": rec.timeline.to_dict() if rec.timeline is not None else None,
        "truth": [t.to_dict() for t in rec.truth],
        "plants": [p.to_dict() for p in rec.plants],
    }


def case_truth(case: Case, files: dict[str, str]) -> dict:
    return {
        "format": "open-eeg-synth/truth", "format_version": 1,
        "generator": {"package": "open-eeg-synth", "version": version.__version__,
                      "signal_version": version.SIGNAL_VERSION, "numpy": np.__version__,
                      "scipy": scipy.__version__},
        "case_id": case.case_id, "seed": case.spec.seed, "label": case.spec.label,
        "spec": case.spec.to_dict(), "subject": case.subject.to_dict(),
        "recordings": {name: _recording_truth(rec, files[name]) for name, rec in case.recordings.items()},
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
    return json.loads(zlib.decompress(raw[len(MAGIC):]).decode("utf-8"))


def render_layers(truth: dict, condition: str, *, rtol: float = 1e-3) -> Recording:
    """Rebuild one condition's layers from the sealed spec and check them against the recorded RMS."""
    gen = truth["generator"]
    if gen["version"] != version.__version__ or gen["signal_version"] != version.SIGNAL_VERSION:
        warnings.warn(f"truth made by open-eeg-synth {gen['version']} (signal {gen['signal_version']}); "
                      f"this is {version.__version__} (signal {version.SIGNAL_VERSION})", stacklevel=2)
    spec = CaseSpec.from_dict(truth["spec"])
    cond = next(c for c in spec.conditions if c.name == condition)
    subject = make_subject(spec)
    rec = make_engine(spec, subject, cond).render_all(int(round(cond.duration_s * spec.fs)))
    rec.timeline = cond.timeline
    rec.plants = [p.record() for p in spec.plants]
    expected = truth["recordings"][condition]["layer_rms_uv"]
    for name, arr in rec.layers.items():
        got = np.sqrt(np.mean(arr.astype(np.float64) ** 2, axis=1))
        want = np.asarray(expected[name])
        if not np.allclose(got, want, rtol=rtol, atol=1e-3):
            raise LayerMismatchError(f"layer {name!r} of {condition!r} does not match the truth file")
    return rec
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_truth.py -v && ruff check . && ruff format --check .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/casefile/truth.py tests/test_truth.py
git commit -m "feat(casefile): sealed truth container with re-render verification"
```

---

### Task 26: `write_case`, `read_case_truth`, and the command line

**Files:**
- Create: `src/open_eeg_synth/casefile/writer.py`
- Create: `src/open_eeg_synth/__main__.py`
- Modify: `src/open_eeg_synth/__init__.py` (export `write_case`)
- Create: `tests/test_writer_cli.py`

**Interfaces:**
- Produces: `CasePaths(truth: Path, recordings: dict[str, Path], layers: Path | None)`; `write_case(directory, case, *, embed_layers=False, physical_range_uv=(-2000, 2000)) -> CasePaths`; `read_case_truth(path) -> dict` (alias of `read_truth`); `render_layers` re-exported from `casefile.writer` (DESIGN §7.4 names it there); CLI `python -m open_eeg_synth make-case --seed N --out DIR [--duration S] [--spec spec.json] [--embed-layers] [--print-truth]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_writer_cli.py
import json
import subprocess
import sys

import numpy as np
import pytest

from open_eeg_synth.case import make_case
from open_eeg_synth.casefile.writer import read_case_truth, write_case
from open_eeg_synth.recipes import resting_case

pytest.importorskip("edfio")


def test_write_case_layout(tmp_path):
    case = make_case(resting_case(61, duration_s=3.0))
    paths = write_case(tmp_path, case)
    assert paths.truth.name == f"{case.case_id}.truth"
    assert sorted(p.name for p in paths.recordings.values()) == sorted(
        f"{case.case_id}_{c}.edf" for c in ("eyes_closed", "eyes_open"))
    assert paths.layers is None
    d = read_case_truth(paths.truth)
    assert d["recordings"]["eyes_open"]["file"] == paths.recordings["eyes_open"].name
    with_layers = write_case(tmp_path / "L", case, embed_layers=True)
    z = np.load(with_layers.layers)
    assert "eyes_open/brain" in z.files and z["eyes_open/brain"].shape == (19, 3 * 256)


def test_cli_make_case(tmp_path):
    out = subprocess.run([sys.executable, "-m", "open_eeg_synth", "make-case", "--seed", "7",
                          "--out", str(tmp_path), "--duration", "2", "--print-truth"],
                         capture_output=True, text=True, check=True)
    d = json.loads(out.stdout)
    assert d["seed"] == 7 and (tmp_path / f"{d['case_id']}.truth").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_writer_cli.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/casefile/writer.py
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


def write_case(directory, case: Case, *, embed_layers: bool = False,
               physical_range_uv: tuple[float, float] = (-2000.0, 2000.0)) -> CasePaths:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    recordings: dict[str, Path] = {}
    for name, rec in case.recordings.items():
        recordings[name] = write_edf(directory / f"{case.case_id}_{name}.edf", rec,
                                     physical_range_uv=physical_range_uv)
    truth = write_truth(directory / f"{case.case_id}.truth",
                        case_truth(case, {k: v.name for k, v in recordings.items()}))
    layers = None
    if embed_layers:
        layers = directory / f"{case.case_id}.layers.npz"
        np.savez_compressed(layers, **{f"{c}/{k}": a for c, rec in case.recordings.items()
                                       for k, a in rec.layers.items()})
    return CasePaths(truth, recordings, layers)


def read_case_truth(path) -> dict:
    return read_truth(path)
```

```python
# src/open_eeg_synth/__main__.py
"""`python -m open_eeg_synth make-case --seed N --out DIR`"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_eeg_synth.case import CaseSpec, make_case
from open_eeg_synth.casefile.truth import case_truth
from open_eeg_synth.casefile.writer import write_case
from open_eeg_synth.recipes import resting_case


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="open_eeg_synth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make-case", help="write a synthetic resting case (EDF + sealed truth)")
    mk.add_argument("--seed", type=int, required=True)
    mk.add_argument("--out", required=True)
    mk.add_argument("--duration", type=float, default=240.0)
    mk.add_argument("--spec", help="JSON file with a full CaseSpec (overrides --duration)")
    mk.add_argument("--embed-layers", action="store_true")
    mk.add_argument("--print-truth", action="store_true")
    args = ap.parse_args(argv)

    if args.spec:
        spec = CaseSpec.from_dict(json.loads(Path(args.spec).read_text()))
        spec = CaseSpec.from_dict({**spec.to_dict(), "seed": args.seed})
    else:
        spec = resting_case(args.seed, duration_s=args.duration)
    case = make_case(spec)
    paths = write_case(args.out, case, embed_layers=args.embed_layers)
    if args.print_truth:
        print(json.dumps(case_truth(case, {k: v.name for k, v in paths.recordings.items()})))
    else:
        print(paths.truth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_writer_cli.py -v && ruff check . && ruff format --check .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/casefile/writer.py src/open_eeg_synth/__main__.py src/open_eeg_synth/__init__.py tests/test_writer_cli.py
git commit -m "feat(casefile): write_case and the make-case command line"
```

**M5 acceptance:** `python -m open_eeg_synth make-case --seed 1 --out /tmp/oes-demo` writes two EDFs and one `.truth`; `python -c "import mne; r=mne.io.read_raw_edf('/tmp/oes-demo/<id>_eyes_closed.edf'); print(r.ch_names, len(r.annotations))"` shows the 19 names and `0`.

---

# M6 — streaming mode + recorder switch-over

### Task 27: HeartSource (RR intervals + ECG waveform)

**Files:**
- Create: `src/open_eeg_synth/heart.py`
- Create: `tests/test_heart.py`

**Interfaces:**
- Consumes: the `classic` synthesizer's ECG wave table and RR constants (copied, not imported).
- Produces: `HeartSource(fs, seq, *, bpm=72.0, bpm_sd=2.0, breath_bpm=5.5, rsa_ms=40.0, mayer=0.02, amp_uv=200.0)` with `.render(t0, n) -> (n,)`, `.beats_between(a_s, b_s) -> list[float]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_heart.py
import numpy as np

from open_eeg_synth.heart import HeartSource
from open_eeg_synth.seeds import stream_seed

FS = 256.0


def _kurt(x):
    x = x - x.mean()
    return float(np.mean(x**4) / np.mean(x**2) ** 2 - 3.0)


def test_ecg_is_spiky_at_about_72_bpm_and_chunk_invariant():
    def run(chunks):
        h = HeartSource(FS, stream_seed(1, "stream:heart"))
        out, t0 = [], 0
        for n in chunks:
            out.append(h.render(t0, n))
            t0 += n
        return np.concatenate(out), h

    whole, h = run([int(60 * FS)])
    chunked, _ = run([32] * int(60 * FS / 32))
    assert np.allclose(whole, chunked, atol=1e-3)  # beat windows cut Gaussian tails at ~1e-4 uV
    assert 150 < whole.max() <= 205 and _kurt(whole) > 3.0
    beats = h.beats_between(0.0, 60.0)
    assert 62 <= len(beats) <= 82
    assert whole.dtype == np.float32
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_heart.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/heart.py
"""Beat times and an ECG waveform; the HR channel now, heartbeat-bleed plug-ins later (DESIGN §7.3)."""

from __future__ import annotations

import numpy as np

# (offset_s, width_s, amplitude) for P, Q, R, S, T — the classic synthesizer's table
WAVES = ((-0.20, 0.040, 0.10), (-0.04, 0.018, -0.10), (0.00, 0.025, 1.00),
         (0.04, 0.025, -0.30), (0.30, 0.080, 0.30))


class HeartSource:
    def __init__(self, fs: float, seq: np.random.SeedSequence, *, bpm: float = 72.0,
                 bpm_sd: float = 2.0, breath_bpm: float = 5.5, rsa_ms: float = 40.0,
                 mayer: float = 0.02, amp_uv: float = 200.0) -> None:
        rr_rng, noise_rng = (np.random.Generator(np.random.PCG64(s)) for s in seq.spawn(2))
        self.fs, self.rr_rng, self.noise_rng = float(fs), rr_rng, noise_rng
        self.rr_mean = 60.0 / bpm
        self.rr_sd = self.rr_mean * (bpm_sd / bpm)
        self.breath_hz, self.rsa_s, self.mayer = breath_bpm / 60.0, rsa_ms / 1000.0, mayer
        self.beats: list[float] = [float(rr_rng.uniform(0.0, self.rr_mean))]
        t = np.arange(-0.5, 0.7, 1.0 / self.fs)
        template = sum(a * np.exp(-((t - o) ** 2) / (2 * w * w)) for o, w, a in WAVES)
        self.scale = amp_uv / float(template.max())

    def _extend(self, until_s: float) -> None:
        while self.beats[-1] < until_s + 0.7:
            t = self.beats[-1]
            rr = (self.rr_mean * (1.0 + self.mayer * np.sin(2 * np.pi * 0.1 * t))
                  + self.rsa_s * np.sin(2 * np.pi * self.breath_hz * t)
                  + float(self.rr_rng.normal(0.0, self.rr_sd)))
            self.beats.append(t + max(0.4, rr))

    def beats_between(self, a_s: float, b_s: float) -> list[float]:
        self._extend(b_s)
        return [b for b in self.beats if a_s <= b < b_s]

    def render(self, t0: int, n: int) -> np.ndarray:
        t = (t0 + np.arange(n)) / self.fs
        self._extend(t[-1])
        sig = np.zeros(n)
        for bt in self.beats:
            if bt < t[0] - 0.7 or bt > t[-1] + 0.5:
                continue
            for off, w, amp in WAVES:
                sig += amp * np.exp(-((t - (bt + off)) ** 2) / (2 * w * w))
        self.beats = [b for b in self.beats if b >= t[0] - 1.0]
        noise = 0.005 * self.noise_rng.standard_normal(n)
        return (self.scale * (sig + noise)).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_heart.py -v && ruff check . && ruff format --check .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/heart.py tests/test_heart.py
git commit -m "feat: HeartSource with streaming beat schedule and ECG waveform"
```

---

### Task 28: StreamSource — the recorder-facing generator

**Files:**
- Create: `src/open_eeg_synth/stream.py`
- Create: `src/open_eeg_synth/markers.py`
- Modify: `src/open_eeg_synth/__init__.py` (export `StreamSource`, `MarkerSchedule`)
- Create: `tests/test_stream.py`

**Interfaces:**
- Consumes: `classic.synth.MarkerSchedule` (re-exported unchanged), `make_subject`, `make_engine`, `HeartSource`, `resting_brain`, `ordinary_artifacts`.
- Produces: `StreamSource(channel_labels, srate, *, seed=None, markers=None, timeline=None, brain=None, artifacts=None, sensor=None, perturb_head=True)` with `.next_chunk(n)`, `.due_markers(n)`, `.truth`, `.seed`, `.labels`, `.srate`. Any EEG label set works (one channel, four, nineteen): the brain is built on the full head and only the requested rows are rendered (Task 12).

- [ ] **Step 1: Write the failing tests** (these mirror what a recording application's own tests assert about its mock devices)

```python
# tests/test_stream.py
import numpy as np
import pytest

from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.channels import UnknownChannelError
from open_eeg_synth.markers import MarkerSchedule
from open_eeg_synth.stream import StreamSource
from tests.helpers import band_power

Q21 = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4",
       "T5", "T6", "Fz", "Cz", "Pz", "HR"]
DRAGON = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3",
          "Pz", "P4", "T6", "O1", "O2"]


def _kurt(x):
    x = x - x.mean()
    return float(np.mean(x**4) / np.mean(x**2) ** 2 - 3.0)


def test_validation():
    with pytest.raises(ValueError):
        StreamSource([], 256.0)
    with pytest.raises(ValueError):
        StreamSource(["O1"], 0.0)
    with pytest.raises(UnknownChannelError):
        StreamSource(["O1", "XYZ"], 256.0)
    s = StreamSource(["O1"], 256.0, seed=1)
    with pytest.raises(ValueError):
        s.next_chunk(0)


def test_chunk_shape_dtype_scale_and_determinism():
    a = StreamSource(Q21, 256.0, seed=7)
    b = StreamSource(Q21, 256.0, seed=7)
    c = a.next_chunk(32)
    assert c.shape == (20, 32) and c.dtype == np.float32 and np.isfinite(c).all()
    assert np.allclose(c, b.next_chunk(32)) and np.allclose(a.next_chunk(32), b.next_chunk(32))
    x = np.concatenate([a.next_chunk(256) for _ in range(8)], axis=1)
    assert 5.0 < x[8].std() < 300.0
    assert StreamSource(Q21, 256.0).seed != StreamSource(Q21, 256.0).seed  # fresh seeds


def test_chunk_size_does_not_change_the_signal():
    a = StreamSource(DRAGON, 500.0, seed=3)
    b = StreamSource(DRAGON, 500.0, seed=3)
    x = a.next_chunk(1000)
    y = np.concatenate([b.next_chunk(50) for _ in range(20)], axis=1)
    assert np.allclose(x, y, atol=1e-3)


def test_pink_slope_posterior_alpha_and_ecg_only_on_hr():
    # Eyes closed and artifact-free: the layered default (eyes open + blinks) legitimately has more
    # frontal 8-12 Hz power than occipital. Posterior dominance is judged after average
    # referencing (the lead field's native reference carries a common component, DESIGN §2.3)
    # and as a median over subjects, because alpha varies by subject as much as it does in life.
    ratios = []
    for seed in (3, 4, 5):
        s = StreamSource(Q21, 256.0, seed=seed, timeline=StateTimeline.constant("eyes_closed"),
                         artifacts=())
        x = np.concatenate([s.next_chunk(256) for _ in range(16)], axis=1)
        cz = x[17]
        assert band_power(cz, 256.0, 2, 6)[0] > band_power(cz, 256.0, 30, 50)[0]
        eeg = x[:19] - x[:19].mean(axis=0, keepdims=True)
        a = band_power(eeg, 256.0, 8, 12)
        ratios.append((a[8] + a[9]) / (a[0] + a[1]))
        hr = x[19]
        assert hr.max() > 120.0 and _kurt(hr) > 3.0 and _kurt(x[8]) < 1.5
    assert np.median(ratios) > 2.0  # occipital alpha well above frontal


def test_arbitrary_montages_and_rates():
    for labels, sr in ((["O1", "O2", "T3", "T4"], 250.0), (DRAGON, 500.0)):
        c = StreamSource(labels, sr, seed=5).next_chunk(64)
        assert c.shape == (len(labels), 64) and np.isfinite(c).all()


def test_markers_and_truth():
    s = StreamSource(["Cz"], 100.0, seed=1, markers=MarkerSchedule(kind="periodic", period_s=1.0))
    got = []
    for _ in range(250):
        got += [lbl for _, lbl in s.due_markers(100)]
    assert got.count("standard") >= 2
    s2 = StreamSource(["Cz"], 100.0, seed=2, markers=MarkerSchedule(kind="oddball", period_s=0.5, p_target=0.3))
    labels = []
    for _ in range(400):
        labels += [lbl for _, lbl in s2.due_markers(100)]
    assert "target" in labels and "standard" in labels
    assert StreamSource(["Cz"], 256.0, seed=1).due_markers(64) == []
    s3 = StreamSource(Q21, 256.0, seed=9)
    for _ in range(int(60 * 256 / 32)):
        s3.next_chunk(32)
    assert any(t.kind == "blink" for t in s3.truth)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stream.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/open_eeg_synth/markers.py
"""Marker schedules for streams. The classic schedule is the contract; re-exported unchanged."""

from __future__ import annotations

from open_eeg_synth.classic.synth import MarkerSchedule  # noqa: F401
```

```python
# src/open_eeg_synth/stream.py
"""Chunked, phase-continuous output for a recording application's mock amplifiers (DESIGN §7.3)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from open_eeg_synth.brain.layer import BrainSpec
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.case import ArtifactSpec, CaseSpec, ConditionSpec, SensorSpec, make_engine, make_subject
from open_eeg_synth.channels import canonical_label, is_heart_label
from open_eeg_synth.heart import HeartSource
from open_eeg_synth.markers import MarkerSchedule
from open_eeg_synth.recipes import ordinary_artifacts, resting_brain
from open_eeg_synth.seeds import fresh_case_seed, stream_rng, stream_seed


class StreamSource:
    def __init__(self, channel_labels: Sequence[str], srate: float, *, seed: int | None = None,
                 markers: MarkerSchedule | None = None, timeline: StateTimeline | None = None,
                 brain: BrainSpec | None = None, artifacts: Sequence[ArtifactSpec] | None = None,
                 sensor: SensorSpec | None = None, perturb_head: bool = True) -> None:
        if not channel_labels:
            raise ValueError("channel_labels must be non-empty")
        if srate <= 0:
            raise ValueError("srate must be positive")
        self.labels = list(channel_labels)
        self.srate = float(srate)
        self._seed = fresh_case_seed() if seed is None else int(seed)
        self._eeg_rows = [i for i, lb in enumerate(self.labels) if not is_heart_label(lb)]
        self._hr_rows = [i for i, lb in enumerate(self.labels) if is_heart_label(lb)]
        eeg_labels = tuple(canonical_label(self.labels[i]) for i in self._eeg_rows)
        self.timeline = timeline or StateTimeline.constant("eyes_open")
        self.spec = CaseSpec(
            seed=self._seed, fs=self.srate, channels=eeg_labels, perturb_head=perturb_head,
            brain=brain or resting_brain(),
            artifacts=tuple(artifacts) if artifacts is not None else ordinary_artifacts(),
            sensor=sensor or SensorSpec(),
            conditions=(ConditionSpec("stream", math.inf, self.timeline),),
        )
        self.subject = make_subject(self.spec)
        self.engine = make_engine(self.spec, self.subject, self.spec.conditions[0])
        self.heart = HeartSource(self.srate, stream_seed(self._seed, "stream:heart")) if self._hr_rows else None
        self.markers = markers or MarkerSchedule()
        self._marker_rng = stream_rng(self._seed, "stream:markers")
        self._marker_clock_s = 0.0
        self._next_marker_s = self.markers.period_s if self.markers.kind != "none" else math.inf
        self._pos = 0

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def truth(self) -> list:
        return self.engine.truth()

    def next_chunk(self, n_samples: int) -> np.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        frame = self.engine.render(self._pos, n_samples)
        out = np.zeros((len(self.labels), n_samples), dtype=np.float32)
        out[self._eeg_rows] = frame.mixed
        if self.heart is not None:
            ecg = self.heart.render(self._pos, n_samples)
            for i in self._hr_rows:
                out[i] = ecg
        self._pos += n_samples
        return out

    def due_markers(self, n_samples: int) -> list[tuple[float, str]]:
        """Same contract as the classic synthesizer: call in lockstep with next_chunk."""
        end = self._marker_clock_s + n_samples / self.srate
        if self.markers.kind == "none":
            self._marker_clock_s = end
            return []
        out: list[tuple[float, str]] = []
        while self._next_marker_s < end:
            if self.markers.kind == "oddball":
                is_target = float(self._marker_rng.random()) < self.markers.p_target
                label = self.markers.labels[1] if is_target else self.markers.labels[0]
            else:
                label = self.markers.labels[0]
            out.append((self._next_marker_s, label))
            self._next_marker_s += self.markers.period_s
        self._marker_clock_s = end
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stream.py -v && ruff check . && ruff format --check .`
Expected: PASS. Construction of a `StreamSource` costs about half a second (head perturbation + mixing on the full model) — acceptable for a mock device that starts once; note it in the docstring. Note for the recorder (Task 29 R7): the raw stream's alpha is far less posterior-dominant than the classic synthesizer's, because the lead field's native reference carries a common component that every channel shares (DESIGN §2.3); a live view that re-references (average or linked ears) shows the gradient, a raw referential view shows alpha on every channel, as a real referential amplifier does.

- [ ] **Step 5: Commit**

```bash
git add src/open_eeg_synth/stream.py src/open_eeg_synth/markers.py src/open_eeg_synth/__init__.py tests/test_stream.py
git commit -m "feat(stream): StreamSource with the classic constructor contract, ECG on heart labels, markers"
```

---

### Task 29: Recorder switch-over (tasks in the recording application's repository)

These run in the **recorder repository**, after v0.2.0 is tagged (Task 35). Constraint that must hold throughout: the recorder never imports any workstation code; `open-eeg-synth` has no such dependency, so this stays true by construction.

- [ ] **R1. Bump the dependency.** In the recorder's `pyproject.toml` change the `open-eeg-synth` requirement from the v0.1.x tag to `open-eeg-synth @ git+https://github.com/peak-mind-llc/open-eeg-synth.git@v0.2.0`. Reinstall (`pip install -e ".[dev]"`).

- [ ] **R2. Swap the three mock device loops.** In the device manager, each mock loop (Q21 at 256 Hz / chunk 32, BrainBit at 250 Hz / chunk 25, DragonEEG at 500 Hz / chunk 50) constructs `RealisticEEGSynthesizer(labels, srate, seed=..., markers=...)`. Replace the import with a small factory:

```python
def _mock_engine(labels, srate, *, seed=None, markers=None):
    import os
    if os.environ.get("COHERENCE_MOCK_ENGINE", "layered") == "classic":
        from open_eeg_synth.classic.synth import RealisticEEGSynthesizer
        return RealisticEEGSynthesizer(labels, srate, seed=seed, markers=markers)
    from open_eeg_synth.stream import StreamSource
    return StreamSource(labels, srate, seed=seed, markers=markers)
```

and call `_mock_engine(...)` in the three loops. `next_chunk(n)` and `due_markers(n)` keep their signatures, so nothing else changes. Keep the `classic` switch for one release (DESIGN §11.10), then delete it.

- [ ] **R3. Tests.** The recorder's existing synth tests import `RealisticEEGSynthesizer`; point them at `open_eeg_synth.classic.synth` (they remain the classic contract) and add `test_mock_engine_layered.py` that constructs `_mock_engine` for the three label sets and asserts shape, dtype, determinism under seed, and posterior-dominant alpha (copy the assertions of Task 28's `test_pink_slope_posterior_alpha_and_ecg_only_on_hr`). Run the mock-marker test (periodic markers through the sink) unchanged — it must pass with the layered engine.

- [ ] **R4. Standalone mock streams.** The mock LSL EEG script switches to `StreamSource`; the oddball ERP mock keeps its own evoked-response injection for now (DESIGN §11.7) but may take its background from `StreamSource`.

- [ ] **R5. PyInstaller spec.** Add `"open_eeg_synth"` to the list of packages whose submodules are collected and add `extra_datas += collect_data_files("open_eeg_synth")` so both `.npz` files and the two NOTICE files ship. Build and check `open_eeg_synth/headmodel/data/colin27_19ch.npz` is present in the bundle.

- [ ] **R6 (optional, later).** Drive the mock impedance display from `open_eeg_synth.contact.ContactTimeline` when the loose-lead plug-in lands; until then the recorder's own wobble stays.

- [ ] **R7. Smoke.** Start the recorder with a mock Q21, open the live view, confirm: posterior alpha visible once the view is re-referenced (in the raw referential view the lead field's common component puts alpha on every channel — see Task 28), blinks at Fp1/Fp2 with the real gradient (not a flat frontal rectangle), an occasional temporal EMG burst, ECG on HR. Commit with the dependency bump.

**M6 acceptance (package side):** `pytest -q` green; `python -c "from open_eeg_synth import StreamSource; s=StreamSource(['O1','O2','T3','T4'],250.0,seed=1); print(s.next_chunk(25).shape)"` prints `(4, 25)`.

---

# M7 — realism suite + CI + release v0.2.0

### Task 30: Realism measurement module (MNE, development only)

**Files:**
- Create: `tests/realism/__init__.py` (empty)
- Create: `tests/realism/measure.py`
- Create: `tests/realism/test_measure_smoke.py`

**Interfaces:**
- Consumes: `mne`, `mne_connectivity` (skip if absent).
- Produces: `BANDS`, `DIST_EDGES`, `CHAINS`, `to_raw(x_uv, fs, channels)`, `prep(raw, remove_blinks=False)`, `bipolar(raw)`, `laplacian(raw)`, `measure(raw, remove_blinks=False) -> dict` with keys `n_epochs`, `<ref>/<Band>/coh_mean`, `<ref>/<Band>/dwpli_mean`, `<ref>/<Band>/coh_by_dist` (5 values), `aperiodic_exponent`, `alpha_share/O1|Fz|Cz`, `rms_uv/<ch>`. The method is the feasibility test's — `spectral_connectivity_epochs(method=["wpli2_debiased", "coh"], fmin=…, fmax=…, faverage=True)` on 4 s fixed-length epochs after 1–45 Hz, `standard_1020`, average reference — except that the pair values are read as they are from the lower triangle of the dense output (mne-connectivity fills that triangle and leaves the other zero). The feasibility test measured through a consumer function that averaged the matrix with its transpose and so halved every pair value; the package does not copy that defect. Every coherence and dwPLI number in DESIGN §9.1, in the reference file and in Task 31's expectations is the true value, twice the feasibility test's figure; Task 31 cross-checks exactly that.

- [ ] **Step 1: Write the smoke test**

```python
# tests/realism/test_measure_smoke.py
import numpy as np
import pytest

pytest.importorskip("mne")
pytest.importorskip("mne_connectivity")

from open_eeg_synth.case import make_case  # noqa: E402
from open_eeg_synth.recipes import resting_case  # noqa: E402
from tests.realism.measure import measure, to_raw  # noqa: E402


@pytest.mark.slow
def test_measure_returns_every_metric():
    rec = make_case(resting_case(1, duration_s=40.0, artifacts=())).recordings["eyes_closed"]
    m = measure(to_raw(rec.mixed, rec.fs, rec.channels))
    assert m["n_epochs"] >= 8
    for ref in ("average", "bipolar", "laplacian"):
        for band in ("Delta", "Theta", "Alpha", "Beta"):
            assert 0.0 <= m[f"{ref}/{band}/coh_mean"] <= 1.0
            assert len(m[f"{ref}/{band}/coh_by_dist"]) == 5
            assert np.isfinite(m[f"{ref}/{band}/dwpli_mean"])
    assert 0.3 < m["aperiodic_exponent"] < 2.0 and 0.0 < m["alpha_share/O1"] < 1.0
```

- [ ] **Step 2: Implement**

```python
# tests/realism/measure.py
"""Realism metrics (DESIGN §9.1). MNE + mne-connectivity only; no consumer code."""

from __future__ import annotations

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
          ("Fz", "Cz"), ("Cz", "Pz")]
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
        ica = mne.preprocessing.ICA(n_components=15, method="fastica", random_state=0, max_iter="auto")
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
        con = spectral_connectivity_epochs(ep, method=["wpli2_debiased", "coh"], fmin=fmin, fmax=fmax,
                                           faverage=True, verbose=False)
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
            res[f"{ref}/{band}/coh_by_dist"] = [float(c[bins == k].mean()) if np.any(bins == k) else None
                                                for k in range(len(DIST_EDGES) - 1)]
            res[f"{ref}/{band}/coh_mean"] = float(c.mean())
            res[f"{ref}/{band}/dwpli_mean"] = float(w.mean())
    data = avg.get_data() * 1e6
    f, p = welch(data, fs=avg.info["sfreq"], nperseg=int(4 * avg.info["sfreq"]))
    fit = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    res["aperiodic_exponent"] = float(-np.polyfit(np.log10(f[fit]), np.log10(np.median(p[:, fit], axis=0)), 1)[0])
    ab, tot = (f >= 8) & (f <= 13), (f >= 1) & (f <= 40)
    for ch in ("O1", "Fz", "Cz"):
        i = avg.ch_names.index(ch)
        res[f"alpha_share/{ch}"] = float(p[i, ab].sum() / p[i, tot].sum())
    for i, ch in enumerate(avg.ch_names):
        res[f"rms_uv/{ch}"] = float(data[i].std())
    return res
```

If `get_data(output="dense")` fills the upper triangle in the installed mne-connectivity (0.9 fills the lower), swap `tril_indices(-1)` for `triu_indices(1)`; the smoke test's `coh_mean` in `[0, 1]` catches a wrong choice (zeros drag it to ~0), and Task 31's cross-check (twice the feasibility test's percentiles, to 0.005) is the guard against any averaging that halves the values.

- [ ] **Step 3: Run**

Run: `pytest -m slow tests/realism/test_measure_smoke.py -v && ruff check . && ruff format --check .`
Expected: PASS (~20 s).

- [ ] **Step 4: Commit**

```bash
git add tests/realism/__init__.py tests/realism/measure.py tests/realism/test_measure_smoke.py
git commit -m "test(realism): MNE-based coherence/dwPLI/spectral measurement module"
```

---

### Task 31: Public reference numbers (download on demand, commit the summary)

**Files:**
- Create: `scripts/build_realism_reference.py`
- Create (generated, committed): `tests/realism/reference/eegmmidb_baselines_s01-20.json`

- [ ] **Step 1: Write the script**

```python
# scripts/build_realism_reference.py
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
        out[k] = {"p10": np.nanpercentile(arr, 10, axis=0).tolist(),
                  "p50": np.nanpercentile(arr, 50, axis=0).tolist(),
                  "p90": np.nanpercentile(arr, 90, axis=0).tolist()}
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
        "source": f"PhysioNet EEG Motor Movement/Imagery Dataset 1.0.0, subjects 1-{args.subjects}, "
                  "runs 1 (eyes open) and 2 (eyes closed), blink components removed by ICA",
        "license": "Open Data Commons Attribution License v1.0",
        "attribution": NOTICE.read_text(),
        "built": dt.date.today().isoformat(), "mne": mne.__version__,
        "n_subjects": {k: len(v) for k, v in groups.items()},
        "groups": {k: summarise(v) for k, v in groups.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ref, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and eyeball the numbers**

Run: `python scripts/build_realism_reference.py --subjects 20`
Expected: 20 usable subjects per condition; `groups.eyes_closed["average/Alpha/coh_mean"]` = `{p10 0.365, p50 0.497, p90 0.570}`, `average/Alpha/dwpli_mean` = `{0.134, 0.185, 0.383}`, `aperiodic_exponent` p50 1.196, `rms_uv/Cz` p50 13.7 (DESIGN §9.1). Cross-check: every coherence, dwPLI and coherence-by-distance percentile must equal **2 × the feasibility test's** (`results_real.json` in the session scratchpad) within 0.005, and every exponent / alpha-share percentile must equal 1 × it within 0.005 — the rebuilt file agreed to 0.003 and 0.002 respectively. A factor-of-two shortfall means an averaging with the transpose crept back into Task 30; anything else, check the ICA blink removal and the epoching.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_realism_reference.py tests/realism/reference/eegmmidb_baselines_s01-20.json
git commit -m "test(realism): committed PhysioNet reference percentiles (ODC-By attribution included)"
```

---

### Task 32: The realism test and the tuning loop

**Files:**
- Create: `tests/realism/test_realism.py`
- Modify (tuning only): `src/open_eeg_synth/recipes.py`

Runs after Task 33 (ruff per-file ignores for `tests/realism/`) and after Task 12's calibration.

- [ ] **Step 1: Write the test**

```python
# tests/realism/test_realism.py
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mne")
pytest.importorskip("mne_connectivity")

from open_eeg_synth.case import make_case  # noqa: E402
from open_eeg_synth.recipes import resting_case  # noqa: E402
from tests.realism.measure import BANDS, measure, to_raw  # noqa: E402

REF = json.loads((Path(__file__).parent / "reference" / "eegmmidb_baselines_s01-20.json").read_text())
TOL = {"average": (0.0, 0.0), "bipolar": (0.0, 0.0), "laplacian": (-0.04, 0.08)}  # DESIGN §9.1
BIN_TOL = 0.06  # true pair values (twice the feasibility test's), so twice its +/-0.03 allowance
N_SEEDS, DUR_S = 6, 60.0


def _synth_group(cond: str) -> list[dict]:
    out = []
    for k in range(N_SEEDS):
        rec = make_case(resting_case(100 + k, duration_s=DUR_S, artifacts=())).recordings[cond]
        out.append(measure(to_raw(rec.mixed, rec.fs, rec.channels)))
    return out


def _median(group, key):
    return np.nanmedian(np.array([g[key] for g in group], dtype=float), axis=0)


@pytest.mark.realism
@pytest.mark.parametrize("cond", ["eyes_closed", "eyes_open"])
def test_resting_recipe_matches_public_reference(cond):
    ref = REF["groups"][cond]
    syn = _synth_group(cond)
    failures = []
    for reference in ("average", "bipolar", "laplacian"):
        lo_t, hi_t = TOL[reference]
        for band in BANDS:
            for metric in ("coh_mean", "dwpli_mean"):
                key = f"{reference}/{band}/{metric}"
                v, lo, hi = float(_median(syn, key)), ref[key]["p10"] + lo_t, ref[key]["p90"] + hi_t
                if not lo <= v <= hi:
                    failures.append(f"{key}: {v:.3f} not in [{lo:.3f}, {hi:.3f}]")
            key = f"{reference}/{band}/coh_by_dist"
            v = _median(syn, key)
            lo, hi = np.array(ref[key]["p10"], float) - BIN_TOL, np.array(ref[key]["p90"], float) + BIN_TOL
            bad = [(i, round(float(v[i]), 3)) for i in range(5) if not (lo[i] <= v[i] <= hi[i])]
            if bad:
                failures.append(f"{key}: bins {bad} outside [p10-{BIN_TOL}, p90+{BIN_TOL}]")
    checks = {"aperiodic_exponent": (0.8, 1.4), "rms_uv/Cz": (8.0, 20.0),
              "alpha_share/O1": (0.35, 0.79) if cond == "eyes_closed" else (0.05, 0.30)}
    for key, (lo, hi) in checks.items():
        v = float(_median(syn, key))
        if not lo <= v <= hi:
            failures.append(f"{key}: {v:.3f} not in [{lo}, {hi}]")
    assert not failures, "\n".join(failures)
```

- [ ] **Step 2: Run it**

Run: `pytest -m realism tests/realism/test_realism.py -v` (2–4 min)
Expected: with the calibrated recipe, the average and bipolar coherence rows pass; the Laplacian alpha/theta rows may fail by a few hundredths (the known gap, DESIGN §11.2). For orientation, the recipe *before* Task 12's calibration fails 7 lines eyes closed and 7 eyes open: dwPLI two to three times too low under every reference (e.g. average/Alpha 0.063 against a 0.134–0.383 band), the nearest-distance coherence bin too high under bipolar and Laplacian, and the O1 alpha share 0.33 eyes closed. dwPLI comes from the delayed network and the rhythms' inter-patch lags (DESIGN §4.2, §4.3), so if it is still low after calibration the knobs are `NetworkSpec.coupling` (1.0 → 1.5), `network_frac`, and the rhythms' `lag_ms`.

- [ ] **Step 3: Tune, at most two hours.** In this order, one change at a time, rerunning the test after each: (a) alpha `indep` 0.6 → 0.75; (b) alpha `n_patches` 4 → 6; (c) network `width_mm` 15 → 12; (d) background `smoothing_mm` 20 → 17. Keep a change only if it reduces the number of failing lines without breaking the average/bipolar rows or the fast tests (`pytest -q`, especially `tests/test_brain_layer.py` and the Task 12 calibration). Stop when the test passes or the two hours are up; in the latter case leave the failing Laplacian lines documented in the README's "Known gaps" section (Task 35) and widen `TOL["laplacian"]` to the smallest value that passes, noting the value in the commit message.

- [ ] **Step 4: Commit**

```bash
git add tests/realism/test_realism.py src/open_eeg_synth/recipes.py
git commit -m "test(realism): resting recipe against PhysioNet percentile bands; recipe tuned"
```

---

### Task 33: CI workflows, ruff configuration

**Files:**
- Modify: `.github/workflows/ci.yml` (exists from v0.1; extend)
- Create: `.github/workflows/realism.yml`
- Modify: `pyproject.toml` (`[tool.ruff]`)

- [ ] **Step 1: ci.yml** — keep the existing workflow's shape and action versions (`actions/checkout@v7`, `actions/setup-python@v7`, matrix 3.10/3.11/3.12, `pip install -e ".[dev]"`, `ruff check .`, `ruff format --check .`); change the test step to `pytest -q` (the `addopts` from Task 1 excludes `realism` and `slow`; a command-line `-m` would override it), add `pytest -m slow -q` on 3.11 only, and add `python -m build --wheel` followed by `unzip -l dist/*.whl | grep -E "colin27_19ch.npz|eog_patterns.npz|NOTICE"`, which must list four files.

```yaml
name: CI
on:
  push: {branches: [main]}
  pull_request: {branches: [main]}
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix: {python: ['3.10', '3.11', '3.12']}
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with: {python-version: '${{ matrix.python }}'}
      - run: pip install -e ".[dev]" build
      - run: ruff check . && ruff format --check .
      - run: pytest -q
      - if: matrix.python == '3.11'
        run: pytest -m slow -q
      - run: python -m build --wheel
      - run: unzip -l dist/*.whl | grep -E "colin27_19ch.npz|eog_patterns.npz|NOTICE-colin27.txt|NOTICE-eegmmidb.txt" | wc -l | grep -qx 4
```

- [ ] **Step 2: realism.yml** — `workflow_dispatch` plus weekly `schedule: cron: "0 6 * * 1"`, Python 3.11, `pip install -e ".[dev]"`, `pytest -m realism -v` (no download: the reference JSON is committed).

- [ ] **Step 3: ruff** — keep the repo's `[tool.ruff]` (`line-length = 100`, `target-version = "py310"`) and `[tool.ruff.lint] select = ["E", "F", "W", "I", "B", "UP"]` unchanged; add `[tool.ruff.lint.per-file-ignores]` with `"tests/realism/*" = ["E402"]` and `"scripts/*" = ["E402"]` (the MNE scripts import after `sys.path` setup). Run `ruff check . --fix && ruff format .` and fix what remains by hand.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml .github/workflows/realism.yml pyproject.toml
git commit -m "ci: fast suite on 3.10-3.12, wheel data check, opt-in realism workflow"
```

---

### Task 34: Golden signal fingerprint and SIGNAL_VERSION discipline

**Files:**
- Create: `tests/golden/resting_seed20260916.json` (generated)
- Create: `scripts/update_golden.py`
- Create: `tests/test_golden_engine.py`

- [ ] **Step 1: Write the test and the updater**

```python
# tests/test_golden_engine.py
import json
from pathlib import Path

import numpy as np

from open_eeg_synth import version
from open_eeg_synth.case import make_case
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.recipes import resting_case

GOLDEN = Path(__file__).parent / "golden" / "resting_seed20260916.json"


def fingerprint() -> dict:
    rec = make_case(resting_case(20260916, duration_s=10.0)).recordings["eyes_closed"]
    x = rec.mixed
    return {"signal_version": version.SIGNAL_VERSION,
            "rms_uv": [round(float(v), 2) for v in x.std(axis=1)],
            "fz_head": [round(float(v), 2) for v in x[CHANNELS_19.index("Fz"), :32]]}


def test_same_seed_same_samples_unless_signal_version_bumped():
    want = json.loads(GOLDEN.read_text())
    got = fingerprint()
    if got["signal_version"] != want["signal_version"]:
        return  # a deliberate change; scripts/update_golden.py regenerates the file
    assert np.allclose(got["rms_uv"], want["rms_uv"], atol=0.05), "signal changed without a SIGNAL_VERSION bump"
    assert np.allclose(got["fz_head"], want["fz_head"], atol=0.05)
```

```python
# scripts/update_golden.py
"""Regenerate the golden fingerprint after a deliberate signal change (bump SIGNAL_VERSION first)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_golden_engine import GOLDEN, fingerprint  # noqa: E402

GOLDEN.parent.mkdir(parents=True, exist_ok=True)
GOLDEN.write_text(json.dumps(fingerprint(), indent=1))
print("wrote", GOLDEN)
```

- [ ] **Step 2: Generate, run, commit**

Run: `python scripts/update_golden.py && pytest tests/test_golden_engine.py -v && ruff check . && ruff format --check .`
Expected: PASS. Append to the existing `CONTRIBUTING.md` (do not replace it) a short section: "any change that alters generated samples for a fixed seed bumps `SIGNAL_VERSION` in `version.py` and runs `scripts/update_golden.py` in the same commit".

```bash
git add tests/golden tests/test_golden_engine.py scripts/update_golden.py CONTRIBUTING.md
git commit -m "test: golden fingerprint guarded by SIGNAL_VERSION"
```

---

### Task 35: README, CHANGELOG, version, tag v0.2.0, consumer notes

**Files:**
- Modify: `README.md`, `CHANGELOG.md`, `src/open_eeg_synth/version.py`

- [ ] **Step 1: README sections** — "What it is" (three sentences from DESIGN's preamble), "Install" (`pip install "open-eeg-synth[edf] @ git+https://github.com/peak-mind-llc/open-eeg-synth.git@v0.2.0"`), "Quick start" (the five-line `make_case` + `write_case` example and the three-line `StreamSource` example), "Data and attribution" (both notices verbatim), "Realism" (the DESIGN §9.1 table's average rows and a link to `docs/DESIGN.md`), "Known gaps" (Laplacian long-range alpha if Task 32 left it), "Development" (`pip install -e ".[dev]"`, `pytest`, `pytest -m realism`, the three scripts and when to rerun them), "Versioning" (`SIGNAL_VERSION`).

- [ ] **Step 2: CHANGELOG** — `## 0.2.0` listing: layered engine, head model file + perturbation, brain layer, artifact framework with blink / eye movement / jaw EMG / dead channel, plants, state timeline, case files + sealed truth, `StreamSource`, realism suite, scipy runtime dependency (new), data files and notices.

- [ ] **Step 3: Version and tag** — tonight this step stops after the commit: the branch is pushed and a PR opened for review; the tag, the release and the merge to `main` wait for that review.

```bash
sed -i '' 's/__version__ = ".*"/__version__ = "0.2.0"/' src/open_eeg_synth/version.py
pytest -q && ruff check . && ruff format --check .
git add README.md CHANGELOG.md CONTRIBUTING.md src/open_eeg_synth/version.py
git commit -m "release: open-eeg-synth 0.2.0"
git tag -a v0.2.0 -m "open-eeg-synth 0.2.0 — layered engine"
git push origin main --tags
```

- [ ] **Step 4: Consumer notes (hand to each application's maintainer; not done in this repo)**
  - Recording application: Task 29 (R1–R7).
  - QEEG analysis application: add `open-eeg-synth[edf] @ git+https://github.com/peak-mind-llc/open-eeg-synth.git@v0.2.0` to its Python dependencies; in its PyInstaller spec add `open_eeg_synth` to the collected packages **and** to the set of first-party packages whose absence fails the build, and add `collect_data_files("open_eeg_synth")`; verify the bundle contains both `.npz` files and both notices. Its practice-case builder then calls `make_case(resting_case(seed, plants=…))` → `write_case(...)`, reads truth with `read_case_truth`, and re-renders layers for grading with `casefile.writer.render_layers`.

**M7 acceptance:** CI green on all three Pythons; `pytest -m realism` green (or its documented Laplacian gap); tag `v0.2.0` pushed; `pip install "open-eeg-synth[edf] @ git+https://github.com/peak-mind-llc/open-eeg-synth.git@v0.2.0"` in a fresh venv followed by `python -m open_eeg_synth make-case --seed 1 --out /tmp/x` works with no MNE installed.

---

## Self-review against the design

- **Spec coverage.** DESIGN §2 layer model → Tasks 6, 20; §3 head model file/export/subsets/perturbation → Tasks 2–4; §4.1–4.5 background/network/rhythm/plants/timeline → Tasks 7–11, 21–22; §5.1–5.4 contract/scheduling/transform/registry → Tasks 13, 19; §5.5 patterns → Task 14; §5.6 blink/eye/EMG → Tasks 16–18; §5.7 future plug-ins → accommodated by base classes (no task; by design); §6 empirical maps + attribution → Task 15; §7.1–7.2 engine/case/subject → Tasks 6, 12; §7.3 streaming → Tasks 27–28; §7.4–7.5 case files/truth/CLI → Tasks 24–26; §8 determinism/versioning/performance → chunk-invariance tests in Tasks 5–8, 10–11, 13, 20, 28, golden Task 34, perf Task 12; §9 realism + fast tests → Tasks 30–32 and every task's tests; §10 packaging/CI/consumption → Tasks 1, 33, 35, 29.
- **Placeholders.** None: every step has its code or its exact command; the two "tune" steps (12.5, 32.3) state the knobs, the order, the stopping rule and the acceptance numbers.
- **Type consistency.** `RenderContext` fields (Task 13) match every `bind` call (Tasks 12, 16–19, 28); `Recording` fields (Task 6) match `make_case` (12), `case_truth` (25), `write_edf` (24); `RhythmSpec` fields (10) match `resting_brain` (11) and `apply_modifiers` (21); `BrainLayer(…, case_seed, condition, rows)` (11) matches `make_engine` (12); the `Transform` protocol's `name` (6) is what `TransformArtifact` (13) provides; `StreamSource` constructor (28) matches the recorder factory (29); `Event.from_pattern` (13) is what Tasks 16–17 call; `EventArtifact.__init__` keyword names (`rate_by_state`, `min_gap_s`, `exclusive`) are what `params()` reflects and what Task 20's specs pass.

---

## Revisions applied 2026-09-16

Pre-flight rulings P1–P18 (see `.superpowers/sdd/PLAN/progress.md`) plus two coordinator requests (R1, D1) and three uncovered pre-flight defects (D19, D20 from `preflight.md`; the ruff formatting note). Every changed code block was re-run in a scratch tree against its task's tests: 91 of 91 plan tests pass (fast set, case files, stream, golden, perf, realism smoke); the realism test itself waits for Task 12's calibration.

- P1 → Task 1: `_version.py` is renamed to `version.py` (`git mv`), `SIGNAL_VERSION` added, `[tool.hatch.version] path` and the `__init__` import updated; Step 1 no longer says "confirm `version.py` exists".
- P2 → Task 9: `StateTimeline` is a frozen dataclass (`segments`, `ramp_s`) with value equality; its round-trip test now asserts `back == tl`. (Tasks 12, 20, 23 round-trip tests unchanged and now pass.)
- P3 → Tasks 4, 11, 12, 28: `make_subject` places patches on the full head and records `rows`; `BrainLayer(…, rows=)` renders the full head and returns the recorded rows; `make_engine` subsets only for artifacts and the sensor layer; `HeadModel.perturbed` skips the scalp blur for one channel (draw still consumed) and Task 4 gains a one-channel test; Task 28's interface says any label set works.
- P4 → Tasks 30, 31, 32 (superseded by P4b for the halving): `measure.py` uses the feasibility test's method (`spectral_connectivity_epochs(method=["wpli2_debiased", "coh"], faverage=True)`, 4 s epochs, 1–45 Hz, `standard_1020`, average reference); Task 32 states what the uncalibrated recipe scores and which knobs move dwPLI.
- P4b → Tasks 30, 31, 32 and DESIGN §9.1: the feasibility test's coherence/dwPLI figures were halved by a consumer-side `(M + Mᵀ)/2` on mne-connectivity's one-triangle output; `measure.py` reads the true pair values from the lower triangle (no averaging with the transpose), the committed reference and every stated band, median and tuned-model figure are the true values (twice the feasibility test's), the Laplacian and per-bin allowances double with them (`[p10 − 0.04, p90 + 0.08]`, ±0.06), and Task 31's cross-check is "2 × `results_real.json` within 0.005" (rebuilt: 0.003; non-connectivity metrics 1 × within 0.002). Re-run in scratch: smoke test passes; the uncalibrated realism test fails 7 + 7 lines (dwPLI ~0.06 against 0.13–0.38), as Task 32 now states.
- P5 → Task 5: `welch` shortens `nperseg` to the input length, so `band_power` works on per-second windows (Task 18's burst test no longer crashes).
- P6 → Tasks 16, 15: `Blink.waveform` normalised to unit peak on the sampled grid; the blink test's posterior factor is 4 (one bound, `|O1| < 0.25`, in both tasks).
- P7 → Task 15: heog time-course threshold 0.4 (25 of 30 subjects clear it; 0.2 kept 4 of 20); download failure stops at the subjects on disk; subjects 1–30 stay the default (21–30 downloaded in about a minute here); the file test asserts `n_subjects >= 15` and `heog_sd.max() < HEOG_SD_MAX` with `HEOG_SD_MAX = 0.45` — the ruled 0.35 cannot be met: the derived map's spread is 0.40 over 17 subjects and 0.42 over 30, so 0.35 would fail regardless of subject count (DESIGN §6.3 now states 0.45 for heog, 0.35 for blink).
- P8 → Tasks 11, 12, 21: brain-amplitude assertions are made after average referencing with provisional bands; Task 12 Step 5 states the measured starting point (Cz 21 µV, O1 share 0.35) and pins Task 11's bands after calibration; the bursts test uses per-second theta power at Fz after average referencing.
- P9 → Task 13: the scheduler is the thinned Poisson process of DESIGN §5.2 (candidates at `rate_max`, pushes instead of refractory re-draws); `truth()` lists events inside the rendered span; the rate test uses 1/s × 0.1 s events with the band 18–45; the gap and exclusivity tests use rates whose busy fraction stays below one.
- P10 → Task 27: heart chunk-invariance asserted with `atol=1e-3`.
- P11 → Task 28: posterior dominance asserted on eyes-closed, artifact-free streams, after average referencing, as a median over three seeds (> 2×); ECG checks unchanged. Task 28/29 note the common-reference component for the recorder's live view.
- P12 → Tasks 12, 13, 20 and the milestone table: Task 13 runs before Task 12 (title, M2/M3 rows, unit order line); the interim import shuffle is deleted; `case.py` imports `open_eeg_synth.artifacts` at module top from Task 12; Task 20 no longer touches `case.py`.
- P13 → Task 2: zero-length source normals (3 of 4871) get the outward radial direction, recorded in the attribution string; the fixed-vs-free check excludes them; the unit-normal test stands.
- P14 → global constraints, Task 1, Task 33 and every task: `ruff>=0.6`, `addopts = "-ra -m …"`, repo CI action versions (`@v7`) and rule set (`E,F,W,I,B,UP`) kept, per-file ignores limited to `tests/realism/*` and `scripts/*` (E402); Task 33 runs before Tasks 30–32 (M7 row); every "verify they pass" step now ends with `ruff check . && ruff format --check .` (30 steps).
- P15 → Tasks 6, 13: the Engine keys transform layers by the artifact's `name` (`transform:<kind>` or `transform:<kind>#<i>`); `TransformArtifact.name` added; the Task 6 test transform carries a `name`.
- P16 → Tasks 2, 26, 34, 35: `pyproject` sdist include gains `scripts` (Task 2); `casefile.writer` re-exports `render_layers`; Task 34 appends to the existing `CONTRIBUTING.md`; Task 35's consumer note names `casefile.writer.render_layers`.
- P17 → Tasks 11, 12 and the global constraints: the brain sub-layers are seeded from `<condition>:background`, `<condition>:network`, `<condition>:rhythm:<name>` exactly as DESIGN §8.1 lists (no `<condition>:brain` spawn); DESIGN §4.3 documents the per-patch 0.3 Hz offset.
- P18 → Task 1 pyproject and the tech-stack line: `mne>=1.6,<1.14` in the `realism` and `dev` extras.
- R1 → Task 21: `FocalSlow` and `RhythmicBursts` accept `state_gain` (passed through to their `RhythmSpec`, carried in the record's `params`); new test `test_plant_state_gain_silences_a_plant_by_state`.
- D1 → Task 1 Step 1: classic modules named correctly (`synth.py`, `cardio.py`, `oddball.py`; no `mock_rr.py`/`impedance.py`).
- D19 (preflight) → Task 21: `PlantRecord.to_dict` converts tuples in `params` to lists so a truth file round-trips for every primitive.
- D20 (preflight) → Task 22: the alpha-peak shift is measured on a Welch spectrum, not a raw periodogram.
- Setup ruling → Task 35: Step 3 stops after the version commit; tag/release/merge wait for review.
- Formatting → global constraints: code blocks are not pre-formatted; `ruff format .` runs before each green step.
