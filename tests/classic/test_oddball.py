"""Oddball schedule and event-locked responses."""

from __future__ import annotations

import numpy as np
import pytest

from open_eeg_synth.classic.oddball import (
    ErpComponent,
    ErpInjector,
    OddballParadigm,
    generate_trial_sequence,
    schedule,
)


def test_sequence_starts_and_resumes_with_standards():
    seq = generate_trial_sequence(2000, np.random.default_rng(0))
    assert seq[:2] == ["standard", "standard"]
    for i, cond in enumerate(seq):
        if cond != "standard":
            assert seq[i - 2 : i] == ["standard", "standard"]


def test_sequence_proportions_are_close_to_the_paradigm():
    seq = generate_trial_sequence(6000, np.random.default_rng(1))
    targets = seq.count("target") / len(seq)
    distractors = seq.count("distractor") / len(seq)
    # The run-of-standards rule lowers deviant rates below the nominal 15 %.
    assert 0.08 < targets < 0.15
    assert 0.08 < distractors < 0.15


def test_schedule_is_reproducible_and_well_formed():
    a = schedule(300, np.random.default_rng(2))
    b = schedule(300, np.random.default_rng(2))
    assert a == b
    stimuli = [e for e in a if e.condition != "response"]
    assert len(stimuli) == 300
    onsets = np.array([e.onset_s for e in stimuli])
    gaps = np.diff(onsets)
    assert gaps.min() >= 1.2 - 1e-9 and gaps.max() <= 1.8 + 1e-9
    assert onsets[0] > 2.0
    assert [e.onset_s for e in a] == sorted(e.onset_s for e in a)


def test_responses_follow_most_targets_within_the_reaction_window():
    events = schedule(3000, np.random.default_rng(3))
    targets = [e for e in events if e.condition == "target"]
    responses = [e for e in events if e.condition == "response"]
    assert all(e.code == "99" for e in responses)
    assert 0.85 < len(responses) / len(targets) < 0.95
    target_onsets = np.array([t.onset_s for t in targets])
    for r in responses:
        latency = r.onset_s - target_onsets[target_onsets < r.onset_s].max()
        assert 0.3 - 1e-9 <= latency <= 0.6 + 1e-9


def test_paradigm_validation():
    with pytest.raises(ValueError):
        OddballParadigm(isi_range_s=(0.0, 1.0))
    with pytest.raises(ValueError):
        OddballParadigm(response_probability=1.5)
    with pytest.raises(ValueError):
        OddballParadigm(condition_ratios={"target": 1.0})


LABELS = ["Fz", "Cz", "Pz", "O1"]


def _stream(chunk: int, n_total: int, onset_s: float, condition: str) -> np.ndarray:
    inj = ErpInjector(LABELS, 256.0)
    inj.add_event(onset_s, condition)
    out = np.zeros((len(LABELS), n_total), dtype=np.float32)
    for start in range(0, n_total, chunk):
        block = np.zeros((len(LABELS), min(chunk, n_total - start)), dtype=np.float32)
        inj.apply(block, start)
        out[:, start : start + block.shape[1]] = block
    return out


def test_a_response_that_starts_after_its_chunk_is_still_written():
    # The original stream only wrote a response that began inside the stimulus
    # chunk. With 32-sample chunks the P3b starts ~90 samples later.
    x = _stream(chunk=32, n_total=512, onset_s=0.1, condition="target")
    assert x[LABELS.index("Pz")].max() == pytest.approx(8.0, abs=0.05)
    assert x[LABELS.index("Cz")].max() == pytest.approx(4.8, abs=0.05)
    assert not x[LABELS.index("Fz")].any()
    assert not x[LABELS.index("O1")].any()


def test_chunking_does_not_change_the_result():
    whole = _stream(chunk=512, n_total=512, onset_s=0.1, condition="distractor")
    pieces = _stream(chunk=7, n_total=512, onset_s=0.1, condition="distractor")
    np.testing.assert_array_equal(whole, pieces)
    assert whole[LABELS.index("Fz")].max() == pytest.approx(6.0, abs=0.05)


def test_response_timing_matches_the_component():
    x = _stream(chunk=64, n_total=512, onset_s=0.5, condition="target")
    nonzero = np.flatnonzero(x[LABELS.index("Pz")])
    assert nonzero[0] >= int(0.5 * 256) + int(0.35 * 256)
    assert nonzero[-1] < int(0.5 * 256) + int(0.35 * 256) + int(0.15 * 256)


def test_pending_responses_are_released_once_written():
    inj = ErpInjector(LABELS, 256.0)
    inj.add_event(0.0, "target")
    inj.add_event(0.0, "standard")  # standards carry no response
    assert inj.pending == 1
    inj.apply(np.zeros((len(LABELS), 512), dtype=np.float32), 0)
    assert inj.pending == 0


def test_custom_components_and_missing_channels():
    comp = ErpComponent(0.1, 0.1, 5.0, {"Oz": 1.0, "O1": 0.5})
    inj = ErpInjector(LABELS, 100.0, components={"target": (comp,)})
    inj.add_event(0.0, "target")
    block = np.zeros((len(LABELS), 50), dtype=np.float32)
    inj.apply(block, 0)
    assert block[LABELS.index("O1")].max() == pytest.approx(2.5, abs=0.05)


def test_apply_rejects_a_wrongly_shaped_chunk():
    inj = ErpInjector(LABELS, 256.0)
    with pytest.raises(ValueError):
        inj.apply(np.zeros((32, len(LABELS)), dtype=np.float32), 0)
