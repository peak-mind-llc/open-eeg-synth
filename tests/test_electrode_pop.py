from __future__ import annotations

import numpy as np
import pytest

from open_eeg_synth.artifacts.base import Occupancy, Remedy, RenderContext
from open_eeg_synth.artifacts.registry import make_artifact
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.seeds import stream_rng

FS = 256.0


def _bind(artifact, *, seed: int = 1) -> None:
    head = load_head_model()
    artifact.bind(
        RenderContext(
            head.channels,
            FS,
            head.electrode_pos,
            head,
            StateTimeline.constant("eyes_open"),
            stream_rng(seed, "event"),
            stream_rng(seed, "subject"),
            Occupancy(),
            "artifact:electrode_pop",
        )
    )


def test_registered_electrode_pop_is_a_local_abrupt_decay_with_truth():
    artifact = make_artifact(
        "electrode_pop",
        channel="T3",
        polarity="positive",
        rate_by_state={},
        peak_range_uv=(200.0, 200.0),
        tau_range_s=(0.3, 0.3),
    )
    _bind(artifact)
    event = artifact.make_event(512)

    target = CHANNELS_19.index("T3")
    assert event.onset == 512
    assert event.block.shape[0] == len(CHANNELS_19)
    assert np.flatnonzero(np.max(np.abs(event.block), axis=1)).tolist() == [target]
    assert event.block[target, 0] == pytest.approx(200.0)
    assert np.all(np.diff(event.block[target]) < 0)
    assert 1.0 <= event.block[target, -1] <= 2.1

    truth = event.truth
    assert truth.kind == "electrode_pop"
    assert truth.subtype == "positive"
    assert truth.channels == ("T3",)
    assert truth.peak_uv == pytest.approx(200.0)
    assert truth.remedies == (
        Remedy.MASK_SEGMENT,
        Remedy.MARK_BAD_CHANNEL,
    )
    assert truth.params["tau_s"] == pytest.approx(0.3)
    assert truth.params["duration_s"] == pytest.approx(event.block.shape[1] / FS)


def test_defaults_match_the_declared_v03_design_ranges():
    artifact = make_artifact("electrode_pop", rate_by_state={})
    assert artifact.peak_range_uv == (200.0, 2000.0)
    assert artifact.tau_range_s == (0.2, 2.0)


def test_random_channel_and_waveform_are_seeded():
    def draw():
        artifact = make_artifact("electrode_pop", rate_by_state={})
        _bind(artifact, seed=17)
        return artifact.make_event(0)

    first, second = draw(), draw()
    assert first.truth.channels == second.truth.channels
    assert np.array_equal(first.block, second.block)


def test_unknown_electrode_pop_channel_fails_at_bind():
    artifact = make_artifact("electrode_pop", channel="not-a-channel", rate_by_state={})
    with pytest.raises(ValueError, match="electrode_pop channel"):
        _bind(artifact)
