from __future__ import annotations

import json

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
            f"artifact:{artifact.kind}",
        )
    )


def _render(kind: str, params: dict, chunks: tuple[int, ...], *, seed: int = 1):
    artifact = make_artifact(kind, **params)
    _bind(artifact, seed=seed)
    parts = []
    t0 = 0
    for n in chunks:
        parts.append(artifact.render(t0, n))
        t0 += n
    return np.concatenate(parts, axis=1), artifact.truth()


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("sweat_drift", {"side": "left", "rms_uv": 80.0}),
        ("noisy_channel", {"channel": "T3", "rms_uv": 20.0}),
        ("mains_hum", {"freq_hz": 60.0, "base_rms_uv": 10.0}),
    ],
)
def test_continuous_acquisition_artifacts_are_seeded_and_chunk_invariant(kind, params):
    whole, whole_truth = _render(kind, params, (4096,), seed=19)
    chunked, chunked_truth = _render(kind, params, (511, 777, 1024, 1784), seed=19)
    assert np.array_equal(whole, chunked)
    assert whole_truth == chunked_truth
    assert whole_truth[0].onset_s == 0.0
    assert whole_truth[0].offset_s is None


def test_sweat_drift_is_slow_and_frontotemporal_with_declared_truth():
    values, truth = _render("sweat_drift", {"side": "left", "rms_uv": 80.0}, (8192,))
    active = np.flatnonzero(np.max(np.abs(values), axis=1) > 0)
    assert {CHANNELS_19[index] for index in active} == {"Fp1", "F3", "F7", "T3"}
    spectrum = np.abs(np.fft.rfft(values[CHANNELS_19.index("Fp1")])) ** 2
    freqs = np.fft.rfftfreq(values.shape[1], 1 / FS)
    assert (
        spectrum[(freqs >= 0.05) & (freqs <= 0.5)].mean()
        > 20 * spectrum[(freqs >= 1.0) & (freqs <= 4.0)].mean()
    )
    assert truth[0].remedies == (Remedy.RAISE_HIGH_PASS, Remedy.MASK_SEGMENT)
    assert truth[0].params["side"] == "left"


def test_contact_noise_is_a_finite_irregular_channel_local_event():
    artifact = make_artifact(
        "contact_noise",
        channel="T3",
        rate_by_state={},
        rms_range_uv=(45.0, 45.0),
        duration_range_s=(0.8, 0.8),
    )
    _bind(artifact, seed=23)
    event = artifact.make_event(1024)
    active = np.flatnonzero(np.max(np.abs(event.block), axis=1) > 0)
    assert active.tolist() == [CHANNELS_19.index("T3")]
    waveform = event.block[active[0]]
    assert waveform.size == round(0.8 * FS)
    assert np.std(waveform) == pytest.approx(45.0, rel=0.05)
    assert np.max(np.abs(np.diff(waveform))) > 2 * np.std(waveform)
    assert event.truth.remedies == (Remedy.MASK_SEGMENT, Remedy.MARK_BAD_CHANNEL)
    assert event.truth.params["rms_uv"] == pytest.approx(45.0)


def test_noisy_channel_is_persistent_broadband_and_channel_local():
    values, truth = _render("noisy_channel", {"channel": "T3", "rms_uv": 20.0}, (8192,), seed=31)
    target = CHANNELS_19.index("T3")
    active = np.flatnonzero(np.max(np.abs(values), axis=1) > 0)
    assert active.tolist() == [target]
    assert np.std(values[target]) == pytest.approx(20.0, rel=0.08)
    assert truth[0].channels == ("T3",)
    assert truth[0].remedies == (Remedy.MARK_BAD_CHANNEL, Remedy.INTERPOLATE)


def test_mains_hum_has_requested_fundamental_and_only_valid_harmonics():
    values, truth = _render(
        "mains_hum",
        {
            "freq_hz": 60.0,
            "base_rms_uv": 10.0,
            "am_depth": 0.0,
            "harmonic_ratio": 0.1,
        },
        (8192,),
        seed=41,
    )
    spectrum = np.abs(np.fft.rfft(values[0]))
    freqs = np.fft.rfftfreq(values.shape[1], 1 / FS)

    def amplitude_at(hz: float) -> float:
        return float(spectrum[np.argmin(np.abs(freqs - hz))])

    assert amplitude_at(60.0) > 50 * np.median(spectrum)
    assert amplitude_at(120.0) > 10 * np.median(spectrum)
    assert truth[0].params["harmonics_hz"] == [60.0, 120.0]
    assert truth[0].remedies == (Remedy.NOTCH, Remedy.RE_REFERENCE)


@pytest.mark.parametrize("freq", [0, 55, 70, float("nan")])
def test_mains_frequency_must_be_50_or_60_hz(freq):
    with pytest.raises(ValueError, match="50 or 60"):
        make_artifact("mains_hum", freq_hz=freq)


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("sweat_drift", {"side": "bilateral", "rms_uv": 70.0}),
        ("contact_noise", {"channel": "T3", "rate_by_state": {}}),
        ("noisy_channel", {"channel": "T4", "rms_uv": 18.0}),
        ("mains_hum", {"freq_hz": 60.0, "base_rms_uv": 9.0}),
    ],
)
def test_acquisition_artifact_params_are_json_safe_and_reconstruct(kind, params):
    artifact = make_artifact(kind, **params)
    encoded = json.loads(json.dumps(artifact.params()))
    rebuilt = make_artifact(kind, **encoded)
    assert json.loads(json.dumps(rebuilt.params())) == encoded
