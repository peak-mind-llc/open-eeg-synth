"""Chunked, phase-continuous output for a recording application's mock amplifiers (DESIGN §7.3).

``StreamSource`` mirrors ``classic.RealisticEEGSynthesizer``'s constructor (channel labels, sample
rate, seed, markers) so a recording application swaps one import to move its mock amplifiers onto
the layered head-model engine. Construction costs about half a second (head perturbation plus
mixing-matrix smoothing on the full head model) — fine for a mock device that starts once, not for
building one per test. Chunk size does not change the signal (DESIGN §8.2): unlike ``classic``,
which drew blinks per chunk, samples here depend only on how many have been rendered so far.

Note for the recording application: the raw stream's alpha is far less posterior-dominant than the
classic synthesizer's, because the lead field's native reference carries a common component that
every channel shares (DESIGN §2.3); a live view that re-references (average or linked ears) shows
the gradient, a raw referential view shows alpha on every channel, as a real referential amplifier
does.

``.truth`` grows for as long as the stream runs — it is every event scheduled so far, and nothing
here caches or discards it. A recording application that streams for hours should read ``.truth``
incrementally (e.g. record the length already consumed) rather than re-copy the whole list on every
chunk.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from open_eeg_synth.artifacts.base import TruthRecord
from open_eeg_synth.brain.layer import BrainSpec
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.case import (
    ArtifactSpec,
    CaseSpec,
    ConditionSpec,
    SensorSpec,
    make_engine,
    make_subject,
)
from open_eeg_synth.channels import canonical_label, is_heart_label
from open_eeg_synth.heart import HeartSource
from open_eeg_synth.markers import MarkerSchedule
from open_eeg_synth.recipes import ordinary_artifacts, resting_brain
from open_eeg_synth.seeds import fresh_case_seed, stream_rng, stream_seed


class StreamSource:
    """A chunked, phase-continuous stand-in for a real amplifier (DESIGN §7.3).

    Wraps a full-head :class:`~open_eeg_synth.case.CaseSpec`/:class:`~open_eeg_synth.engine.Engine`
    (brain, sensor noise, ordinary artifacts by default) plus a
    :class:`~open_eeg_synth.heart.HeartSource` for any heart-rate label, behind the same
    constructor and call shape as ``classic.RealisticEEGSynthesizer``. Call :meth:`next_chunk`
    (and, in lockstep, :meth:`due_markers`) repeatedly as new samples are needed; ``.truth`` and
    ``.seed`` are read at any time.

    ``channel_labels`` must not repeat an EEG electrode (CaseSpec rejects duplicate channels
    after alias canonicalisation); heart-rate labels are the exception and may repeat freely.
    """

    def __init__(
        self,
        channel_labels: Sequence[str],
        srate: float,
        *,
        seed: int | None = None,
        markers: MarkerSchedule | None = None,
        timeline: StateTimeline | None = None,
        brain: BrainSpec | None = None,
        artifacts: Sequence[ArtifactSpec] | None = None,
        sensor: SensorSpec | None = None,
        perturb_head: bool = True,
    ) -> None:
        if not channel_labels:
            raise ValueError("channel_labels must be non-empty")
        if not math.isfinite(srate) or srate <= 0:
            raise ValueError(f"srate must be finite and positive, got {srate!r}")
        self.labels = list(channel_labels)
        self.srate = float(srate)
        self._seed = fresh_case_seed() if seed is None else int(seed)
        self._eeg_rows = [i for i, lb in enumerate(self.labels) if not is_heart_label(lb)]
        self._hr_rows = [i for i, lb in enumerate(self.labels) if is_heart_label(lb)]
        eeg_labels = tuple(canonical_label(self.labels[i]) for i in self._eeg_rows)
        self.timeline = timeline or StateTimeline.constant("eyes_open")
        self.spec = CaseSpec(
            seed=self._seed,
            fs=self.srate,
            channels=eeg_labels,
            perturb_head=perturb_head,
            brain=brain or resting_brain(),
            artifacts=tuple(artifacts) if artifacts is not None else ordinary_artifacts(),
            sensor=sensor or SensorSpec(),
            conditions=(ConditionSpec("stream", math.inf, self.timeline),),
        )
        self.subject = make_subject(self.spec)
        self.engine = make_engine(self.spec, self.subject, self.spec.conditions[0])
        self.heart = (
            HeartSource(self.srate, stream_seed(self._seed, "stream:heart"))
            if self._hr_rows
            else None
        )
        self.markers = markers or MarkerSchedule()
        self._marker_rng = stream_rng(self._seed, "stream:markers")
        self._marker_clock_s = 0.0
        self._next_marker_s = self.markers.period_s if self.markers.kind != "none" else math.inf
        self._pos = 0

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def truth(self) -> list[TruthRecord]:
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
        """Same contract as the classic synthesizer: call in lockstep with next_chunk.

        Reimplements ``classic.synth``'s marker-clock logic here rather than importing and
        calling it, because ``open_eeg_synth.classic`` stays byte-identical (it is the recording
        application's original output, frozen) and so cannot take a dependency on this package's
        engine or RNG plumbing.
        """
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
