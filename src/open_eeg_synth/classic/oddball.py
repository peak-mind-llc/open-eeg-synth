"""Oddball paradigm schedule and simulated event-related potentials.

Moved from Coherence Recorder's standalone mock ERP stream, without its
streaming (LSL) plumbing. Two deliberate changes from that script:

* Every random draw comes from a caller-supplied ``numpy.random.Generator``,
  so a schedule is reproducible from a seed.
* A response is carried across chunk boundaries. The original added a
  response only when it *began* inside the chunk that held the stimulus
  marker. With 32-sample chunks at 256 Hz and a 280-350 ms latency, that
  condition was never true, so the original stream carried no responses.

The paradigm defaults match the Peak Mind oddball paradigm: 70 % standard,
15 % target, 15 % distractor; inter-stimulus interval 1.2-1.8 s; at least two
standards before a deviant; a button press 300-600 ms after 90 % of targets.
The response shapes are illustrative, not physiological: a P3b-like positive
half-sine at Pz (60 % of it at Cz) after targets and a P3a-like one at Fz
(50 % at Cz) after distractors.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np

CONDITION_RATIOS: Mapping[str, float] = {"standard": 0.70, "target": 0.15, "distractor": 0.15}
CONDITION_CODES: Mapping[str, str] = {"standard": "1", "target": "2", "distractor": "3"}
RESPONSE_CONDITION = "response"
RESPONSE_CODE = "99"


@dataclasses.dataclass(frozen=True)
class OddballParadigm:
    """Timing and proportions of an auditory or visual oddball task."""

    condition_ratios: Mapping[str, float] = dataclasses.field(
        default_factory=lambda: dict(CONDITION_RATIOS)
    )
    condition_codes: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: dict(CONDITION_CODES)
    )
    response_code: str = RESPONSE_CODE
    isi_range_s: tuple[float, float] = (1.2, 1.8)
    min_standards_before_deviant: int = 2
    response_rt_range_s: tuple[float, float] = (0.3, 0.6)
    response_probability: float = 0.90
    lead_in_s: float = 2.0

    def __post_init__(self) -> None:
        if "standard" not in self.condition_ratios:
            raise ValueError("condition_ratios must include 'standard'")
        if any(r < 0 for r in self.condition_ratios.values()):
            raise ValueError("condition_ratios must be non-negative")
        if sum(self.condition_ratios.values()) <= 0:
            raise ValueError("condition_ratios must not all be zero")
        missing = set(self.condition_ratios) - set(self.condition_codes)
        if missing:
            raise ValueError(f"no marker code for condition(s): {sorted(missing)}")
        lo, hi = self.isi_range_s
        if not 0 < lo <= hi:
            raise ValueError("isi_range_s must satisfy 0 < low <= high")
        lo, hi = self.response_rt_range_s
        if not 0 <= lo <= hi:
            raise ValueError("response_rt_range_s must satisfy 0 <= low <= high")
        if not 0.0 <= self.response_probability <= 1.0:
            raise ValueError("response_probability must be within [0, 1]")
        if self.min_standards_before_deviant < 0:
            raise ValueError("min_standards_before_deviant must be >= 0")


@dataclasses.dataclass(frozen=True)
class OddballEvent:
    """One marker: a stimulus (standard, target, distractor) or a response."""

    onset_s: float
    condition: str
    code: str


def generate_trial_sequence(
    n_trials: int, rng: np.random.Generator, paradigm: OddballParadigm | None = None
) -> list[str]:
    """Conditions for ``n_trials`` trials, honouring the minimum run of standards."""
    p = paradigm or OddballParadigm()
    if n_trials < 0:
        raise ValueError("n_trials must be >= 0")
    conditions = list(p.condition_ratios)
    weights = np.array([p.condition_ratios[c] for c in conditions], dtype=np.float64)
    weights = weights / weights.sum()
    sequence: list[str] = []
    run_of_standards = 0
    for _ in range(n_trials):
        if run_of_standards < p.min_standards_before_deviant:
            condition = "standard"
        else:
            condition = conditions[int(rng.choice(len(conditions), p=weights))]
        sequence.append(condition)
        run_of_standards = run_of_standards + 1 if condition == "standard" else 0
    return sequence


def schedule(
    n_trials: int, rng: np.random.Generator, paradigm: OddballParadigm | None = None
) -> list[OddballEvent]:
    """Stimulus and response markers for a run of ``n_trials``, sorted by onset."""
    p = paradigm or OddballParadigm()
    conditions = generate_trial_sequence(n_trials, rng, p)
    events: list[OddballEvent] = []
    t = p.lead_in_s
    for condition in conditions:
        t += float(rng.uniform(*p.isi_range_s))
        events.append(OddballEvent(t, condition, p.condition_codes[condition]))
        if condition == "target" and float(rng.random()) < p.response_probability:
            rt = float(rng.uniform(*p.response_rt_range_s))
            events.append(OddballEvent(t + rt, RESPONSE_CONDITION, p.response_code))
    events.sort(key=lambda e: e.onset_s)
    return events


@dataclasses.dataclass(frozen=True)
class ErpComponent:
    """A positive half-sine starting ``latency_s`` after the stimulus."""

    latency_s: float
    width_s: float
    amplitude_uv: float
    channel_gains: Mapping[str, float]

    def waveform(self, srate: float) -> np.ndarray:
        n = max(1, int(self.width_s * srate))
        return (self.amplitude_uv * np.sin(np.linspace(0.0, np.pi, n))).astype(np.float32)


ERP_COMPONENTS: Mapping[str, tuple[ErpComponent, ...]] = {
    "target": (ErpComponent(0.35, 0.15, 8.0, {"Pz": 1.0, "Cz": 0.6}),),
    "distractor": (ErpComponent(0.28, 0.12, 6.0, {"Fz": 1.0, "Cz": 0.5}),),
}


class ErpInjector:
    """Adds event-locked responses to a stream of ``(n_channels, n_samples)`` chunks.

    Register each stimulus with :meth:`add_event`, then pass every chunk, in
    order, to :meth:`apply`. A response that runs past the end of one chunk
    continues in the next, so the result does not depend on chunk size.
    """

    def __init__(
        self,
        channel_labels: Sequence[str],
        srate: float,
        components: Mapping[str, Sequence[ErpComponent]] | None = None,
    ) -> None:
        if srate <= 0:
            raise ValueError("srate must be positive")
        self.labels = [str(label).strip() for label in channel_labels]
        self.srate = float(srate)
        self.components = ERP_COMPONENTS if components is None else components
        self._index = {label: i for i, label in enumerate(self.labels)}
        # (first sample, waveform, [(channel index, gain), ...])
        self._pending: list[tuple[int, np.ndarray, list[tuple[int, float]]]] = []

    @property
    def pending(self) -> int:
        """Responses registered but not yet fully written."""
        return len(self._pending)

    def add_event(self, onset_s: float, condition: str) -> None:
        for comp in self.components.get(condition, ()):
            targets = [
                (self._index[ch], float(gain))
                for ch, gain in comp.channel_gains.items()
                if ch in self._index
            ]
            if not targets:
                continue
            start = int(onset_s * self.srate) + int(comp.latency_s * self.srate)
            self._pending.append((start, comp.waveform(self.srate), targets))

    def apply(self, chunk: np.ndarray, chunk_start_sample: int) -> None:
        """Add every response overlapping this chunk, in place."""
        if chunk.ndim != 2 or chunk.shape[0] != len(self.labels):
            raise ValueError("chunk must be shaped (n_channels, n_samples)")
        c0 = int(chunk_start_sample)
        c1 = c0 + chunk.shape[1]
        still_pending = []
        for start, wave, targets in self._pending:
            end = start + len(wave)
            lo, hi = max(start, c0), min(end, c1)
            if lo < hi:
                piece = wave[lo - start : hi - start]
                for idx, gain in targets:
                    chunk[idx, lo - c0 : hi - c0] += gain * piece
            if end > c1:
                still_pending.append((start, wave, targets))
        self._pending = still_pending
