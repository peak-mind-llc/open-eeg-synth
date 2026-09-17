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
        """Plain-Python-typed (JSON-safe) and unrounded, so ``from_dict(to_dict(r)) == r``."""
        return {
            "kind": self.kind,
            "subtype": self.subtype,
            "side": self.side,
            "channels": list(self.channels),
            "onset_s": float(self.onset_s),
            "offset_s": None if self.offset_s is None else float(self.offset_s),
            "peak_uv": float(self.peak_uv),
            "remedies": [r.value for r in self.remedies],
            "layer": self.layer,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: dict) -> TruthRecord:
        return cls(
            d["kind"],
            d["subtype"],
            d["side"],
            tuple(d["channels"]),
            d["onset_s"],
            d["offset_s"],
            d["peak_uv"],
            tuple(Remedy(r) for r in d["remedies"]),
            d["layer"],
            dict(d.get("params", {})),
        )


class Occupancy:
    """Busy spans (samples) shared by every exclusive artifact of one engine.

    Also coordinates the exclusive artifacts' scheduling itself (§5.2): a per-artifact schedule
    driven purely by that artifact's own block boundary is not chunk- or call-order-invariant,
    because whichever artifact happens to claim a span first depends on how the render calls
    happen to be chunked and interleaved. `advance_exclusive` instead advances every exclusive
    artifact registered here together, resolving strictly in global onset order, so the result
    depends only on the final `until` requested and not on how many smaller steps got there or
    which artifact's render() call triggered them (DESIGN §8.2).
    """

    def __init__(self) -> None:
        self.spans: list[tuple[int, int]] = []
        self._exclusive: list[EventArtifact] = []
        self._advanced_to = 0

    def add_exclusive(self, art: EventArtifact) -> None:
        if art not in self._exclusive:
            self._exclusive.append(art)

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

    def advance_exclusive(self, until: int) -> None:
        """Advance every exclusive artifact sharing this occupancy up to `until`, in onset order.

        Each artifact peeks its own next candidate (drawing from its own rng only when it needs
        a fresh one); whichever peeked onset is chronologically earliest is committed first. Ties
        break on registration order, which is fixed at bind() time and so is independent of
        render-call order or chunk size.
        """
        if until <= self._advanced_to:
            return
        while True:
            best_onset: int | None = None
            best_art: EventArtifact | None = None
            for art in self._exclusive:
                onset = art._peek_onset(until)
                if onset is not None and (best_onset is None or onset < best_onset):
                    best_onset, best_art = onset, art
            if best_art is None:
                break
            best_art._consume_next()
        self._advanced_to = until


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

    def __post_init__(self) -> None:
        if self.block.ndim != 2:
            raise ValueError(f"Event.block must be (n_ch, L), got shape {self.block.shape}")

    @property
    def end(self) -> int:
        return self.onset + int(self.block.shape[1])

    @classmethod
    def from_pattern(
        cls, onset: int, pattern: np.ndarray, waveform: np.ndarray, truth: TruthRecord
    ) -> Event:
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

    def __init__(
        self, *, rate_by_state: dict[str, float], min_gap_s: float = 0.0, exclusive: bool = False
    ) -> None:
        self.rate_by_state = dict(rate_by_state)
        self.min_gap_s = float(min_gap_s)
        self.exclusive = bool(exclusive)

    @property
    def name(self) -> str:
        return self.layer_name

    def bind(self, ctx: RenderContext) -> None:
        """Attach a render context and (re-)initialise all scheduler state.

        Safe to call again to rebind (e.g. reuse of a plug-in instance across conditions): any
        events already scheduled or drawn against a previous context are discarded rather than
        leaking into the new one.
        """
        self.ctx = ctx
        self.fs = ctx.fs
        self.n_ch = len(ctx.channels)
        self.layer_name = ctx.layer_name
        self._rate_max = max(self.rate_by_state.values(), default=0.0)
        self._pending: list[Event] = []
        self._truth: list[TruthRecord] = []
        self._truth_onset: list[int] = []
        self._cand_s = 0.0
        self._next: tuple[float, bool] | None = None
        self._earliest = 0
        self._rendered = 0
        if self.exclusive:
            ctx.occupancy.add_exclusive(self)

    @abstractmethod
    def make_event(self, onset: int) -> Event: ...

    def _peek_onset(self, until: int) -> int | None:
        """Ensure the next scheduling candidate is drawn; return its onset sample if < until."""
        if self._rate_max <= 0.0:
            return None
        if self._next is None:
            rng = self.ctx.rng
            self._cand_s += float(rng.exponential(1.0 / self._rate_max))
            p = self.ctx.timeline.rate(self.rate_by_state, self._cand_s) / self._rate_max
            self._next = (self._cand_s, bool(rng.random() < p))
        t_s, _accept = self._next
        onset = int(round(t_s * self.fs))
        return onset if onset < until else None

    def _consume_next(self) -> None:
        """Commit the currently peeked candidate: reject it, or build and place its Event.

        The exclusive push (DESIGN §5.2) happens here, against whatever this artifact's own or
        a peer's earlier commit has already claimed on the shared Occupancy, so the Event this
        artifact keeps and reports through truth() always reflects its final onset, never the
        pre-push candidate.
        """
        t_s, accept = self._next
        self._next = None
        onset = int(round(t_s * self.fs))
        if not accept:
            return
        onset = max(onset, self._earliest)
        ev = self.make_event(onset)
        if self.exclusive:
            moved = self.ctx.occupancy.next_free(onset, ev.block.shape[1])
            if moved != onset:
                truth = ev.truth
                if truth.offset_s is None:
                    new_truth = replace(truth, onset_s=moved / self.fs)
                else:
                    dur = truth.offset_s - truth.onset_s
                    new_truth = replace(
                        truth, onset_s=moved / self.fs, offset_s=moved / self.fs + dur
                    )
                ev = replace(ev, onset=moved, truth=new_truth)
            self.ctx.occupancy.add(ev.onset, ev.end)
        self._pending.append(ev)
        self._truth.append(ev.truth)
        self._truth_onset.append(ev.onset)
        self._earliest = ev.end + int(round(self.min_gap_s * self.fs))

    def _schedule_until(self, until: int) -> None:
        """Thinned Poisson (DESIGN §5.2): candidates at rate_max, kept with p = rate(state)/max.

        An accepted onset inside the previous event's refractory gap is pushed to the end of
        the gap, never dropped, so the nominal rate holds as long as rate x (length + gap)
        stays well below one. Candidates and waveforms are drawn in onset order, so chunking
        cannot change them. An exclusive artifact delegates to the shared Occupancy instead of
        scheduling only against its own block boundary, so two exclusive artifacts stay
        chunk- and call-order-invariant together, not just individually.
        """
        if self._rate_max <= 0.0:
            return
        if self.exclusive:
            self.ctx.occupancy.advance_exclusive(until)
            return
        while self._peek_onset(until) is not None:
            self._consume_next()

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
        """Events whose onset lies inside the rendered span (a pushed event may start later).

        Filters on the stored integer onset sample, not a re-derived `onset_s * fs`, since that
        float round trip does not always recover the original sample exactly.
        """
        return [
            t
            for t, onset in zip(self._truth, self._truth_onset, strict=True)
            if onset < self._rendered
        ]


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
