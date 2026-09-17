"""The artifact plug-in contract (DESIGN §5.1-5.3)."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
        self._span_owners: list[object] = []  # parallel to spans; the artifact that added each
        self._exclusive: list[EventArtifact] = []
        self._advanced_to = 0
        # Every owner (artifact) whose span has ever been pruned (see _prune_stale_spans), kept
        # forever once added - not just the most recent prune - so add_exclusive can refuse a
        # same-occupancy rebind whenever this occupancy has EVER forgotten a peer's history, not
        # only when the most recent prune happened to touch it.
        self._pruned_owners: set[object] = set()

    def add_exclusive(self, art: EventArtifact) -> None:
        """Register `art` as a participant, unless it already is one (a same-occupancy rebind).

        A genuinely new artifact cannot join after this occupancy has already advanced: the
        artifacts already on it may have committed spans and reported truth for that span, and
        a late joiner scheduling into it would be attempting to retroactively share time that is
        already settled.

        A same-occupancy rebind is allowed only if pruning has never discarded a *peer's* span.
        `EventArtifact.bind`'s rebind path resets `art`'s own scheduler to sample 0 and drops
        only `art`'s own spans (`drop_spans_of`), relying on every other participant's
        already-committed spans still being in `self.spans` so `art`'s freshly-scheduled events
        get pushed clear of them, exactly as a fresh instance sharing this occupancy would be.
        Once `_prune_stale_spans` has thrown away a peer's span, that safety net is gone: `art`
        replaying from 0 can schedule straight through history this occupancy no longer
        remembers (confirmed with a reproducer: 115 of 122 replayed events overlapped the peer).
        A rebind that only ever pruned its *own* spans is unaffected - dropping them was already
        going to happen anyway - so a single-artifact occupancy keeps rebinding freely forever.

        Deliberately does nothing else and raises before appending, so `EventArtifact.bind` can
        call this first, before it changes any of its own or the (possible) old Occupancy's
        state: a refused join must leave everything exactly as it was, not half-migrated.
        """
        if art in self._exclusive:
            stale_peers = self._pruned_owners - {art}
            if stale_peers:
                raise ValueError(
                    f"{art!r} cannot rebind to this Occupancy: pruning has already discarded "
                    f"span(s) belonging to {len(stale_peers)} other artifact(s) sharing it, so "
                    "resuming from sample 0 could overlap history this Occupancy no longer "
                    "remembers; bind a fresh Occupancy instead"
                )
            return
        if self._advanced_to > 0:
            raise ValueError(
                f"{art!r} cannot join this Occupancy: it has already advanced to sample "
                f"{self._advanced_to}, and only artifacts bound before the first render may "
                "share it"
            )
        self._exclusive.append(art)

    def remove_exclusive(self, art: EventArtifact) -> None:
        if art in self._exclusive:
            self._exclusive.remove(art)

    def drop_spans_of(self, owner: object) -> None:
        """Remove every span `owner` itself added: a same-occupancy rebind's own stale spans
        must not linger and keep pushing that artifact's freshly-scheduled events around."""
        kept = [
            (s, o) for s, o in zip(self.spans, self._span_owners, strict=True) if o is not owner
        ]
        self.spans = [s for s, _ in kept]
        self._span_owners = [o for _, o in kept]

    def next_free(self, start: int, length: int) -> int:
        moved = True
        while moved:
            moved = False
            for a, b in self.spans:
                if start < b and start + length > a:
                    start, moved = b, True
        return start

    def add(self, start: int, end: int, owner: object = None) -> None:
        self.spans.append((start, end))
        self._span_owners.append(owner)

    def advance_exclusive(self, until: int) -> None:
        """Advance every exclusive artifact sharing this occupancy up to `until`, in onset order.

        Each artifact peeks its own next candidate (drawing from its own rng only when it needs
        a fresh one); whichever peeked onset is chronologically earliest is committed first. Ties
        break on registration order, which is fixed at bind() time and so is independent of
        render-call order or chunk size.

        Always re-derives progress from each artifact's own state rather than short-circuiting
        on a shared "already advanced" flag: an artifact already caught up to (or past) `until`
        just peeks its cached candidate and finds it is not `< until`, which costs an O(1) check,
        so nothing is skipped for an artifact that legitimately still needs advancing (e.g. one
        whose own scheduler state was just reset by a same-occupancy rebind). `_prune_stale_spans`
        below does use a shared watermark, but only to decide which already-committed spans are
        safe to forget, never to decide what gets scheduled next.
        """
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
        self._advanced_to = max(self._advanced_to, until)
        self._prune_stale_spans()

    def _prune_stale_spans(self) -> None:
        """Drop spans that can never overlap a future event, so a long stream's `spans` (and
        `next_free`'s scan over it) does not grow without bound.

        Each registered exclusive artifact's own floor for its next onset is
        `max(art._earliest, round(art._next[0] * art.fs))`:

        - `_earliest` (the earliest sample *it* could next place an event at) only ever
          increases - each commit sets it to that event's end plus its gap, never earlier than
          the onset just used, which was itself never before the previous `_earliest`.
        - `_next`, once `advance_exclusive` has returned, is populated for every artifact with
          `_rate_max > 0`: the loop's last iteration peeks every registered artifact's candidate
          (via `_peek_onset`) before finding none is `< until` and stopping, so every such
          artifact has a cached, not-yet-committed candidate at that point. That candidate can
          only be committed at or after where it was peeked (`_consume_next` clamps to
          `_earliest`, never earlier), or superseded by an even later one if thinned away, so its
          onset is itself a safe floor.

        Without the `_next` term, a peer that fires rarely (or whose only nonzero rate is for a
        state this stream's timeline never visits) holds the watermark at its own long-past - or
        permanently zero - `_earliest`, and pruning for the *whole* Occupancy stalls even though
        every artifact's true next possible onset has long since moved on.

        The smallest such floor among the artifacts that can still schedule at all (`_rate_max >
        0`; a permanently silent artifact places no floor) is the watermark: a span that ends at
        or before it is behind every future onset and can be forgotten - `next_free`'s collision
        check needs `start < end`, and every future `start` is now `>= end`. This never changes
        which onset gets accepted (chunk-invariance, DESIGN §8.2, is untouched), but it does
        change what a *same-occupancy rebind* can safely assume - see `add_exclusive` - so every
        owner whose span is dropped here is recorded in `_pruned_owners` forever, not just for
        this call.
        """
        if not self._exclusive:
            return
        floors: list[int] = []
        for art in self._exclusive:
            if art._rate_max <= 0.0:
                continue
            floor = art._earliest
            if art._next is not None:
                floor = max(floor, int(round(art._next[0] * art.fs)))
            floors.append(floor)
        if not floors:
            return
        watermark = min(floors)
        dropped_owners = {
            owner
            for span, owner in zip(self.spans, self._span_owners, strict=True)
            if span[1] <= watermark
        }
        if not dropped_owners:
            return
        self._pruned_owners |= dropped_owners
        kept = [
            (span, owner)
            for span, owner in zip(self.spans, self._span_owners, strict=True)
            if span[1] > watermark
        ]
        self.spans = [span for span, _ in kept]
        self._span_owners = [owner for _, owner in kept]


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
        # So truth() is well-defined even before bind(), instead of an AttributeError.
        self._truth: list[TruthRecord] = []
        self._truth_onset: list[int] = []
        self._rendered = 0

    @property
    def name(self) -> str:
        return self.layer_name

    def bind(self, ctx: RenderContext) -> None:
        """Attach a render context and (re-)initialise all scheduler state.

        Safe to call again to rebind (e.g. reuse of a plug-in instance across conditions): any
        events already scheduled or drawn against a previous context are discarded rather than
        leaking into the new one. Rebinding to a *different* Occupancy than before removes this
        artifact from the old one's exclusive list, so the old occupancy stops consulting a
        context that no longer belongs to it. Rebinding to the *same* Occupancy as before also
        drops this artifact's own previously-committed spans from it, so its freshly-scheduled
        events are not pushed around by its own stale ones. Joining a *new* Occupancy (whether
        binding for the first time or rebinding elsewhere) is validated by
        `Occupancy.add_exclusive` *before* any of the above happens: it rejects a genuinely new
        artifact joining an occupancy that has already advanced, and does so without mutating
        anything, so a refused bind leaves this artifact exactly as it was - still bound to, and
        working against, its previous context - rather than half-migrated.
        """
        old_occupancy = self.ctx.occupancy if hasattr(self, "ctx") else None
        if self.exclusive:
            ctx.occupancy.add_exclusive(self)  # validates (and may raise) before anything else
        self.ctx = ctx
        self.fs = ctx.fs
        self.n_ch = len(ctx.channels)
        self.layer_name = ctx.layer_name
        self._rate_max = max(self.rate_by_state.values(), default=0.0)
        self._pending: list[Event] = []
        self._truth = []
        self._truth_onset = []
        self._cand_s = 0.0
        self._next: tuple[float, bool] | None = None
        self._earliest = 0
        self._rendered = 0
        if self.exclusive:
            if old_occupancy is ctx.occupancy:
                ctx.occupancy.drop_spans_of(self)
            elif old_occupancy is not None:
                old_occupancy.remove_exclusive(self)

    @abstractmethod
    def make_event(self, onset: int) -> Event:
        """Draw one event starting at sample `onset` (DESIGN §5.2).

        The returned Event's `onset` must equal the `onset` given: nothing may start before it,
        and a delayed start is made with leading zeros in the block, not by moving the Event.
        The scheduler checks this on every build, including the rebuild after an exclusive push
        to a later onset, and raises ValueError (naming the plug-in's kind) otherwise. Draw
        everything from `self.ctx.rng`, in a fixed order, so that a rebuild is reproducible.
        """

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

    def _build(self, onset: int) -> Event:
        """Call make_event and enforce its contract: the Event it returns must start exactly at
        the onset it was given. Without this, a plug-in whose Event starts before (or after) its
        onset can make the exclusive rebuild loop below spin forever: `next_free` is asked about
        a span that does not start where the scheduler thinks it does, can report it as still
        colliding after a "move", and the plug-in rebuilds an equally-offset Event at the new
        onset forever.
        """
        ev = self.make_event(onset)
        if ev.onset != onset:
            raise ValueError(
                f"{self.kind!r} plug-in's make_event(onset={onset}) returned an Event starting "
                f"at {ev.onset}; make_event must return an Event whose onset equals the onset "
                "it was given"
            )
        return ev

    def _consume_next(self) -> None:
        """Commit the currently peeked candidate: reject it, or build and place its Event.

        An exclusive push (DESIGN §5.2) rebuilds the event at its pushed onset rather than
        patching the truth record after the fact, so `make_event` always sees - and can react
        to - the event's actual final onset (state, amplitude, anything else it might compute
        from `onset`), never a pre-push candidate it was never really scheduled at. Occupancy
        collisions are resolved against whatever this artifact's own or a peer's earlier commit
        has already claimed on the shared Occupancy, in the coordinator's global onset order
        (`Occupancy.advance_exclusive`), so how many times this rebuilds (and so how many rng
        draws it costs) is itself deterministic, not dependent on chunk size or call order.
        """
        t_s, accept = self._next
        self._next = None
        onset = int(round(t_s * self.fs))
        if not accept:
            return
        onset = max(onset, self._earliest)
        ev = self._build(onset)
        if self.exclusive:
            while True:
                moved = self.ctx.occupancy.next_free(ev.onset, ev.block.shape[1])
                if moved == ev.onset:
                    break
                ev = self._build(moved)
            self.ctx.occupancy.add(ev.onset, ev.end, owner=self)
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
