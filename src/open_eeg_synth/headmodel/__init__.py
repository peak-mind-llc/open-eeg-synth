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
