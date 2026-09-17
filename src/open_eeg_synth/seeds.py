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
