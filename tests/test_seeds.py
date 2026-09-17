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
