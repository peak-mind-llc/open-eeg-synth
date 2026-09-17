from __future__ import annotations

import open_eeg_synth


def test_dir_lists_the_public_api():
    """``dir(open_eeg_synth)`` must offer every name in ``__all__``, whether or not it has been
    lazily loaded yet (``__getattr__``, DESIGN §10)."""
    assert set(open_eeg_synth.__all__) <= set(dir(open_eeg_synth))


def test_dir_excludes_implementation_helpers():
    """``__dir__`` used to fall back to ``set(globals())``, which also surfaced this module's own
    plumbing (the lazy-loading table, its imports) as if they were part of the public API."""
    noise = {"_LAZY", "importlib", "TYPE_CHECKING", "Any"}
    assert noise.isdisjoint(dir(open_eeg_synth))


def test_dir_includes_an_already_loaded_submodule():
    """A submodule that has actually been imported (here, ``open_eeg_synth.classic``, loaded via
    the lazy table's own ``importlib.import_module``) is real state on the package and belongs in
    ``dir()``, unlike a bare helper name."""
    import open_eeg_synth.classic  # noqa: F401  (the import is the point; sets the attribute)

    assert "classic" in dir(open_eeg_synth)


def test_dir_returns_a_sorted_list_with_no_duplicates():
    out = dir(open_eeg_synth)
    assert isinstance(out, list)
    assert out == sorted(out)
    assert len(out) == len(set(out))
