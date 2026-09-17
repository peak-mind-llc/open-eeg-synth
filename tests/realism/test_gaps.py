from __future__ import annotations

from tests.realism.gaps import Failure, Gap, is_excused, passing_gaps, unexpected

EC_LAP = ("eyes_closed", "laplacian", "Theta", "coh_by_dist")
EO_BETA = ("eyes_open", "bipolar", "Beta", "coh_mean")
GAPS = {EC_LAP: Gap("focal theta", bins=(0,)), EO_BETA: Gap("unbinned")}


def test_binned_gap_excuses_only_its_bins():
    assert is_excused(Failure(EC_LAP, (0,), "bin 0"), GAPS)
    assert not is_excused(Failure(EC_LAP, (3,), "bin 3"), GAPS)
    assert not is_excused(Failure(EC_LAP, (0, 3), "bins 0 and 3"), GAPS)
    assert unexpected([Failure(EC_LAP, (0, 4), "m")], GAPS) == [Failure(EC_LAP, (0, 4), "m")]


def test_binned_failure_needs_named_bins_and_the_same_line():
    unnamed = {EC_LAP: Gap("no bins named")}
    assert not is_excused(Failure(EC_LAP, (0,), "m"), unnamed)
    other_cond = ("eyes_open", *EC_LAP[1:])
    assert not is_excused(Failure(other_cond, (0,), "m"), GAPS)
    assert is_excused(Failure(EO_BETA, (), "m"), GAPS)


def test_passing_gaps_reports_listed_lines_that_did_not_fail():
    assert passing_gaps([], GAPS, "eyes_closed") == [EC_LAP]
    assert passing_gaps([Failure(EC_LAP, (0,), "m")], GAPS, "eyes_closed") == []
    # a failure in another bin does not count as the gap failing
    assert passing_gaps([Failure(EC_LAP, (2,), "m")], GAPS, "eyes_closed") == [EC_LAP]
    assert passing_gaps([Failure(EO_BETA, (), "m")], GAPS, "eyes_open") == []
    assert passing_gaps([], GAPS, "eyes_open") == [EO_BETA]
