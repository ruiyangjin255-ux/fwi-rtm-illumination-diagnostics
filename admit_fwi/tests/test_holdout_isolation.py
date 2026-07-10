from __future__ import annotations

import pytest

from admit_fwi.diagnostics.shot_partition import assert_audit_isolation, interleaved_audit_split, interleaved_groups


def test_interleaved_groups_cover_all_shots_once() -> None:
    shots = list(range(12))
    groups = interleaved_groups(shots, 4)
    flattened = sorted(shot for group in groups.values() for shot in group)
    assert flattened == shots
    assert groups[0] == [0, 4, 8]


def test_audit_split_is_disjoint() -> None:
    split = interleaved_audit_split(list(range(12)), audit_fold=1, num_folds=4)
    assert set(split.audit_shots).isdisjoint(split.inversion_shots)
    assert split.audit_shots == [1, 5, 9]
    assert_audit_isolation(split.inversion_shots, split)


def test_audit_leak_raises() -> None:
    split = interleaved_audit_split(list(range(8)), audit_fold=0, num_folds=4)
    with pytest.raises(ValueError):
        assert_audit_isolation([0, 1, 2], split)

