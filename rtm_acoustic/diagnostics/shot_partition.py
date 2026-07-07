from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShotSplit:
    shots: list[int]
    audit_fold: int
    num_folds: int
    inversion_shots: list[int]
    audit_shots: list[int]


def interleaved_groups(shots: list[int], num_groups: int) -> dict[int, list[int]]:
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    groups = {idx: [] for idx in range(num_groups)}
    for shot_index, shot in enumerate(shots):
        groups[shot_index % num_groups].append(int(shot))
    return groups


def interleaved_audit_split(shots: list[int], audit_fold: int, num_folds: int = 4) -> ShotSplit:
    if num_folds <= 1:
        raise ValueError("num_folds must be greater than 1")
    if audit_fold < 0 or audit_fold >= num_folds:
        raise ValueError("audit_fold must be in [0, num_folds)")
    audit = [int(shot) for idx, shot in enumerate(shots) if idx % num_folds == audit_fold]
    inversion = [int(shot) for idx, shot in enumerate(shots) if idx % num_folds != audit_fold]
    if not audit:
        raise ValueError("audit split is empty")
    if not inversion:
        raise ValueError("inversion split is empty")
    return ShotSplit(
        shots=[int(shot) for shot in shots],
        audit_fold=int(audit_fold),
        num_folds=int(num_folds),
        inversion_shots=inversion,
        audit_shots=audit,
    )


def assert_audit_isolation(used_shots: list[int], split: ShotSplit) -> None:
    leaked = sorted(set(int(shot) for shot in used_shots) & set(split.audit_shots))
    if leaked:
        raise ValueError(f"audit shots leaked into inversion diagnostics: {leaked}")

