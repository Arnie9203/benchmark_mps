"""Formula definitions and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


class Formula:
    """Base class for logical formulas."""

    def size(self) -> int:
        raise NotImplementedError

    def depth(self) -> int:
        raise NotImplementedError

    def atoms(self) -> set[str]:
        raise NotImplementedError

    def has_eg(self) -> bool:
        return False

    def eval(self, predicates: Mapping[str, Sequence[bool]]) -> list[bool]:
        raise NotImplementedError


@dataclass(frozen=True)
class TrueConst(Formula):
    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1

    def eval(self, predicate: Sequence[bool]) -> list[bool]:
        return [True for _ in predicate]


@dataclass(frozen=True)
class Atom(Formula):
    name: str = "atom"

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1

    def atoms(self) -> set[str]:
        return {self.name}

    def eval(self, predicates: Mapping[str, Sequence[bool]]) -> list[bool]:
        if self.name not in predicates:
            raise ValueError(f"Missing predicate values for atom '{self.name}'")
        return list(predicates[self.name])


@dataclass(frozen=True)
class Not(Formula):
    child: Formula

    def size(self) -> int:
        return 1 + self.child.size()

    def depth(self) -> int:
        return 1 + self.child.depth()

    def has_eg(self) -> bool:
        return self.child.has_eg()

    def atoms(self) -> set[str]:
        return self.child.atoms()

    def eval(self, predicates: Mapping[str, Sequence[bool]]) -> list[bool]:
        values = self.child.eval(predicates)
        return [not value for value in values]


@dataclass(frozen=True)
class Next(Formula):
    child: Formula

    def size(self) -> int:
        return 1 + self.child.size()

    def depth(self) -> int:
        return 1 + self.child.depth()

    def has_eg(self) -> bool:
        return self.child.has_eg()

    def eval(self, predicate: Sequence[bool]) -> list[bool]:
        values = self.child.eval(predicate)
        if not values:
            return []
        return values[1:] + [False]


@dataclass(frozen=True)
class Next(Formula):
    child: Formula

    def size(self) -> int:
        return 1 + self.child.size()

    def depth(self) -> int:
        return 1 + self.child.depth()

    def has_eg(self) -> bool:
        return self.child.has_eg()

    def atoms(self) -> set[str]:
        return self.child.atoms()

    def eval(self, predicates: Mapping[str, Sequence[bool]]) -> list[bool]:
        values = self.child.eval(predicates)
        if not values:
            return []
        return values[1:] + [False]


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def has_eg(self) -> bool:
        return self.left.has_eg() or self.right.has_eg()

    def atoms(self) -> set[str]:
        return self.left.atoms() | self.right.atoms()

    def eval(self, predicates: Mapping[str, Sequence[bool]]) -> list[bool]:
        left_vals = self.left.eval(predicates)
        right_vals = self.right.eval(predicates)
        return [l and r for l, r in zip(left_vals, right_vals)]


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def has_eg(self) -> bool:
        return self.left.has_eg() or self.right.has_eg()

    def atoms(self) -> set[str]:
        return self.left.atoms() | self.right.atoms()

    def eval(self, predicates: Mapping[str, Sequence[bool]]) -> list[bool]:
        left_vals = self.left.eval(predicates)
        right_vals = self.right.eval(predicates)
        return [l or r for l, r in zip(left_vals, right_vals)]


@dataclass(frozen=True)
class Eventually(Formula):
    child: Formula

    def size(self) -> int:
        return 1 + self.child.size()

    def depth(self) -> int:
        return 1 + self.child.depth()

    def has_eg(self) -> bool:
        return self.child.has_eg()

    def eval(self, predicate: Sequence[bool]) -> list[bool]:
        values = self.child.eval(predicate)
        result = [False] * len(values)
        running_any = False
        for idx in range(len(values) - 1, -1, -1):
            running_any = running_any or values[idx]
            result[idx] = running_any
        return result


@dataclass(frozen=True)
class Globally(Formula):
    child: Formula

    def size(self) -> int:
        return 1 + self.child.size()

    def depth(self) -> int:
        return 1 + self.child.depth()

    def has_eg(self) -> bool:
        return self.child.has_eg()

    def eval(self, predicate: Sequence[bool]) -> list[bool]:
        values = self.child.eval(predicate)
        result = [False] * len(values)
        running_all = True
        for idx in range(len(values) - 1, -1, -1):
            running_all = running_all and values[idx]
            result[idx] = running_all
        return result


@dataclass(frozen=True)
class EG(Formula):
    child: Formula

    def size(self) -> int:
        return 1 + self.child.size()

    def depth(self) -> int:
        return 1 + self.child.depth()

    def has_eg(self) -> bool:
        return True

    def atoms(self) -> set[str]:
        return self.child.atoms()

    def eval(self, predicates: Mapping[str, Sequence[bool]]) -> list[bool]:
        values = self.child.eval(predicates)
        # Eventually always: exists k>=n such that for all m>=k values[m] is True.
        suffix_all = [False] * len(values)
        running_all = True
        for idx in range(len(values) - 1, -1, -1):
            running_all = running_all and values[idx]
            suffix_all[idx] = running_all
        # For each position, EG holds if any future suffix is all True.
        result = [False] * len(values)
        seen_true = False
        for idx in range(len(values) - 1, -1, -1):
            if suffix_all[idx]:
                seen_true = True
            result[idx] = seen_true
        return result
