"""Formula specifications for the benchmark suite."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


TriValue = bool | None


class Formula:
    def eval(self, predicates: Mapping[str, Sequence[TriValue]]) -> list[TriValue]:
        raise NotImplementedError


@dataclass(frozen=True)
class Atom(Formula):
    name: str

    def eval(self, predicates: Mapping[str, Sequence[TriValue]]) -> list[TriValue]:
        if self.name not in predicates:
            raise ValueError(f"Missing predicate values for atom '{self.name}'")
        return list(predicates[self.name])


@dataclass(frozen=True)
class Not(Formula):
    child: Formula

    def eval(self, predicates: Mapping[str, Sequence[TriValue]]) -> list[TriValue]:
        values = self.child.eval(predicates)
        out: list[TriValue] = []
        for val in values:
            if val is None:
                out.append(None)
            else:
                out.append(not val)
        return out


@dataclass(frozen=True)
class Next(Formula):
    child: Formula

    def eval(self, predicates: Mapping[str, Sequence[TriValue]]) -> list[TriValue]:
        values = self.child.eval(predicates)
        if not values:
            return []
        return values[1:] + [None]


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula

    def eval(self, predicates: Mapping[str, Sequence[TriValue]]) -> list[TriValue]:
        left_vals = self.left.eval(predicates)
        right_vals = self.right.eval(predicates)
        out: list[TriValue] = []
        for left, right in zip(left_vals, right_vals):
            if left is False or right is False:
                out.append(False)
            elif left is True and right is True:
                out.append(True)
            else:
                out.append(None)
        return out


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula

    def eval(self, predicates: Mapping[str, Sequence[TriValue]]) -> list[TriValue]:
        left_vals = self.left.eval(predicates)
        right_vals = self.right.eval(predicates)
        out: list[TriValue] = []
        for left, right in zip(left_vals, right_vals):
            if left is True or right is True:
                out.append(True)
            elif left is False and right is False:
                out.append(False)
            else:
                out.append(None)
        return out


@dataclass(frozen=True)
class Implies(Formula):
    left: Formula
    right: Formula

    def eval(self, predicates: Mapping[str, Sequence[TriValue]]) -> list[TriValue]:
        return Or(Not(self.left), self.right).eval(predicates)


@dataclass(frozen=True)
class Globally(Formula):
    child: Formula

    def eval(self, predicates: Mapping[str, Sequence[TriValue]]) -> list[TriValue]:
        values = self.child.eval(predicates)
        result: list[TriValue] = [None] * len(values)
        running_false = False
        running_unknown = False
        for idx in range(len(values) - 1, -1, -1):
            val = values[idx]
            if val is False:
                running_false = True
            elif val is None:
                running_unknown = True
            if running_false:
                result[idx] = False
            elif running_unknown:
                result[idx] = None
            else:
                result[idx] = True
        return result


@dataclass(frozen=True)
class EG(Formula):
    child: Formula

    def eval(self, predicates: Mapping[str, Sequence[TriValue]]) -> list[TriValue]:
        values = self.child.eval(predicates)
        suffix_all: list[TriValue] = [None] * len(values)
        running_false = False
        running_unknown = False
        for idx in range(len(values) - 1, -1, -1):
            val = values[idx]
            if val is False:
                running_false = True
            elif val is None:
                running_unknown = True
            if running_false:
                suffix_all[idx] = False
            elif running_unknown:
                suffix_all[idx] = None
            else:
                suffix_all[idx] = True

        result: list[TriValue] = [None] * len(values)
        seen_true = False
        seen_unknown = False
        for idx in range(len(values) - 1, -1, -1):
            val = suffix_all[idx]
            if val is True:
                seen_true = True
            elif val is None:
                seen_unknown = True
            if seen_true:
                result[idx] = True
            elif seen_unknown:
                result[idx] = None
            else:
                result[idx] = False
        return result


def build_formulas() -> dict[str, Formula]:
    e_low = Atom("e_low")
    e_spike = Atom("e_spike")
    stab = Atom("stab")
    corr_small = Atom("corr_small")
    gap_down = Atom("gap_down")
    order = Atom("order")
    adv = Atom("adv")
    adv_drop = Atom("adv_drop")

    return {
        "Phi1": EG(And(e_low, Not(e_spike))),
        "Phi2": Globally(Implies(stab, Next(stab))),
        "Phi3": Or(EG(corr_small), Globally(order)),
        "Phi4": EG(And(corr_small, And(Next(corr_small), Not(gap_down)))),
        "Phi5": EG(And(adv, Not(adv_drop))),
    }
