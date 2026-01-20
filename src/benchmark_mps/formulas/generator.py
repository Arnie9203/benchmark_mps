"""Formula generators for complexity sweeps."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark_mps.formulas.formula import And, Atom, EG, Formula, Not, Or


@dataclass(frozen=True)
class FormulaSpec:
    name: str
    formula: Formula


def build_formula_suite() -> list[FormulaSpec]:
    atom = Atom()
    suite: list[FormulaSpec] = [
        FormulaSpec(name="atom", formula=atom),
        FormulaSpec(name="not_atom", formula=Not(atom)),
        FormulaSpec(name="and_atom", formula=And(atom, atom)),
        FormulaSpec(name="or_atom", formula=Or(atom, atom)),
        FormulaSpec(name="eg_atom", formula=EG(atom)),
        FormulaSpec(name="eg_and", formula=EG(And(atom, atom))),
        FormulaSpec(name="nest_eg", formula=EG(EG(atom))),
        FormulaSpec(name="deep_combo", formula=EG(And(Not(atom), Or(atom, EG(atom))))),
    ]
    return suite
