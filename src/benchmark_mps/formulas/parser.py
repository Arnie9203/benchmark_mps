"""Formula parser for LCL-style syntax."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark_mps.formulas.formula import (
    And,
    Atom,
    EG,
    Eventually,
    Formula,
    Globally,
    Implies,
    Next,
    Not,
    Or,
    TrueConst,
)


@dataclass(frozen=True)
class Token:
    kind: str
    value: str


def _tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    idx = 0
    while idx < len(text):
        char = text[idx]
        if char.isspace():
            idx += 1
            continue
        if char == "-" and idx + 1 < len(text) and text[idx + 1] == ">":
            tokens.append(Token(kind="IMPLIES", value="->"))
            idx += 2
            continue
        if char in ("(", ")", "!", "&", "|"):
            tokens.append(Token(kind=char, value=char))
            idx += 1
            continue
        if char.isalpha() or char == "_":
            start = idx
            while idx < len(text) and (text[idx].isalnum() or text[idx] == "_"):
                idx += 1
            word = text[start:idx]
            tokens.append(Token(kind="WORD", value=word))
            continue
        raise ValueError(f"Unexpected character '{char}' at position {idx}")
    return tokens


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Token | None:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _consume(self, expected: str | None = None) -> Token:
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of input")
        if expected is not None and token.kind != expected:
            raise ValueError(f"Expected '{expected}', got '{token.kind}'")
        self.pos += 1
        return token

    def parse(self) -> Formula:
        expr = self._parse_implies()
        if self._peek() is not None:
            token = self._peek()
            raise ValueError(f"Unexpected token '{token.value}'")
        return expr

    def _parse_implies(self) -> Formula:
        expr = self._parse_or()
        token = self._peek()
        if token is not None and token.kind == "IMPLIES":
            self._consume("IMPLIES")
            rhs = self._parse_implies()
            return Implies(expr, rhs)
        return expr

    def _parse_or(self) -> Formula:
        expr = self._parse_and()
        while True:
            token = self._peek()
            if token is not None and token.kind == "|":
                self._consume("|")
                rhs = self._parse_and()
                expr = Or(expr, rhs)
            else:
                break
        return expr

    def _parse_and(self) -> Formula:
        expr = self._parse_unary()
        while True:
            token = self._peek()
            if token is not None and token.kind == "&":
                self._consume("&")
                rhs = self._parse_unary()
                expr = And(expr, rhs)
            else:
                break
        return expr

    def _parse_unary(self) -> Formula:
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of input")
        if token.kind == "!":
            self._consume("!")
            return Not(self._parse_unary())
        if token.kind == "WORD":
            word = token.value
            if word == "not":
                self._consume("WORD")
                return Not(self._parse_unary())
            if word == "X":
                self._consume("WORD")
                return Next(self._parse_unary())
            if word == "E":
                self._consume("WORD")
                return Eventually(self._parse_unary())
            if word == "G":
                self._consume("WORD")
                return Globally(self._parse_unary())
            if word == "EG":
                self._consume("WORD")
                return EG(self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> Formula:
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of input")
        if token.kind == "(":
            self._consume("(")
            expr = self._parse_or()
            self._consume(")")
            return expr
        if token.kind == "WORD":
            word = token.value
            self._consume("WORD")
            if word.lower() == "true":
                return TrueConst()
            return Atom(name=word)
        raise ValueError(f"Unexpected token '{token.value}'")


def parse_formula(text: str) -> Formula:
    tokens = _tokenize(text)
    parser = _Parser(tokens)
    return parser.parse()
