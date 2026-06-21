"""Core data types shared across the linter."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class Issue:
    severity: Severity
    line: int | None  # None for file-level issues with no single line
    message: str
    code: str  # short stable identifier, e.g. "FROM001", useful for CI filtering

    def __str__(self) -> str:
        loc = f"line {self.line}" if self.line is not None else "general"
        return f"[{self.severity.value}] {loc}: {self.message} ({self.code})"


@dataclass
class LintResult:
    path: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def infos(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.INFO]

    @property
    def ok(self) -> bool:
        """True if there are no ERROR-level issues. Warnings/infos don't fail a lint."""
        return len(self.errors) == 0

    def __bool__(self) -> bool:
        return self.ok
