"""Public API for owist-modelfile-lint.

Typical usage:

    from owist_modelfile_lint import lint

    result = lint("Modelfile")
    if not result.ok:
        for issue in result.issues:
            print(issue)
"""

from __future__ import annotations

from pathlib import Path

from . import checks
from .issues import Issue, LintResult, Severity
from .parser import parse


def lint(path: str | Path) -> LintResult:
    """Lint a Modelfile on disk.

    Args:
        path: path to the Modelfile.

    Returns:
        LintResult with `.ok`, `.issues`, `.errors`, `.warnings`, `.infos`.
    """
    path = Path(path)
    result = LintResult(path=str(path))

    if not path.exists():
        result.issues.append(
            Issue(Severity.ERROR, None, f"file not found: {path}", "FILE001")
        )
        return result

    text = path.read_text(encoding="utf-8", errors="replace")
    return _lint_text(text, base_dir=path.parent, path_label=str(path))


def lint_text(text: str, base_dir: str | Path = ".") -> LintResult:
    """Lint Modelfile content already in memory (e.g. generated programmatically).

    Args:
        text: the raw Modelfile content.
        base_dir: directory used to resolve relative FROM/ADAPTER paths.

    Returns:
        LintResult.
    """
    return _lint_text(text, base_dir=Path(base_dir), path_label="<text>")


def _lint_text(text: str, base_dir: Path, path_label: str) -> LintResult:
    result = LintResult(path=path_label)

    parsed = parse(text)
    result.issues.extend(checks.check_parse_errors(parsed.parse_errors))

    instructions = parsed.instructions
    result.issues.extend(checks.check_from(instructions, base_dir))
    result.issues.extend(checks.check_parameters(instructions))
    result.issues.extend(checks.check_template(instructions))
    result.issues.extend(checks.check_system(instructions))
    result.issues.extend(checks.check_adapter(instructions, base_dir))
    result.issues.extend(checks.check_message(instructions))

    result.issues.sort(key=lambda i: (i.line is None, i.line or 0))
    return result
