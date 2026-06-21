"""Command-line interface for owist-modelfile-lint.

Usage:
    modelfile-lint Modelfile
    modelfile-lint path/to/Modelfile --quiet
    modelfile-lint Modelfile --json
"""

from __future__ import annotations

import argparse
import json
import sys

from .issues import Severity
from .lint_api import lint

_COLOR = {
    Severity.ERROR: "\033[31m",
    Severity.WARNING: "\033[33m",
    Severity.INFO: "\033[36m",
}
_RESET = "\033[0m"


def _supports_color() -> bool:
    return sys.stdout.isatty()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="modelfile-lint",
        description="Validate an Ollama Modelfile before running `ollama create`.",
    )
    parser.add_argument("path", help="path to the Modelfile to lint")
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="only print errors, suppress warnings/info",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="output machine-readable JSON instead of text",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="disable colored output",
    )
    args = parser.parse_args(argv)

    result = lint(args.path)

    if args.json:
        payload = {
            "path": result.path,
            "ok": result.ok,
            "issues": [
                {
                    "severity": i.severity.value,
                    "line": i.line,
                    "message": i.message,
                    "code": i.code,
                }
                for i in result.issues
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0 if result.ok else 1

    use_color = _supports_color() and not args.no_color
    issues_to_show = result.errors if args.quiet else result.issues

    if not issues_to_show:
        if result.ok:
            print(f"✓ {result.path}: no issues found")
        return 0 if result.ok else 1

    for issue in issues_to_show:
        loc = f"line {issue.line}" if issue.line is not None else "general"
        if use_color:
            color = _COLOR.get(issue.severity, "")
            print(f"{color}[{issue.severity.value}]{_RESET} {loc}: {issue.message}  ({issue.code})")
        else:
            print(f"[{issue.severity.value}] {loc}: {issue.message}  ({issue.code})")

    print()
    n_err = len(result.errors)
    n_warn = len(result.warnings)
    summary = f"{n_err} error(s), {n_warn} warning(s)"
    if result.ok:
        print(f"✓ {result.path}: {summary} — would pass `ollama create`")
    else:
        print(f"✗ {result.path}: {summary}")

    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
