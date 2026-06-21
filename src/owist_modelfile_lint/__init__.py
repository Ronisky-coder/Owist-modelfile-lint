"""owist-modelfile-lint — static validation for Ollama Modelfiles.

    from owist_modelfile_lint import lint

    result = lint("Modelfile")
    if not result.ok:
        for issue in result.issues:
            print(issue)
"""

from .issues import Issue, LintResult, Severity
from .lint_api import lint, lint_text

__all__ = ["lint", "lint_text", "Issue", "LintResult", "Severity"]

__version__ = "0.1.0"
