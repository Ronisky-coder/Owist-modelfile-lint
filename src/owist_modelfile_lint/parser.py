"""Parser for Ollama Modelfile syntax.

Modelfile format (per https://ollama.readthedocs.io/en/modelfile/):
    # comment
    INSTRUCTION arguments

Instructions are case-insensitive. They can appear in any order, though
FROM is conventionally first. Values can be a bare word/line, or a
triple-quoted multi-line string using \"\"\" ... \"\"\".
"""

from __future__ import annotations

from dataclasses import dataclass, field

KNOWN_INSTRUCTIONS = {
    "FROM",
    "PARAMETER",
    "TEMPLATE",
    "SYSTEM",
    "ADAPTER",
    "LICENSE",
    "MESSAGE",
}


@dataclass
class Instruction:
    """A single parsed Modelfile instruction."""

    keyword: str  # normalized uppercase, e.g. "FROM"
    raw_keyword: str  # as written in the file, e.g. "from"
    args: str  # everything after the keyword, whitespace-trimmed
    line: int  # 1-indexed line number where the instruction starts
    end_line: int  # 1-indexed line number where it ends (multi-line strings)


@dataclass
class ParseResult:
    instructions: list[Instruction] = field(default_factory=list)
    parse_errors: list[tuple[int, str]] = field(default_factory=list)  # (line, message)


def parse(text: str) -> ParseResult:
    """Parse raw Modelfile text into a list of instructions.

    This is a line-oriented parser, not a full grammar implementation,
    but it correctly handles triple-quoted multi-line string values
    (used by TEMPLATE, SYSTEM, LICENSE) and '#' comments.
    """
    lines = text.splitlines()
    result = ParseResult()

    i = 0
    n = len(lines)
    while i < n:
        raw_line = lines[i]
        stripped = raw_line.strip()

        # Skip blank lines and full-line comments
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # Split into keyword + rest
        parts = stripped.split(None, 1)
        keyword_raw = parts[0]
        keyword = keyword_raw.upper()
        rest = parts[1] if len(parts) > 1 else ""

        if keyword not in KNOWN_INSTRUCTIONS:
            result.parse_errors.append(
                (i + 1, f"unrecognized instruction '{keyword_raw}'")
            )
            i += 1
            continue

        start_line = i + 1

        # Handle triple-quoted multi-line values: KEYWORD """...."""
        if '"""' in rest:
            first_marker = rest.index('"""')
            before = rest[:first_marker]
            after_first = rest[first_marker + 3 :]

            if '"""' in after_first:
                # Closes on the same line
                close_idx = after_first.index('"""')
                value = after_first[:close_idx].strip()
                end_line = start_line
            else:
                # Spans multiple lines — accumulate until closing """
                value_lines = [after_first]
                j = i + 1
                closed = False
                while j < n:
                    if '"""' in lines[j]:
                        close_idx = lines[j].index('"""')
                        value_lines.append(lines[j][:close_idx])
                        closed = True
                        end_line = j + 1
                        i = j
                        break
                    value_lines.append(lines[j])
                    j += 1
                if not closed:
                    result.parse_errors.append(
                        (start_line, f"unterminated triple-quoted string in {keyword}")
                    )
                    end_line = n
                    i = n - 1
                value = "\n".join(value_lines).strip()

            args = (before.strip() + " " + value).strip() if before.strip() else value
            result.instructions.append(
                Instruction(
                    keyword=keyword,
                    raw_keyword=keyword_raw,
                    args=args,
                    line=start_line,
                    end_line=end_line,
                )
            )
            i += 1
            continue

        # Single-line instruction
        result.instructions.append(
            Instruction(
                keyword=keyword,
                raw_keyword=keyword_raw,
                args=rest.strip(),
                line=start_line,
                end_line=start_line,
            )
        )
        i += 1

    return result
