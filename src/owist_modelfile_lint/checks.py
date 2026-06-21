"""Individual lint checks for each Modelfile instruction type.

Each check function takes the full parsed instruction list (and the
Modelfile's directory, for resolving relative paths) and returns a list
of Issue objects. Keeping each check independent makes it easy to test,
disable, or extend one rule without touching the others.
"""

from __future__ import annotations

from pathlib import Path

from .gguf_sniff import sniff_gguf_header
from .issues import Issue, Severity
from .params import PARAMETER_SPECS, closest_param_name
from .parser import Instruction

VALID_MESSAGE_ROLES = {"system", "user", "assistant"}


def check_from(instructions: list[Instruction], base_dir: Path) -> list[Issue]:
    issues: list[Issue] = []
    from_instrs = [i for i in instructions if i.keyword == "FROM"]

    if not from_instrs:
        issues.append(
            Issue(Severity.ERROR, None, "no FROM instruction found — it is required", "FROM001")
        )
        return issues

    if len(from_instrs) > 1:
        for extra in from_instrs[1:]:
            issues.append(
                Issue(
                    Severity.ERROR,
                    extra.line,
                    "duplicate FROM instruction — only one is allowed",
                    "FROM002",
                )
            )

    first = instructions[0] if instructions else None
    if first is not None and first.keyword != "FROM":
        issues.append(
            Issue(
                Severity.WARNING,
                from_instrs[0].line,
                "FROM is conventionally the first instruction in a Modelfile "
                "(Ollama allows any order, but tooling and humans expect FROM first)",
                "FROM003",
            )
        )

    target = from_instrs[0]
    value = target.args.strip()

    if not value:
        issues.append(Issue(Severity.ERROR, target.line, "FROM has no value", "FROM004"))
        return issues

    looks_like_path = value.startswith((".", "/", "~")) or value.lower().endswith(
        (".gguf", ".safetensors")
    )

    if looks_like_path:
        resolved = (base_dir / value).expanduser() if not value.startswith("/") else Path(value).expanduser()
        if not resolved.exists():
            issues.append(
                Issue(
                    Severity.ERROR,
                    target.line,
                    f"FROM path '{value}' does not exist (resolved to '{resolved}')",
                    "FROM005",
                )
            )
        elif resolved.is_file() and value.lower().endswith(".gguf"):
            header = sniff_gguf_header(resolved)
            if not header.is_gguf:
                issues.append(
                    Issue(
                        Severity.ERROR,
                        target.line,
                        f"FROM path '{value}' does not look like a valid GGUF file: {header.error}",
                        "FROM006",
                    )
                )
        elif resolved.is_dir():
            # Safetensors directory build — just confirm it has weight files
            has_safetensors = any(resolved.glob("*.safetensors"))
            has_config = (resolved / "config.json").exists()
            if not has_safetensors:
                issues.append(
                    Issue(
                        Severity.WARNING,
                        target.line,
                        f"FROM directory '{value}' contains no .safetensors files",
                        "FROM007",
                    )
                )
            if not has_config:
                issues.append(
                    Issue(
                        Severity.WARNING,
                        target.line,
                        f"FROM directory '{value}' has no config.json — Ollama needs this "
                        "to detect the model architecture",
                        "FROM008",
                    )
                )
    # else: it's a library model reference like "llama3.2" or "llama3.2:8b" —
    # nothing to check locally, Ollama resolves these against its registry.

    return issues


def check_parameters(instructions: list[Instruction]) -> list[Issue]:
    issues: list[Issue] = []
    seen_non_repeatable: dict[str, Instruction] = {}

    for instr in (i for i in instructions if i.keyword == "PARAMETER"):
        parts = instr.args.split(None, 1)
        if len(parts) < 2:
            issues.append(
                Issue(
                    Severity.ERROR,
                    instr.line,
                    f"PARAMETER '{instr.args}' is missing a value "
                    "(expected: PARAMETER <name> <value>)",
                    "PARAM001",
                )
            )
            continue

        key, raw_value = parts[0], parts[1].strip().strip('"')
        spec = PARAMETER_SPECS.get(key.lower())

        if spec is None:
            suggestion = closest_param_name(key)
            hint = f" (did you mean '{suggestion}'?)" if suggestion else ""
            issues.append(
                Issue(
                    Severity.ERROR,
                    instr.line,
                    f"PARAMETER '{key}' is not a recognized Ollama parameter{hint}",
                    "PARAM002",
                )
            )
            continue

        if not spec.allow_repeat:
            if key.lower() in seen_non_repeatable:
                prev = seen_non_repeatable[key.lower()]
                issues.append(
                    Issue(
                        Severity.WARNING,
                        instr.line,
                        f"PARAMETER '{key}' was already set on line {prev.line} — "
                        "the later value will silently win",
                        "PARAM003",
                    )
                )
            seen_non_repeatable[key.lower()] = instr

        # Type check
        if spec.value_type == "int":
            try:
                num = int(raw_value)
            except ValueError:
                issues.append(
                    Issue(
                        Severity.ERROR,
                        instr.line,
                        f"PARAMETER '{key}' expects an integer, got '{raw_value}'",
                        "PARAM004",
                    )
                )
                continue
            _check_range(issues, instr, key, num, spec)
        elif spec.value_type == "float":
            try:
                num = float(raw_value)
            except ValueError:
                issues.append(
                    Issue(
                        Severity.ERROR,
                        instr.line,
                        f"PARAMETER '{key}' expects a number, got '{raw_value}'",
                        "PARAM005",
                    )
                )
                continue
            _check_range(issues, instr, key, num, spec)
        # string type ('stop') — no further validation beyond non-empty
        elif spec.value_type == "string" and not raw_value:
            issues.append(
                Issue(
                    Severity.ERROR,
                    instr.line,
                    f"PARAMETER '{key}' has an empty value",
                    "PARAM006",
                )
            )

    return issues


def _check_range(issues: list[Issue], instr: Instruction, key: str, num: float, spec) -> None:
    if spec.typical_range is None:
        return
    low, high = spec.typical_range
    if not (low <= num <= high):
        issues.append(
            Issue(
                Severity.WARNING,
                instr.line,
                f"PARAMETER '{key}' value {num} is outside the typical range "
                f"[{low}, {high}] (default: {spec.default}) — this is valid syntax "
                "but likely unintentional",
                "PARAM007",
            )
        )


def check_template(instructions: list[Instruction]) -> list[Issue]:
    issues: list[Issue] = []
    templates = [i for i in instructions if i.keyword == "TEMPLATE"]
    from_instrs = [i for i in instructions if i.keyword == "FROM"]

    if len(templates) > 1:
        for extra in templates[1:]:
            issues.append(
                Issue(
                    Severity.WARNING,
                    extra.line,
                    "multiple TEMPLATE instructions found — only the last one applies",
                    "TPL001",
                )
            )

    if not templates:
        from_value = from_instrs[0].args.strip().lower() if from_instrs else ""
        is_local_path = from_value.startswith((".", "/", "~")) or from_value.endswith(
            (".gguf", ".safetensors")
        )
        # Any local model — whatever it's named, however many of these you
        # publish — has no guarantee Ollama can auto-detect its chat format.
        # Auto-detection is keyed off metadata baked into the GGUF/config,
        # not the filename, so there is no name list that could answer this
        # reliably. We warn whenever a local model has no explicit TEMPLATE,
        # full stop, and let the human (who knows their own model) decide.
        if is_local_path:
            issues.append(
                Issue(
                    Severity.WARNING,
                    None,
                    "no TEMPLATE instruction found and FROM points to a local file/directory "
                    "— Ollama's chat-template auto-detection depends on metadata in the "
                    "model itself, not the filename, and may fail for custom or "
                    "less-common architectures, producing a model with no chat "
                    "formatting at all. Add an explicit TEMPLATE if you're not certain, "
                    "or confirm with `ollama show <model> --template` after creating it.",
                    "TPL002",
                )
            )
        return issues

    last_template = templates[-1]
    value = last_template.args
    if "{{" not in value or "}}" not in value:
        issues.append(
            Issue(
                Severity.WARNING,
                last_template.line,
                "TEMPLATE does not appear to contain any Go template variables "
                "(e.g. {{ .Prompt }}) — the model may not receive user input correctly",
                "TPL003",
            )
        )
    elif ".Prompt" not in value and ".Response" not in value:
        issues.append(
            Issue(
                Severity.WARNING,
                last_template.line,
                "TEMPLATE does not reference {{ .Prompt }} or {{ .Response }} — "
                "double check this is intentional",
                "TPL004",
            )
        )

    return issues


def check_system(instructions: list[Instruction]) -> list[Issue]:
    issues: list[Issue] = []
    systems = [i for i in instructions if i.keyword == "SYSTEM"]
    if len(systems) > 1:
        for extra in systems[1:]:
            issues.append(
                Issue(
                    Severity.WARNING,
                    extra.line,
                    "multiple SYSTEM instructions found — only the last one applies",
                    "SYS001",
                )
            )
    for s in systems:
        value = s.args.strip()
        # Strip a single layer of surrounding double-quotes, e.g. SYSTEM ""
        # (triple-quoted values are already unwrapped by the parser)
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1]
        if not value.strip():
            issues.append(
                Issue(Severity.WARNING, s.line, "SYSTEM message is empty", "SYS002")
            )
    return issues


def check_adapter(instructions: list[Instruction], base_dir: Path) -> list[Issue]:
    issues: list[Issue] = []
    adapters = [i for i in instructions if i.keyword == "ADAPTER"]
    from_instrs = [i for i in instructions if i.keyword == "FROM"]

    if adapters and not from_instrs:
        issues.append(
            Issue(
                Severity.ERROR,
                adapters[0].line,
                "ADAPTER requires a FROM instruction specifying the base model "
                "the adapter was tuned from",
                "ADP001",
            )
        )

    for instr in adapters:
        value = instr.args.strip()
        if not value:
            issues.append(Issue(Severity.ERROR, instr.line, "ADAPTER has no value", "ADP002"))
            continue
        resolved = (base_dir / value).expanduser() if not value.startswith("/") else Path(value).expanduser()
        if not resolved.exists():
            issues.append(
                Issue(
                    Severity.ERROR,
                    instr.line,
                    f"ADAPTER path '{value}' does not exist (resolved to '{resolved}')",
                    "ADP003",
                )
            )
        elif resolved.is_file() and value.lower().endswith(".gguf"):
            header = sniff_gguf_header(resolved)
            if not header.is_gguf:
                issues.append(
                    Issue(
                        Severity.ERROR,
                        instr.line,
                        f"ADAPTER path '{value}' does not look like a valid GGUF file: {header.error}",
                        "ADP004",
                    )
                )
    return issues


def check_message(instructions: list[Instruction]) -> list[Issue]:
    issues: list[Issue] = []
    for instr in (i for i in instructions if i.keyword == "MESSAGE"):
        parts = instr.args.split(None, 1)
        if len(parts) < 2:
            issues.append(
                Issue(
                    Severity.ERROR,
                    instr.line,
                    f"MESSAGE '{instr.args}' is missing content "
                    "(expected: MESSAGE <role> <content>)",
                    "MSG001",
                )
            )
            continue
        role = parts[0].lower()
        if role not in VALID_MESSAGE_ROLES:
            issues.append(
                Issue(
                    Severity.ERROR,
                    instr.line,
                    f"MESSAGE role '{parts[0]}' is invalid — must be one of "
                    f"{sorted(VALID_MESSAGE_ROLES)}",
                    "MSG002",
                )
            )
    return issues


def check_parse_errors(parse_errors: list[tuple[int, str]]) -> list[Issue]:
    return [
        Issue(Severity.ERROR, line, message, "PARSE001")
        for line, message in parse_errors
    ]
