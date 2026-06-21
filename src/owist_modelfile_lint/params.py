"""Known Ollama PARAMETER keys, their types, and sane-range hints.

Sourced from the official Modelfile reference:
https://ollama.readthedocs.io/en/modelfile/

`typical_range` is not a hard validity constraint (Ollama itself won't
reject out-of-range values) — it's used to emit WARNINGs for values that
are syntactically valid but very likely a mistake (e.g. temperature 17).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParamSpec:
    name: str
    value_type: str  # "int" | "float" | "string"
    default: str
    typical_range: tuple[float, float] | None  # inclusive, None = unbounded/string
    allow_repeat: bool  # True for params like 'stop' that may appear multiple times
    description: str


PARAMETER_SPECS: dict[str, ParamSpec] = {
    "mirostat": ParamSpec(
        "mirostat", "int", "0", (0, 2), False,
        "Enable Mirostat sampling (0=disabled, 1=Mirostat, 2=Mirostat 2.0)",
    ),
    "mirostat_eta": ParamSpec(
        "mirostat_eta", "float", "0.1", (0.0, 1.0), False,
        "Learning rate for Mirostat feedback response speed",
    ),
    "mirostat_tau": ParamSpec(
        "mirostat_tau", "float", "5.0", (0.0, 10.0), False,
        "Balance between coherence and diversity",
    ),
    "num_ctx": ParamSpec(
        "num_ctx", "int", "2048", (1, 2_000_000), False,
        "Context window size in tokens",
    ),
    "repeat_last_n": ParamSpec(
        "repeat_last_n", "int", "64", (-1, 1_000_000), False,
        "How far back to look to prevent repetition (0=disabled, -1=num_ctx)",
    ),
    "repeat_penalty": ParamSpec(
        "repeat_penalty", "float", "1.1", (0.0, 3.0), False,
        "Penalty strength for repeated tokens",
    ),
    "temperature": ParamSpec(
        "temperature", "float", "0.8", (0.0, 2.0), False,
        "Sampling temperature; higher = more creative/random",
    ),
    "seed": ParamSpec(
        "seed", "int", "0", None, False,
        "Random seed for reproducible generation",
    ),
    "stop": ParamSpec(
        "stop", "string", "", None, True,
        "Stop sequence; may be specified multiple times",
    ),
    "tfs_z": ParamSpec(
        "tfs_z", "float", "1", (1.0, 5.0), False,
        "Tail free sampling factor (1.0 disables it)",
    ),
    "num_predict": ParamSpec(
        "num_predict", "int", "128", (-2, 10_000_000), False,
        "Max tokens to generate (-1=infinite, -2=fill context)",
    ),
    "top_k": ParamSpec(
        "top_k", "int", "40", (0, 1000), False,
        "Top-k sampling cutoff",
    ),
    "top_p": ParamSpec(
        "top_p", "float", "0.9", (0.0, 1.0), False,
        "Top-p (nucleus) sampling cutoff",
    ),
    "min_p": ParamSpec(
        "min_p", "float", "0.0", (0.0, 1.0), False,
        "Minimum relative probability cutoff",
    ),
}

# For "did you mean...?" suggestions on typos
_KNOWN_KEYS = list(PARAMETER_SPECS.keys())


def closest_param_name(name: str) -> str | None:
    """Return the closest known parameter name for a likely typo, or None."""
    import difflib

    matches = difflib.get_close_matches(name.lower(), _KNOWN_KEYS, n=1, cutoff=0.6)
    return matches[0] if matches else None
