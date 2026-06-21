# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-06-19

### Added
- Initial release.
- `lint()` / `lint_text()` Python API.
- `modelfile-lint` CLI with `--quiet`, `--json`, `--no-color` flags.
- Checks for `FROM` (existence, GGUF magic bytes, safetensors directory shape, duplicates).
- Checks for `PARAMETER` (known keys with typo suggestions, type validation, typical-range warnings, non-repeatable duplicates).
- Checks for `TEMPLATE` (missing on unrecognized local models, missing template variables).
- Checks for `SYSTEM` (empty, duplicates).
- Checks for `ADAPTER` (requires `FROM`, path existence, GGUF validity).
- Checks for `MESSAGE` (valid roles, content present).
- Parser-level checks for unrecognized instructions and unterminated triple-quoted strings.
