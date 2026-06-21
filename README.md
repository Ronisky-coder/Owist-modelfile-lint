# owist-modelfile-lint

Static validation for [Ollama](https://ollama.com) Modelfiles. Catch broken
`FROM` paths, invalid `PARAMETER` values, and missing `TEMPLATE`s **before**
you run `ollama create` and get a cryptic Go error three minutes into a
model build.

Built by [Openwist AI](https://openwist.kesug.com), maker of the
[LimitAI](https://huggingface.co/Coder-rony) open model family.

## The problem

`ollama create` parses your Modelfile on the Go side and fails late:

```
Error: invalid file magic
```

That's it. No line number, no hint about which instruction caused it, and
you find out only after Ollama has already started reading your (possibly
multi-gigabyte) model file. A typo'd `PARAMETER` key gets silently ignored
instead of erroring. A missing `TEMPLATE` on an unrecognized base model
ships a model with no chat formatting at all, and you don't notice until
it responds with garbage.

`owist-modelfile-lint` reads your Modelfile *before* any of that, the same
way `ruff` or `eslint` check source code before you run it.

## Install

```bash
pip install owist-modelfile-lint
```

## Usage

### CLI

```bash
modelfile-lint ./Modelfile
```

```
[ERROR]   line 1: FROM path './sophia-q4.gguf' does not exist  (FROM005)
[ERROR]   line 5: PARAMETER 'temprature' is not a recognized Ollama parameter (did you mean 'temperature'?)  (PARAM002)
[WARNING] line 7: PARAMETER 'temperature' value 5.7 is outside the typical range [0.0, 2.0] (default: 0.8) — this is valid syntax but likely unintentional  (PARAM007)
[WARNING] general: no TEMPLATE instruction found and FROM points to a local file/directory — Ollama's chat-template auto-detection may fail for unrecognized architectures, producing a model with no chat formatting at all. Consider adding an explicit TEMPLATE.  (TPL002)

✗ ./Modelfile: 2 error(s), 2 warning(s)
```

Exit code is `0` when there are no errors (warnings don't fail the check),
`1` otherwise — so it's a drop-in CI or pre-commit gate:

```bash
modelfile-lint ./Modelfile || exit 1
```

Other flags:

```bash
modelfile-lint ./Modelfile --quiet      # errors only, suppress warnings
modelfile-lint ./Modelfile --json       # machine-readable output
modelfile-lint ./Modelfile --no-color
```

### Python API

```python
from owist_modelfile_lint import lint

result = lint("Modelfile")

if not result.ok:
    for issue in result.issues:
        print(issue)
    raise SystemExit(1)
```

```python
from owist_modelfile_lint import lint_text

# lint content that doesn't exist on disk yet, e.g. generated programmatically
result = lint_text("""
FROM llama3.2
PARAMETER temperature 0.7
SYSTEM You are a helpful assistant.
""")
print(result.ok)  # True
```

`LintResult` gives you `.ok`, `.issues`, `.errors`, `.warnings`, `.infos`,
and is truthy/falsy based on `.ok` so `if result:` works too.

## What it checks

| Instruction | Checks |
|---|---|
| `FROM` | required and present exactly once; conventionally first; if it's a local path, the path exists; if it's a `.gguf` file, the magic bytes actually say `GGUF`; if it's a directory, it has `.safetensors` weights and a `config.json` |
| `PARAMETER` | key is a real Ollama parameter (with "did you mean...?" suggestions for typos); value is the right type (int/float/string); value is in the typical sane range; duplicate non-repeatable parameters |
| `TEMPLATE` | present when the base model isn't one Ollama can auto-detect; contains actual Go template variables (`{{ .Prompt }}`, `{{ .Response }}`) when present |
| `SYSTEM` | not empty; warns on duplicates |
| `ADAPTER` | requires a `FROM`; path exists; GGUF adapters are validated the same way as `FROM` |
| `MESSAGE` | role is one of `system` / `user` / `assistant`; has content |
| structure | unrecognized instructions, unterminated `"""` strings |

This is **static** analysis — it never loads model weights, runs Ollama, or
needs a GPU. A GGUF check reads 24 bytes of the file header, nothing more.

## What it deliberately does not do

- It does not validate that your `TEMPLATE` Go-template syntax is
  *semantically* correct for the model's actual chat format — that
  requires knowing what the base model expects, which is out of scope
  for a static linter.
- It does not check model *quality* — see
  [`tinyeval`](https://github.com/Ronisky-coder) (planned) for that.
- It does not talk to the Ollama daemon or registry. Library model
  references like `FROM llama3.2` are accepted as-is without checking
  whether that tag exists.

## Why "owist"

Short for [Openwist AI](https://openwist.kesug.com) — we build the
[LimitAI](https://huggingface.co/Coder-rony) open model family (Anan,
Sophia) and got tired of debugging our own Modelfiles by trial and error.

## License

MIT
