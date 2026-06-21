import struct
from pathlib import Path

from owist_modelfile_lint import lint, lint_text
from owist_modelfile_lint.issues import Severity


def _write_gguf(path: Path) -> None:
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 5))
        f.write(struct.pack("<Q", 2))
        f.write(b"\x00" * 50)


# ---- clean cases ----

def test_clean_library_model_passes():
    result = lint_text("""
FROM llama3.2
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM You are a helpful assistant.
""")
    assert result.ok
    assert result.errors == []


def test_clean_gguf_model_with_template_passes(tmp_path):
    gguf_path = tmp_path / "model.gguf"
    _write_gguf(gguf_path)
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text(f"""
FROM ./model.gguf
TEMPLATE \"\"\"{{{{ if .System }}}}{{{{ .System }}}}{{{{ end }}}}{{{{ .Prompt }}}}\"\"\"
PARAMETER temperature 0.7
PARAMETER stop "</s>"
PARAMETER stop "<|end|>"
SYSTEM You are Sophia.
""")
    result = lint(modelfile)
    assert result.ok, [str(i) for i in result.issues]


def test_multiple_stop_params_not_flagged_as_duplicate():
    result = lint_text("""
FROM llama3.2
PARAMETER stop "a"
PARAMETER stop "b"
PARAMETER stop "c"
""")
    dup_issues = [i for i in result.issues if i.code == "PARAM003"]
    assert dup_issues == []


# ---- FROM checks ----

def test_missing_from_is_error():
    result = lint_text("PARAMETER temperature 0.7\n")
    assert not result.ok
    assert any(i.code == "FROM001" for i in result.errors)


def test_nonexistent_gguf_path_is_error(tmp_path):
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("FROM ./does-not-exist.gguf\n")
    result = lint(modelfile)
    assert not result.ok
    assert any(i.code == "FROM005" for i in result.errors)


def test_fake_gguf_wrong_magic_is_error(tmp_path):
    fake = tmp_path / "fake.gguf"
    fake.write_bytes(b"NOTAREALGGUFFILE" + b"\x00" * 20)
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("FROM ./fake.gguf\n")
    result = lint(modelfile)
    assert not result.ok
    assert any(i.code == "FROM006" for i in result.errors)


def test_duplicate_from_is_error():
    result = lint_text("FROM llama3.2\nFROM mistral\n")
    assert not result.ok
    assert any(i.code == "FROM002" for i in result.errors)


def test_safetensors_dir_without_config_warns(tmp_path):
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"\x00" * 10)
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("FROM ./my-model\n")
    result = lint(modelfile)
    assert any(i.code == "FROM008" for i in result.warnings)


# ---- PARAMETER checks ----

def test_typo_parameter_is_error_with_suggestion():
    result = lint_text("FROM llama3.2\nPARAMETER temprature 0.7\n")
    assert not result.ok
    err = next(i for i in result.errors if i.code == "PARAM002")
    assert "temperature" in err.message


def test_out_of_range_value_is_warning_not_error():
    result = lint_text("FROM llama3.2\nPARAMETER temperature 5.7\n")
    assert result.ok  # warnings don't fail
    assert any(i.code == "PARAM007" for i in result.warnings)


def test_wrong_type_value_is_error():
    result = lint_text("FROM llama3.2\nPARAMETER num_ctx not_a_number\n")
    assert not result.ok
    assert any(i.code == "PARAM004" for i in result.errors)


def test_missing_parameter_value_is_error():
    result = lint_text("FROM llama3.2\nPARAMETER temperature\n")
    assert not result.ok
    assert any(i.code == "PARAM001" for i in result.errors)


def test_duplicate_non_repeatable_param_warns():
    result = lint_text("FROM llama3.2\nPARAMETER temperature 0.5\nPARAMETER temperature 0.8\n")
    assert any(i.code == "PARAM003" for i in result.warnings)


# ---- TEMPLATE checks ----

def test_missing_template_on_local_gguf_warns(tmp_path):
    gguf_path = tmp_path / "model.gguf"
    _write_gguf(gguf_path)
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("FROM ./model.gguf\n")
    result = lint(modelfile)
    assert any(i.code == "TPL002" for i in result.warnings)


def test_missing_template_on_known_library_model_does_not_warn():
    result = lint_text("FROM llama3.2\n")
    assert not any(i.code == "TPL002" for i in result.warnings)


def test_template_without_variables_warns():
    result = lint_text('FROM llama3.2\nTEMPLATE """just plain text no vars"""\n')
    assert any(i.code == "TPL003" for i in result.warnings)


# ---- SYSTEM checks ----

def test_empty_system_warns():
    result = lint_text('FROM llama3.2\nSYSTEM ""\n')
    assert any(i.code == "SYS002" for i in result.warnings)


# ---- ADAPTER checks ----

def test_adapter_without_from_is_error():
    result = lint_text("ADAPTER ./adapter.gguf\n")
    # also triggers FROM001, but should specifically include ADP001
    assert any(i.code == "ADP001" for i in result.errors)


def test_adapter_nonexistent_path_is_error(tmp_path):
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("FROM llama3.2\nADAPTER ./missing-adapter.gguf\n")
    result = lint(modelfile)
    assert any(i.code == "ADP003" for i in result.errors)


# ---- MESSAGE checks ----

def test_invalid_message_role_is_error():
    result = lint_text("FROM llama3.2\nMESSAGE wizard hello\n")
    assert any(i.code == "MSG002" for i in result.errors)


def test_valid_message_roles_pass():
    result = lint_text(
        "FROM llama3.2\n"
        "MESSAGE user hi\n"
        "MESSAGE assistant hello\n"
        "MESSAGE system be nice\n"
    )
    assert not any(i.code in ("MSG001", "MSG002") for i in result.issues)


# ---- file-level / misc ----

def test_missing_file_returns_error():
    result = lint("/this/path/does/not/exist/Modelfile")
    assert not result.ok
    assert any(i.code == "FILE001" for i in result.errors)


def test_result_is_truthy_when_ok():
    result = lint_text("FROM llama3.2\n")
    assert bool(result) is True


def test_result_is_falsy_when_errors():
    result = lint_text("PARAMETER temperature 0.7\n")  # no FROM
    assert bool(result) is False


def test_issues_sorted_by_line():
    result = lint_text("FROM llama3.2\nFROM mistral\nPARAMETER badkeyxyz 1\n")
    lines = [i.line for i in result.issues if i.line is not None]
    assert lines == sorted(lines)
