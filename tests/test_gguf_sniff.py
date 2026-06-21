import struct
from pathlib import Path

from owist_modelfile_lint.gguf_sniff import sniff_gguf_header


def _write_gguf(path: Path, magic: bytes = b"GGUF", version: int = 3,
                 tensor_count: int = 5, kv_count: int = 2, extra: bytes = b"\x00" * 50) -> None:
    with open(path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<Q", tensor_count))
        f.write(struct.pack("<Q", kv_count))
        f.write(extra)


def test_valid_gguf_header(tmp_path):
    p = tmp_path / "model.gguf"
    _write_gguf(p, version=3, tensor_count=10, kv_count=4)
    info = sniff_gguf_header(p)
    assert info.is_gguf
    assert info.version == 3
    assert info.tensor_count == 10
    assert info.metadata_kv_count == 4
    assert info.error is None


def test_wrong_magic_bytes(tmp_path):
    p = tmp_path / "fake.gguf"
    _write_gguf(p, magic=b"NOTG")
    info = sniff_gguf_header(p)
    assert not info.is_gguf
    assert "magic bytes" in info.error


def test_file_too_small(tmp_path):
    p = tmp_path / "tiny.gguf"
    p.write_bytes(b"GGUF")
    info = sniff_gguf_header(p)
    assert not info.is_gguf
    assert "too small" in info.error


def test_missing_file(tmp_path):
    p = tmp_path / "does_not_exist.gguf"
    info = sniff_gguf_header(p)
    assert not info.is_gguf
    assert info.error is not None
