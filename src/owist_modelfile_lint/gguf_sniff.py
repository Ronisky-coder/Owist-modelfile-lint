"""Minimal GGUF file header sniffing.

We deliberately do NOT depend on llama.cpp, gguf-py, or load any tensor
data. We only need to answer: "does this look like a real GGUF file?"
and that only requires reading the first few bytes.

GGUF format (https://github.com/ggerganov/ggml/blob/master/docs/gguf.md):
    offset 0, 4 bytes : magic = b"GGUF"
    offset 4, 4 bytes : version (uint32, little-endian)
    offset 8, 8 bytes : tensor_count (uint64, little-endian)
    offset 16, 8 bytes: metadata_kv_count (uint64, little-endian)
    ... key-value metadata follows, not parsed here
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

GGUF_MAGIC = b"GGUF"


@dataclass
class GgufHeaderInfo:
    is_gguf: bool
    version: int | None = None
    tensor_count: int | None = None
    metadata_kv_count: int | None = None
    error: str | None = None


def sniff_gguf_header(path: Path) -> GgufHeaderInfo:
    """Read just enough of a file to confirm it's a valid GGUF and report
    basic header fields. Does not read tensor data or full metadata.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(24)
    except OSError as e:
        return GgufHeaderInfo(is_gguf=False, error=f"could not read file: {e}")

    if len(head) < 24:
        return GgufHeaderInfo(is_gguf=False, error="file is too small to be a valid GGUF")

    magic = head[0:4]
    if magic != GGUF_MAGIC:
        return GgufHeaderInfo(
            is_gguf=False,
            error=f"magic bytes are {magic!r}, expected b'GGUF'",
        )

    try:
        version = struct.unpack("<I", head[4:8])[0]
        tensor_count = struct.unpack("<Q", head[8:16])[0]
        metadata_kv_count = struct.unpack("<Q", head[16:24])[0]
    except struct.error as e:
        return GgufHeaderInfo(is_gguf=True, error=f"header fields malformed: {e}")

    return GgufHeaderInfo(
        is_gguf=True,
        version=version,
        tensor_count=tensor_count,
        metadata_kv_count=metadata_kv_count,
    )
