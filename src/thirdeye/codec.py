from __future__ import annotations

from typing import Any

import msgpack
import zstandard as zstd


def encode_event(event: dict[str, Any]) -> bytes:
    packed = msgpack.packb(event, use_bin_type=True)
    return zstd.ZstdCompressor().compress(packed)


def decode_event(frame: bytes) -> dict[str, Any]:
    packed = zstd.ZstdDecompressor().decompress(frame)
    return msgpack.unpackb(packed, raw=False)
