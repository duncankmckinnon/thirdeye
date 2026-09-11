from __future__ import annotations

import pytest

from thirdeye.codec import decode_event, encode_event

# -- zstd magic bytes for frame detection --
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


class TestRoundtrip:
    def test_simple_event(self):
        event = {"t": "user_message", "ts": "2026-04-30T17:00:00.000Z", "seq": 5, "data": "hello"}
        frame = encode_event(event)
        assert isinstance(frame, bytes)
        assert decode_event(frame) == event

    def test_minimal_event(self):
        event = {"t": "x", "ts": "now", "seq": 0}
        assert decode_event(encode_event(event)) == event

    def test_event_without_data_key(self):
        event = {"t": "ping", "ts": "2026-04-30T00:00:00.000Z", "seq": 42}
        assert decode_event(encode_event(event)) == event

    def test_event_with_none_data(self):
        event = {"t": "noop", "ts": "now", "seq": 0, "data": None}
        assert decode_event(encode_event(event)) == event

    def test_empty_string_data(self):
        event = {"t": "empty", "ts": "now", "seq": 0, "data": ""}
        assert decode_event(encode_event(event)) == event

    def test_large_seq(self):
        event = {"t": "x", "ts": "now", "seq": 2**31}
        assert decode_event(encode_event(event)) == event


class TestNestedData:
    def test_nested_dict(self):
        event = {
            "t": "tool_call",
            "ts": "2026-04-30T17:00:00.000Z",
            "seq": 3,
            "data": {"name": "Read", "args": {"path": "x.py", "lines": [1, 2, 3]}},
        }
        assert decode_event(encode_event(event)) == event

    def test_deeply_nested(self):
        event = {
            "t": "deep",
            "ts": "now",
            "seq": 0,
            "data": {"a": {"b": {"c": {"d": [1, 2, {"e": True}]}}}},
        }
        assert decode_event(encode_event(event)) == event

    def test_list_data(self):
        event = {"t": "multi", "ts": "now", "seq": 0, "data": [1, "two", 3.0, None, True]}
        assert decode_event(encode_event(event)) == event


class TestBinaryData:
    def test_binary_bytes(self):
        event = {"t": "blob", "ts": "now", "seq": 0, "data": b"\x00\x01\xff"}
        assert decode_event(encode_event(event)) == event

    def test_empty_bytes(self):
        event = {"t": "blob", "ts": "now", "seq": 0, "data": b""}
        assert decode_event(encode_event(event)) == event


class TestZstdFrame:
    def test_output_starts_with_zstd_magic(self):
        frame = encode_event({"t": "x", "ts": "now", "seq": 0})
        assert frame[:4] == ZSTD_MAGIC

    def test_each_encode_is_independent_frame(self):
        frame_a = encode_event({"t": "a", "ts": "now", "seq": 0})
        frame_b = encode_event({"t": "b", "ts": "now", "seq": 1})
        assert frame_a[:4] == ZSTD_MAGIC
        assert frame_b[:4] == ZSTD_MAGIC
        # Each frame is decodable independently
        assert decode_event(frame_a)["t"] == "a"
        assert decode_event(frame_b)["t"] == "b"

    def test_concatenated_frames_are_not_single_decode(self):
        frame_a = encode_event({"t": "a", "ts": "now", "seq": 0})
        frame_b = encode_event({"t": "b", "ts": "now", "seq": 1})
        # Concatenating two independent frames and decoding should only
        # return the first event (zstd decompresses the first frame).
        combined = frame_a + frame_b
        result = decode_event(combined)
        assert result["t"] == "a"


class TestEncodeReturnType:
    def test_returns_bytes(self):
        result = encode_event({"t": "x", "ts": "now", "seq": 0})
        assert isinstance(result, bytes)

    def test_compressed_smaller_than_naive(self):
        event = {"t": "verbose", "ts": "now", "seq": 0, "data": "a" * 1000}
        frame = encode_event(event)
        import msgpack

        raw = msgpack.packb(event, use_bin_type=True)
        assert len(frame) < len(raw)


class TestDecodeErrors:
    def test_decode_garbage_raises(self):
        with pytest.raises(Exception):
            decode_event(b"not a zstd frame")

    def test_decode_empty_raises(self):
        with pytest.raises(Exception):
            decode_event(b"")

    def test_decode_truncated_frame_raises(self):
        frame = encode_event({"t": "x", "ts": "now", "seq": 0})
        with pytest.raises(Exception):
            decode_event(frame[:4])

    def test_decode_valid_zstd_but_invalid_msgpack_raises(self):
        import zstandard as zstd

        compressor = zstd.ZstdCompressor()
        bad_frame = compressor.compress(b"\xc1")  # 0xc1 is never-used msgpack byte
        with pytest.raises(Exception):
            decode_event(bad_frame)


class TestUnicodeAndSpecialValues:
    def test_unicode_data(self):
        event = {"t": "msg", "ts": "now", "seq": 0, "data": "Hello, world! \u2603 \U0001f600"}
        assert decode_event(encode_event(event)) == event

    def test_boolean_values(self):
        event = {"t": "flag", "ts": "now", "seq": 0, "data": {"ok": True, "err": False}}
        assert decode_event(encode_event(event)) == event

    def test_float_values(self):
        event = {"t": "metric", "ts": "now", "seq": 0, "data": 3.14159}
        result = decode_event(encode_event(event))
        assert abs(result["data"] - 3.14159) < 1e-10

    def test_negative_int(self):
        event = {"t": "x", "ts": "now", "seq": 0, "data": -42}
        assert decode_event(encode_event(event)) == event
