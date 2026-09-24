"""Minimal TFRecord reader, so the conversion needs protobuf but not tensorflow."""
import struct


def records(path):
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                return
            n = struct.unpack("<Q", head[:8])[0]
            data = f.read(n)
            f.read(4)
            yield data
