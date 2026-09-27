#!/usr/bin/env python3
"""Print the GGUF metadata keys that decide how big a session's KV state is.

Only reads the header and the key-value section, so it costs no model load and never touches the
tensor data, which matters when the model is tens of GiB.

Usage:
    gguf_meta.py model.gguf [key_substring ...]
"""

import struct
import sys

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
          6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
STRING = 8
ARRAY = 9
KEEP = 6  # how many array elements to show


def read_value(f, t):
    if t == STRING:
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")
    if t == ARRAY:
        at = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        vs = []
        for i in range(n):
            v = read_value(f, at)
            if i < KEEP:
                vs.append(v)
        if n > KEEP:
            return f"[{n} values of type {at}]"
        return vs
    if t not in SCALAR:
        raise ValueError(f"unknown gguf value type {t}")
    return struct.unpack(SCALAR[t], f.read(struct.calcsize(SCALAR[t])))[0]


def main():
    path = sys.argv[1]
    filters = sys.argv[2:]
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            print(f"not a gguf file: {magic!r}")
            return 1
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        print(f"gguf version {version}, {n_tensors} tensors, {n_kv} metadata keys")
        for _ in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]
            key = f.read(klen).decode("utf-8", "replace")
            vtype = struct.unpack("<I", f.read(4))[0]
            val = read_value(f, vtype)
            if not filters or any(s in key for s in filters):
                print(f"{key} = {val}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
