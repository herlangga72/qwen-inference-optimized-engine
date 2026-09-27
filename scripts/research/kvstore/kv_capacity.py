#!/usr/bin/env python3
"""Size a parked KV store from a model's own GGUF metadata.

Reads only the header and key-value section, so it needs no model load.

The capacity assumption this computes is: ten sessions, each holding the full context size of the
loaded model, with K cached as q8_0 and V cached as planar3_0.

Usage:
    kv_capacity.py model.gguf [context_tokens] [n_sessions]
"""

import re
import struct
import sys

# bytes per cached element. q8_0 is 34 bytes per 32 elements, planar3_0 is 98 bytes per 256
# (f16 norm plus 96 bytes of 3 bit coordinates), which is why they are not whole numbers.
BYTES_PER_ELEMENT = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "planar3_0": 98 / 256,
}


def bytes_per_element(t):
    return BYTES_PER_ELEMENT[t]


SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
          6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_value(f, t):
    if t == 8:
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")
    if t == 9:
        at = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        vals = [read_value(f, at) for _ in range(n)]
        return vals if n <= 6 else f"[{n} values]"
    return struct.unpack(SCALAR[t], f.read(struct.calcsize(SCALAR[t])))[0]


def read_meta(path):
    meta = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a gguf file")
        struct.unpack("<I", f.read(4))[0]
        struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]
            key = f.read(klen).decode()
            t = struct.unpack("<I", f.read(4))[0]
            meta[key] = read_value(f, t)
    return meta


def gib(n):
    return n / (1024 ** 3)


def mib(n):
    return n / (1024 ** 2)


def main():
    path = sys.argv[1]
    n_sessions = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    meta = read_meta(path)
    arch = meta["general.architecture"]

    def get(suffix, default=None):
        return meta.get(f"{arch}.{suffix}", default)

    n_layer = get("block_count")
    n_head_kv = get("attention.head_count_kv")
    k_len = get("attention.key_length")
    v_len = get("attention.value_length")
    interval = get("full_attention_interval", 1)
    ctx_native = get("context_length")
    ctx = int(sys.argv[2]) if len(sys.argv) > 2 else ctx_native

    n_full = n_layer // interval
    n_state = n_layer - n_full

    print(f"model {meta.get('general.name')} arch {arch}")
    print(f"  layers {n_layer}, full attention {n_full} (1 every {interval}), state layers {n_state}")
    print(f"  kv heads {n_head_kv}, key_length {k_len}, value_length {v_len}")
    print(f"  context: model {ctx_native}, using {ctx}")

    k_bytes_per_tok = n_full * n_head_kv * k_len * bytes_per_element("q8_0")
    v_bytes_per_tok = n_full * n_head_kv * v_len * bytes_per_element("planar3_0")
    kv_bytes_per_tok = k_bytes_per_tok + v_bytes_per_tok
    kv_f16_per_tok = n_full * n_head_kv * (k_len + v_len) * bytes_per_element("f16")

    print(f"  K q8_0      {k_bytes_per_tok:8.0f} B/token"
          f"  ({n_full} layers x {n_head_kv} heads x {k_len} dims x {bytes_per_element('q8_0'):.4f} B)")
    print(f"  V planar3_0 {v_bytes_per_tok:8.0f} B/token"
          f"  ({n_full} layers x {n_head_kv} heads x {v_len} dims x {bytes_per_element('planar3_0'):.4f} B)")
    print(f"  total       {kv_bytes_per_tok:8.0f} B/token"
          f"  (f16 kv would be {kv_f16_per_tok:.0f}, ratio {kv_f16_per_tok / kv_bytes_per_tok:.2f}x)")

    # the fixed per session term: the recurrent state of the non attention layers. measured for this
    # model at 62.8 MiB in docs/research/17-device-to-disk-path-results.md
    base = 62.8 * 1024 * 1024

    per_session = base + kv_bytes_per_tok * ctx
    print(f"  per session at {ctx} tokens: {mib(per_session):9.1f} MiB"
          f" = {gib(per_session):.3f} GiB  (base {mib(base):.1f} MiB + kv {mib(kv_bytes_per_tok * ctx):.1f} MiB)")
    print(f"  {n_sessions} sessions: {gib(per_session * n_sessions):.2f} GiB"
          f"  (f16 kv would be {gib((base + kv_f16_per_tok * ctx) * n_sessions):.2f} GiB)")


if __name__ == "__main__":
    main()
