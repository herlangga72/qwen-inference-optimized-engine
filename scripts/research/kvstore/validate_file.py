#!/usr/bin/env python3
"""Independent validator for a session KV store file.

Reads with plain buffered reads and validates every framed block's crc, then diffs
files against each other. Deliberately shares no code with the C++ reader, so a bug
in that reader cannot show up here as a passing result.

Layout (see docs/research/18-direct-io-framing-results.md):
    block 0        : prefix, raw
    block i > 0    : [u32 used][u32 crc32(payload area)][u32 gen][payload][zero padding]

A block whose header reads used=0 is reported as a hole. It is not an empty frame: the reader
rejects it with EBADMSG, and treating it as valid is what made a lost write look like a framing
difference for most of docs/research/22.

gen is a per save generation. A block left over from an earlier save of the same file is a valid
frame with a matching crc, so the crc cannot see it and the generation is what does. Mixed
generations are reported as stale blocks.
"""

import struct
import sys
import zlib
from collections import Counter

BLOCK = 4096
HDR = 12


def parse(path):
    data = open(path, "rb").read()
    if len(data) % BLOCK:
        print(f"  {path}: length {len(data)} is not a block multiple")
    blocks = []
    for i in range(len(data) // BLOCK):
        raw = data[i * BLOCK:(i + 1) * BLOCK]
        if i == 0:
            blocks.append(("prefix", raw))
            continue
        used, crc = struct.unpack_from("<II", raw, 0)
        gen = struct.unpack_from("<I", raw, 8)[0]
        if used == 0:
            # the reader rejects this with EBADMSG, so it is a hole, never an empty frame
            blocks.append(("hole", used, crc, raw))
            continue
        if used > BLOCK - HDR:
            blocks.append(("bad_used", used, crc, raw))
            continue
        calc = zlib.crc32(raw[HDR:HDR + used]) & 0xFFFFFFFF
        blocks.append(("blk", used, crc, calc, raw[HDR:HDR + used], gen))
    return data, blocks


def runs_of(idx):
    """Collapse block indices into runs, so damage reads as [9761-9765, 12961-12966]."""
    out = []
    for i in idx:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [f"{a}-{b}" if a != b else str(a) for a, b in out]


def report(path):
    data, blocks = parse(path)
    bad = [i for i, b in enumerate(blocks)
           if b[0] in ("hole", "bad_used") or (b[0] == "blk" and b[2] != b[3])]
    payload = b"".join(b[4] for b in blocks if b[0] == "blk")
    # a block of zeros reads as used=0, crc=0, and crc32(b"") is 0, so it only looks valid if it is
    # treated as an empty frame. count it as damage, the way the reader does.
    hole = [i for i, b in enumerate(blocks) if b[0] in ("hole", "bad_used")]
    # the generation of the save is whatever most blocks carry, so anything else is a leftover
    gens = Counter(b[5] for b in blocks if b[0] == "blk")
    majority = gens.most_common(1)[0][0] if gens else 0
    stale = [i for i, b in enumerate(blocks) if b[0] == "blk" and b[5] != majority]
    print(f"  {path}: bytes={len(data)} blocks={len(blocks)} bad_blk={len(bad)}"
          f" first_bad={bad[0] if bad else -1} payload={len(payload)}"
          f" holes={len(hole)} gens={len(gens)} stale={len(stale)}"
          f" payload_crc={zlib.crc32(payload) & 0xFFFFFFFF:08x}")
    if stale:
        print(f"    stale block runs: {runs_of(stale[:64])} (gen {majority} expected)")
    if hole:
        print(f"    hole runs: {runs_of(hole)}")
        print(f"    first hole block offset={hole[0] * BLOCK}"
              f" payload position={sum(b[1] for b in blocks[:hole[0]] if b[0] == 'blk')}")
    if bad:
        i = bad[0]
        b = blocks[i]
        if b[0] == "blk":
            print(f"    first bad block {i}: used={b[1]} stored_crc={b[2]:08x} calc_crc={b[3]:08x}")
        else:
            print(f"    first bad block {i}: {b[0]} used={b[1]} crc={b[2]:08x}")
    return payload


def main():
    paths = sys.argv[1:]
    print("validate:")
    payloads = [report(p) for p in paths]
    print("pairwise:")
    for a in range(len(paths)):
        for b in range(a + 1, len(paths)):
            x, y = payloads[a], payloads[b]
            n = min(len(x), len(y))
            first = next((i for i in range(n) if x[i] != y[i]), -1)
            diff = sum(1 for i in range(n) if x[i] != y[i])
            print(f"  {a} vs {b}: len {len(x)} {len(y)} first={first} diff={diff}")


if __name__ == "__main__":
    main()
