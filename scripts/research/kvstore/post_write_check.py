#!/usr/bin/env python3
"""Does the write path lose whole blocks?

Writes a file the same way the store sink does, one aligned pwrite per block, then reads it back
with plain buffered reads and counts blocks that did not come back as written. Every block holds
its own pattern, so a zero block, a misplaced block and a torn block are all distinguishable.

The first run showed O_DIRECT damaging 5 of 10 files and buffered writes damaging none, on a
volume with a 4 KB native block size, written 512 bytes at a time. So this sweeps the block size
to see whether matching the device block size removes the damage.

See docs/research/23-direct-io-loses-blocks-results.md

Usage:
    post_write_check.py [iterations] [total_bytes]
"""

import mmap
import os
import struct
import sys

CASES = [
    (512, True, True),
    (512, True, False),
    (4096, True, True),
    (4096, True, False),
    (512, False, True),
    (4096, False, True),
]


def cases_from_env():
    blocks = [int(b) for b in os.environ.get("KVIO_BLOCKS", "512,4096").split(",")]
    out = []
    for b in blocks:
        out.append((b, True, True))
        out.append((b, True, False))
        if os.environ.get("KVIO_BUFFERED"):
            out.append((b, False, True))
    return out


def make_pattern(nblk, block):
    buf = bytearray()
    for i in range(nblk):
        # 8 bytes is the smallest unit the block header uses, so keep it a whole number of headers
        buf += struct.pack("<Q", i + 1) * (block // 8)
    return bytes(buf)


def write_file(path, pattern, block, direct, do_fsync):
    nblk = len(pattern) // block
    # an mmap gives page aligned memory, which O_DIRECT needs, and a memoryview slice of it keeps
    # that alignment plus the block aligned offset
    mm = mmap.mmap(-1, len(pattern))
    mm.write(pattern)
    v = memoryview(mm)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | (os.O_DIRECT if direct else 0)
    fd = os.open(path, flags, 0o644)
    try:
        for i in range(nblk):
            os.pwrite(fd, v[i * block:(i + 1) * block], i * block)
        if do_fsync:
            os.fsync(fd)
    finally:
        os.close(fd)
        v.release()
        mm.close()


def classify(got, pattern, i, block, nblk):
    blk = got[i * block:(i + 1) * block]
    exp = pattern[i * block:(i + 1) * block]
    if blk == b"\x00" * block:
        return "zeros"
    if i > 0 and blk == pattern[(i - 1) * block:i * block]:
        return "previous block"
    if i + 1 < nblk and blk == pattern[(i + 1) * block:(i + 2) * block]:
        return "next block"
    same = sum(1 for a, b in zip(blk, exp) if a == b)
    return f"{same}/{block} bytes match"


def check(path, pattern, block):
    nblk = len(pattern) // block
    with open(path, "rb") as f:
        got = f.read()
    if len(got) != len(pattern):
        return None, got, f"length {len(got)} != {len(pattern)}"
    bad = []
    for i in range(nblk):
        if got[i * block:(i + 1) * block] != pattern[i * block:(i + 1) * block]:
            bad.append(i)
    return bad, got, None


def runs(idx):
    out = []
    for i in idx:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [f"{a}-{b}" if a != b else str(a) for a, b in out]


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 28096000
    outdir = os.environ.get("KVIO_DIR", "/home/herlanggays/.jcode/scratch/kvstore/holes")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "probe.bin")

    for block, direct, do_fsync in cases_from_env():
        nblk = total // block
        pattern = make_pattern(nblk, block)
        bad_runs = 0
        total_bad = 0
        kinds = {}
        for it in range(iters):
            try:
                write_file(path, pattern, block, direct, do_fsync)
            except OSError as e:
                print(f"block={block} direct={direct} fsync={do_fsync}: write failed: {e}")
                break
            bad, got, err = check(path, pattern, block)
            if err:
                print(f"block={block} direct={direct} fsync={do_fsync}: {err}")
                break
            if bad:
                bad_runs += 1
                total_bad += len(bad)
                for i in bad[:6]:
                    k = classify(got, pattern, i, block, nblk)
                    kinds[k] = kinds.get(k, 0) + 1
                print(f"  block={block} direct={direct} fsync={do_fsync} iter={it}"
                      f" bad={len(bad)} runs={runs(bad)}")
        print(f"block={block} direct={direct} fsync={do_fsync}:"
              f" {bad_runs} of {iters} runs damaged, {total_bad} blocks")
        if kinds:
            print(f"    kinds: {kinds}")


if __name__ == "__main__":
    main()
