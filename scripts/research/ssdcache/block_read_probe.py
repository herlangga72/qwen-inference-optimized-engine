"""SSD prompt cache: what a restore actually costs, at the granularity the store uses.

The store reads and writes one 4096 byte O_DIRECT block per syscall
(src/llama-io.cpp, next_block). A pread of 1 MiB measures the device, not the path, so
this probe drives the block sized pattern and reports both.

It also confirms the residency policy: O_DIRECT must not grow the page cache.

Run: python3 scripts/research/ssdcache/block_read_probe.py [mib]
"""

import mmap
import os
import sys
import time

SCRATCH = os.environ.get("SSDCACHE_DIR", "/home/herlanggays/.jcode/scratch/ssd-cache/disk")
BLOCK = 4096
MIB = 1 << 20


def cached_mib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("Cached:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def aligned_buf(size):
    # mmap gives page aligned memory, which O_DIRECT requires for the buffer
    return mmap.mmap(-1, (size + BLOCK - 1) // BLOCK * BLOCK)


def write_blocks(path, size, buf):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    try:
        for off in range(0, size, BLOCK):
            os.pwrite(fd, memoryview(buf)[:BLOCK], off)
        os.fsync(fd)
    finally:
        os.close(fd)


def read_blocks(path, size, buf):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        n = 0
        for off in range(0, size, BLOCK):
            n += len(os.pread(fd, BLOCK, off))
        return n
    finally:
        os.close(fd)


def read_chunked(path, size, buf, chunk):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        n = 0
        for off in range(0, size, chunk):
            n += len(os.pread(fd, chunk, off))
        return n
    finally:
        os.close(fd)


def drop_cache(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def main():
    size_mib = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    os.makedirs(SCRATCH, exist_ok=True)
    path = os.path.join(SCRATCH, "blocks.bin")
    size = size_mib * MIB
    n_blocks = size // BLOCK

    buf = aligned_buf(BLOCK)
    chunk = aligned_buf(1 << 20)

    print(f"file {size_mib} MiB, block {BLOCK} B, {n_blocks} blocks, {path}")
    print(f"Cached before {cached_mib():.0f} MiB")

    t = time.time()
    write_blocks(path, size, buf)
    tw = time.time() - t
    print(f"  O_DIRECT write, 1 block per pwrite   {tw:6.3f} s  {size_mib / tw:7.1f} MiB/s"
          f"  {tw / n_blocks * 1e6:6.2f} us/block   Cached {cached_mib():.0f} MiB")

    for label, fn in (("1 block per pread", lambda: read_blocks(path, size, buf)),
                      ("1 MiB per pread", lambda: read_chunked(path, size, buf, 1 << 20))):
        drop_cache(path)
        t = time.time()
        n = fn()
        td = time.time() - t
        assert n == size, (n, size)
        print(f"  O_DIRECT read,  {label:18s} {td:6.3f} s  {size_mib / td:7.1f} MiB/s"
              f"  {td / n_blocks * 1e6:6.2f} us/block   Cached {cached_mib():.0f} MiB")

    drop_cache(path)
    t = time.time()
    read_blocks(path, size, buf)
    td = time.time() - t
    rate = size_mib / td
    pp_us = 1e6 / 120.0
    print()
    print(f"restore model at {rate:.1f} MiB/s, 1 block per pread, pp 120 t/s ({pp_us:.0f} us/token)")
    print(f"  {'tokens':>8} {'on disk':>10} {'restore':>10} {'us/token':>10} {'pp/restore':>11}")
    for tokens in (4000, 13500, 64000, 262144):
        payload = tokens * 7400 * 1.016 + 62.8 * MIB
        sec = payload / MIB / rate
        us_tok = sec / tokens * 1e6
        print(f"  {tokens:8d} {payload / MIB:9.1f}M {sec * 1000:9.1f}ms {us_tok:10.2f} {pp_us / us_tok:11.0f}")
    print()
    print(f"  pp reference: {pp_us:.0f} us/token at 120 t/s")


if __name__ == "__main__":
    main()
