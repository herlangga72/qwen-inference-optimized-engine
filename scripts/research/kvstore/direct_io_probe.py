"""O_DIRECT mechanics for the KV store file sink.

The residency policy requires O_DIRECT, and O_DIRECT requires aligned offsets and
lengths. Layer 1 appends variable length records, and KV pages are variable sized,
so this probe answers three things the spec could not:

  1. what alignment f2fs actually demands, for size and for offset
  2. what alignment costs, per record versus batched into blocks
  3. whether a reused staging buffer keeps RSS flat on a large write

Run: python3 scripts/research/kvstore/direct_io_probe.py
"""

import mmap
import os
import struct
import sys
import time
import zlib

SCRATCH = "/home/herlanggays/.jcode/scratch/kvstore/direct"
BLOCK = 4096
HDR = 8


def rss_kib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


def aligned_buf(size):
    return mmap.mmap(-1, size)


def try_direct(path, offset, length, keep=False):
    """Write `length` bytes at `offset` with O_DIRECT. Returns None or the errno."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o644)
    try:
        buf = aligned_buf(length)
        os.pwrite(fd, buf, offset)
        os.fsync(fd)
        return None
    except OSError as e:
        return e.errno
    finally:
        os.close(fd)
        if not keep:
            pass


def probe_alignment(path):
    print("alignment probe, O_DIRECT write of N bytes at offset 0")
    results = []
    for n in (64, 128, 256, 512, 1024, 4096):
        err = try_direct(path, 0, n)
        results.append((n, err))
        print("  size %5d  %s" % (n, "ok" if err is None else "errno %d" % err))
    print("alignment probe, 4096 bytes at offset N")
    for off in (0, 64, 512, 1024, 4096):
        err = try_direct(path, off, 4096)
        print("  offset %5d  %s" % (off, "ok" if err is None else "errno %d" % err))
    return results


def layer1_records(n):
    """Records shaped like the layer 1 status log: len, crc, payload."""
    out = []
    epoch = 0
    for i in range(n):
        epoch += 1
        payload = struct.pack("<QI", epoch, i % 9) + b"\0" * 24
        out.append(struct.pack("<II", len(payload), zlib.crc32(payload) & 0xFFFFFFFF) + payload)
    return out


def write_per_record(path, records):
    """One O_DIRECT write per record, padded to a block. What a naive sink does."""
    if os.path.exists(path):
        os.unlink(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o644)
    off = 0
    written = 0
    try:
        for r in records:
            buf = aligned_buf(BLOCK)
            buf[:len(r)] = r
            buf[len(r):BLOCK] = b"\0" * (BLOCK - len(r))
            os.pwrite(fd, buf, off)
            off += BLOCK
            written += BLOCK
        os.fsync(fd)
    finally:
        os.close(fd)
    return written


def write_batched(path, records):
    """Accumulate into one reused aligned block, flush when full. What the sink should do."""
    if os.path.exists(path):
        os.unlink(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o644)
    block = aligned_buf(BLOCK)
    fill = 0
    off = 0
    written = 0
    flushes = 0
    try:
        for r in records:
            if fill + len(r) > BLOCK:
                block[fill:BLOCK] = b"\0" * (BLOCK - fill)
                os.pwrite(fd, block, off)
                off += BLOCK
                written += BLOCK
                flushes += 1
                fill = 0
            block[fill:fill + len(r)] = r
            fill += len(r)
        if fill > 0:
            block[fill:BLOCK] = b"\0" * (BLOCK - fill)
            os.pwrite(fd, block, off)
            off += BLOCK
            written += BLOCK
            flushes += 1
        os.fsync(fd)
    finally:
        os.close(fd)
    return written, flushes


def write_batched_hdr(path, records):
    """Block framed append. Each block carries its own used length and crc, so the
    padding can be told apart from a torn tail."""
    if os.path.exists(path):
        os.unlink(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o644)
    block = aligned_buf(BLOCK)
    fill = HDR
    off = 0
    written = 0
    flushes = 0

    def flush(fill, off):
        struct.pack_into("<II", block, 0, fill - HDR,
                         zlib.crc32(bytes(block[HDR:fill])) & 0xFFFFFFFF)
        block[fill:BLOCK] = b"\0" * (BLOCK - fill)
        os.pwrite(fd, block, off)
        return BLOCK

    try:
        for r in records:
            if fill + len(r) > BLOCK:
                written += flush(fill, off)
                off += BLOCK
                flushes += 1
                fill = HDR
            block[fill:fill + len(r)] = r
            fill += len(r)
        if fill > HDR:
            written += flush(fill, off)
            flushes += 1
        os.fsync(fd)
    finally:
        os.close(fd)
    return written, flushes


def parse_blocks(buf):
    """Reader for the framed layout: walk blocks, validate the header, then the records."""
    out = []
    for boff in range(0, len(buf), BLOCK):
        blk = buf[boff:boff + BLOCK]
        if len(blk) < BLOCK:
            break
        used, crc = struct.unpack_from("<II", blk, 0)
        if used == 0 or HDR + used > BLOCK:
            break
        if zlib.crc32(blk[HDR:HDR + used]) & 0xFFFFFFFF != crc:
            break
        o = HDR
        end = HDR + used
        while o + 8 <= end:
            ln = struct.unpack_from("<I", blk, o)[0]
            c = struct.unpack_from("<I", blk, o + 4)[0]
            o += 8
            if ln == 0 or o + ln > end:
                return out
            payload = blk[o:o + ln]
            if zlib.crc32(payload) & 0xFFFFFFFF != c:
                return out
            o += ln
            out.append(payload)
    return out


def read_back(path, length):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    out = bytearray()
    try:
        buf = aligned_buf(BLOCK)
        for off in range(0, length, BLOCK):
            # preadv writes into the page aligned mmap, which O_DIRECT requires
            os.preadv(fd, [buf], off)
            out += bytes(buf)
    finally:
        os.close(fd)
    return bytes(out)


def parse_records(buf):
    out = []
    o = 0
    while o + 8 <= len(buf):
        ln, o = struct.unpack_from("<I", buf, o)[0], o + 4
        c, o = struct.unpack_from("<I", buf, o)[0], o + 4
        if ln == 0 or ln > len(buf) - o:
            break
        payload = buf[o:o + ln]
        if zlib.crc32(payload) & 0xFFFFFFFF != c:
            break
        o += ln
        out.append(payload)
    return out


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    p = os.path.join(SCRATCH, "probe.bin")

    probe_alignment(p)

    n = 2000
    records = layer1_records(n)
    logical = sum(len(r) for r in records)

    print()
    print("%d records, %d logical bytes, %.1f bytes per record"
          % (n, logical, logical / n))

    t0 = time.time()
    per_rec = write_per_record(p, records)
    t_per = time.time() - t0
    print("  per record, one O_DIRECT write each: %8d bytes on disk  %.2fx  %.3f s"
          % (per_rec, per_rec / logical, t_per))

    t0 = time.time()
    batched, flushes = write_batched(p, records)
    t_batch = time.time() - t0
    print("  batched into blocks, %3d flushes:      %8d bytes on disk  %.2fx  %.3f s"
          % (flushes, batched, batched / logical, t_batch))

    raw = read_back(p, batched)
    got_naive = parse_records(raw)
    print("  naive scan over the batched file: %d of %d records, byte exact: %s"
          % (len(got_naive), n, len(got_naive) == n))

    framed_path = p + ".framed"
    framed, fflushes = write_batched_hdr(framed_path, records)
    got = parse_blocks(read_back(framed_path, framed))
    ok = len(got) == n and all(got[i] == records[i][8:] for i in range(n))
    print("  block framed, %3d flushes:            %8d bytes on disk  %.2fx  %.3f s"
          % (fflushes, framed, framed / logical, t_batch))
    print("  block framed readback: %d of %d records, byte exact: %s"
          % (len(got), n, ok))

    # a torn tail: zero the last block, which the header check must reject
    torn = len(records) - 7
    tp = framed_path + ".torn"
    partial, _ = write_batched_hdr(tp, records[:torn])
    block = aligned_buf(BLOCK)
    fd = os.open(tp, os.O_WRONLY | os.O_DIRECT)
    os.pwrite(fd, block, partial - BLOCK)
    os.close(fd)
    got_t = parse_blocks(read_back(tp, partial))
    print("  torn last block, framed: %d of %d records recovered"
          % (len(got_t), torn))

    # RSS across a large write with one reused buffer
    before = rss_kib()
    big = os.path.join(SCRATCH, "big.bin")
    fd = os.open(big, os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o644)
    buf = aligned_buf(1 << 20)
    total = 512 << 20
    t0 = time.time()
    for off in range(0, total, 1 << 20):
        os.pwrite(fd, buf, off)
    os.fsync(fd)
    os.close(fd)
    dt = time.time() - t0
    after = rss_kib()
    print()
    print("  %d MiB through one reused 1 MiB buffer: %.1f MiB/s, RSS %d -> %d KiB (delta %d)"
          % (total >> 20, (total >> 20) / dt, before, after, after - before))

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
