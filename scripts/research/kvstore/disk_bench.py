"""Disk tier measurement for the session KV store.

The policy is that KV never lives in system RAM, only in VRAM or on disk. On
Linux a plain write to a file lands in the page cache, which is system RAM, so
the interesting question is what the direct path costs.

Measures buffered and O_DIRECT write and read, then converts to the park and
unpark cost for one session.

Run: python3 scripts/research/kvstore/disk_bench.py
"""

import mmap
import os
import sys
import time

SCRATCH = "/home/herlanggays/.jcode/scratch/kvstore/disk"
SIZE_MIB = int(os.environ.get("DB_MIB", "512"))

# measured in E0 for the 0.8B, see docs/research/15-e0-park-unpark-results.md
SESSION_MIB = 35.0


def mem_avail_mib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


def cached_mib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("Cached:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


def time_it(fn):
    t0 = time.time()
    fn()
    return time.time() - t0


def buffered_write(path, data):
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def buffered_read(path):
    with open(path, "rb") as f:
        while f.read(1 << 20):
            pass


def direct_write(path, size):
    if os.path.exists(path):
        os.unlink(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o644)
    try:
        buf = mmap.mmap(-1, 1 << 20)
        for i in range(0, size, 1 << 20):
            os.pwrite(fd, buf, i)
        os.fsync(fd)
    finally:
        os.close(fd)


def direct_read(path, size):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        buf = mmap.mmap(-1, 1 << 20)
        for i in range(0, size, 1 << 20):
            os.pread(fd, 1 << 20, i)
    finally:
        os.close(fd)


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    size = SIZE_MIB << 20
    path = os.path.join(SCRATCH, "kv.bin")

    print("size %d MiB, target %s" % (SIZE_MIB, path))
    print("MemAvailable %.0f MiB, Cached %.0f MiB before" % (mem_avail_mib(), cached_mib()))

    data = bytes(size)

    results = {}

    t = time_it(lambda: buffered_write(path, data))
    results["buffered write+fsync"] = t
    print("  buffered write+fsync  %7.3f s  %7.1f MiB/s   Cached now %.0f MiB"
          % (t, SIZE_MIB / t, cached_mib()))

    t = time_it(lambda: buffered_read(path))
    results["buffered read (cached)"] = t
    print("  buffered read cached  %7.3f s  %7.1f MiB/s" % (t, SIZE_MIB / t))

    try:
        t = time_it(lambda: direct_write(path, size))
        results["direct write"] = t
        print("  O_DIRECT write        %7.3f s  %7.1f MiB/s   Cached now %.0f MiB"
              % (t, SIZE_MIB / t, cached_mib()))
    except OSError as e:
        print("  O_DIRECT write unsupported: %s" % e)

    try:
        t = time_it(lambda: direct_read(path, size))
        results["direct read"] = t
        print("  O_DIRECT read         %7.3f s  %7.1f MiB/s" % (t, SIZE_MIB / t))
    except OSError as e:
        print("  O_DIRECT read unsupported: %s" % e)

    # drop our own pages and read again, to see the cold buffered cost
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except OSError as e:
        print("  fadvise DONTNEED failed: %s" % e)
    t = time_it(lambda: buffered_read(path))
    print("  buffered read cold    %7.3f s  %7.1f MiB/s" % (t, SIZE_MIB / t))

    print()
    print("one %0.0f MiB session" % SESSION_MIB)
    for k, t in sorted(results.items()):
        per = t * SESSION_MIB / SIZE_MIB * 1000.0
        print("  %-22s %8.1f ms" % (k, per))
    print("  compare: a full re-prefill of 1339 tokens on the 0.8B is about 2700 ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
