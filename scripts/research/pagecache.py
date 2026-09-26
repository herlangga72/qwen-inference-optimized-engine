#!/usr/bin/env python3
"""Report how much of a file is resident in the page cache.

Uses mincore(2) on a private read-only mapping, so it needs no root and does not
touch the file content. Used by scripts/research/bench-stage.sh.

usage: pagecache.py FILE
"""
import ctypes
import os
import sys


def resident(path: str) -> tuple[int, int]:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]

    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    addr = libc.mmap(None, size, 1, 0x02, fd, 0)
    if addr is None or addr == 2**64 - 1:
        raise OSError(ctypes.get_errno(), "mmap failed")

    pages = size // os.sysconf("SC_PAGESIZE")
    vec = ctypes.create_string_buffer(pages)
    if libc.mincore(ctypes.c_void_p(addr), size, vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore failed")

    return sum(1 for b in vec.raw if b & 1), pages


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    path = sys.argv[1]
    got, total = resident(path)
    print(f"pagecache: {path}")
    print(f"pagecache: resident {got}/{total} pages = {100.0*got/total:.1f}% "
          f"({got*4096/2**30:.2f} GiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
