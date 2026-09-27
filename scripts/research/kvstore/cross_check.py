"""Cross check: read a status log written by the C++ sink with the Python reader.

Two independent implementations of the same format. The C++ side
(`src/llama-io.cpp`, exercised by `direct_sink_check.cpp`) writes the file, and this
script parses it with `sim.parse_blocks` and compares against the records it
regenerates from the same rules. The crc32 in C++ is the zlib crc32, so a
disagreement means one of the two is wrong.

Run: python3 scripts/research/kvstore/cross_check.py [path]
"""

import os
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sim

DEFAULT_PATH = "/home/herlanggays/.jcode/scratch/kvstore/direct/log.bin.clean"
N_RECORDS = 200
OP_ALLOC = 3


def expected_records(n):
    """The same records the C++ driver writes, built with the Python rules."""
    out = []
    for i in range(n):
        epoch = i + 1
        args = (sim.u32(epoch) + sim.u32(epoch * 3) + sim.u32(epoch % 2) +
                sim.u32(epoch // 4) + sim.u32(4))
        rec = sim.record(epoch, OP_ALLOC, args)
        out.append((epoch, OP_ALLOC, args, rec))
    return out


def raw_payload_stream(raw):
    """Concatenate the payload areas of the blocks, checking each header with zlib."""
    stream = bytearray()
    blocks = 0
    for off in range(0, len(raw) - sim.BLOCK + 1, sim.BLOCK):
        blk = raw[off:off + sim.BLOCK]
        used, c = struct.unpack_from("<II", blk, 0)
        if used == 0 or sim.BLOCK_HDR + used > sim.BLOCK:
            break
        area = blk[sim.BLOCK_HDR:sim.BLOCK_HDR + used]
        if zlib.crc32(area) & 0xFFFFFFFF != c:
            break
        stream += area
        blocks += 1
    return bytes(stream), blocks


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    if not os.path.exists(path):
        print("missing %s, run direct_sink_check first" % path)
        return 1

    raw = open(path, "rb").read()
    ok = True

    def check(name, pass_):
        nonlocal ok
        print("  %-40s %s" % (name, "ok" if pass_ else "FAIL"))
        ok = ok and pass_

    print("cross check: C++ writer, Python reader")
    print("  file %s, %d bytes" % (path, len(raw)))

    check("size is a whole number of blocks", len(raw) % sim.BLOCK == 0)
    check("block 0 is the status header", len(raw) >= sim.BLOCK)

    # the header the C++ side wrote, validated by the Python rule
    class D:
        def __init__(self, data):
            self.data = data

        def read_at(self, off, n):
            return bytes(self.data[off:off + n])

    base = sim.read_status_base(D(raw))
    check("status base parsed from the C++ header", base == sim.BLOCK)
    check("base is block aligned", base % sim.BLOCK == 0)

    got, clean = sim.parse_blocks(raw[base:])
    check("block stream parsed to the end", clean)

    want = expected_records(N_RECORDS)
    check("record count matches", len(got) == len(want))

    same = (len(got) == len(want) and
            all(got[i][0] == want[i][0] and got[i][1] == want[i][1] and got[i][2] == want[i][2]
                for i in range(len(want))))
    check("epochs, ops and args match Python", same)

    # strongest form: the payload stream must be the record bytes, in order
    stream, blocks = raw_payload_stream(raw[base:])
    joined = b"".join(w[3] for w in want)
    check("payload stream equals the records, byte for byte", stream == joined)
    print("  %-40s %d blocks, %d payload bytes" % ("framing", blocks, len(stream)))

    print()
    print("  overall: %s" % ("pass" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
