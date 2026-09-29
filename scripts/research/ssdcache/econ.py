"""Economics of an SSD backed prompt cache for agentic workloads.

Every input is either a measured number from docs/research or a computed one from
scripts/research/kvstore/kv_capacity.py. Nothing here loads a model.

The question this answers: for a token that is already in the cache, what does it cost
to restore compared to what it costs to process again?

Run: python3 scripts/research/ssdcache/econ.py
"""

import argparse

MIB = 1024.0 * 1024.0
GIB = 1024.0 * MIB


def main():
    ap = argparse.ArgumentParser()
    # prompt processing, tokens/s. 120 is measured in the degraded GPU window
    # (docs/research/26-prefill-vulkan-profile.md, commit 4e94b1937). 230 and 254.8 are
    # the same model and tree measured healthy.
    ap.add_argument("--pp", type=float, default=120.0)
    # attention KV bytes per token for the layers that hold KV, K q8_0 + V planar3_0.
    # from kv_capacity.py on Qwen3.6-35B-A3B.
    ap.add_argument("--kv-bytes-per-token", type=float, default=7400.0)
    # the gated delta net state, fixed per session, not per token.
    ap.add_argument("--rs-mib", type=float, default=62.8)
    # 4 KB block framing overhead measured on the store file, 54875 blocks for 27640948
    # payload bytes.
    ap.add_argument("--framing", type=float, default=1.016)
    # measured on the current volume, scripts/research/ssdcache/block_read_probe.c:
    # one 4 KB pread per block, as llama_io_read_direct did before the window, is 185 MiB/s
    # (21 us per read). Batching to 1 MiB measures 1810 MiB/s of raw I/O.
    #
    # the store itself, after the syscall window, the slice-by-8 crc32 and the widened device
    # copy, measures 843 MiB/s reading and 432 MiB/s saving on CPU. those are the numbers to use
    # for what the cache costs, and the raw device rates are what the path could reach.
    # see docs/research/29-store-io-batching-results.md
    ap.add_argument("--read-mib-s", type=float, default=843.0, help="the store, after batching")
    ap.add_argument("--read-mib-s-blk", type=float, default=185.0, help="4 KB reads, one per block")
    ap.add_argument("--write-mib-s", type=float, default=432.0, help="the store, write plus verify")
    ap.add_argument("--write-mib-s-blk", type=float, default=273.0, help="4 KB writes, one per block")
    # device to staging buffer, 33 MiB in 2.65 ms both ways,
    # docs/research/17-device-to-disk-path-results.md
    ap.add_argument("--xfer-gib-s", type=float, default=24.4)
    # agentic loop shape
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--prefix0", type=int, default=4000)
    ap.add_argument("--growth", type=int, default=500)
    args = ap.parse_args()

    print("== per cached token ==")
    kv_b = args.kv_bytes_per_token * args.framing
    pp_us = 1e6 / args.pp
    xfer_us = kv_b / (args.xfer_gib_s * GIB) * 1e6

    print(f"  prompt processing        {pp_us:9.1f} us/token   ({args.pp:.0f} t/s)")
    for label, rmib, wmib in (("as it was, 4 KB per syscall", args.read_mib_s_blk, args.write_mib_s_blk),
                              ("the store, after batching", args.read_mib_s, args.write_mib_s)):
        read_us = kv_b / (rmib * MIB) * 1e6
        write_us = kv_b / (wmib * MIB) * 1e6
        print(f"  {label}")
        print(f"    ssd read               {read_us:9.2f} us/token   ({rmib:7.0f} MiB/s)")
        print(f"    ssd write              {write_us:9.2f} us/token   ({wmib:7.0f} MiB/s)")
        print(f"    restore, read + xfer   {read_us + xfer_us:9.2f} us/token")
        print(f"    pp / restore                    {pp_us / (read_us + xfer_us):8.0f}x")

    print()
    print("== agentic loop: prefix0 then growth per turn, transcript resent each turn ==")
    lens = [args.prefix0 + args.growth * i for i in range(args.turns)]
    total = sum(lens)

    pp_all_s = total / args.pp
    # with the cache only the newly appended tokens need processing
    pp_new_s = sum(min(args.growth, n) for n in lens) / args.pp
    print(f"  turns {args.turns}, prefix {args.prefix0} + {args.growth}/turn, {total} token-turns")

    print()
    print("  no cache")
    print(f"    pp                       {total:9d} tokens  {pp_all_s:8.1f} s")

    for name, mode in (("whole state per turn", "blob"), ("attention blocks + one rs", "block")):
        attn_bytes = sum(n * args.kv_bytes_per_token for n in lens) * args.framing
        rs_bytes = args.turns * args.rs_mib * MIB
        if mode == "block":
            write_bytes = args.growth * args.turns * args.kv_bytes_per_token * args.framing + rs_bytes
        else:
            write_bytes = attn_bytes + rs_bytes
        read_bytes = attn_bytes + rs_bytes

        print()
        print(f"  with cache, {name}")
        print(f"    pp (new tokens only)     {args.growth * args.turns:9d} tokens  {pp_new_s:8.1f} s")
        for label, rmib, wmib in (("4 KB per syscall, as it was", args.read_mib_s_blk, args.write_mib_s_blk),
                                  ("the store, after batching", args.read_mib_s, args.write_mib_s)):
            io_s = (write_bytes / (wmib * MIB) +
                    read_bytes / (rmib * MIB) +
                    (write_bytes + read_bytes) / (args.xfer_gib_s * GIB))
            total_s = pp_new_s + io_s
            print(f"    {label}")
            print(f"      ssd write              {write_bytes / GIB:9.3f} GiB  {write_bytes / (wmib * MIB):8.2f} s")
            print(f"      ssd read               {read_bytes / GIB:9.3f} GiB  {read_bytes / (rmib * MIB):8.2f} s")
            print(f"      device transfer        {2 * (write_bytes + read_bytes) / GIB:9.3f} GiB  {(write_bytes + read_bytes) / (args.xfer_gib_s * GIB):8.2f} s")
            print(f"      total                  {total_s:20.1f} s")
            print(f"      loop speedup                     {pp_all_s / total_s:8.1f}x")
            print(f"      io share of the saved time       {io_s / (pp_all_s - total_s) * 100:8.2f}%")

    print()
    print("== footprint, 20 live conversations, no trimming ==")
    for n in (4000, 16000, 64000, 262144):
        attn = n * args.kv_bytes_per_token * args.framing
        one = attn + args.rs_mib * MIB
        print(f"  {n:7d} tokens  attn {attn / MIB:8.1f} MiB + rs {args.rs_mib * MIB / MIB:6.1f} MiB"
              f" = {one / MIB:8.1f} MiB each, {20 * one / GIB:6.2f} GiB for 20")

    print()
    print("== disk traffic to hold one growing conversation, cumulative writes ==")
    for mode in ("blob", "block"):
        if mode == "blob":
            w = sum(n * args.kv_bytes_per_token for n in lens) * args.framing + args.turns * args.rs_mib * MIB
        else:
            w = args.growth * args.turns * args.kv_bytes_per_token * args.framing + args.turns * args.rs_mib * MIB
        final_attn = lens[-1] * args.kv_bytes_per_token * args.framing
        print(f"  {mode:6s} writes {w / GIB:7.2f} GiB, residency {((final_attn + args.rs_mib * MIB) / MIB):8.1f} MiB,"
              f" write amplification {w / (final_attn + args.rs_mib * MIB):6.1f}x")


if __name__ == "__main__":
    main()
