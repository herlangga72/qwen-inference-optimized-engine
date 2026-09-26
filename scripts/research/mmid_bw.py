#!/usr/bin/env python3
"""Turn test-backend-ops perf output into effective bandwidth for MUL_MAT / MUL_MAT_ID.

usage: mmid_bw.py FILE...

For MUL_MAT_ID the interesting question is whether the kernel reads each selected expert once
(union bound) or re-reads the expert weights for every token (per token bound). Both are printed,
so the true behaviour can be read off the pair.
"""
import re
import sys
from collections import defaultdict

# bytes per weight for the quant types that matter here
BPW = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q4_0": 18/32, "q4_1": 20/32, "q5_0": 22/32, "q5_1": 24/32, "q8_0": 34/32,
    "q2_K": 84/256, "q3_K": 110/256, "q4_K": 144/256, "q5_K": 176/256, "q6_K": 210/256,
    "iq2_xxs": 66/256, "iq2_xs": 74/256, "iq2_s": 82/256, "iq3_xxs": 98/256, "iq3_s": 110/256,
    "iq1_s": 50/256, "iq1_m": 56/256, "iq4_nl": 18/32, "iq4_xs": 136/256, "mxfp4": 17/32,
}

CASE = re.compile(r"(MUL_MAT_ID|MUL_MAT)\(([^)]*)\):\s+\d+ runs\s+-\s+([\d.]+) us/run\s+-\s+([\d.]+) MFLOP/run")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def fields(s: str) -> dict:
    out = {}
    for part in s.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def distinct(n_mats: int, draws: int) -> float:
    return n_mats * (1.0 - (1.0 - 1.0/n_mats) ** draws)


def main() -> int:
    rows = defaultdict(dict)
    for path in sys.argv[1:]:
        for line in open(path, errors="replace"):
            m = CASE.search(ANSI.sub("", line))
            if not m:
                continue
            op, args, us, mflop = m.group(1), fields(m.group(2)), float(m.group(3)), float(m.group(4))
            ta = args.get("type_a")
            if ta not in BPW:
                continue
            n = int(args.get("n", 1))
            n_used = int(args.get("n_used", 1))
            n_mats = int(args.get("n_mats", 1))
            k = int(args["k"])
            mm = int(args["m"])
            bw = BPW[ta]
            if op == "MUL_MAT_ID":
                per_expert = mm * k * bw
                union = distinct(n_mats, n * n_used) * per_expert
                maxtok = n * n_used * per_expert
            else:
                union = maxtok = mm * k * bw
            gbs_u = union / (us * 1e-6) / 1e9
            gbs_m = maxtok / (us * 1e-6) / 1e9
            rows[(op, ta, n_mats, n_used, mm, k)][n] = (us, mflop, gbs_u, gbs_m)

    for key in sorted(rows, key=lambda t: (t[0], t[1], t[2], t[4], t[5])):
        op, ta, n_mats, n_used, mm, k = key
        print(f"{op} type_a={ta} n_mats={n_mats} n_used={n_used} m={mm} k={k} "
              f"(expert={mm*k*BPW[ta]/1024:.0f} KiB)")
        print(f"  {'n':>3} {'us/run':>10} {'MFLOP':>9} {'GB/s union':>11} {'GB/s pertok':>12}")
        for n in sorted(rows[key]):
            us, mflop, gu, gm = rows[key][n]
            print(f"  {n:>3} {us:>10.2f} {mflop:>9.2f} {gu:>11.2f} {gm:>12.2f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
