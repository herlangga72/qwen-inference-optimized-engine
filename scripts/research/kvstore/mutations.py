"""Mutation checks for the layer 1 harness.

Each mutation removes or inverts one rule, then re-runs the crash matrix. A rule
that does not change the outcome was not being tested, which is the point: the
result table in docs/research/14-session-kv-store-format-results.md is only
evidence if the rules are individually falsifiable.

Run: python3 scripts/research/kvstore/mutations.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sim
import scenarios
from sim import Budget, Crash, Disk


def execute_pages_last(recs, limit, tear, tear_len=None):
    """Log records are durable; page writes are deferred to the end of the run."""
    budget = Budget(limit, tear, tear_len)
    pages = Disk(budget)
    pages.data = bytearray(sim.pages_header())
    status = Disk(budget)
    status.data = bytearray(sim.status_header(sim.STATUS_HEADER))

    pending = []
    for i, r in enumerate(recs):
        try:
            if r.kind == "page":
                pending.append((r.addr, r.data))
            else:
                status.write_at(r.addr, r.data)
        except Crash:
            return pages, status, i
    for addr, data in pending:
        try:
            pages.write_at(addr, data)
        except Crash:
            return pages, status, len(recs)
    return pages, status, len(recs)


def parse_blocks_naive(buf):
    """Scan records across the whole buffer, ignoring the block framing. This is the
    layer 1 rule as it stood before the atomic unit became a block."""
    out = []
    o = 0
    while o + 8 <= len(buf):
        ln, o = sim.rd_u32(buf, o)
        c, o = sim.rd_u32(buf, o)
        if ln == 0 or ln > len(buf) - o:
            break
        payload = buf[o:o + ln]
        if sim.crc32(payload) != c:
            break
        o += ln
        epoch, _ = sim.rd_u64(payload, 0)
        op, _ = sim.rd_u32(payload, 8)
        out.append((epoch, op, payload[12:]))
    return out, True


def read_slot_no_crc(pages, slot):
    """Return the generation without checking the crc or the slot id."""
    raw = pages.read_at(sim.SLOT_BASE + slot * sim.SLOT_SIZE, sim.SLOT_SIZE)
    if len(raw) < sim.SLOT_SIZE:
        return None
    return sim.struct.unpack_from("<I", raw, 4)[0]


def recover_no_epoch_guard(pages, status):
    """Same recovery, but stray records outside an epoch are applied instead of
    refused, so the epoch bracket stops being a visibility rule."""
    sim.check_pages_head(pages)
    base = sim.read_status_base(status)
    recs, _clean = sim.parse_blocks(status.read_at(base, len(status.data) - base))

    state = sim.empty_state()
    ops = []
    current = None
    committed = 0
    for epoch, op, args in recs:
        if op == sim.OP_BEGIN:
            current = epoch
            ops = []
        elif op == sim.OP_COMMIT:
            if current is not None and epoch == current:
                for (e, o, a) in ops:
                    sim.apply_op(state, e, o, a)
                committed = epoch
                ops = []
                current = None
        else:
            ops.append((epoch, op, args))
    return state, committed, None


def quiet_matrix(plan, label, expect_failures):
    saved = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        _ok, failures, _ = scenarios.run_matrix(plan, label, expect_failures=expect_failures)
    finally:
        sys.stdout.close()
        sys.stdout = saved
    return failures


def main():
    results = []

    print("baseline")
    _ok, base_fail, _ = scenarios.run_matrix(scenarios.build_safe(), "safe")
    _ok, base_unsafe, _ = scenarios.run_matrix(scenarios.build_unsafe_reuse(),
                                               "unsafe", expect_failures=True)
    print("    safe failures=%d unsafe detections=%d" % (len(base_fail), len(base_unsafe)))
    results.append(len(base_fail) == 0 and len(base_unsafe) > 0)

    print("mutation: page writes deferred to the end")
    original = scenarios.execute
    scenarios.execute = execute_pages_last
    fails = quiet_matrix(scenarios.build_safe(), "x", expect_failures=True)
    scenarios.execute = original
    print("    safe failures=%d" % len(fails))
    results.append(len(fails) > 0)

    print("mutation: record scan that ignores the block framing")
    original = sim.parse_blocks
    sim.parse_blocks = parse_blocks_naive
    fails = quiet_matrix(scenarios.build_safe(), "x", expect_failures=True)
    sim.parse_blocks = original
    print("    safe failures=%d" % len(fails))
    results.append(len(fails) > 0)

    print("mutation: page crc check removed")
    original = sim.read_slot
    sim.read_slot = read_slot_no_crc
    fails = quiet_matrix(scenarios.build_safe(), "x", expect_failures=False)
    unsafe = quiet_matrix(scenarios.build_unsafe_reuse(), "x", expect_failures=True)
    sim.read_slot = original
    print("    safe failures=%d unsafe detections=%d" % (len(fails), len(unsafe)))
    print("    unchanged, so the generation comparison is a second independent catch")
    results.append(len(fails) == 0 and len(unsafe) > 0)

    print("mutation: stray records applied outside an epoch")
    original = sim.recover
    sim.recover = recover_no_epoch_guard
    fails = quiet_matrix(scenarios.build_safe(), "x", expect_failures=True)
    sim.recover = original
    print("    safe failures=%d" % len(fails))
    results.append(len(fails) > 0)

    print()
    print("overall: %s" % ("pass" if all(results) else "FAIL"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
