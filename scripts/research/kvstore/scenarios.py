"""Session KV store harness: crash matrix over the layer 1 format.

Three plan shapes:

  safe        a checkpoint before any slot reuse
  unsafe      reuse of a slot the newest durable checkpoint still references,
              which must be detected
  stale_rs    a bind without a matching recurrent write, which must be detected

The atomic unit is a block. The ledger therefore has one entry per block flush,
not one per record, which is what the file can actually do under O_DIRECT.

The expected states come from a second, independent application of the same
logical operations, so the log encoding and the model can disagree.

Run: python3 scripts/research/kvstore/scenarios.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sim
from sim import Budget, Crash, Disk, RecoveryError


class Rec:
    __slots__ = ("epoch", "kind", "addr", "data")

    def __init__(self, epoch, kind, addr, data):
        self.epoch = epoch
        self.kind = kind
        self.addr = addr
        self.data = data


def blob(seed):
    return bytes([seed & 0xFF]) * sim.PAYLOAD


def ref_apply(state, epoch, op):
    """Independent model of the op semantics, on logical values."""
    slots, seqs, rs, rs_epoch, seq_epoch = state
    k = op[0]
    if k == "alloc":
        _, slot, gen, layer, block = op
        slots[slot] = (gen, layer, block)
    elif k == "free":
        _, slot, gen = op
        cur = slots.get(slot)
        if cur is not None and cur[0] == gen:
            del slots[slot]
    elif k == "move":
        _, src, src_gen, dst, dst_gen = op
        cur = slots.get(src)
        if cur is not None and cur[0] == src_gen:
            slots[dst] = (dst_gen, cur[1], cur[2])
            del slots[src]
            for seq, cells in seqs.items():
                seqs[seq] = [(p,
                              dst if (s, g) == (src, src_gen) else s,
                              dst_gen if (s, g) == (src, src_gen) else g)
                             for (p, s, g) in cells]
    elif k == "seq_rm":
        _, seq, p0, p1 = op
        seqs[seq] = [c for c in seqs.get(seq, []) if not (p0 <= c[0] < p1)]
        seq_epoch[seq] = epoch
    elif k == "seq_bind":
        _, seq, slot, gen, pos = op
        lst = seqs.setdefault(seq, [])
        lst.append((pos, slot, gen))
        lst.sort()
        seq_epoch[seq] = epoch
    elif k == "rs_put":
        _, seq, content, ep = op
        rs[seq] = content
        rs_epoch[seq] = ep


class Plan:
    def __init__(self):
        self.recs = []
        self.ref = sim.empty_state()
        self.epoch = 0
        self.log = sim.BlockLog(sim.STATUS_HEADER)
        self.root_bump = sim.ROOT_BASE
        self.commit_block = {}
        self.block_rec = {}
        self.begin_block_off = {}
        self.root_offsets = []
        self.epoch_ops = {}

    # ledger helpers

    def _emit(self, kind, addr, data):
        self.recs.append(Rec(self.epoch, kind, addr, data))
        return len(self.recs) - 1

    def _drain_closed(self):
        for (off, blk) in self.log.take_closed():
            ri = self._emit("block", off, blk)
            self.block_rec[(off - sim.STATUS_HEADER) // sim.BLOCK] = ri

    def _log(self, data):
        bi, _off = self.log.append(data)
        self._drain_closed()
        return bi

    def flush_block(self):
        self.log.close()
        self._drain_closed()

    def finish(self):
        """Flush the final partial block. Ordered after every page write in the plan."""
        self.flush_block()

    def _ref(self, logical):
        self.epoch_ops.setdefault(self.epoch, []).append(logical)
        ref_apply(self.ref, self.epoch, logical)

    # epochs

    def begin(self, fresh=False):
        if fresh:
            # start the epoch in a fresh block, so a base offset can point at it
            self.flush_block()
        self.epoch += 1
        self.begin_block_off[self.epoch] = self.log.cur_off
        self._log(sim.record(self.epoch, sim.OP_BEGIN))

    def commit(self):
        bi = self._log(sim.record(self.epoch, sim.OP_COMMIT))
        self.commit_block[self.epoch] = bi

    # pages and ops

    def write_slot(self, slot, gen, layer, block, cells, content):
        data = sim.slot_record(slot, gen, layer, block, cells, content)
        self._emit("page", sim.SLOT_BASE + slot * sim.SLOT_SIZE, data)

    def alloc(self, slot, gen, layer, block, n_cells):
        args = sim.u32(slot) + sim.u32(gen) + sim.u32(layer) + sim.u32(block) + sim.u32(n_cells)
        self._log(sim.record(self.epoch, sim.OP_ALLOC, args))
        self._ref(("alloc", slot, gen, layer, block))

    def free(self, slot, gen):
        self._log(sim.record(self.epoch, sim.OP_FREE, sim.u32(slot) + sim.u32(gen)))
        self._ref(("free", slot, gen))

    def move(self, src, src_gen, dst, dst_gen):
        args = sim.u32(src) + sim.u32(src_gen) + sim.u32(dst) + sim.u32(dst_gen)
        self._log(sim.record(self.epoch, sim.OP_MOVE, args))
        self._ref(("move", src, src_gen, dst, dst_gen))

    def seq_rm(self, seq, p0, p1):
        args = sim.u32(seq) + sim.i32(p0) + sim.i32(p1)
        self._log(sim.record(self.epoch, sim.OP_SEQ_RM, args))
        self._ref(("seq_rm", seq, p0, p1))

    def seq_bind(self, seq, slot, gen, pos):
        args = sim.u32(seq) + sim.u32(slot) + sim.u32(gen) + sim.i32(pos)
        self._log(sim.record(self.epoch, sim.OP_SEQ_BIND, args))
        self._ref(("seq_bind", seq, slot, gen, pos))

    def rs_put(self, seq, content, ep):
        args = sim.u32(seq) + sim.u32(len(content)) + sim.u32(ep) + content
        self._log(sim.record(self.epoch, sim.OP_RS_PUT, args))
        self._ref(("rs_put", seq, content, ep))

    def checkpoint(self):
        snap = sim.snapshot(self.ref)
        page = sim.root_page(*snap)
        off = self.root_bump
        self.root_bump += len(page)
        self.root_offsets.append(off)
        self._emit("page", off, page)
        self._log(sim.record(self.epoch, sim.OP_CHECKPOINT,
                             sim.u64(off) + sim.u32(len(page))))

    def set_base(self, off):
        self._emit("hdr", 0, sim.status_header(off))


def put_block(plan, l0, l1, gen, block, pos0, seq, mask, seed, bind=True):
    cells = [(pos0 + i, mask) for i in range(sim.PAGE_TOKENS)]
    plan.write_slot(l0, gen, 0, block, cells, blob(seed))
    plan.write_slot(l1, gen, 1, block, cells, blob(seed + 1))
    plan.alloc(l0, gen, 0, block, sim.PAGE_TOKENS)
    plan.alloc(l1, gen, 1, block, sim.PAGE_TOKENS)
    if bind:
        for i in range(sim.PAGE_TOKENS):
            plan.seq_bind(seq, l0, gen, pos0 + i)


M_SEQ0 = 1
M_SEQ1 = 2

# tear prefix lengths swept by the matrix: one that cuts inside a block header, one
# that cuts inside the payload, and sim.BLOCK // 2 which may cover the payload whole
BLOCK_TEAR_SMALL = 4
BLOCK_TEAR_MID = 32


def build_safe():
    """Checkpoint before every slot reuse, and the recurrent write in the same epoch."""
    p = Plan()

    # epoch 1: seq 0 prefill, two blocks
    p.begin()
    put_block(p, 0, 1, 10, 0, 0, 0, M_SEQ0, 0xA0)
    put_block(p, 2, 3, 11, 1, 4, 0, M_SEQ0, 0xA2)
    p.commit()

    # epoch 2: seq 0 decode, third block
    p.begin()
    put_block(p, 4, 5, 12, 2, 8, 0, M_SEQ0, 0xA4)
    p.commit()

    # epoch 3: checkpoint
    p.begin()
    p.checkpoint()
    p.commit()

    # epoch 4: seq 1 arrives, with its recurrent blob
    p.begin()
    put_block(p, 6, 7, 20, 0, 0, 1, M_SEQ1, 0xB0)
    p.rs_put(1, blob(0xB1), 4)
    p.commit()

    # epoch 5: evict seq 0, free its slots
    p.begin()
    p.seq_rm(0, 0, 12)
    for slot, gen in ((0, 10), (1, 10), (2, 11), (3, 11), (4, 12), (5, 12)):
        p.free(slot, gen)
    p.commit()

    # epoch 6: context compaction on seq 1, recurrent rewritten in the same epoch
    p.begin()
    p.seq_rm(1, 0, 2)
    p.rs_put(1, blob(0xB2), 6)
    p.commit()

    # epoch 7: checkpoint, so slots 0..5 are no longer referenced by a root
    p.begin()
    p.checkpoint()
    p.commit()

    # epoch 8: defrag, slot 6 into slot 0, reuse is now legal
    p.begin()
    p.write_slot(0, 30, 0, 0, [(2, M_SEQ1), (3, M_SEQ1), (0, 0), (0, 0)], blob(0xC0))
    p.move(6, 20, 0, 30)
    p.commit()

    # epoch 9: log compaction. The epoch starts in a fresh block so the base offset
    # can point at a block boundary, and the block is flushed before the base moves:
    # a base that points at an unflushed block loses everything behind it.
    p.begin(fresh=True)
    p.checkpoint()
    p.commit()
    p.flush_block()
    p.set_base(p.begin_block_off[9])

    # epoch 10: seq 1 decode, with its recurrent blob
    p.begin()
    put_block(p, 8, 9, 40, 1, 4, 1, M_SEQ1, 0xD0)
    p.rs_put(1, blob(0xD1), 10)
    p.commit()

    # epoch 11: middle range compaction, which a recurrent state cannot do on its
    # own, so the blob has to be rewritten in the same epoch
    p.begin()
    p.seq_rm(1, 5, 7)
    p.rs_put(1, blob(0xD2), 11)
    p.commit()

    p.finish()
    return p


def build_unsafe_reuse():
    """A move into a slot the newest durable checkpoint still references."""
    p = Plan()

    p.begin()
    put_block(p, 0, 1, 10, 0, 0, 0, M_SEQ0, 0xA0)
    put_block(p, 2, 3, 11, 1, 4, 0, M_SEQ0, 0xA2)
    p.commit()

    p.begin(fresh=True)
    p.checkpoint()
    p.commit()
    p.flush_block()
    p.set_base(p.begin_block_off[2])

    # dst is slot 2, live and referenced by the checkpoint
    p.begin()
    p.write_slot(2, 99, 0, 1, [(4, M_SEQ0), (5, M_SEQ0), (0, 0), (0, 0)], blob(0xEE))
    p.move(2, 11, 2, 99)
    p.commit()

    p.finish()
    return p


def build_stale_rs():
    """A bind whose recurrent blob was written at an earlier epoch."""
    p = Plan()

    p.begin()
    put_block(p, 0, 1, 10, 0, 0, 0, M_SEQ0, 0xA0)
    p.rs_put(0, blob(0xA1), 1)
    p.commit()

    p.begin()
    put_block(p, 2, 3, 11, 1, 4, 0, M_SEQ0, 0xA2)
    p.rs_put(0, blob(0xA3), 1)  # stale: should be 2
    p.commit()

    p.finish()
    return p


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def execute(recs, limit, tear, tear_len=None):
    budget = Budget(limit, tear, tear_len)
    pages = Disk(budget)
    pages.data = bytearray(sim.pages_header())
    status = Disk(budget)
    status.data = bytearray(sim.status_header(sim.STATUS_HEADER))

    for i, r in enumerate(recs):
        try:
            if r.kind == "page":
                pages.write_at(r.addr, r.data)
            else:
                status.write_at(r.addr, r.data)
        except Crash:
            return pages, status, i
    return pages, status, len(recs)


def expected_state(plan, durable):
    """The state at the newest epoch whose commit block was flushed."""
    state = sim.empty_state()
    for epoch in sorted(plan.commit_block):
        ri = plan.block_rec.get(plan.commit_block[epoch])
        if ri is None or ri >= durable:
            break
        for op in plan.epoch_ops.get(epoch, []):
            ref_apply(state, epoch, op)
    return state


def total_bytes(plan):
    return sum(len(r.data) for r in plan.recs)


def run_matrix(plan, label, expect_failures=False):
    n = len(plan.recs)
    runs = 0
    failures = []
    seen_epochs = set()
    reclaimed = 0
    # tear_len is swept because a torn write can either cut into a block's payload or
    # cover it whole, and the second case leaves the block valid.
    cases = [(False, None)] + [(True, tl) for tl in (BLOCK_TEAR_SMALL, BLOCK_TEAR_MID, sim.BLOCK // 2)]
    for tear, tear_len in cases:
        for limit in range(0, n + 1):
            pages, status, durable = execute(plan.recs, limit, tear, tear_len)
            runs += 1
            if sim.read_status_base(status) != sim.STATUS_HEADER:
                reclaimed += 1
            # the crashing op is fully lost, or it fully landed when the torn prefix
            # covered its whole payload area; nothing in between is observable
            allowed = [sim.snapshot(expected_state(plan, durable))]
            if tear:
                allowed.append(sim.snapshot(expected_state(plan, durable + 1)))
            try:
                got, committed, _cp = sim.recover(pages, status)
            except RecoveryError as e:
                failures.append((tear, tear_len, limit, "recovery error: %s" % e))
                continue
            seen_epochs.add(committed)
            if sim.snapshot(got) not in allowed:
                failures.append((tear, tear_len, limit, "state mismatch"))

    ok = bool(failures) if expect_failures else not failures
    print("%-16s runs=%-5d blocks=%-4d bytes=%-6d states=%-3d reclaimed=%-3d failures=%-3d %s"
          % (label, runs, n, total_bytes(plan), len(seen_epochs), reclaimed, len(failures),
             "ok" if ok else "FAILED"))
    if failures and not expect_failures:
        for f in failures[:5]:
            print("    tear=%s len=%s limit=%d %s" % f)
    return ok, failures, sorted(seen_epochs)


def corruption_control():
    """Corrupt one byte of a committed slot page, recovery must refuse."""
    plan = build_safe()
    pages, status, _durable = execute(plan.recs, None, False)
    off = sim.SLOT_BASE + 8 * sim.SLOT_SIZE + 20
    pages.data[off] ^= 0xFF
    try:
        sim.recover(pages, status)
    except RecoveryError as e:
        print("%-16s refused: %s" % ("corrupt_page", e))
        return True
    print("%-16s FAILED: corruption was accepted" % "corrupt_page")
    return False


def root_corruption_control():
    """Corrupt the newest root page, recovery must refuse rather than load it."""
    plan = build_safe()
    pages, status, _durable = execute(plan.recs, None, False)
    off = plan.root_offsets[-1] + 6
    pages.data[off] ^= 0xFF
    try:
        sim.recover(pages, status)
    except RecoveryError as e:
        print("%-16s refused: %s" % ("corrupt_root", e))
        return True
    print("%-16s FAILED: corruption was accepted" % "corrupt_root")
    return False


def dead_root_control():
    """A superseded root is unreachable after a durable base advance, so it may
    be damaged. This asserts the log compaction actually took effect."""
    plan = build_safe()
    pages, status, _durable = execute(plan.recs, None, False)
    off = plan.root_offsets[0] + 6
    pages.data[off] ^= 0xFF
    try:
        sim.recover(pages, status)
    except RecoveryError as e:
        print("%-16s FAILED: dead root was still read: %s" % ("dead_root", e))
        return False
    print("%-16s unreachable, not read" % "dead_root")
    return True


def block_header_control():
    """Zero the header of the newest block at or after the base. The records inside
    are intact, so only the block header can catch this. Recovery must lose that
    block, not read through it."""
    plan = build_safe()
    pages, status, _durable = execute(plan.recs, None, False)
    base = sim.read_status_base(status)
    live = [r.addr for r in plan.recs if r.kind == "block" and r.addr >= base]
    if not live:
        print("%-16s SKIPPED, no live blocks" % "block_header")
        return False
    victim = live[-1]
    status.data[victim:victim + 8] = b"\0" * 8
    try:
        _got, committed, _cp = sim.recover(pages, status)
    except RecoveryError as e:
        print("%-16s refused: %s" % ("block_header", e))
        return True
    if committed < 11:
        print("%-16s lost the tail, back to epoch %d" % ("block_header", committed))
        return True
    print("%-16s FAILED: read past a zeroed block header to epoch %d" % ("block_header", committed))
    return False


def main():
    results = []

    ok, fails, epochs = run_matrix(build_safe(), "safe")
    results.append(ok)
    print("    safe recovered states: %s" % (epochs,))

    ok, fails, _ = run_matrix(build_unsafe_reuse(), "unsafe_reuse", expect_failures=True)
    results.append(ok)
    if fails:
        print("    detection example: tear=%s len=%s limit=%d %s" % fails[0])

    ok, fails, _ = run_matrix(build_stale_rs(), "stale_rs", expect_failures=True)
    results.append(ok)
    if fails:
        print("    detection example: tear=%s len=%s limit=%d %s" % fails[-1])

    results.append(corruption_control())
    results.append(root_corruption_control())
    results.append(dead_root_control())
    results.append(block_header_control())

    print()
    print("overall: %s" % ("pass" if all(results) else "FAIL"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
