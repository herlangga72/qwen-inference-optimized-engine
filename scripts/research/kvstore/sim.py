"""Session KV store: layer 1 format and recovery.

Pure bytes, no model, no backend. Models the two files from the spec and the
recovery algorithm, plus a simulated disk that can lose writes to test crash
behaviour.

The atomic unit is a block, not a record, because O_DIRECT needs 512 byte
alignment of offset and length on this filesystem
(docs/research/18-direct-io-framing-results.md). Records are packed into blocks
and never straddle a block boundary, so a crash loses whole blocks.
"""

import struct
import zlib

# O_DIRECT needs 512 byte alignment of offset and length, so a block is the
# atomic unit and the slot size is padded to one.
BLOCK     = 512
BLOCK_HDR = 8

HEADER_SIZE = BLOCK
SLOT_BASE   = HEADER_SIZE
MAX_SLOTS   = 32
PAGE_TOKENS = 4
N_LAYERS    = 2
N_CTX       = 64
PAYLOAD     = 8          # stand-in for a K or V row
N_SEQ_MAX   = 8

SLOT_PAYLOAD = 8 + PAGE_TOKENS * 8 + PAGE_TOKENS * PAYLOAD
SLOT_SIZE    = BLOCK     # padded, so one slot write is one whole block
ROOT_BASE    = SLOT_BASE + MAX_SLOTS * SLOT_SIZE

STATUS_HEADER = BLOCK    # the first block is the status file header
STATUS_MAGIC  = 0x4B565354  # "KVST"
PAGES_MAGIC   = 0x4B565047  # "KVPG"
FORMAT_VER    = 1

# ops
OP_BEGIN      = 1
OP_COMMIT     = 2
OP_ALLOC      = 3
OP_FREE       = 4
OP_MOVE       = 5
OP_SEQ_RM     = 6
OP_SEQ_BIND   = 7
OP_RS_PUT     = 8
OP_CHECKPOINT = 9


class Crash(Exception):
    pass


class Budget:
    """Shared crash budget. One budget across all files, so one crash stops everything.

    limit = number of atomic writes that succeed. tear = the write that reaches the
    limit lands a prefix of tear_len bytes instead of the whole block.

    The prefix length matters. A prefix long enough to cover a block's whole payload
    area leaves the block intact and valid, so a torn write can fully land. The matrix
    sweeps the length to cover both outcomes.
    """

    def __init__(self, limit=None, tear=False, tear_len=None):
        self.limit = limit
        self.tear = tear
        self.tear_len = BLOCK // 2 if tear_len is None else tear_len
        self.used = 0

    def take(self):
        """Consume one atomic write. True when this write is the torn one."""
        if self.limit is not None and self.used >= self.limit:
            raise Crash()
        self.used += 1
        return self.limit is not None and self.tear and self.used == self.limit


class Disk:
    """Byte addressed file. Atomic writes are whole blocks."""

    def __init__(self, budget=None):
        self.data = bytearray()
        self.budget = budget if budget is not None else Budget()

    def _grow(self, end):
        if end > len(self.data):
            self.data.extend(b"\0" * (end - len(self.data)))

    def write_at(self, off, buf):
        for i in range(0, len(buf), BLOCK):
            chunk = buf[i:i + BLOCK]
            torn = self.budget.take()
            self._grow(off + i + len(chunk))
            if torn:
                half = chunk[:max(1, min(self.budget.tear_len, len(chunk)))]
                self.data[off + i:off + i + len(half)] = half
                raise Crash()
            self.data[off + i:off + i + len(chunk)] = chunk

    def read_at(self, off, n):
        if off >= len(self.data):
            return b""
        return bytes(self.data[off:off + n])

    def clip(self, n):
        self.data = self.data[:n]


def crc32(b):
    return zlib.crc32(b) & 0xFFFFFFFF


def u32(v):
    return struct.pack("<I", v & 0xFFFFFFFF)


def i32(v):
    return struct.pack("<i", v)


def u64(v):
    return struct.pack("<Q", v)


def rd_u32(b, o):
    return struct.unpack_from("<I", b, o)[0], o + 4


def rd_i32(b, o):
    return struct.unpack_from("<i", b, o)[0], o + 4


def rd_u64(b, o):
    return struct.unpack_from("<Q", b, o)[0], o + 8


def pad_block(b):
    """Pad to a whole number of blocks, so an O_DIRECT write is legal."""
    rem = len(b) % BLOCK
    if rem:
        b = b + b"\0" * (BLOCK - rem)
    return b


# ---------------------------------------------------------------------------
# pages file
# ---------------------------------------------------------------------------

def pages_header():
    body = u32(PAGES_MAGIC) + u32(FORMAT_VER) + u32(PAGE_TOKENS) + u32(MAX_SLOTS)
    body += u32(N_LAYERS) + u32(N_CTX) + u32(N_SEQ_MAX)
    body += u32(0x04030201)  # endian marker
    body += u32(crc32(body))
    return body + b"\0" * (HEADER_SIZE - len(body))


def slot_payload(layer, block, cells, blob):
    out = u32(layer) + u32(block)
    for pos, mask in cells:
        out += i32(pos) + u32(mask)
    for i in range(PAGE_TOKENS):
        out += blob[i * PAYLOAD:(i + 1) * PAYLOAD].ljust(PAYLOAD, b"\0")
    return out


def slot_record(slot, gen, layer, block, cells, blob):
    p = slot_payload(layer, block, cells, blob)
    return pad_block(u32(slot) + u32(gen) + u32(crc32(p)) + p)


def encode_root(slots, seqs, rs, rs_epoch, seq_epoch):
    out = u32(len(slots))
    for slot in sorted(slots):
        gen, layer, block = slots[slot]
        out += u32(slot) + u32(gen) + u32(layer) + u32(block)
    out += u32(len(seqs))
    for seq in sorted(seqs):
        out += u32(seq) + u32(len(seqs[seq]))
        for pos, slot, gen in seqs[seq]:
            out += i32(pos) + u32(slot) + u32(gen)
    out += u32(len(rs))
    for seq in sorted(rs):
        blob = rs[seq]
        out += u32(seq) + u32(len(blob)) + u32(rs_epoch.get(seq, 0))
        out += blob.ljust(PAYLOAD, b"\0")
    out += u32(len(seq_epoch))
    for seq in sorted(seq_epoch):
        out += u32(seq) + u32(seq_epoch[seq])
    return out


def decode_root(b):
    slots, seqs, rs, rs_epoch, seq_epoch = {}, {}, {}, {}, {}
    o = 0
    n, o = rd_u32(b, o)
    for _ in range(n):
        slot, o = rd_u32(b, o)
        gen, o = rd_u32(b, o)
        layer, o = rd_u32(b, o)
        block, o = rd_u32(b, o)
        slots[slot] = (gen, layer, block)
    n, o = rd_u32(b, o)
    for _ in range(n):
        seq, o = rd_u32(b, o)
        cnt, o = rd_u32(b, o)
        cells = []
        for _ in range(cnt):
            pos, o = rd_i32(b, o)
            slot, o = rd_u32(b, o)
            gen, o = rd_u32(b, o)
            cells.append((pos, slot, gen))
        seqs[seq] = cells
    n, o = rd_u32(b, o)
    for _ in range(n):
        seq, o = rd_u32(b, o)
        ln, o = rd_u32(b, o)
        ep, o = rd_u32(b, o)
        blob = b[o:o + PAYLOAD]
        o += PAYLOAD
        rs[seq] = blob
        rs_epoch[seq] = ep
    n, o = rd_u32(b, o)
    for _ in range(n):
        seq, o = rd_u32(b, o)
        ep, o = rd_u32(b, o)
        seq_epoch[seq] = ep
    return slots, seqs, rs, rs_epoch, seq_epoch


def root_page(slots, seqs, rs, rs_epoch, seq_epoch):
    """The root lives in the pages file as a padded, self describing blob."""
    body = encode_root(slots, seqs, rs, rs_epoch, seq_epoch)
    return pad_block(u32(crc32(body)) + u32(len(body)) + body)


def decode_root_page(raw):
    if len(raw) < 8:
        return None
    c, o = rd_u32(raw, 0)
    ln, o = rd_u32(raw, 4)
    if 8 + ln > len(raw):
        return None
    body = raw[8:8 + ln]
    if crc32(body) != c:
        return None
    return decode_root(body)


# ---------------------------------------------------------------------------
# status file: block framed append only log
# ---------------------------------------------------------------------------

def status_header(base_off):
    body = u32(STATUS_MAGIC) + u32(FORMAT_VER) + u64(base_off)
    out = body + u32(crc32(body))
    return out + b"\0" * (STATUS_HEADER - len(out))


def read_status_base(disk):
    hdr = disk.read_at(0, STATUS_HEADER)
    if len(hdr) < STATUS_HEADER:
        return STATUS_HEADER
    body = hdr[:16]
    magic, _ = rd_u32(hdr, 0)
    ver, _ = rd_u32(hdr, 4)
    base, _ = rd_u64(hdr, 8)
    c, _ = rd_u32(hdr, 16)
    if magic != STATUS_MAGIC or ver != FORMAT_VER or c != crc32(body):
        return STATUS_HEADER
    if base < STATUS_HEADER or base > len(disk.data) or base % BLOCK != 0:
        return STATUS_HEADER
    return base


class BlockLog:
    """Packs records into fixed blocks with a per block header.

    Planning and writing share this class, so the offsets the plan embeds are the
    offsets the writer produces.

        block := [u32 used][u32 crc32(payload area)][records][zero padding]
    """

    def __init__(self, base_off):
        self.cur_off = base_off
        self.cur = bytearray(BLOCK)
        self.fill = BLOCK_HDR
        self.index = -1
        self.closed = []

    def append(self, rec):
        if len(rec) > BLOCK - BLOCK_HDR:
            raise ValueError("record larger than a block")
        if self.fill + len(rec) > BLOCK:
            self.close()
        if self.fill == BLOCK_HDR:
            self.index += 1
        off = self.cur_off + self.fill
        self.cur[self.fill:self.fill + len(rec)] = rec
        self.fill += len(rec)
        return self.index, off

    def close(self):
        if self.fill == BLOCK_HDR:
            return
        used = self.fill - BLOCK_HDR
        struct.pack_into("<II", self.cur, 0, used,
                         crc32(bytes(self.cur[BLOCK_HDR:self.fill])))
        self.cur[self.fill:BLOCK] = b"\0" * (BLOCK - self.fill)
        self.closed.append((self.cur_off, bytes(self.cur)))
        self.cur = bytearray(BLOCK)
        self.fill = BLOCK_HDR
        self.cur_off += BLOCK

    def take_closed(self):
        out = self.closed
        self.closed = []
        return out


def record(epoch, op, args=b""):
    payload = u64(epoch) + u32(op) + args
    return u32(len(payload)) + u32(crc32(payload)) + payload


def parse_blocks(buf):
    """Walk the block framed log. Returns (records, clean).

    A block is accepted only if its header crc covers its payload area and every
    record inside it is intact and inside that area. The first bad block ends the
    stream: padding and a torn tail are different things here.
    """
    recs = []
    o = 0
    while o + BLOCK <= len(buf):
        blk = buf[o:o + BLOCK]
        used, c = struct.unpack_from("<II", blk, 0)
        if used == 0 or BLOCK_HDR + used > BLOCK:
            return recs, False
        if crc32(blk[BLOCK_HDR:BLOCK_HDR + used]) != c:
            return recs, False
        p = BLOCK_HDR
        end = BLOCK_HDR + used
        while p + 8 <= end:
            ln, rc = struct.unpack_from("<II", blk, p)
            p += 8
            if ln == 0 or p + ln > end:
                return recs, False
            payload = blk[p:p + ln]
            if crc32(payload) != rc:
                return recs, False
            p += ln
            epoch, _ = rd_u64(payload, 0)
            op, _ = rd_u32(payload, 8)
            recs.append((epoch, op, payload[12:]))
        o += BLOCK
    return recs, True


# ---------------------------------------------------------------------------
# reference state
# ---------------------------------------------------------------------------

def empty_state():
    return {}, {}, {}, {}, {}


def apply_op(state, epoch, op, args):
    slots, seqs, rs, rs_epoch, seq_epoch = state
    if op == OP_ALLOC:
        slot = struct.unpack_from("<I", args, 0)[0]
        gen = struct.unpack_from("<I", args, 4)[0]
        layer = struct.unpack_from("<I", args, 8)[0]
        block = struct.unpack_from("<I", args, 12)[0]
        slots[slot] = (gen, layer, block)
    elif op == OP_FREE:
        slot = struct.unpack_from("<I", args, 0)[0]
        gen = struct.unpack_from("<I", args, 4)[0]
        cur = slots.get(slot)
        if cur is not None and cur[0] == gen:
            del slots[slot]
    elif op == OP_MOVE:
        src = struct.unpack_from("<I", args, 0)[0]
        src_gen = struct.unpack_from("<I", args, 4)[0]
        dst = struct.unpack_from("<I", args, 8)[0]
        dst_gen = struct.unpack_from("<I", args, 12)[0]
        cur = slots.get(src)
        if cur is not None and cur[0] == src_gen:
            slots[dst] = (dst_gen, cur[1], cur[2])
            del slots[src]
            for seq in seqs:
                seqs[seq] = [(p, dst if (s == src and g == src_gen) else s,
                              dst_gen if (s == src and g == src_gen) else g)
                             for (p, s, g) in seqs[seq]]
    elif op == OP_SEQ_RM:
        seq = struct.unpack_from("<I", args, 0)[0]
        p0 = struct.unpack_from("<i", args, 4)[0]
        p1 = struct.unpack_from("<i", args, 8)[0]
        seqs[seq] = [c for c in seqs.get(seq, []) if not (p0 <= c[0] < p1)]
        seq_epoch[seq] = epoch
    elif op == OP_SEQ_BIND:
        seq = struct.unpack_from("<I", args, 0)[0]
        slot = struct.unpack_from("<I", args, 4)[0]
        gen = struct.unpack_from("<I", args, 8)[0]
        pos = struct.unpack_from("<i", args, 12)[0]
        lst = seqs.setdefault(seq, [])
        lst.append((pos, slot, gen))
        lst.sort()
        seq_epoch[seq] = epoch
    elif op == OP_RS_PUT:
        seq = struct.unpack_from("<I", args, 0)[0]
        ln = struct.unpack_from("<I", args, 4)[0]
        ep = struct.unpack_from("<I", args, 8)[0]
        rs[seq] = bytes(args[12:12 + ln]).ljust(PAYLOAD, b"\0")
        rs_epoch[seq] = ep
    return state


def snapshot(state):
    slots, seqs, rs, rs_epoch, seq_epoch = state
    return ({k: v for k, v in slots.items()},
            {k: list(v) for k, v in seqs.items()},
            {k: bytes(v) for k, v in rs.items()},
            {k: v for k, v in rs_epoch.items()},
            {k: v for k, v in seq_epoch.items()})


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------

class RecoveryError(Exception):
    pass


def check_pages_head(pages):
    hdr = pages.read_at(0, HEADER_SIZE)
    if len(hdr) < HEADER_SIZE:
        raise RecoveryError("pages header missing")
    magic, _ = rd_u32(hdr, 0)
    if magic != PAGES_MAGIC:
        raise RecoveryError("pages magic mismatch")
    body = hdr[:32]
    c, _ = rd_u32(hdr, 32)
    if c != crc32(body):
        raise RecoveryError("pages header crc mismatch")


def read_slot(pages, slot):
    off = SLOT_BASE + slot * SLOT_SIZE
    raw = pages.read_at(off, SLOT_SIZE)
    if len(raw) < 12:
        return None
    sid, _ = rd_u32(raw, 0)
    gen, _ = rd_u32(raw, 4)
    c, _ = rd_u32(raw, 8)
    p = raw[12:12 + SLOT_PAYLOAD]
    if sid != slot or crc32(p) != c:
        return None
    return gen


def recover(pages, status):
    """Recovery per the spec. Raises RecoveryError when a rule fails closed."""
    check_pages_head(pages)

    base = read_status_base(status)
    buf = status.read_at(base, len(status.data) - base)
    recs, _clean = parse_blocks(buf)

    state = empty_state()
    epoch_ops = []
    current = None
    last_checkpoint = None
    committed = 0

    for epoch, op, args in recs:
        if op == OP_BEGIN:
            current = epoch
            epoch_ops = []
        elif op == OP_COMMIT:
            if current is not None and epoch == current:
                for (e, o, a) in epoch_ops:
                    apply_op(state, e, o, a)
                cp = [a for (e, o, a) in epoch_ops if o == OP_CHECKPOINT]
                if cp:
                    a = cp[-1]
                    off, _ = rd_u64(a, 0)
                    ln, _ = rd_u32(a, 8)
                    raw = pages.read_at(off, ln)
                    if len(raw) < ln:
                        raise RecoveryError("checkpoint root truncated")
                    st = decode_root_page(raw)
                    if st is None:
                        raise RecoveryError("checkpoint root crc mismatch")
                    state = st
                    last_checkpoint = epoch
                committed = epoch
                epoch_ops = []
                current = None
        else:
            if current is None:
                raise RecoveryError("record outside an epoch")
            epoch_ops.append((epoch, op, args))

    slots, seqs, rs, rs_epoch, seq_epoch = state
    for seq in sorted(seqs):
        for (_pos, slot, gen) in seqs[seq]:
            if slot not in slots:
                raise RecoveryError("seq %d references unknown slot %d" % (seq, slot))
            if slots[slot][0] != gen:
                raise RecoveryError("seq %d slot %d generation mismatch" % (seq, slot))
            if read_slot(pages, slot) != gen:
                raise RecoveryError("seq %d slot %d page crc or generation mismatch" % (seq, slot))

    for seq in sorted(seqs):
        prev = None
        for (pos, _slot, _gen) in seqs[seq]:
            if pos < 0 or pos >= N_CTX:
                raise RecoveryError("seq %d position %d out of range" % (seq, pos))
            if prev is not None and pos <= prev:
                raise RecoveryError("seq %d positions not strictly increasing" % seq)
            prev = pos
    for seq in sorted(rs):
        if seq not in seqs:
            continue
        if rs_epoch.get(seq, 0) != seq_epoch.get(seq, 0):
            raise RecoveryError("seq %d recurrent epoch %d != attention epoch %d"
                                % (seq, rs_epoch.get(seq, 0), seq_epoch.get(seq, 0)))

    return state, committed, last_checkpoint
