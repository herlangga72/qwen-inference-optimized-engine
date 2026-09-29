#include "llama-io.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_WIN32)
// O_DIRECT is what keeps KV out of the page cache, and it is not wired up here. Failing
// loudly is better than silently writing through the cache and breaking the policy.
#error "llama_io_write_direct needs O_DIRECT, which is not implemented for Windows"
#else
#include <fcntl.h>
#include <unistd.h>
#endif

void llama_io_write_i::write_string(const std::string & str) {
    uint32_t str_size = str.size();

    write(&str_size,  sizeof(str_size));
    write(str.data(), str_size);
}

void llama_io_read_i::read_string(std::string & str) {
    uint32_t str_size;
    read(&str_size, sizeof(str_size));

    std::vector<char> buf(str_size);
    read(buf.data(), str_size);

    str.assign(buf.data(), str_size);
}

// zlib crc32, same polynomial and same initial and final xor, so the Python harness and
// this code produce identical bytes.
//
// Sliced by eight. The byte at a time loop this replaced ran at 519 MiB/s, which held the whole
// store there: the reader checksums every block, so it could not exceed that, and the writer
// checksums once and the verify pass checksums again. Slicing computes the same value with the
// same tables, one byte of table lookup per byte of input instead of one per byte per bit.
// See docs/research/29-store-io-batching-results.md
uint32_t llama_io_crc32(const void * data, size_t size) {
    static uint32_t table[8][256];

    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            table[0][i] = c;
        }
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = table[0][i];
            for (int k = 1; k < 8; ++k) {
                c = table[0][c & 0xFF] ^ (c >> 8);
                table[k][i] = c;
            }
        }
        init = true;
    }

    const uint8_t * p = (const uint8_t *) data;
    size_t left = size;

    uint32_t c = 0xFFFFFFFFu;

    // the eight bytes are folded in lowest address first, which is what a little endian load
    // gives. this tree is x86-64 only.
    while (left >= 8) {
        uint64_t v = 0;
        memcpy(&v, p, sizeof(v));
        v ^= c;

        c = table[7][(v      ) & 0xFF] ^
            table[6][(v >>  8) & 0xFF] ^
            table[5][(v >> 16) & 0xFF] ^
            table[4][(v >> 24) & 0xFF] ^
            table[3][(v >> 32) & 0xFF] ^
            table[2][(v >> 40) & 0xFF] ^
            table[1][(v >> 48) & 0xFF] ^
            table[0][(v >> 56) & 0xFF];

        p    += 8;
        left -= 8;
    }

    while (left > 0) {
        c = table[0][(c ^ *p) & 0xFF] ^ (c >> 8);
        p++;
        left--;
    }

    return c ^ 0xFFFFFFFFu;
}

static void * io_alloc_block() {
    void * p = nullptr;
    if (posix_memalign(&p, LLAMA_IO_BLOCK, LLAMA_IO_BLOCK) != 0) {
        return nullptr;
    }
    memset(p, 0, LLAMA_IO_BLOCK);
    return p;
}

// one window buffer for both directions. O_DIRECT wants the buffer aligned to the block
// size, so an aligned window is aligned for every block in it.
static void * io_alloc_window() {
    void * p = nullptr;
    if (posix_memalign(&p, LLAMA_IO_BLOCK, LLAMA_IO_WIN) != 0) {
        return nullptr;
    }
    memset(p, 0, LLAMA_IO_WIN);
    return p;
}

// Generations only need to differ between saves of the same file, so a random value is enough, and
// zero is reserved for "not set".
static uint32_t io_new_gen() {
    uint32_t g = 0;
    const int fd = ::open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        if (::read(fd, &g, sizeof(g)) != (ssize_t) sizeof(g)) {
            g = 0;
        }
        ::close(fd);
    }
    if (g == 0) {
        static uint32_t counter = 0;
        g = (uint32_t) (::getpid() * 2654435761u) ^ (++counter * 40503u);
        if (g == 0) {
            g = 1;
        }
    }
    return g;
}

static int io_pwrite_all(int fd, const void * buf, size_t size, size_t off) {
    const uint8_t * p = (const uint8_t *) buf;
    size_t done = 0;

    while (done < size) {
        ssize_t n = ::pwrite(fd, p + done, size - done, (off_t) (off + done));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return errno ? errno : EIO;
        }
        if (n == 0) {
            return EIO;
        }
        done += (size_t) n;
    }

    return 0;
}

//
// llama_io_write_direct
//

llama_io_write_direct::llama_io_write_direct(const std::string & path, bool framed, uint32_t gen) : framed(framed) {
    gen_ = gen ? gen : io_new_gen();
    fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) {
        err = errno ? errno : EIO;
        return;
    }

    blk = (uint8_t *) io_alloc_block();
    if (!blk) {
        err = ENOMEM;
        ::close(fd);
        fd = -1;
        return;
    }

    win = (uint8_t *) io_alloc_window();
    if (!win) {
        err = ENOMEM;
        ::close(fd);
        fd = -1;
        return;
    }

    // block 0 is reserved for the file header
    block_off = LLAMA_IO_BLOCK;
    win_off   = LLAMA_IO_BLOCK;
}

llama_io_write_direct::~llama_io_write_direct() {
    // a missing flush loses the final partial block and whatever is still in the window,
    // so do it rather than report it
    if (!flushed) {
        flush();
    }
    if (fd >= 0) {
        ::close(fd);
    }
    free(blk);
    free(win);
}

void llama_io_write_direct::set_prefix(const void * data, size_t size) {
    if (err || !blk || !win) {
        return;
    }
    if (size > LLAMA_IO_BLOCK) {
        err = EINVAL;
        return;
    }
    if (win_fill != 0 || block_off != LLAMA_IO_BLOCK) {
        // the prefix is block 0, so it cannot follow any other block
        err = EINVAL;
        return;
    }

    memset(blk, 0, LLAMA_IO_BLOCK);
    memcpy(blk, data, size);

    // block 0 goes into the window with the frames that follow it, so a file is opened with
    // one write rather than two
    win_off = 0;
    append_window(blk, LLAMA_IO_BLOCK);
}

void llama_io_write_direct::append_window(const void * src, size_t size) {
    if (err || !win || size == 0) {
        return;
    }

    if (win_fill + size > LLAMA_IO_WIN) {
        flush_window();
        if (err) {
            return;
        }
    }

    memcpy(win + win_fill, src, size);
    win_fill += size;

    if (win_fill == LLAMA_IO_WIN) {
        flush_window();
    }
}

void llama_io_write_direct::flush_window() {
    if (err || !win || win_fill == 0) {
        return;
    }

    err = io_pwrite_all(fd, win, win_fill, win_off);
    if (err) {
        return;
    }

    win_off  += win_fill;
    win_fill  = 0;
}

void llama_io_write_direct::put_block(size_t off, size_t used) {
    if (err || !blk || !win) {
        return;
    }
    // frames go out in order, so the offset the caller expects is the one the window is at
    if (off != win_off + win_fill) {
        err = EINVAL;
        return;
    }

    uint32_t u = (uint32_t) used;
    uint32_t c = llama_io_crc32(blk + LLAMA_IO_BLOCK_HDR, used);

    memcpy(blk + 0, &u, sizeof(u));
    memcpy(blk + 4, &c, sizeof(c));
    memcpy(blk + 8, &gen_, sizeof(gen_));
    memset(blk + LLAMA_IO_BLOCK_HDR + used, 0, LLAMA_IO_BLOCK - LLAMA_IO_BLOCK_HDR - used);

    append_window(blk, LLAMA_IO_BLOCK);
    if (err) {
        return;
    }

    blocks++;
}

void llama_io_write_direct::flush_block() {
    if (err || !blk) {
        return;
    }
    if (fill == LLAMA_IO_BLOCK_HDR) {
        return;
    }

    put_block(block_off, fill - LLAMA_IO_BLOCK_HDR);
    if (err) {
        return;
    }

    block_off += LLAMA_IO_BLOCK;
    fill = LLAMA_IO_BLOCK_HDR;
}

void llama_io_write_direct::write(const void * src, size_t size) {
    if (err || !blk || size == 0) {
        return;
    }

    n_logical += size;

    if (!framed) {
        // the caller already padded, so only whole blocks are legal
        if (size % LLAMA_IO_BLOCK) {
            err = EINVAL;
            return;
        }
        // plain writes go straight out, so anything buffered has to land first
        flush_window();
        if (err) {
            return;
        }
        err = io_pwrite_all(fd, src, size, block_off);
        if (err) {
            return;
        }
        block_off += size;
        blocks    += size / LLAMA_IO_BLOCK;
        return;
    }

    const uint8_t * p = (const uint8_t *) src;
    size_t left = size;
    const size_t cap = LLAMA_IO_BLOCK - LLAMA_IO_BLOCK_HDR;

    // A write smaller than one block payload is placed so it does not straddle a boundary, which
    // keeps a logical record inside one frame. A larger write, which is what the tensor path now
    // produces, fills whole blocks and leaves its remainder in the next one, so the byte stream
    // is preserved and the reader, which reassembles across blocks, gets it back unchanged. Only
    // the stream matters here: this reader has no record parser to keep in sync.
    if (left <= cap) {
        if (fill + left > LLAMA_IO_BLOCK) {
            flush_block();
        }
        memcpy(blk + fill, p, left);
        fill += left;
        if (fill == LLAMA_IO_BLOCK) {
            flush_block();
        }
        return;
    }

    // larger than one block payload: flush, then fill whole blocks
    flush_block();

    while (left > cap && !err) {
        memcpy(blk + LLAMA_IO_BLOCK_HDR, p, cap);
        fill = LLAMA_IO_BLOCK;
        flush_block();
        p    += cap;
        left -= cap;
    }

    if (left > 0 && !err) {
        memcpy(blk + LLAMA_IO_BLOCK_HDR, p, left);
        fill = LLAMA_IO_BLOCK_HDR + left;
    }
}

void llama_io_write_direct::write_tensor(ggml_tensor * tensor, size_t offset, size_t size) {
    // one device copy per window, not one per block. on a device backend a 4 KB
    // ggml_backend_tensor_get per block costs more than the copy inside it, which held the whole
    // store at 38 MiB/s on Vulkan while the CPU path did 450. write() splits the stream into
    // blocks and the reader reassembles it, so the copy granularity here does not change what
    // the file means. see docs/research/29-store-io-batching-results.md
    std::vector<uint8_t> tmp(std::min<size_t>(LLAMA_IO_WIN, size));

    for (size_t done = 0; done < size && !err; ) {
        const size_t take = std::min(tmp.size(), size - done);
        ggml_backend_tensor_get(tensor, tmp.data(), offset + done, take);
        write(tmp.data(), take);
        done += take;
    }
}

void llama_io_write_direct::flush() {
    if (!flushed && framed) {
        flush_block();
    }
    flush_window();
    flushed = true;
}

size_t llama_io_verify_direct(const std::string & path, uint32_t gen) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        return (size_t) -1;
    }

    // a window at a time, because one syscall per block is 9.8x off the device and this pass
    // runs over the whole file on every save
    uint8_t * win = (uint8_t *) io_alloc_window();
    if (!win) {
        ::close(fd);
        return (size_t) -1;
    }

    size_t   bad      = 0;
    size_t   off      = LLAMA_IO_BLOCK; // block 0 is the file header, not a frame
    uint32_t expected = gen;
    bool     done     = false;

    while (!done) {
        const ssize_t n = ::pread(fd, win, LLAMA_IO_WIN, (off_t) off);
        if (n < 0) {
            bad = (size_t) -1;
            break;
        }
        if (n == 0) {
            break;
        }

        size_t whole = (size_t) n;
        if (whole < LLAMA_IO_BLOCK) {
            bad++; // a torn tail
            break;
        }
        if (whole % LLAMA_IO_BLOCK) {
            bad++; // a partial block at the tail
        }
        whole -= whole % LLAMA_IO_BLOCK;

        for (size_t b = 0; b < whole; b += LLAMA_IO_BLOCK) {
            const uint8_t * p = win + b;

            uint32_t used = 0;
            uint32_t crc  = 0;
            uint32_t bgen = 0;
            memcpy(&used, p, 4);
            memcpy(&crc,  p + 4, 4);
            memcpy(&bgen, p + 8, 4);

            if (used == 0 || used > LLAMA_IO_BLOCK - LLAMA_IO_BLOCK_HDR ||
                llama_io_crc32(p + LLAMA_IO_BLOCK_HDR, used) != crc) {
                bad++;
            } else if (expected == 0) {
                expected = bgen; // adopt the generation of the first intact frame
            } else if (bgen != expected) {
                bad++; // a frame from a different save of this file
            }
        }

        off += whole;
        done = (size_t) n < LLAMA_IO_WIN;
    }

    free(win);
    ::close(fd);

    return bad;
}

//
// llama_io_read_direct
//

llama_io_read_direct::llama_io_read_direct(const std::string & path, bool framed, uint32_t gen) : framed(framed), gen_(gen) {
    fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        err = errno ? errno : EIO;
        return;
    }

    win = (uint8_t *) io_alloc_window();
    if (!win) {
        err = ENOMEM;
        ::close(fd);
        fd = -1;
        return;
    }

    // block 0 holds the file header
    blk_off = LLAMA_IO_BLOCK;
}

llama_io_read_direct::~llama_io_read_direct() {
    if (fd >= 0) {
        ::close(fd);
    }
    free(win);
}

bool llama_io_read_direct::refill_window() {
    const ssize_t n = ::pread(fd, win, LLAMA_IO_WIN, (off_t) blk_off);
    if (n < 0) {
        err = errno ? errno : EIO;
        return false;
    }
    if (n < (ssize_t) LLAMA_IO_BLOCK) {
        // fewer than one whole block is left, so this is a torn tail and not a payload
        eof     = true;
        win_len = 0;
        win_pos = 0;
        avail   = 0;
        pos     = 0;
        return false;
    }

    // a remainder below one block is a torn tail. it is not served, and the next refill reads
    // it at blk_off, finds it short and rejects it there.
    win_len = (size_t) n - (size_t) n % LLAMA_IO_BLOCK;
    win_pos = 0;

    return true;
}

bool llama_io_read_direct::next_block() {
    if (err || !win) {
        return false;
    }

    if (win_len - win_pos < LLAMA_IO_BLOCK) {
        if (!refill_window()) {
            return false;
        }
    }

    blk = win + win_pos;

    win_pos += LLAMA_IO_BLOCK;
    blk_off += LLAMA_IO_BLOCK;

    if (framed) {
        uint32_t used = 0;
        uint32_t c    = 0;
        uint32_t gen  = 0;
        memcpy(&used, blk + 0, sizeof(used));
        memcpy(&c,    blk + 4, sizeof(c));
        memcpy(&gen,  blk + 8, sizeof(gen));

        if (used == 0 || used > LLAMA_IO_BLOCK - LLAMA_IO_BLOCK_HDR) {
            err = EBADMSG;
            return false;
        }
        if (llama_io_crc32(blk + LLAMA_IO_BLOCK_HDR, used) != c) {
            err = EBADMSG;
            return false;
        }
        // a block from an earlier save of this file is a valid frame with a matching crc, so the
        // generation is the only thing that can reject it
        if (gen_ == 0) {
            gen_ = gen;
        } else if (gen != gen_) {
            err = EBADMSG;
            return false;
        }
        avail = used;
    } else {
        avail = LLAMA_IO_BLOCK;
    }

    pos = 0;
    blocks++;

    return true;
}

void llama_io_read_direct::read_prefix(void * dst, size_t size) {
    if (err || !win) {
        return;
    }
    if (size > LLAMA_IO_BLOCK) {
        err = EINVAL;
        return;
    }

    // the header shares a window with the blocks that follow it, so opening a file costs one
    // read rather than two
    blk_off = 0;
    if (!refill_window()) {
        if (!eof) {
            return; // an error is already set
        }
        err = EBADMSG;
        return;
    }

    memcpy(dst, win, size);

    win_pos = LLAMA_IO_BLOCK;
    blk_off = LLAMA_IO_BLOCK;
    blk     = win;
}

void llama_io_read_direct::read(void * dst, size_t size) {
    const size_t pay = framed ? LLAMA_IO_BLOCK_HDR : 0;

    uint8_t * out = (uint8_t *) dst;
    size_t left = size;

    while (left > 0) {
        if (pos >= avail) {
            if (!next_block()) {
                // failed or at a torn tail: zero fill, and the caller must check good()
                memset(out, 0, left);
                return;
            }
        }

        const size_t take = std::min(left, avail - pos);
        memcpy(out, blk + pay + pos, take);

        pos    += take;
        out    += take;
        left   -= take;
        n_read += take;
    }
}

void llama_io_read_direct::read_tensor(ggml_tensor * tensor, size_t offset, size_t size) {
    // one device copy per window, for the same reason as the write side. read() reassembles
    // across blocks, so the granularity here does not change what the stream means
    std::vector<uint8_t> tmp(std::min<size_t>(LLAMA_IO_WIN, size));

    for (size_t done = 0; done < size && !err; ) {
        const size_t take = std::min(tmp.size(), size - done);
        read(tmp.data(), take);
        ggml_backend_tensor_set(tensor, tmp.data(), offset + done, take);
        done += take;
    }
}
