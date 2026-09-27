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
// this code produce identical bytes
uint32_t llama_io_crc32(const void * data, size_t size) {
    static uint32_t table[256];
    static bool init = false;

    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        init = true;
    }

    uint32_t c = 0xFFFFFFFFu;
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < size; ++i) {
        c = table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
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

    // block 0 is reserved for the file header
    block_off = LLAMA_IO_BLOCK;
}

llama_io_write_direct::~llama_io_write_direct() {
    // a missing flush loses the final partial block, so do it rather than report it
    if (!flushed && framed) {
        flush();
    }
    if (fd >= 0) {
        ::close(fd);
    }
    free(blk);
}

void llama_io_write_direct::set_prefix(const void * data, size_t size) {
    if (err || !blk) {
        return;
    }
    if (size > LLAMA_IO_BLOCK) {
        err = EINVAL;
        return;
    }

    memset(blk, 0, LLAMA_IO_BLOCK);
    memcpy(blk, data, size);

    err = io_pwrite_all(fd, blk, LLAMA_IO_BLOCK, 0);
}

void llama_io_write_direct::put_block(size_t off, size_t used) {
    uint32_t u = (uint32_t) used;
    uint32_t c = llama_io_crc32(blk + LLAMA_IO_BLOCK_HDR, used);

    memcpy(blk + 0, &u, sizeof(u));
    memcpy(blk + 4, &c, sizeof(c));
    memcpy(blk + 8, &gen_, sizeof(gen_));
    memset(blk + LLAMA_IO_BLOCK_HDR + used, 0, LLAMA_IO_BLOCK - LLAMA_IO_BLOCK_HDR - used);

    err = io_pwrite_all(fd, blk, LLAMA_IO_BLOCK, off);
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

    // a logical write never straddles a block boundary. a record that is split cannot be
    // read back, because the block header is what says where the records in it end.
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
    // chunked so host memory stays one block, whatever the tensor size. framed writes keep
    // a chunk inside one block payload, plain writes are whole blocks by contract.
    const size_t chunk = framed ? LLAMA_IO_BLOCK - LLAMA_IO_BLOCK_HDR : LLAMA_IO_BLOCK;
    std::vector<uint8_t> tmp(chunk);

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
    flushed = true;
}

size_t llama_io_verify_direct(const std::string & path, uint32_t gen) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        return (size_t) -1;
    }

    void * blk = io_alloc_block();
    if (!blk) {
        ::close(fd);
        return (size_t) -1;
    }

    const uint8_t * p = (const uint8_t *) blk;
    size_t   bad      = 0;
    size_t   off      = LLAMA_IO_BLOCK; // block 0 is the file header, not a frame
    uint32_t expected = gen;
    for (;;) {
        const ssize_t n = ::pread(fd, blk, LLAMA_IO_BLOCK, (off_t) off);
        if (n == 0) {
            break;
        }
        if (n < 0) {
            bad = (size_t) -1;
            break;
        }
        if (n < (ssize_t) LLAMA_IO_BLOCK) {
            bad++; // a torn tail
            break;
        }

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

        off += LLAMA_IO_BLOCK;
    }

    free(blk);
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

    blk = (uint8_t *) io_alloc_block();
    if (!blk) {
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
    free(blk);
}

bool llama_io_read_direct::next_block() {
    if (err || !blk) {
        return false;
    }

    ssize_t n = ::pread(fd, blk, LLAMA_IO_BLOCK, (off_t) blk_off);
    if (n < 0) {
        err = errno ? errno : EIO;
        return false;
    }
    if (n < (ssize_t) LLAMA_IO_BLOCK) {
        // a short block is a torn tail, not a payload
        eof   = true;
        avail = 0;
        pos   = 0;
        return false;
    }

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
    if (err || !blk) {
        return;
    }
    if (size > LLAMA_IO_BLOCK) {
        err = EINVAL;
        return;
    }

    ssize_t n = ::pread(fd, blk, LLAMA_IO_BLOCK, 0);
    if (n < (ssize_t) LLAMA_IO_BLOCK) {
        err = EBADMSG;
        return;
    }

    memcpy(dst, blk, size);
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
    std::vector<uint8_t> tmp(LLAMA_IO_BLOCK);

    for (size_t done = 0; done < size && !err; ) {
        const size_t take = std::min(tmp.size(), size - done);
        read(tmp.data(), take);
        ggml_backend_tensor_set(tensor, tmp.data(), offset + done, take);
        done += take;
    }
}
