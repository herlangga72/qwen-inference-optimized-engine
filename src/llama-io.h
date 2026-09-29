#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // drop tensor data that has been read but not yet applied (e.g. when a restore fails)
    virtual void discard() {}

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};

// block size for the session KV store files. O_DIRECT needs 512 byte alignment of both
// offset and length on this filesystem, so a block is the atomic unit.
// see docs/research/18-direct-io-framing-results.md
#define LLAMA_IO_BLOCK     4096
#define LLAMA_IO_BLOCK_HDR 12

// Blocks are framed at LLAMA_IO_BLOCK but moved in windows of this size. One syscall per block
// costs about 21 us on this volume, which holds the path at 185 MiB/s reads and 273 MiB/s writes
// whatever the device can do. At 1 MiB it measures 1810 and 1854 MiB/s. Must be a multiple of
// LLAMA_IO_BLOCK. See docs/research/27-ssd-prompt-cache-feasibility.md
#define LLAMA_IO_WIN       (1 << 20)

// zlib compatible crc32, so the Python harness and this code agree byte for byte
uint32_t llama_io_crc32(const void * data, size_t size);

// Read a framed store file back with O_DIRECT and validate every block. Returns the number of
// damaged blocks, or SIZE_MAX if the file cannot be read at all. One reused aligned block buffer, so
// the check costs O(1) host memory and never holds the payload resident.
//
// A write can be lost on the way to the device, and a block that was never written reads back as
// whatever held it before. When that previous occupant is an earlier save of the same path, which
// has the same layout, the stale bytes are a structurally valid frame with a matching crc, so the
// per block crc alone cannot see it and the generation is what does. Pass the generation the save
// used, or 0 to adopt the file's own first block. See
// docs/research/23-write-path-loses-blocks-results.md
size_t llama_io_verify_direct(const std::string & path, uint32_t gen);

// Direct I/O writer, one reused aligned block buffer, so host memory stays O(1).
//
// framed = true  packs the byte stream into blocks with a per block header:
//
//     block := [u32 used][u32 crc32(payload area)][u32 gen][payload][zero padding]
//
//   which is what the status log needs, since a bare record stream cannot tell its
//   own padding apart from a torn tail.
//
//   gen is a per save generation, so a block left over from an earlier save of the same file is
//   rejected even though it is a valid frame. Zero is never a valid generation.
//
// framed = false writes whole blocks as given, which is what the pages file needs,
// where each slot is already padded to a block multiple by its own layout.
class llama_io_write_direct : public llama_io_write_i {
public:
    llama_io_write_direct(const std::string & path, bool framed, uint32_t gen = 0);
    ~llama_io_write_direct() override;

    void write(const void * src, size_t size) override;
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override;

    // reserve the first block for a file header. call before any write
    void set_prefix(const void * data, size_t size);

    // seal the final partial block and write out the window. required, the destructor
    // reports if it is missing
    void flush();

    size_t n_bytes()  override { return n_logical; }
    size_t n_blocks() const    { return blocks; }
    bool   good()     const    { return err == 0; }
    int    error()    const    { return err; }
    uint32_t gen()    const    { return gen_; }

private:
    void flush_block();

    // framed blocks are appended to the window and written by the window, so one syscall
    // covers LLAMA_IO_WIN / LLAMA_IO_BLOCK blocks instead of one
    void put_block(size_t off, size_t used);
    void append_window(const void * src, size_t size);
    void flush_window();

    int      fd        = -1;
    bool     framed    = true;
    uint8_t * blk      = nullptr;
    size_t   fill      = LLAMA_IO_BLOCK_HDR;
    size_t   block_off = 0;
    size_t   n_logical = 0;
    size_t   blocks    = 0;
    bool     flushed   = false;
    int      err       = 0;
    uint32_t gen_      = 0;

    uint8_t * win      = nullptr;
    size_t    win_fill = 0;   // bytes framed into the window and not yet written
    size_t    win_off  = 0;   // file offset of win[0]
};

// Reader for the same two layouts, streaming one block at a time.
class llama_io_read_direct : public llama_io_read_i {
public:
    llama_io_read_direct(const std::string & path, bool framed, uint32_t gen = 0);
    ~llama_io_read_direct() override;

    void read(void * dst, size_t size) override;
    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override;

    // read block 0, the file header
    void read_prefix(void * dst, size_t size);

    // require every framed block to carry this generation. call after read_prefix, before the first
    // read, so a block left over from an earlier save is rejected instead of restored.
    void set_gen(uint32_t gen) { gen_ = gen; }

    size_t n_bytes()  override { return n_read; }
    size_t n_blocks() const    { return blocks; }
    bool   good()     const    { return err == 0; }
    int    error()    const    { return err; }

private:
    bool next_block();
    // pull the next window at blk_off. false at end of file or on error, and a read shorter
    // than one block is a torn tail
    bool refill_window();

    int      fd       = -1;
    bool     framed   = true;
    uint8_t * win     = nullptr;
    uint8_t * blk     = nullptr;   // the current block, a window inside win
    size_t   win_len  = 0;   // whole blocks available in the window
    size_t   win_pos  = 0;   // bytes of the window already served
    size_t   avail    = 0;   // payload bytes in the current block
    size_t   pos      = 0;   // payload bytes consumed
    size_t   blk_off  = 0;
    size_t   n_read   = 0;
    size_t   blocks   = 0;
    int      err      = 0;
    bool     eof      = false;
    uint32_t gen_     = 0;   // the save generation to require, or 0 to adopt the first frame's
};
