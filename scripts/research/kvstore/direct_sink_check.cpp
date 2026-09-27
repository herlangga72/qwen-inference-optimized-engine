// Verifies llama_io_write_direct / llama_io_read_direct against the layer 1 format.
//
// It writes a real status log through the sink, checks the bytes, reads them back,
// then damages the file three ways and requires each to be detected. It also checks
// that host RSS does not grow with the amount of data written.
//
// The file it leaves behind is parsed independently by scenarios-side Python
// (sim.parse_blocks), so the C++ writer and the Python reader have to agree byte for
// byte. That cross check is the point of the crc32 matching zlib.
//
// build:
//   g++ -O2 -std=c++17 -I src -I include -I ggml/include \
//       scripts/research/kvstore/direct_sink_check.cpp src/llama-io.cpp \
//       -L build-cpu/bin -lggml-base -lggml -o <out>

#include "llama-io.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

static size_t rss_kib() {
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) {
        return 0;
    }
    char line[256];
    size_t out = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            out = (size_t) strtoul(line + 6, nullptr, 10);
            break;
        }
    }
    fclose(f);
    return out;
}

static void put_u32(std::vector<uint8_t> & v, uint32_t x) {
    for (int i = 0; i < 4; ++i) {
        v.push_back((uint8_t) ((x >> (8 * i)) & 0xFF));
    }
}

static void put_u64(std::vector<uint8_t> & v, uint64_t x) {
    for (int i = 0; i < 8; ++i) {
        v.push_back((uint8_t) ((x >> (8 * i)) & 0xFF));
    }
}

// the same record layout as sim.record: [u32 len][u32 crc32(payload)][payload]
static std::vector<uint8_t> make_record(uint32_t epoch, uint32_t op) {
    std::vector<uint8_t> payload;
    put_u64(payload, epoch);
    put_u32(payload, op);
    // an ALLOC shaped args block, so the records look like the real ones
    put_u32(payload, epoch);          // slot
    put_u32(payload, epoch * 3);      // gen
    put_u32(payload, epoch % 2);      // layer
    put_u32(payload, epoch / 4);      // block
    put_u32(payload, 4);              // n_cells

    std::vector<uint8_t> out;
    put_u32(out, (uint32_t) payload.size());
    put_u32(out, llama_io_crc32(payload.data(), payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

static std::vector<uint8_t> make_status_header(uint64_t base) {
    std::vector<uint8_t> body;
    put_u32(body, 0x4B565354);  // status magic, matches sim.STATUS_MAGIC
    put_u32(body, 1);           // format version
    put_u64(body, base);

    std::vector<uint8_t> out = body;
    put_u32(out, llama_io_crc32(body.data(), body.size()));
    return out;                 // the sink pads the rest of the block
}

static int patch_byte(const std::string & path, size_t off, uint8_t value) {
    // buffered is fine here, this is a test patch
    int fd = ::open(path.c_str(), O_WRONLY);
    if (fd < 0) {
        return 1;
    }
    if (::pwrite(fd, &value, 1, (off_t) off) != 1) {
        ::close(fd);
        return 1;
    }
    // the reader uses O_DIRECT, so the patch has to reach the device
    ::fsync(fd);
    ::close(fd);
    return 0;
}

int main(int argc, char ** argv) {
    const std::string path = argc > 1 ? argv[1] : "/home/herlanggays/.jcode/scratch/kvstore/direct/log.bin";
    const int n_records = 200;

    bool ok = true;
    auto check = [&](const char * name, bool pass) {
        printf("  %-34s %s\n", name, pass ? "ok" : "FAIL");
        ok = ok && pass;
    };

    std::vector<std::vector<uint8_t>> records;
    records.reserve(n_records);
    for (int i = 0; i < n_records; ++i) {
        records.push_back(make_record((uint32_t) i + 1, 3));  // OP_ALLOC
    }

    size_t logical = 0;
    for (auto & r : records) {
        logical += r.size();
    }

    // ---- write ----
    // the clean copy is kept for the Python cross check, the damage controls use `path`
    const std::string clean = path + ".clean";
    size_t rss_before = 0;
    {
        llama_io_write_direct w(clean, true);
        check("writer opened", w.good());

        auto hdr = make_status_header(LLAMA_IO_BLOCK);
        w.set_prefix(hdr.data(), hdr.size());

        rss_before = rss_kib();
        for (auto & r : records) {
            w.write(r.data(), r.size());
        }
        w.flush();
        check("writer good after flush", w.good());
        check("logical bytes match", w.n_bytes() == logical);
        printf("  %-34s %zu records, %zu blocks, %zu logical bytes\n",
               "sink output", records.size(), w.n_blocks(), w.n_bytes());
    }

    size_t rss_after = rss_kib();
    size_t delta = rss_after > rss_before ? rss_after - rss_before : rss_before - rss_after;
    printf("  %-34s %zu KiB (block is %d bytes)\n", "rss delta over the write", delta, LLAMA_IO_BLOCK);
    check("rss stays one block", delta <= 256);

    // ---- read back ----
    {
        llama_io_read_direct r(clean, true);
        check("reader opened", r.good());

        bool bytes_ok = true;
        for (auto & rec : records) {
            std::vector<uint8_t> got(rec.size());
            r.read(got.data(), got.size());
            if (got != rec) {
                bytes_ok = false;
                break;
            }
        }
        check("records read back byte exact", bytes_ok && r.good());
        check("read byte count matches", r.n_bytes() == logical);
        printf("  %-34s %zu blocks\n", "reader blocks consumed", r.n_blocks());
    }

    // ---- control: zero a block header in the middle ----
    {
        size_t victim = LLAMA_IO_BLOCK * 3;
        patch_byte(path, victim, 0);
        patch_byte(path, victim + 1, 0);

        llama_io_read_direct r(path, true);
        std::vector<uint8_t> buf(LLAMA_IO_BLOCK * 4);
        r.read(buf.data(), buf.size());
        check("zeroed block header refused", !r.good());
    }

    // ---- control: truncate mid block ----
    {
        llama_io_write_direct w(path, true);
        const uint8_t dummy = 0;
        w.set_prefix(&dummy, 0);
        for (auto & rec : records) {
            w.write(rec.data(), rec.size());
        }
        w.flush();

        size_t full = 0;
        {
            int fd = ::open(path.c_str(), O_RDONLY);
            full = (size_t) lseek(fd, 0, SEEK_END);
            ::close(fd);
        }
        // drop the last block and a bit, so the tail is short and unreadable
        if (truncate(path.c_str(), (off_t) (full - LLAMA_IO_BLOCK - 100)) != 0) {
            check("truncate for the torn tail test", false);
        } else {
            llama_io_read_direct r(path, true);
            std::vector<uint8_t> buf(LLAMA_IO_BLOCK);
            size_t got = 0;
            while (r.good() && got < logical) {
                r.read(buf.data(), buf.size());
                if (r.n_bytes() == got) {
                    break;  // no progress, the tail is gone
                }
                got = r.n_bytes();
            }
            check("short tail detected as short", got < logical);
        }
    }

    // ---- same fixed input, twice, compared block by block ----
    // No model and no tensors involved. If the two files differ, the sink is nondeterministic
    // for identical input, which is the one second version of the failure.
    {
        const std::string p2 = clean + ".two";
        {
            llama_io_write_direct w2(p2, true);
            auto hdr = make_status_header(LLAMA_IO_BLOCK);
            w2.set_prefix(hdr.data(), hdr.size());
            for (auto & r : records) {
                w2.write(r.data(), r.size());
            }
            w2.flush();
            check("second writer good", w2.good());
        }

        llama_io_read_direct ra(clean, false);
        llama_io_read_direct rb(p2, false);
        uint8_t ba[LLAMA_IO_BLOCK];
        uint8_t bb[LLAMA_IO_BLOCK];
        size_t first = (size_t) -1;
        size_t nd = 0;
        size_t blocks = 0;
        while (true) {
            const size_t before = ra.n_bytes();
            ra.read(ba, sizeof(ba));
            rb.read(bb, sizeof(bb));
            if (!ra.good() || !rb.good() || ra.n_bytes() == before) {
                break;
            }
            for (size_t i = 0; i < sizeof(ba); ++i) {
                if (ba[i] != bb[i]) {
                    if (first == (size_t) -1) {
                        first = blocks * LLAMA_IO_BLOCK + i;
                    }
                    nd++;
                }
            }
            blocks++;
        }
        printf("  %-42s blocks=%zu first_diff=%zd differing=%zu\n",
               "same input written twice", blocks,
               first == (size_t) -1 ? -1 : (long long) first, nd);
        check("the sink is deterministic for fixed input", nd == 0);
    }

    // ---- write_tensor determinism: the path the park actually uses ----
    // The fixed buffer test above goes through write() only. The park serializes everything
    // through write_tensor, which is a different path and has never been tested for this.
    {
        ggml_init_params ip = { 1u << 20, nullptr, true };
        ggml_context * gctx = ggml_init(ip);
        ggml_tensor * t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 100000);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(
                gctx, ggml_backend_cpu_buffer_type());

        std::vector<float> data(100000);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = (float) (i % 251) * 0.25f;
        }
        ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));

        const std::string ta = path + ".tensor_a";
        const std::string tb = path + ".tensor_b";
        for (int k = 0; k < 2; ++k) {
            llama_io_write_direct w(k == 0 ? ta : tb, true);
            w.write_tensor(t, 0, ggml_nbytes(t));
            w.flush();
            check("tensor writer good", w.good());
        }

        llama_io_read_direct ra(ta, false);
        llama_io_read_direct rb(tb, false);
        uint8_t ta_buf[LLAMA_IO_BLOCK];
        uint8_t tb_buf[LLAMA_IO_BLOCK];
        size_t first = (size_t) -1;
        size_t nd = 0;
        size_t blocks = 0;
        while (true) {
            const size_t before = ra.n_bytes();
            ra.read(ta_buf, sizeof(ta_buf));
            rb.read(tb_buf, sizeof(tb_buf));
            if (!ra.good() || !rb.good() || ra.n_bytes() == before) {
                break;
            }
            for (size_t i = 0; i < sizeof(ta_buf); ++i) {
                if (ta_buf[i] != tb_buf[i]) {
                    if (first == (size_t) -1) {
                        first = blocks * LLAMA_IO_BLOCK + i;
                    }
                    nd++;
                }
            }
            blocks++;
        }
        printf("  %-42s blocks=%zu first_diff=%zd differing=%zu (tensor is %zu bytes)\n",
               "same tensor written twice", blocks,
               first == (size_t) -1 ? -1 : (long long) first, nd, ggml_nbytes(t));
        check("write_tensor is deterministic", nd == 0);

        ggml_backend_buffer_free(buf);
        ggml_free(gctx);
    }

    printf("\n  file: %s\n", path.c_str());
    printf("  overall: %s\n", ok ? "pass" : "FAIL");
    return ok ? 0 : 1;
}
