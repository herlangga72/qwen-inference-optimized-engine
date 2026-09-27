// End to end check for the disk park path: park a real sequence, unpark it, and
// require the continuation to be identical to never having parked.
//
//   run A  decode the prompt, generate 8 tokens greedily                (never parks)
//   run B  decode the prompt, park to disk, clear, unpark, generate 8   (the test)
//
// A and B must produce the same token ids, and the RSS delta across the park and
// unpark must not scale with the session size.
//
// build:
//   g++ -O2 -std=c++17 -I include -I ggml/include scripts/research/kvstore/park_disk_check.cpp \
//       -L build-cpu/bin -lllama -lggml-base -lggml \
//       -Wl,-rpath,$PWD/build-cpu/bin -o <out>

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>
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

static llama_token argmax(const float * logits, int32_t n_vocab) {
    llama_token best = 0;
    float best_v = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}

// one batch of n tokens on seq 0, starting at pos0, logits on the last
static bool decode(llama_context * ctx, const llama_token * toks, int n, int pos0) {
    llama_batch b = llama_batch_init(n, 0, 1);
    // llama_batch_init treats its first argument as the allocation size, not the count
    b.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        b.token[i]    = toks[i];
        b.pos[i]      = pos0 + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]   = (i == n - 1) ? 1 : 0;
    }
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

static std::vector<llama_token> generate(llama_context * ctx, int n, int pos0) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    std::vector<llama_token> out;
    for (int i = 0; i < n; ++i) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            break;
        }
        const llama_token t = argmax(logits, n_vocab);
        out.push_back(t);
        if (!decode(ctx, &t, 1, pos0 + i)) {
            break;
        }
    }
    return out;
}

// a full context prompt is far larger than one batch, so it goes in chunks
static bool decode_fill(llama_context * ctx, const llama_token * toks, int n, int pos0, int n_batch) {
    for (int off = 0; off < n; off += n_batch) {
        const int take = std::min(n_batch, n - off);
        if (!decode(ctx, toks + off, take, pos0 + off)) {
            return false;
        }
    }
    return true;
}

// the store's block size, LLAMA_IO_BLOCK in src/llama-io.h. a private header, so it is repeated
// here, and it has to match or the frame parsing below reads the wrong offsets.
static const size_t STORE_BLOCK = 4096;

static void quiet_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

static uint64_t fnv1a(const uint8_t * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

int main(int argc, char ** argv) {
    // the loader is loud, and this check prints its own report
    if (!getenv("PARK_VERBOSE")) {
        llama_log_set(quiet_log, nullptr);
    }
    const std::string model_path = argc > 1 ? argv[1]
        : "/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf";
    const std::string store_path = argc > 2 ? argv[2]
        : "/home/herlanggays/.jcode/scratch/kvstore/direct/park.bin";

    const int n_words = argc > 3 ? atoi(argv[3]) : 400;
    const int n_gen   = 8;

    bool ok = true;
    auto check = [&](const char * name, bool pass) {
        printf("  %-42s %s\n", name, pass ? "ok" : "FAIL");
        ok = ok && pass;
    };

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        printf("failed to load %s\n", model_path.c_str());
        return 1;
    }

    // 1 max context: a single session filled to the model's own context length, with the KV cache
    // types the capacity assumption is written against. See
    // docs/superpowers/specs/2026-09-27-session-kv-store-design.md
    const int n_ctx = argc > 4 ? atoi(argv[4]) : 262144;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = n_ctx;
    cparams.n_batch   = 2048;
    cparams.n_seq_max = 1;
    cparams.type_k    = GGML_TYPE_Q8_0;
    cparams.type_v    = GGML_TYPE_PLANAR3_0;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        printf("failed to create context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const char * words[] = { "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
                             "golf", "hotel", "india", "juliet", "kilo", "lima" };

    std::vector<llama_token> prompt;
    if (n_words > 0) {
        std::string text;
        for (int i = 0; i < n_words; ++i) {
            text += words[i % 12];
            text += (i % 13 == 12) ? ".\n" : " ";
        }
        prompt.resize(text.size() + 16);
        const int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                                     prompt.data(), (int32_t) prompt.size(), true, true);
        if (n < 0) {
            printf("tokenize failed\n");
            return 1;
        }
        prompt.resize(n);
    } else {
        // no word count given: fill the context, tokenized in chunks so the buffers stay small
        const int target = n_ctx - n_gen - 8;
        std::vector<llama_token> chunk(4096);
        for (int round = 0; (int) prompt.size() < target; ++round) {
            std::string text;
            for (int i = 0; i < 1024; ++i) {
                const int w = round * 1024 + i;
                text += words[w % 12];
                text += (w % 13 == 12) ? ".\n" : " ";
            }
            const int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                                         chunk.data(), (int32_t) chunk.size(),
                                         /*add_special*/ prompt.empty(), /*parse_special*/ true);
            if (n < 0) {
                printf("tokenize failed at round %d\n", round);
                return 1;
            }
            prompt.insert(prompt.end(), chunk.begin(), chunk.begin() + n);
        }
        prompt.resize(target);
    }
    const int n_prompt = (int) prompt.size();
    printf("  prompt tokens: %d, n_ctx %d, KV K q8_0 V planar3_0\n", n_prompt, n_ctx);

    llama_memory_t mem = llama_get_memory(ctx);

    // ---- run A: never parks ----
    llama_memory_clear(mem, true);
    if (!decode_fill(ctx, prompt.data(), n_prompt, 0, cparams.n_batch)) {
        printf("  run A prompt decode FAILED, n_prompt = %d, n_batch = %d, n_ctx = %d\n",
               n_prompt, cparams.n_batch, cparams.n_ctx);
        check("run A prompt decode", false);
        return 1;
    }
    std::vector<llama_token> gen_a = generate(ctx, n_gen, n_prompt);
    check("run A generated tokens", (int) gen_a.size() == n_gen);

    // ---- run B: park to disk and unpark ----
    llama_memory_clear(mem, true);
    if (!decode_fill(ctx, prompt.data(), n_prompt, 0, cparams.n_batch)) {
        check("run B prompt decode", false);
        return 1;
    }

    // Serialize the same state twice into host buffers, without touching the disk or the
    // sink. If these differ, the engine's serialization is unstable on its own.
    // a hash of the pre-save host serialization, kept instead of the full buffer so the RSS
    // measurement below is not inflated by a retained 27 MB copy
    uint64_t pre_save_hash = 0;
    uint64_t pre_save_len  = 0;
    {
        const size_t sz = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
        std::vector<uint8_t> b1(sz);
        std::vector<uint8_t> b2(sz);
        const size_t n1 = llama_state_seq_get_data_ext(ctx, b1.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t n2 = llama_state_seq_get_data_ext(ctx, b2.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
        size_t first = (size_t) -1;
        size_t ndiff = 0;
        for (size_t i = 0; i < std::min(n1, n2); ++i) {
            if (b1[i] != b2[i]) {
                if (first == (size_t) -1) {
                    first = i;
                }
                ndiff++;
            }
        }
        printf("  %-42s sizes %zu and %zu, first_diff=%zd, differing=%zu\n",
               "two host serializations", n1, n2,
               first == (size_t) -1 ? -1 : (long long) first, ndiff);
        check("host serialization is reproducible", ndiff == 0 && n1 == n2);

        // Now the same serialization with a two second gap. If this one differs, the state
        // changes with time and the medium is irrelevant.
        std::vector<uint8_t> b3(sz);
        std::vector<uint8_t> b4(sz);
        llama_state_seq_get_data_ext(ctx, b3.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
        usleep(2000000);
        llama_state_seq_get_data_ext(ctx, b4.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
        size_t first2 = (size_t) -1;
        size_t ndiff2 = 0;
        for (size_t i = 0; i < sz; ++i) {
            if (b3[i] != b4[i]) {
                if (first2 == (size_t) -1) {
                    first2 = i;
                }
                ndiff2++;
            }
        }
        printf("  %-42s first_diff=%zd, differing=%zu\n",
               "two host serializations 2s apart",
               first2 == (size_t) -1 ? -1 : (long long) first2, ndiff2);
        check("host serialization is stable over time", ndiff2 == 0);

        pre_save_hash = fnv1a(b1.data(), b1.size());
        pre_save_len  = b1.size();
    }

    const size_t rss_before = rss_kib();

    const size_t n_written = llama_state_seq_save_file_direct(ctx, store_path.c_str(), 0,
                                                              prompt.data(), prompt.size());
    check("park wrote state", n_written > 0);

    // save the same state three times in the same process. if all three differ it is raw
    // memory nondeterminism, if 2 and 3 agree with each other but not with 1 then the
    // first save has a side effect on whatever the later ones read.
    {
        const std::string p2 = store_path + ".second";
        const std::string p3 = store_path + ".third";
        const char * paths[3] = { store_path.c_str(), p2.c_str(), p3.c_str() };
        size_t sizes[3] = { 0, 0, 0 };
        for (int s = 0; s < 3; ++s) {
            sizes[s] = llama_state_seq_save_file_direct(ctx, paths[s], 0,
                                                        prompt.data(), prompt.size());
            // the sink does not fsync, so make the write durable before it is read back
            int fd = ::open(paths[s], O_RDONLY);
            if (fd >= 0) {
                ::fsync(fd);
                ::close(fd);
            }
        }
        auto drop_pages = [&](const char * p) {
            // the files are written with O_DIRECT. drop any cached pages first, so the
            // buffered comparison below is not reading stale page cache content.
            int fd = ::open(p, O_RDONLY);
            if (fd >= 0) {
                posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                ::close(fd);
            }
        };
        auto diff_pair = [&](int a, int b, size_t * first_out, size_t * ndiff_out) {
            drop_pages(paths[a]);
            drop_pages(paths[b]);
            FILE * fa = fopen(paths[a], "rb");
            FILE * fb = fopen(paths[b], "rb");
            size_t off = 0, first = (size_t) -1, ndiff = 0;
            if (fa && fb) {
                int ca, cb;
                while ((ca = fgetc(fa)) != EOF && (cb = fgetc(fb)) != EOF) {
                    // every save carries its own generation, so that field differs between two saves
                    // of the same state by design and is not a difference in the data. block 0 is the
                    // file header, where it sits after the caller's four words, and its crc covers
                    // only those four, so only the generation moves between blocks
                    const size_t in_block    = off % STORE_BLOCK;
                    const bool   hdr_block   = (off / STORE_BLOCK) == 0;
                    const bool   gen_field   = hdr_block ? (in_block >= 16 && in_block < 20)
                                                         : (in_block >= 8 && in_block < 12);
                    if (ca != cb && !gen_field) {
                        if (first == (size_t) -1) {
                            first = off;
                        }
                        ndiff++;
                    }
                    off++;
                }
                fclose(fa);
                fclose(fb);
            }
            *first_out = first;
            *ndiff_out = ndiff;
        };
        size_t f12, d12, f23, d23;
        diff_pair(0, 1, &f12, &d12);
        diff_pair(1, 2, &f23, &d23);

        // Compare only the used payload area of each block, ignoring the padding. The reader
        // ignores the padding, so a difference there would not affect a restore.
        auto payload_diff = [&](int a, int b) {
            drop_pages(paths[a]);
            drop_pages(paths[b]);
            FILE * fa = fopen(paths[a], "rb");
            FILE * fb = fopen(paths[b], "rb");
            size_t nd = 0;
            if (!fa || !fb) {
                return nd;
            }
            std::vector<unsigned char> ba(STORE_BLOCK);
            std::vector<unsigned char> bb(STORE_BLOCK);
            // block 0 is the file header, so the frame header offsets below do not apply to it and
            // its generation would read as a payload difference
            if (fread(ba.data(), 1, STORE_BLOCK, fa) != STORE_BLOCK ||
                fread(bb.data(), 1, STORE_BLOCK, fb) != STORE_BLOCK) {
                fclose(fa);
                fclose(fb);
                return nd;
            }
            while (fread(ba.data(), 1, STORE_BLOCK, fa) == STORE_BLOCK &&
                   fread(bb.data(), 1, STORE_BLOCK, fb) == STORE_BLOCK) {
                uint32_t ua = 0, ub = 0;
                memcpy(&ua, ba.data(), 4);
                memcpy(&ub, bb.data(), 4);
                const size_t n = ua < ub ? ua : ub;
                for (size_t i = 12; i < 12 + n && i < STORE_BLOCK; ++i) {
                    if (ba[i] != bb[i]) {
                        nd++;
                    }
                }
                if (ua != ub) {
                    nd++;
                }
            }
            fclose(fa);
            fclose(fb);
            return nd;
        };
        const size_t p12 = payload_diff(0, 1);
        printf("  %-42s padding only: 1v2 %zu, 2v3 %zu\n", "payload only differences",
               p12, payload_diff(1, 2));
        check("the payload area is identical across saves", p12 == 0);

        // Re-serialize after the saves and compare a hash against the snapshot taken before
        // them. If the state changed across the saves, the engine is mutating it.
        {
            const size_t sz2 = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
            std::vector<uint8_t> post(sz2);
            const size_t npost = llama_state_seq_get_data_ext(ctx, post.data(), sz2, 0,
                                                             LLAMA_STATE_SEQ_FLAGS_NONE);
            const uint64_t post_hash = fnv1a(post.data(), npost);
            printf("  %-42s hash %s, sizes %zu and %zu\n",
                   "host serialization before vs after saves",
                   post_hash == pre_save_hash ? "equal" : "DIFFERENT",
                   (size_t) pre_save_len, npost);
            check("the state is unchanged across the saves",
                  post_hash == pre_save_hash && npost == (size_t) pre_save_len);
        }
        printf("  %-42s sizes %zu %zu %zu\n", "three saves", sizes[0], sizes[1], sizes[2]);
        printf("  %-42s 1v2 first=%zd diff=%zu | 2v3 first=%zd diff=%zu\n",
               "pairwise differences", f12 == (size_t) -1 ? -1 : (long long) f12, d12,
               f23 == (size_t) -1 ? -1 : (long long) f23, d23);
        check("saving the same state is reproducible", d12 == 0 && d23 == 0 &&
                                                      sizes[0] == sizes[1] && sizes[1] == sizes[2]);
    }

    // Stop before any unpark. Every save after this point is a consecutive save with nothing in
    // between, so the sink's window hash comparison is unambiguous about which round it is.
    if (getenv("PARK_NO_UNPARK")) {
        printf("\n  PARK_NO_UNPARK set: stopping before the unpark\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        printf("  overall: %s\n", ok ? "pass" : "FAIL");
        return ok ? 0 : 1;
    }

    struct stat st;
    const bool have_stat = stat(store_path.c_str(), &st) == 0;
    check("store file padded to blocks", have_stat && st.st_size % STORE_BLOCK == 0);
    printf("  %-42s %zu logical bytes, %lld on disk\n", "park size",
           n_written, have_stat ? (long long) st.st_size : -1LL);

    // drop the sequence, so the unpark has to come off the disk
    llama_memory_seq_rm(mem, 0, -1, -1);
    const llama_pos pos_max_after_rm = llama_memory_seq_pos_max(mem, 0);
    printf("  %-42s pos_max after seq_rm = %d\n", "arena state", (int) pos_max_after_rm);
    check("sequence empty after seq_rm", pos_max_after_rm == -1);

    std::vector<llama_token> tokens_out(prompt.size());
    size_t n_out = 0;
    size_t n_read = llama_state_seq_load_file_direct(ctx, store_path.c_str(), 0,
                                                     tokens_out.data(), tokens_out.size(),
                                                     &n_out);
    if (n_read == 0) {
        // the restore failed, so try again and report whether it is the file or the
        // destination that is the problem
        llama_memory_seq_rm(mem, 0, -1, -1);
        n_out = 0;
        const size_t n_retry = llama_state_seq_load_file_direct(ctx, store_path.c_str(), 0,
                                                                tokens_out.data(), tokens_out.size(),
                                                                &n_out);
        printf("  %-42s first=0 retry=%zu\n", "unpark retry", n_retry);
        n_read = n_retry;
    }
    const size_t rss_after = rss_kib();

    check("unpark read state", n_read > 0);
    check("token count round tripped", n_out == prompt.size());
    check("tokens round tripped", tokens_out == prompt);

    std::vector<llama_token> gen_b = generate(ctx, n_gen, n_prompt);
    check("run B generated tokens", (int) gen_b.size() == n_gen);
    check("continuation identical to never parking", gen_a == gen_b);

    if (gen_a != gen_b) {
        printf("  A:");
        for (auto t : gen_a) printf(" %d", t);
        printf("\n  B:");
        for (auto t : gen_b) printf(" %d", t);
        printf("\n");
    }

    const size_t delta = rss_after > rss_before ? rss_after - rss_before : rss_before - rss_after;
    printf("  %-42s %zu KiB while the state is %zu KiB (informational, diagnostics run inside this window)\n",
           "rss delta across park and unpark", delta, n_written / 1024);

    // ---- the RSS check, isolated: one park and unpark with nothing else in the window ----
    {
        const std::string path3 = store_path + ".rss";
        const size_t r0 = rss_kib();
        const size_t n3 = llama_state_seq_save_file_direct(ctx, path3.c_str(), 0,
                                                           prompt.data(), prompt.size());
        llama_memory_seq_rm(mem, 0, -1, -1);
        std::vector<llama_token> t3(prompt.size());
        size_t n3out = 0;
        llama_state_seq_load_file_direct(ctx, path3.c_str(), 0, t3.data(), t3.size(), &n3out);
        const size_t r1 = rss_kib();
        const size_t d3 = r1 > r0 ? r1 - r0 : r0 - r1;
        printf("  %-42s %zu KiB while the state is %zu KiB\n",
               "rss delta, isolated park and unpark", d3, n3 / 1024);
        check("rss does not scale with the state", d3 < 4096);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("\n  overall: %s\n", ok ? "pass" : "FAIL");
    return ok ? 0 : 1;
}
