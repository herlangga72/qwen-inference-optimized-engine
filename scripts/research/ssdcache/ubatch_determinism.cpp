// Determinism gate for the SSD prompt cache: does the same token prefix
// processed under different ubatch/chunk splits serialize to the same sequence
// state? If it does not, a cache hit reproduces the computation that wrote the
// entry, not a recomputation, and the cache's transparency claim has to weaken.
//
//   A  n_ubatch 512, prompt decoded in chunks of 512
//   B  n_ubatch 512, prompt decoded in chunks of 128
//   C  n_ubatch 128, prompt decoded in chunks of 512
//   D  n_ubatch 256, prompt decoded in chunks of 256
//
// Three environment variables turn the same harness into probes for document 31:
//   UBDET_PAINT=<byte>  paint the stack with that byte before every llama_decode (test 5)
//   UBDET_CSR=1         print MXCSR at entry and after each prompt decode (test 6)
//   UBDET_VERBOSE=1     do not silence the library log
//
// The state a run produces also depends on whether the Vulkan driver is loaded, even at
// n_gpu_layers = 0, because the backend registry initialises the device at startup. See
// document 31, test 6: VK_ICD_FILENAMES=/nonexistent/icd.json makes every run, thread
// count and configuration agree on one state, and it agrees with a GGML_VULKAN=OFF build.
//
// Control: serialize twice inside every configuration and compare. Document 22
// shows a naive harness here can be wrong, so the control is mandatory: if it
// reports a difference, the harness is unusable before the gate means anything.
//
// Every configuration also generates 8 greedy tokens from its own state, so a
// state difference would also show up as a behavioural difference.
//
// build:
//   g++ -O2 -std=c++17 -I include -I ggml/include \
//       scripts/research/ssdcache/ubatch_determinism.cpp \
//       -L build-vk/bin -lllama -lggml-base -lggml \
//       -Wl,-rpath,$PWD/build-vk/bin -o <out>

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/types.h>

#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
// MXCSR is per thread and the FPU modes in it change results: bit 15 flush to
// zero, bit 6 denormals are zero. Print it when UBDET_CSR is set, so a run can
// say whether the Vulkan driver left the CPU in a different mode.
static void print_csr(const char * where) {
    if (!getenv("UBDET_CSR")) {
        return;
    }
    const unsigned int csr = _mm_getcsr();
    printf("  mxcsr %-22s 0x%08x ftz=%d daz=%d\n", where, csr,
           (csr >> 15) & 1, (csr >> 6) & 1);
}
#else
static void print_csr(const char *) {}
#endif

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

// The stack analogue of MALLOC_PERTURB_, which document 31 uses for the heap.
// Paint the calling thread's stack with a chosen byte before a decode: if an
// uninitialized stack read is what makes runs disagree, a deterministic stack
// should make them agree, and changing the byte should change the answer.
// UBDET_PAINT=<byte> enables it; depth 128 frames of 16 KiB covers 2 MiB, well
// inside the 8 MiB default stack.
static volatile unsigned char g_paint_sink;

static void paint_stack(unsigned char v, int depth) {
    volatile unsigned char buf[16384];
    for (size_t i = 0; i < sizeof(buf); ++i) {
        buf[i] = v;
    }
    if (depth > 0) {
        paint_stack(v, depth - 1);
    }
    g_paint_sink = buf[(size_t) depth % sizeof(buf)];
}

// one llama_decode call over n tokens on seq 0, positions pos0..pos0+n-1.
// logits are requested only for the last token when want_logits is set.
static bool decode_chunk(llama_context * ctx, const llama_token * toks, int n, int pos0,
                         bool want_logits) {
    const char * pv = getenv("UBDET_PAINT");
    if (pv) {
        paint_stack((unsigned char) strtoul(pv, nullptr, 0), 128);
    }
    llama_batch b = llama_batch_init(n, 0, 1);
    b.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        b.token[i]        = toks[i];
        b.pos[i]          = pos0 + i;
        b.n_seq_id[i]     = 1;
        b.seq_id[i][0]    = 0;
        b.logits[i]       = (want_logits && i == n - 1) ? 1 : 0;
    }
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

// the whole prompt, in explicit chunks of chunk_size, no internal help
static bool decode_prompt(llama_context * ctx, const llama_token * toks, int n, int chunk_size) {
    for (int off = 0; off < n; off += chunk_size) {
        const int take  = (n - off < chunk_size) ? n - off : chunk_size;
        const bool last = (off + take == n);
        if (!decode_chunk(ctx, toks + off, take, off, last)) {
            return false;
        }
    }
    return true;
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
        if (!decode_chunk(ctx, &t, 1, pos0 + i, true)) {
            break;
        }
    }
    return out;
}

struct Diff {
    size_t n_diff  = 0;
    size_t first   = (size_t) -1;
    bool   len_eq  = true;
};

static Diff compare(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    Diff d;
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    d.len_eq = (a.size() == b.size());
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            if (d.first == (size_t) -1) {
                d.first = i;
            }
            d.n_diff++;
        }
    }
    if (!d.len_eq) {
        d.n_diff += (a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
    }
    return d;
}

static void print_tokens(const char * label, const std::vector<llama_token> & toks) {
    printf("%s", label);
    for (size_t i = 0; i < toks.size(); ++i) {
        printf("%s%d", i ? " " : "", (int) toks[i]);
    }
    printf("\n");
}

int main(int argc, char ** argv) {
    if (!getenv("UBDET_VERBOSE")) {
        llama_log_set(quiet_log, nullptr);
    }

    const std::string model_path = argc > 1 ? argv[1]
        : "/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf";
    const int n_prompt_target = argc > 2 ? atoi(argv[2]) : 1500;
    const int n_threads_arg   = argc > 3 ? atoi(argv[3]) : 0;  // 0 leaves the library default
    const int n_ctx_arg       = argc > 4 ? atoi(argv[4]) : 2048;
    const char * only_cfg     = argc > 5 ? argv[5] : nullptr;  // run one configuration alone
    const int n_gen           = 8;

    print_csr("main entry");
    llama_backend_init();
    print_csr("after backend_init");

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU: small footprint, no GTT pressure
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        printf("failed to load %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Word-list prompt, as in scripts/research/kvstore/park_disk_check.cpp. Built once,
    // before any context exists, so every configuration sees the identical token vector.
    const char * words[] = { "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
                             "golf", "hotel", "india", "juliet", "kilo", "lima" };
    std::vector<llama_token> prompt;
    {
        std::string text;
        for (int i = 0; i < 2000; ++i) {
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
        if ((int) prompt.size() > n_prompt_target) {
            prompt.resize(n_prompt_target);
        }
    }
    const int n_prompt = (int) prompt.size();
    printf("model: %s\n", model_path.c_str());
    printf("prompt tokens: %d (target %d), first %d last %d\n",
           n_prompt, n_prompt_target, (int) prompt[0], (int) prompt[n_prompt - 1]);
    printf("n_ctx %d n_threads %d n_vocab %d\n",
           n_ctx_arg, n_threads_arg, (int) llama_vocab_n_tokens(vocab));
    printf("\n");

    struct Cfg {
        const char * name;
        int          ubatch;
        int          chunk;
    };
    const Cfg cfgs[] = {
        { "A",  512, 512 },
        { "B",  512, 128 },
        { "C",  128, 512 },
        { "D",  256, 256 },
        { "A2", 512, 512 }, // A repeated in a second fresh context: is an identical rerun stable?
    };
    const int n_cfg = (int) (sizeof(cfgs) / sizeof(cfgs[0]));

    std::vector<std::vector<uint8_t>> states(n_cfg);
    std::vector<std::vector<llama_token>> gens(n_cfg);
    std::vector<int> selected;
    for (int c = 0; c < n_cfg; ++c) {
        if (!only_cfg || strcmp(only_cfg, cfgs[c].name) == 0) {
            selected.push_back(c);
        }
    }

    for (int ci = 0; ci < (int) selected.size(); ++ci) {
        const int c = selected[ci];
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx     = n_ctx_arg;
        cparams.n_batch   = 2048;
        cparams.n_ubatch  = cfgs[c].ubatch;
        cparams.n_seq_max = 1;
        if (n_threads_arg > 0) {
            cparams.n_threads       = n_threads_arg;
            cparams.n_threads_batch = n_threads_arg;
        }
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            printf("config %s: failed to create context\n", cfgs[c].name);
            return 1;
        }

        if (!decode_prompt(ctx, prompt.data(), n_prompt, cfgs[c].chunk)) {
            printf("config %s: prompt decode FAILED (ubatch %d chunk %d)\n",
                   cfgs[c].name, cfgs[c].ubatch, cfgs[c].chunk);
            llama_free(ctx);
            return 1;
        }
        {
            char lbl[64];
            snprintf(lbl, sizeof lbl, "after prompt %s", cfgs[c].name);
            print_csr(lbl);
        }

        // Logits after the prompt are the semantic output of the used state. They are not
        // polluted by never-written cache regions, so hashing them separates "the computation
        // differs" from "the serialized blob contains unused bytes that differ".
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        const float * lg = llama_get_logits_ith(ctx, -1);
        const uint64_t logits_hash = lg ? fnv1a((const uint8_t *) lg,
                                                (size_t) n_vocab * sizeof(float)) : 0;
        const llama_token first_tok = lg ? argmax(lg, n_vocab) : -1;

        // serialize twice: the control. Same config, same state, back to back.
        const size_t sz = llama_state_seq_get_size_ext(ctx, 0, 0);
        std::vector<uint8_t> b1(sz);
        std::vector<uint8_t> b2(sz);
        const size_t n1 = llama_state_seq_get_data_ext(ctx, b1.data(), sz, 0, 0);
        const size_t n2 = llama_state_seq_get_data_ext(ctx, b2.data(), sz, 0, 0);
        b1.resize(n1);
        b2.resize(n2);
        const Diff ctrl = compare(b1, b2);

        printf("config %s ubatch=%d chunk=%d threads=%d state_size=%zu hash=%016llx len=%zu"
               " ctrl_hash=%016llx ctrl_len=%zu ctrl_diff=%zu first=%zd"
               " logits_hash=%016llx first_tok=%d\n",
               cfgs[c].name, cfgs[c].ubatch, cfgs[c].chunk, cparams.n_threads, sz,
               (unsigned long long) fnv1a(b1.data(), b1.size()), b1.size(),
               (unsigned long long) fnv1a(b2.data(), b2.size()), b2.size(),
               ctrl.n_diff, ctrl.first == (size_t) -1 ? (ssize_t) -1 : (ssize_t) ctrl.first,
               (unsigned long long) logits_hash, (int) first_tok);

        gens[c] = generate(ctx, n_gen, n_prompt);
        states[c] = std::move(b1);
        llama_free(ctx);
    }

    printf("\n");
    for (int ci = 0; ci < (int) selected.size(); ++ci) {
        const int c = selected[ci];
        char label[64];
        snprintf(label, sizeof(label), "gen %s: ", cfgs[c].name);
        print_tokens(label, gens[c]);
    }

    printf("\n");
    printf("pair    bytes     diff       first_offset  fraction\n");
    const size_t bytes = states[selected.empty() ? 0 : selected[0]].size();
    for (int si = 0; si < (int) selected.size(); ++si) {
        for (int sj = si + 1; sj < (int) selected.size(); ++sj) {
            const int i = selected[si];
            const int j = selected[sj];
            const Diff d = compare(states[i], states[j]);
            printf("%-3s-%-3s %9zu %9zu  %12zd  %.6f%%\n", cfgs[i].name, cfgs[j].name, bytes,
                   d.n_diff, d.first == (size_t) -1 ? (ssize_t) -1 : (ssize_t) d.first,
                   bytes ? 100.0 * (double) d.n_diff / (double) bytes : 0.0);
        }
    }

    // coarse location map: differing bytes per 1 MiB bucket, so a difference that lives in one
    // tensor region is distinguishable from one spread across the whole blob
    printf("\nbucket map, differing bytes per 1 MiB bucket (nonzero buckets only, up to 10)\n");
    const size_t BUCKET = 1024 * 1024;
    for (int si = 0; si < (int) selected.size(); ++si) {
        for (int sj = si + 1; sj < (int) selected.size(); ++sj) {
            const int i = selected[si];
            const int j = selected[sj];
            std::vector<size_t> buckets((bytes + BUCKET - 1) / BUCKET, 0);
            const size_t n = states[i].size() < states[j].size() ? states[i].size() : states[j].size();
            for (size_t k = 0; k < n; ++k) {
                if (states[i][k] != states[j][k]) {
                    buckets[k / BUCKET]++;
                }
            }
            size_t nonzero = 0;
            for (size_t v : buckets) {
                nonzero += (v != 0);
            }
            printf("%-3s-%-3s nonzero_buckets=%zu/%zu  ", cfgs[i].name, cfgs[j].name,
                   nonzero, buckets.size());
            int shown = 0;
            for (size_t b = 0; b < buckets.size() && shown < 10; ++b) {
                if (buckets[b]) {
                    printf("[%zu MiB]=%zu ", b, buckets[b]);
                    shown++;
                }
            }
            if ((size_t) shown < nonzero) {
                printf("...");
            }
            printf("\n");
        }
    }

    // verdict, machine readable
    bool all_identical = true;
    for (int si = 0; si < (int) selected.size(); ++si) {
        for (int sj = si + 1; sj < (int) selected.size(); ++sj) {
            if (compare(states[selected[si]], states[selected[sj]]).n_diff != 0) {
                all_identical = false;
            }
        }
    }
    printf("\nVERDICT: %s\n", all_identical ? "IDENTICAL" : "DIFFERS");

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
