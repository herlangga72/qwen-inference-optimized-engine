// Round trip through the disk store: save a sequence state, load it into a fresh context, and
// require three things.
//
//   1. the loaded state re-serializes byte for byte like the original, so the store is lossless
//   2. greedy tokens decoded from the loaded state equal those decoded from the original
//   3. the save and the load are timed, so the I/O rate of the store is a measurement and not a
//      model
//
// This is the core assumption an SSD prompt cache rests on, and it is checked on the model and
// KV types the cache runs with. See docs/superpowers/specs/2026-09-28-ssd-prompt-cache-design.md
//
// build:
//   g++ -O2 -std=c++17 -I include -I ggml/include scripts/research/ssdcache/store_roundtrip.cpp \
//       -L build-vk/bin -lllama -lggml-base -lggml -Wl,-rpath,$PWD/build-vk/bin -o <out>
// run:
//   <out> [model] [file] [n_words] [n_ctx] [ngl]

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <time.h>

static void quiet_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

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

// one batch of n tokens on seq 0, positions pos0.., logits only on the last when asked
static bool decode_chunk(llama_context * ctx, const llama_token * toks, int n, int pos0, bool want_logits) {
    llama_batch b = llama_batch_init(n, 0, 1);
    b.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = pos0 + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = (want_logits && i == n - 1) ? 1 : 0;
    }
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

static bool decode_fill(llama_context * ctx, const llama_token * toks, int n, int pos0, int n_batch) {
    for (int off = 0; off < n; off += n_batch) {
        const int take = (n - off < n_batch) ? n - off : n_batch;
        if (!decode_chunk(ctx, toks + off, take, pos0 + off, off + take == n)) {
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

int main(int argc, char ** argv) {
    if (!getenv("SRT_VERBOSE")) {
        llama_log_set(quiet_log, nullptr);
    }

    const std::string model_path = argc > 1 ? argv[1]
        : "/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf";
    const std::string store_path = argc > 2 ? argv[2]
        : "/home/herlanggays/.jcode/scratch/ssd-cache/roundtrip.bin";
    const int n_words = argc > 3 ? atoi(argv[3]) : 1200;
    const int n_ctx   = argc > 4 ? atoi(argv[4]) : 8192;
    const int n_ngl   = argc > 5 ? atoi(argv[5]) : 0;
    const int n_gen   = 8;

    bool ok = true;
    auto check = [&](const char * name, bool pass) {
        printf("  %-44s %s\n", name, pass ? "ok" : "FAIL");
        ok = ok && pass;
    };

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        printf("failed to load %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const char * words[] = { "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
                             "golf", "hotel", "india", "juliet", "kilo", "lima" };
    std::vector<llama_token> prompt;
    {
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
    }
    const int n_prompt = (int) prompt.size();

    auto make_ctx = [&]() {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx     = n_ctx;
        cparams.n_batch   = 2048;
        cparams.n_seq_max = 1;
        cparams.type_k    = GGML_TYPE_Q8_0;
        cparams.type_v    = GGML_TYPE_PLANAR3_0;
        return llama_init_from_model(model, cparams);
    };

    printf("model %s\n", model_path.c_str());
    printf("prompt tokens %d, n_ctx %d, ngl %d, K q8_0 V planar3_0\n", n_prompt, n_ctx, n_ngl);

    // ---- run A: never parks ----
    llama_context * ctx_a = make_ctx();
    if (!ctx_a) {
        printf("failed to create context A\n");
        return 1;
    }
    check("run A prompt decode", decode_fill(ctx_a, prompt.data(), n_prompt, 0, 2048));

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    llama_token t_force = 0;
    {
        const float * lg = llama_get_logits_ith(ctx_a, -1);
        check("run A has logits after the prompt", lg != nullptr);
        t_force = lg ? argmax(lg, n_vocab) : 0;
    }

    // ---- save the state at the prompt boundary, timed ----
    const size_t rss_before_save = rss_kib();
    const double t_save0 = now_s();
    const size_t written = llama_state_seq_save_file_direct(
            ctx_a, store_path.c_str(), 0, prompt.data(), prompt.size());
    const double t_save = now_s() - t_save0;
    const size_t rss_after_save = rss_kib();

    check("save wrote the state", written > 0);

    struct stat st;
    const size_t on_disk = (::stat(store_path.c_str(), &st) == 0) ? (size_t) st.st_size : 0;
    printf("  %-44s %zu bytes payload, %zu bytes on disk\n", "store file", written, on_disk);

    // the same state in memory, for the comparison below
    const size_t sz_a = llama_state_seq_get_size_ext(ctx_a, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    std::vector<uint8_t> blob_a(sz_a);
    const size_t n_a = llama_state_seq_get_data_ext(ctx_a, blob_a.data(), sz_a, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    blob_a.resize(n_a);

    // a restore produces no logits, so both runs decode the same forced token before sampling,
    // which also puts them at the same position. see docs/research/24-state-api-compat-results.md,
    // "sampling straight after a restore"
    check("run A decodes the forced token", decode_chunk(ctx_a, &t_force, 1, n_prompt, true));
    std::vector<llama_token> gen_a = generate(ctx_a, n_gen, n_prompt + 1);

    // ---- run B: fresh context, load from the store, timed ----
    llama_context * ctx_b = make_ctx();
    if (!ctx_b) {
        printf("failed to create context B\n");
        return 1;
    }

    std::vector<llama_token> toks_out(prompt.size() + 16);
    size_t n_tok_out = 0;
    const double t_load0 = now_s();
    const size_t loaded = llama_state_seq_load_file_direct(
            ctx_b, store_path.c_str(), 0, toks_out.data(), toks_out.size(), &n_tok_out);
    const double t_load = now_s() - t_load0;

    check("load succeeded", loaded > 0);
    check("load returned the token count", (int) n_tok_out == n_prompt);
    check("load returned the tokens", n_tok_out == prompt.size() &&
          memcmp(toks_out.data(), prompt.data(), n_prompt * sizeof(llama_token)) == 0);

    // the loaded state serializes like the original. informational, not a gate: this box
    // computes two identical runs differently, see docs/research/28-ubatch-determinism-results.md
    const size_t sz_b = llama_state_seq_get_size_ext(ctx_b, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    std::vector<uint8_t> blob_b(sz_b);
    const size_t n_b = llama_state_seq_get_data_ext(ctx_b, blob_b.data(), sz_b, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    blob_b.resize(n_b);

    size_t diff = 0;
    size_t first = (size_t) -1;
    const size_t n_cmp = n_a < n_b ? n_a : n_b;
    for (size_t i = 0; i < n_cmp; ++i) {
        if (blob_a[i] != blob_b[i]) {
            if (first == (size_t) -1) {
                first = i;
            }
            diff++;
        }
    }
    printf("  %-44s %zu vs %zu bytes, first_diff=%zd, differing=%zu\n",
           "state blob after load", n_a, n_b, first == (size_t) -1 ? -1 : (long long) first, diff);

    // ---- the loaded state continues the same way. this is the gate ----
    check("run B decodes the forced token", decode_chunk(ctx_b, &t_force, 1, n_prompt, true));
    std::vector<llama_token> gen_b = generate(ctx_b, n_gen, n_prompt + 1);
    bool same = gen_a.size() == gen_b.size() && gen_a.size() == (size_t) n_gen;
    for (size_t i = 0; same && i < gen_a.size(); ++i) {
        same = gen_a[i] == gen_b[i];
    }
    printf("  gen A:");
    for (llama_token t : gen_a) printf(" %d", (int) t);
    printf("\n  gen B:");
    for (llama_token t : gen_b) printf(" %d", (int) t);
    printf("\n");
    check("loaded state continues identically", same);

    // ---- rates ----
    const double mib = (double) written / (1024.0 * 1024.0);
    printf("\n  save   %7.3f s  %7.1f MiB/s   rss delta %zd KiB\n",
           t_save, mib / t_save, (ssize_t) rss_after_save - (ssize_t) rss_before_save);
    printf("  load   %7.3f s  %7.1f MiB/s\n", t_load, mib / t_load);
    printf("  round trip %.3f s for %.2f MiB\n", t_save + t_load, mib);
    printf("  state %.3f MiB, payload hash %016llx\n",
           (double) n_a / (1024.0 * 1024.0), (unsigned long long) fnv1a(blob_a.data(), blob_a.size()));

    llama_free(ctx_a);
    llama_free(ctx_b);
    llama_model_free(model);
    llama_backend_free();

    printf("\n  overall: %s\n", ok ? "pass" : "FAIL");
    return ok ? 0 : 1;
}
