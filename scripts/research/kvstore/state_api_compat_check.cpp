// Cross checks the two public state file APIs against each other, which is the boundary the
// store format claim depends on:
//
//   llama_state_seq_save_file / llama_state_seq_load_file          the shipping pair, used by
//                                                                  --slot-save-path and the
//                                                                  prompt cache, host buffered
//   llama_state_seq_save_file_direct / _load_file_direct           the store pair, O_DIRECT,
//                                                                  block framed
//
// Each API is checked against itself first, then each file is loaded by the other API. The
// claim under test is that the store format is a strict superset of LLAMA_STATE_SEQ v3, which
// only holds if the payloads interoperate.
//
// build:
//   g++ -O2 -std=c++17 -I include -I ggml/include scripts/research/kvstore/state_api_compat_check.cpp \
//       -L build-cpu/bin -lllama -lggml-base -lggml \
//       -Wl,-rpath,$PWD/build-cpu/bin -o <out>

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void quiet_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
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

static bool decode(llama_context * ctx, const llama_token * toks, int n, int pos0) {
    llama_batch b = llama_batch_init(n, 0, 1);
    b.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = pos0 + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = (i == n - 1) ? 1 : 0;
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

static std::string toks(const std::vector<llama_token> & v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%s%d", i ? " " : "", (int) v[i]);
        s += buf;
    }
    return s;
}

struct attempt {
    size_t ret = 0;
    std::vector<llama_token> gen;
    bool loaded = false;
};

static attempt load_and_generate(llama_model * model, size_t n_prompt, llama_token probe,
                                 const char * path, bool direct) {
    // a fresh context per load. an earlier version of this check reused one context for every
    // restore and the second restore in the same context either failed or came back shifted,
    // which is a different question from whether the file itself is good.
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 2048;
    cparams.n_batch   = 1024;
    cparams.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        printf("failed to create a context for the load\n");
        exit(1);
    }
    std::vector<llama_token> toks(n_prompt + 16);
    size_t n_out = 0;
    attempt a;
    a.ret = direct
        ? llama_state_seq_load_file_direct(ctx, path, 0, toks.data(), toks.size(), &n_out)
        : llama_state_seq_load_file(ctx, path, 0, toks.data(), toks.size(), &n_out);
    a.loaded = a.ret > 0;
    if (a.loaded) {
        // decode one fresh token before sampling. a restore does not produce logits, and this is
        // what a server does when the next message arrives on a restored session.
        if (decode(ctx, &probe, 1, (int) n_out)) {
            a.gen = generate(ctx, 8, (int) n_out + 1);
        }
    }
    llama_free(ctx);
    return a;
}

int main(int argc, char ** argv) {
    if (!getenv("PARK_VERBOSE")) {
        llama_log_set(quiet_log, nullptr);
    }
    const std::string model_path = argc > 1 ? argv[1]
        : "/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf";
    const std::string dir = argc > 2 ? argv[2] : "/home/herlanggays/.jcode/scratch/kvstore/compat";
    const int n_words = argc > 3 ? atoi(argv[3]) : 400;

    const std::string ship_path   = dir + "/ship.bin";
    const std::string direct_path = dir + "/direct.bin";

    bool ok = true;
    auto check = [&](const char * name, bool pass) {
        printf("  %-44s %s\n", name, pass ? "ok" : "FAIL");
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
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 2048;
    cparams.n_batch   = 1024;
    cparams.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        printf("failed to create context\n");
        return 1;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string text;
    const char * words[] = { "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
                             "golf", "hotel", "india", "juliet", "kilo", "lima" };
    for (int i = 0; i < n_words; ++i) {
        text += words[i % 12];
        text += (i % 13 == 12) ? ".\n" : " ";
    }
    std::vector<llama_token> prompt(4096);
    int n_prompt = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                                  prompt.data(), (int32_t) prompt.size(), true, true);
    if (n_prompt < 0) {
        printf("tokenize failed\n");
        return 1;
    }
    prompt.resize(n_prompt);

    // the reference: this state, then one fresh token decoded into it, then 8 tokens sampled.
    // the probe decode is what makes the comparison meaningful, since a restore produces no
    // logits of its own and sampling needs one.
    const llama_token probe = prompt[0];
    llama_memory_clear(mem, true);
    if (!decode(ctx, prompt.data(), n_prompt, 0)) {
        printf("reference decode failed\n");
        return 1;
    }
    if (!decode(ctx, &probe, 1, n_prompt)) {
        printf("reference probe decode failed\n");
        return 1;
    }
    const std::vector<llama_token> ref = generate(ctx, 8, n_prompt + 1);
    printf("  prompt tokens: %d, probe token: %d, reference continuation: %s\n",
           n_prompt, (int) probe, toks(ref).c_str());

    // put the sequence back to exactly the prompt before saving. the save records the token
    // count it is handed, and the loader rejects a state whose recorded count disagrees with
    // what it finds, which is how this check first produced n_stream mismatch.
    llama_memory_seq_rm(mem, 0, -1, -1);
    if (!decode(ctx, prompt.data(), n_prompt, 0)) {
        printf("re-decode before save failed\n");
        return 1;
    }

    // save the same state through both public APIs. the order is selectable because a first run
    // of this check showed the direct file failing to restore when the shipping save ran first,
    // which would mean one of the saves has a side effect on the state the other one captures.
    size_t ship_saved = 0;
    size_t direct_saved = 0;
    const bool direct_first = getenv("SAVE_DIRECT_FIRST") != nullptr;
    if (direct_first) {
        direct_saved = llama_state_seq_save_file_direct(ctx, direct_path.c_str(), 0,
                                                        prompt.data(), prompt.size());
        ship_saved = llama_state_seq_save_file(ctx, ship_path.c_str(), 0,
                                               prompt.data(), prompt.size());
    } else {
        ship_saved = llama_state_seq_save_file(ctx, ship_path.c_str(), 0,
                                               prompt.data(), prompt.size());
        direct_saved = llama_state_seq_save_file_direct(ctx, direct_path.c_str(), 0,
                                                        prompt.data(), prompt.size());
    }
    printf("  save order: %s\n", direct_first ? "direct first" : "shipping first");
    printf("  shipping save returned %zu, direct save returned %zu\n", ship_saved, direct_saved);
    check("shipping save wrote a file", ship_saved > 0);
    check("direct save wrote a file", direct_saved > 0);

    // each API against its own file
    const attempt ship_ship = load_and_generate(model, prompt.size(), probe, ship_path.c_str(), false);
    printf("  shipping save -> shipping load: returned %zu, continuation %s\n",
           ship_ship.ret, ship_ship.loaded ? toks(ship_ship.gen).c_str() : "not loaded");
    check("shipping save -> shipping load", ship_ship.loaded && ship_ship.gen == ref);

    const attempt dir_dir = load_and_generate(model, prompt.size(), probe, direct_path.c_str(), true);
    printf("  direct save -> direct load: returned %zu, continuation %s\n",
           dir_dir.ret, dir_dir.loaded ? toks(dir_dir.gen).c_str() : "not loaded");
    check("direct save -> direct load", dir_dir.loaded && dir_dir.gen == ref);

    // the boundary: each API against the other API's file
    const attempt ship_dir = load_and_generate(model, prompt.size(), probe, ship_path.c_str(), true);
    const attempt dir_ship = load_and_generate(model, prompt.size(), probe, direct_path.c_str(), false);

    printf("  boundary, shipping file through the direct loader: returned %zu, %s\n",
           ship_dir.ret, ship_dir.loaded ? ("continuation " + toks(ship_dir.gen)).c_str() : "not loaded");
    printf("  boundary, direct file through the shipping loader: returned %zu, %s\n",
           dir_ship.ret, dir_ship.loaded ? ("continuation " + toks(dir_ship.gen)).c_str() : "not loaded");
    check("shipping file readable by the direct loader", ship_dir.loaded && ship_dir.gen == ref);
    check("direct file readable by the shipping loader", dir_ship.loaded && dir_ship.gen == ref);

    printf("  overall: %s\n", ok ? "pass" : "FAIL");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
