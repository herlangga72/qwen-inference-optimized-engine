#include "llama-vocab.h"

#include "ggml.h"
#include "gguf.h"
#include "llama-impl.h"
#include "llama-model-loader.h"

#include "unicode.h"

#include <algorithm>
#include <cassert>
#include <cstdarg>
#include <cstring>
#include <cstdlib>
#include <forward_list>
#include <limits>
#include <queue>
#include <set>
#include <unordered_map>

//
// tokenizers
//

struct llm_tokenizer {
    llm_tokenizer() {}
    virtual ~llm_tokenizer() = default;
};

struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

static_assert(std::is_trivially_copyable<llm_symbol>::value, "llm_symbol is not trivially copyable");

//
// BPE tokenizer
// adapted from https://github.com/cmp-nct/ggllm.cpp [MIT License]
// tried to simplify unicode stuff, so most likely does not work 100% correctly!
//

// TODO: there are a lot of common parts between spm and bpe tokenizers, should be refactored and reused

template<typename T, typename Container = std::vector<T>, typename Compare = std::less<typename Container::value_type>>
class llama_priority_queue : public std::priority_queue<T, Container, Compare> {
public:
    using std::priority_queue<T, Container, Compare>::priority_queue;

    T pop_move() {
        T item = std::move(this->c.front());
        std::pop_heap(this->c.begin(), this->c.end(), this->comp);
        this->c.pop_back();
        return item;
    }

    void pop() =  delete;
};

struct llm_bigram_bpe {
    struct comparator {
        bool operator()(const llm_bigram_bpe & l, const llm_bigram_bpe & r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = llama_priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    std::string text;
    int rank;
    size_t size;
};

struct llm_tokenizer_bpe : llm_tokenizer {
    llm_tokenizer_bpe(const llama_vocab & vocab) {
        GGML_ASSERT(vocab.get_type() == LLAMA_VOCAB_TYPE_BPE);
        switch (vocab.get_pre_type()) {
            case LLAMA_VOCAB_PRE_TYPE_QWEN2:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_QWEN35:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            default:
                // default regex for BPE tokenization pre-processing
                regex_exprs = {
                    "[\\p{P}\\$\\+<=>\\^~\\|]+",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                    "\\p{N}+",
                    "[0-9][0-9][0-9]",
                };
                break;
        }
    }

    std::vector<std::string> regex_exprs;
};

struct llm_tokenizer_bpe_session {
    llm_tokenizer_bpe_session(const llama_vocab & vocab, const llm_tokenizer_bpe & tokenizer) : vocab(vocab), tokenizer(tokenizer) {}

    virtual ~llm_tokenizer_bpe_session() = default;

    static void append(const llama_token token_id, std::vector<llama_token> & output)  {
        output.push_back(token_id);
    }

    bool append_bos(std::vector<llama_token> & output) const {
        if (vocab.get_add_bos()) {
            GGML_ASSERT(vocab.token_bos() != LLAMA_TOKEN_NULL);
            output.push_back(vocab.token_bos());
            return true;
        }
        return false;
    }

    bool append_eos(std::vector<llama_token> & output) const {
        if (vocab.get_add_eos()) {
            GGML_ASSERT(vocab.token_eos() != LLAMA_TOKEN_NULL);
            output.push_back(vocab.token_eos());
            return true;
        }
        return false;
    }

    void check_double_bos_eos(const std::vector<llama_token> & output) const {
        if (vocab.get_add_bos() && output.size() >= 2 && output[1] == vocab.token_bos()) {
            LLAMA_LOG_WARN(
                "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                "Are you sure this is what you want?\n", __FUNCTION__);
        }
        if (vocab.get_add_eos() && output.size() >= 2 && *(output.end()-2) == vocab.token_eos()) {
            LLAMA_LOG_WARN(
                "%s: Added a EOS token to the prompt as specified by the model but the prompt "
                "also ends with a EOS token. So now the final prompt ends with 2 EOS tokens. "
                "Are you sure this is what you want?\n", __FUNCTION__);
        }
    }

    virtual void tokenize(const std::string & text, std::vector<llama_token> & output) {
        int final_prev_index = -1;
        const auto word_collection = unicode_regex_split(text, tokenizer.regex_exprs);

        symbols_final.clear();

        for (const auto & word : word_collection) {
            work_queue = llm_bigram_bpe::queue();
            symbols.clear();

            int index = 0;
            size_t offset = 0;

            //if (vocab.tokenizer_ignore_merges && vocab.token_to_id.find(word) != vocab.token_to_id.end()) {
            if (vocab.get_ignore_merges() && vocab.text_to_token(word) != LLAMA_TOKEN_NULL) {
                symbols.emplace_back(llm_symbol{-1, -1, word.c_str(), word.size()});
                offset = word.size();
            }

            while (offset < word.size()) {
                llm_symbol sym;
                size_t char_len = std::min(word.size() - offset, (size_t) unicode_len_utf8(word[offset]));
                sym.text = word.c_str() + offset;
                sym.n = char_len;
                offset += sym.n;
                sym.prev = index - 1;
                sym.next = offset == word.size() ? -1 : index + 1;
                index++;
                symbols.emplace_back(sym);
            }
            for (int i = 1; i < (int) symbols.size(); ++i) {
                add_new_bigram(i - 1, i);
            }

            // build token(s)
            while (!work_queue.empty()) {
                auto bigram = work_queue.pop_move();

                auto & left_symbol = symbols[bigram.left];
                auto & right_symbol = symbols[bigram.right];

                if (left_symbol.n == 0 || right_symbol.n == 0) {
                    continue;
                }
                std::string left_token = std::string(left_symbol.text, left_symbol.n);
                std::string right_token = std::string(right_symbol.text, right_symbol.n);
                if (left_token + right_token != bigram.text) {
                    continue;  // Skip this bigram if it's outdated
                }

                // merge the right sym into the left one
                left_symbol.n += right_symbol.n;
                right_symbol.n = 0;

                // remove the right sym from the chain
                left_symbol.next = right_symbol.next;
                if (right_symbol.next >= 0) {
                    symbols[right_symbol.next].prev = bigram.left;
                }

                add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
                add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
            }

            // add the finished tokens to the final list keeping correct order for next and prev
            for (auto & sym : symbols) {
                if (sym.n > 0) {
                    sym.prev = final_prev_index;
                    sym.next = -1;
                    if (final_prev_index != -1) {
                        symbols_final[final_prev_index].next = symbols_final.size();
                    }
                    symbols_final.emplace_back(sym);
                    final_prev_index = symbols_final.size() - 1;
                }
            }
        }

        symbols = symbols_final;

        if (!symbols.empty()) {
            for (int i = 0; i != -1; i = symbols[i].next) {
                auto & symbol = symbols[i];
                if (symbol.n == 0) {
                    continue;
                }

                const std::string str = std::string(symbol.text, symbol.n);
                const auto token = vocab.text_to_token(str);

                if (token == LLAMA_TOKEN_NULL) {
                    for (auto j = str.begin(); j != str.end(); ++j) {
                        llama_token token_multibyte = vocab.text_to_token(std::string(1, *j));
                        if (token_multibyte != LLAMA_TOKEN_NULL) {
                            output.push_back(token_multibyte);
                        }
                    }
                } else {
                    output.push_back(token);
                }
            }
        }
    }

private:
    void add_new_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }
        std::string left_token  = std::string(symbols[left].text,  symbols[left].n);
        std::string right_token = std::string(symbols[right].text, symbols[right].n);

        int rank_found = -1;

        rank_found = vocab.find_bpe_rank(left_token, right_token);

        if (rank_found < 0) {
            return;
        }

        llm_bigram_bpe bigram;

        bigram.left  = left;
        bigram.right = right;
        bigram.text  = left_token + right_token;
        bigram.size  = left_token.size() + right_token.size();
        bigram.rank  = rank_found;

        work_queue.push(bigram);
    }

    const llama_vocab & vocab;
    const llm_tokenizer_bpe & tokenizer;

    std::vector<llm_symbol> symbols;
    std::vector<llm_symbol> symbols_final;
    llm_bigram_bpe::queue work_queue;
};

//
// impl
//

typedef enum FRAGMENT_BUFFER_VARIANT_TYPE {
    FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN,
    FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT
} FRAGMENT_BUFFER_VARIANT_TYPE;

struct fragment_buffer_variant {
    fragment_buffer_variant(llama_token _token)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN),
        token(_token),
        raw_text(_dummy),
        offset(0),
        length(0) {}

    fragment_buffer_variant(const std::string & _raw_text, int64_t _offset, int64_t _length)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT),
        token((llama_token) - 1),
        raw_text(_raw_text),
        offset(_offset),
        length(_length){
            GGML_ASSERT(_offset >= 0);
            GGML_ASSERT(_length >= 1);
            GGML_ASSERT(offset + length <= raw_text.length());
        }

    const FRAGMENT_BUFFER_VARIANT_TYPE type;
    const llama_token token;
    const std::string _dummy;
    const std::string & raw_text;
    const uint64_t offset;
    const uint64_t length;
};

struct llama_vocab::impl {
    uint32_t n_token_types = 0; // for BERT-style token types

    std::string tokenizer_model;
    std::string tokenizer_pre;

    enum llama_vocab_type     type     = LLAMA_VOCAB_TYPE_NONE;
    enum llama_vocab_pre_type pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;

    int max_token_len = 0; // used for optimizing longest token search

    // default LLaMA special tokens
    // TODO: should we set all of these to LLAMA_TOKEN_NULL?
    llama_token special_bos_id  = 1;
    llama_token special_eos_id  = 2;
    llama_token special_eot_id  = LLAMA_TOKEN_NULL;
    llama_token special_eom_id  = LLAMA_TOKEN_NULL;
    llama_token special_unk_id  = 0;
    llama_token special_sep_id  = LLAMA_TOKEN_NULL;
    llama_token special_pad_id  = LLAMA_TOKEN_NULL;
    llama_token special_mask_id = LLAMA_TOKEN_NULL;

    llama_token linefeed_id = 13;

    // fim tokens
    llama_token special_fim_pre_id = LLAMA_TOKEN_NULL;
    llama_token special_fim_suf_id = LLAMA_TOKEN_NULL;
    llama_token special_fim_mid_id = LLAMA_TOKEN_NULL;
    llama_token special_fim_pad_id = LLAMA_TOKEN_NULL;
    llama_token special_fim_rep_id = LLAMA_TOKEN_NULL; // repo
    llama_token special_fim_sep_id = LLAMA_TOKEN_NULL; // file separator

    // tokenizer flags
    bool add_space_prefix           = false;
    bool add_bos                    = false;
    bool add_eos                    = false;
    bool add_sep                    = false;
    bool ignore_merges              = false;
    bool clean_spaces               = false;  // clean_up_tokenization_spaces
    bool remove_extra_whitespaces   = false;
    bool escape_whitespaces         = true;
    bool treat_whitespace_as_suffix = false;

    // BertNormalizer options
    llama_vocab::normalizer_options normalizer_opts;

    std::unordered_map<std::string, llama_token> token_to_id;
    std::vector<token_data>                      id_to_token;

    std::vector<llama_token> cache_special_tokens;
    std::vector<std::string> cache_token_to_piece; // llama_token_to_piece(special = true);
    struct pair_hash {
        size_t operator()(const std::pair<std::string, std::string> & p) const {
            return std::hash<std::string>{}(p.first) ^  //create some hash for pair
                   (std::hash<std::string>{}(p.second) << 1);
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> bpe_ranks;

    // set of all tokens that cause "end of generation"
    std::set<llama_token> special_eog_ids;

    std::vector<llama_token> suppress_tokens;

    std::unique_ptr<llm_tokenizer> tokenizer;

    std::vector<char> precompiled_charsmap;

    impl(const llama_vocab & vocab) : vocab(vocab) {
    }

    ~impl() = default;

    void load(llama_model_loader & ml, const LLM_KV & kv);

    enum llama_vocab_type get_type() const;

    std::string type_name() const;

    bool is_normal      (llama_token id) const;
    bool is_unknown     (llama_token id) const;
    bool is_control     (llama_token id) const;
    bool is_byte        (llama_token id) const;
    bool is_user_defined(llama_token id) const;
    bool is_unused      (llama_token id) const;
    bool is_eog         (llama_token id) const;

    uint8_t token_to_byte(llama_token id) const;

    llama_token_attr token_get_attr(llama_token id) const;

    void init_tokenizer(enum llama_vocab_type type);

    void tokenizer_st_partition(std::forward_list<fragment_buffer_variant> & buffer, bool parse_special) const;

    std::string token_to_piece_for_cache(
                  llama_token   token,
                         bool   special) const;


    std::vector<llama_token> tokenize(
            const std::string & raw_text,
                         bool   add_special,
                         bool   parse_special = false) const;

    int32_t tokenize(
                   const char * text,
                      int32_t   text_len,
                  llama_token * tokens,
                      int32_t   n_tokens_max,
                         bool   add_special,
                         bool   parse_special) const;

    // does not write null-terminator to buf
    int32_t token_to_piece(
                  llama_token   token,
                         char * buf,
                      int32_t   length,
                      int32_t   lstrip,
                         bool   special) const;

    // use cached data
    const std::string & token_to_piece(llama_token token) const;

    int32_t detokenize(
            const llama_token * tokens,
                      int32_t   n_tokens,
                         char * text,
                      int32_t   text_len_max,
                         bool   remove_special,
                         bool   unparse_special) const;

    std::string detokenize(
            const std::vector<llama_token> & tokens,
                                      bool   special) const;

    void print_info() const;

private:
    const llama_vocab & vocab;
};

void llama_vocab::impl::load(llama_model_loader & ml, const LLM_KV & kv) {
    struct gguf_context * ctx = ml.metadata;

    // determine vocab type
    {
        ml.get_key(LLM_KV_TOKENIZER_MODEL, tokenizer_model);
        ml.get_key(LLM_KV_TOKENIZER_PRE,   tokenizer_pre, false);

        ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, n_token_types, false);

        if (tokenizer_model == "no_vocab" || tokenizer_model == "none") {
            type = LLAMA_VOCAB_TYPE_NONE;

            // default special tokens
            special_bos_id  = LLAMA_TOKEN_NULL;
            special_eos_id  = LLAMA_TOKEN_NULL;
            special_unk_id  = LLAMA_TOKEN_NULL;
            special_sep_id  = LLAMA_TOKEN_NULL;
            special_pad_id  = LLAMA_TOKEN_NULL;
            special_mask_id = LLAMA_TOKEN_NULL;
            linefeed_id     = LLAMA_TOKEN_NULL;

            // read vocab size from metadata
            uint32_t n_tokens = 0;
            if (ml.get_key(LLM_KV_VOCAB_SIZE, n_tokens, false)) {
                LLAMA_LOG_WARN("%s: adding %u dummy tokens\n", __func__, n_tokens);
                id_to_token.resize(n_tokens);
            }

            return;
        }

        if (tokenizer_model == "gpt2") {
            type = LLAMA_VOCAB_TYPE_BPE;

            // read bpe merges and populate bpe ranks
            const int merges_keyidx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_MERGES).c_str());

            if (merges_keyidx == -1) {
                throw std::runtime_error("cannot find tokenizer merges in model file\n");
            } else {
                if (gguf_get_kv_type(ctx, merges_keyidx) != GGUF_TYPE_ARRAY ||
                    gguf_get_arr_type(ctx, merges_keyidx) != GGUF_TYPE_STRING) {
                    throw std::runtime_error(format("invalid gguf type for %s", kv(LLM_KV_TOKENIZER_MERGES).c_str()));
                }
                const int n_merges = gguf_get_arr_n(ctx, merges_keyidx);
                for (int i = 0; i < n_merges; i++) {
                    const std::string word = gguf_get_arr_str(ctx, merges_keyidx, i);
                    //GGML_ASSERT(unicode_cpts_from_utf8(word).size() > 0);

                    std::string first;
                    std::string second;

                    const size_t pos = word.find(' ', 1);

                    if (pos != std::string::npos) {
                        first  = word.substr(0, pos);
                        second = word.substr(pos + 1);
                    }

                    bpe_ranks.emplace(std::make_pair(first, second), i);
                }
            }

            // default special tokens
            special_bos_id  = 11;
            special_eos_id  = 11;
            special_unk_id  = LLAMA_TOKEN_NULL;
            special_sep_id  = LLAMA_TOKEN_NULL;
            special_pad_id  = LLAMA_TOKEN_NULL;
            special_mask_id = LLAMA_TOKEN_NULL;
        } else if (tokenizer_model == "test") {
            type = LLAMA_VOCAB_TYPE_TEST;

            // default special tokens
            special_bos_id  = LLAMA_TOKEN_NULL;
            special_eos_id  = LLAMA_TOKEN_NULL;
            special_unk_id  = LLAMA_TOKEN_NULL;
            special_sep_id  = LLAMA_TOKEN_NULL;
            special_pad_id  = LLAMA_TOKEN_NULL;
            special_mask_id = LLAMA_TOKEN_NULL;
        } else {
            throw std::runtime_error(format("unknown tokenizer: '%s'", tokenizer_model.c_str()));
        }

        // for now, only BPE models have pre-tokenizers
        if (type == LLAMA_VOCAB_TYPE_BPE) {
            add_space_prefix = false;
            escape_whitespaces = false;
            clean_spaces = true;
            if (tokenizer_pre.empty()) {
                LLAMA_LOG_WARN("%s: missing pre-tokenizer type, using: 'default'\n", __func__);
                LLAMA_LOG_WARN("%s:                                             \n", __func__);
                LLAMA_LOG_WARN("%s: ************************************        \n", __func__);
                LLAMA_LOG_WARN("%s: GENERATION QUALITY WILL BE DEGRADED!        \n", __func__);
                LLAMA_LOG_WARN("%s: CONSIDER REGENERATING THE MODEL             \n", __func__);
                LLAMA_LOG_WARN("%s: ************************************        \n", __func__);
                LLAMA_LOG_WARN("%s:                                             \n", __func__);
                pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            } else if (tokenizer_pre == "default") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            } else if (tokenizer_pre == "qwen2") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_QWEN2;
                clean_spaces = false;
            } else if (tokenizer_pre == "qwen35") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_QWEN35;
                clean_spaces = false;
            } else {
                throw std::runtime_error(format("unknown pre-tokenizer type: '%s'", tokenizer_pre.c_str()));
            }
        } else {
            pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
        }

        ml.get_key(LLM_KV_TOKENIZER_ADD_PREFIX,      add_space_prefix,         false);
        ml.get_key(LLM_KV_TOKENIZER_REMOVE_EXTRA_WS, remove_extra_whitespaces, false);
    }

    const int token_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_LIST).c_str());
    if (token_idx == -1) {
        throw std::runtime_error("cannot find tokenizer vocab in model file\n");
    }

    if (gguf_get_kv_type(ctx, token_idx) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(ctx, token_idx) != GGUF_TYPE_STRING) {
        throw std::runtime_error(format("invalid gguf type for %s", kv(LLM_KV_TOKENIZER_LIST).c_str()));
    }

    const uint32_t n_tokens = gguf_get_arr_n(ctx, token_idx);

    const float * scores = nullptr;
    const int * iscores = nullptr;
    const int score_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_SCORES).c_str());
    if (score_idx != -1) {
        const gguf_type kv_type = gguf_get_kv_type(ctx, score_idx);
        const gguf_type arr_type = kv_type == GGUF_TYPE_ARRAY ? gguf_get_arr_type(ctx, score_idx) : GGUF_TYPE_COUNT;
        if (arr_type != GGUF_TYPE_INT32 &&
            arr_type != GGUF_TYPE_FLOAT32) {
            throw std::runtime_error(format("invalid gguf type for %s", kv(LLM_KV_TOKENIZER_SCORES).c_str()));
        }
        const uint32_t n_scores = gguf_get_arr_n(ctx, score_idx);
        if (n_scores < n_tokens) {
            throw std::runtime_error("Index out of array bounds for scores (" + std::to_string(n_scores) + " < " + std::to_string(n_tokens) + ")\n");
        }
        if (arr_type == GGUF_TYPE_INT32) {
            iscores = (const int *) gguf_get_arr_data(ctx, score_idx);
        } else {
            scores = (const float * ) gguf_get_arr_data(ctx, score_idx);
        }
    }

    const int * toktypes = nullptr;
    const int toktype_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE).c_str());
    if (toktype_idx != -1) {
        if (gguf_get_kv_type(ctx, toktype_idx) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(ctx, toktype_idx) != GGUF_TYPE_INT32) {
            throw std::runtime_error(format("invalid gguf type for %s", kv(LLM_KV_TOKENIZER_TOKEN_TYPE).c_str()));
        }
        const uint32_t n_toktypes = gguf_get_arr_n(ctx, toktype_idx);
        if (n_toktypes < n_tokens) {
            throw std::runtime_error("Index out of array bounds for toktypes (" + std::to_string(n_toktypes) + " < " + std::to_string(n_tokens) + ")\n");
        }
        toktypes = (const int * ) gguf_get_arr_data(ctx, toktype_idx);
    }

    id_to_token.resize(n_tokens);

    for (uint32_t i = 0; i < n_tokens; i++) {
        std::string word = gguf_get_arr_str(ctx, token_idx, i);
        if (word.empty()) {
            LLAMA_LOG_WARN("%s: empty token at index %u\n", __func__, i);
            word = "[EMPTY_" + std::to_string(i) + "]";
        }

        token_to_id[word] = i;
        max_token_len = std::max(max_token_len, (int) word.size());

        auto & token_data = id_to_token[i];
        token_data.text  = std::move(word);
        if (scores) {
            token_data.score = scores[i];
        } else if (iscores) {
            token_data.score = static_cast<float>(iscores[i]);
        } else {
            token_data.score = 0.0f;
        }
        token_data.attr  = LLAMA_TOKEN_ATTR_NORMAL;

        if (toktypes) {  //TODO: remove, required until per token attributes are available from GGUF file
            switch(toktypes[i]) {
                case LLAMA_TOKEN_TYPE_UNKNOWN:      token_data.attr = LLAMA_TOKEN_ATTR_UNKNOWN;      break;
                case LLAMA_TOKEN_TYPE_UNUSED:       token_data.attr = LLAMA_TOKEN_ATTR_UNUSED;       break;
                case LLAMA_TOKEN_TYPE_NORMAL:       token_data.attr = LLAMA_TOKEN_ATTR_NORMAL;       break;
                case LLAMA_TOKEN_TYPE_CONTROL:      token_data.attr = LLAMA_TOKEN_ATTR_CONTROL;      break;
                case LLAMA_TOKEN_TYPE_USER_DEFINED: token_data.attr = LLAMA_TOKEN_ATTR_USER_DEFINED; break;
                case LLAMA_TOKEN_TYPE_BYTE:         token_data.attr = LLAMA_TOKEN_ATTR_BYTE;         break;
                case LLAMA_TOKEN_TYPE_UNDEFINED:    token_data.attr = LLAMA_TOKEN_ATTR_UNDEFINED;    break;
                default:                            token_data.attr = LLAMA_TOKEN_ATTR_UNDEFINED;    break;
            }
        }
    }
    GGML_ASSERT(id_to_token.size() == token_to_id.size());

    init_tokenizer(type);

    // determine the newline token
    {
        const std::vector<int> ids = tokenize("\n", false);

        //GGML_ASSERT(!ids.empty() && "model vocab missing newline token");
        if (ids.empty()) {
            LLAMA_LOG_WARN("%s: model vocab missing newline token, using special_pad_id instead\n", __func__);
            linefeed_id = special_pad_id;
        } else {
            linefeed_id = ids[0];
        }
    }

    // special tokens
    {
        const std::vector<std::pair<enum llm_kv, int32_t &>> special_token_types = {
            { LLM_KV_TOKENIZER_BOS_ID,     special_bos_id     },
            { LLM_KV_TOKENIZER_EOS_ID,     special_eos_id     },
            { LLM_KV_TOKENIZER_EOT_ID,     special_eot_id     },
            { LLM_KV_TOKENIZER_EOM_ID,     special_eom_id     },
            { LLM_KV_TOKENIZER_UNK_ID,     special_unk_id     },
            { LLM_KV_TOKENIZER_SEP_ID,     special_sep_id     },
            { LLM_KV_TOKENIZER_PAD_ID,     special_pad_id     },
            { LLM_KV_TOKENIZER_MASK_ID,    special_mask_id    },
            { LLM_KV_TOKENIZER_FIM_PRE_ID, special_fim_pre_id },
            { LLM_KV_TOKENIZER_FIM_SUF_ID, special_fim_suf_id },
            { LLM_KV_TOKENIZER_FIM_MID_ID, special_fim_mid_id },
            { LLM_KV_TOKENIZER_FIM_PAD_ID, special_fim_pad_id },
            { LLM_KV_TOKENIZER_FIM_REP_ID, special_fim_rep_id },
            { LLM_KV_TOKENIZER_FIM_SEP_ID, special_fim_sep_id },

            // deprecated
            { LLM_KV_TOKENIZER_PREFIX_ID, special_fim_pre_id },
            { LLM_KV_TOKENIZER_SUFFIX_ID, special_fim_suf_id },
            { LLM_KV_TOKENIZER_MIDDLE_ID, special_fim_mid_id },
        };

        for (const auto & it : special_token_types) {
            const std::string & key = kv(std::get<0>(it));
            int32_t & id = std::get<1>(it);

            if (id >= 0 && static_cast<size_t>(id) >= id_to_token.size()) {
                LLAMA_LOG_WARN("%s: default special token '%s' = %d out of vocab range, disabling\n",
                    __func__, key.c_str(), id);
                id = LLAMA_TOKEN_NULL;
            }

            uint32_t new_id;
            if (!ml.get_key(std::get<0>(it), new_id, false)) {
                continue;
            }
            if (new_id >= id_to_token.size()) {
                LLAMA_LOG_WARN("%s: bad special token: '%s' = %u, using default id %d\n",
                    __func__, key.c_str(), new_id, id);
            } else {
                id = new_id;
            }
        }

        // Handle add_bos, add_eos and add_sep
        {
            bool temp = true;

            if (ml.get_key(LLM_KV_TOKENIZER_ADD_BOS, temp, false)) {
                add_bos = temp;
            }
            if (ml.get_key(LLM_KV_TOKENIZER_ADD_EOS, temp, false)) {
                add_eos = temp;
            }
            if (ml.get_key(LLM_KV_TOKENIZER_ADD_SEP, temp, false)) {
                add_sep = temp;
            }
        }

        // BertNormalizer options
        ml.get_key(LLM_KV_TOKENIZER_NORMALIZER_LOWERCASE,     normalizer_opts.lowercase,     false);
        normalizer_opts.strip_accents = normalizer_opts.lowercase;
        ml.get_key(LLM_KV_TOKENIZER_NORMALIZER_STRIP_ACCENTS, normalizer_opts.strip_accents, false);

        // suppress tokens
        {
            const int suppress_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_SUPPRESS_TOKENS).c_str());
            if (suppress_idx != -1) {
                if (gguf_get_kv_type(ctx, suppress_idx) != GGUF_TYPE_ARRAY ||
                    gguf_get_arr_type(ctx, suppress_idx) != GGUF_TYPE_INT32) {
                    throw std::runtime_error(format("invalid gguf type for %s", kv(LLM_KV_TOKENIZER_SUPPRESS_TOKENS).c_str()));
                }
                const int n = gguf_get_arr_n(ctx, suppress_idx);
                const int32_t * data = (const int32_t *) gguf_get_arr_data(ctx, suppress_idx);
                // drop out-of-range ids
                suppress_tokens.reserve(n);
                for (int i = 0; i < n; ++i) {
                    const int32_t id = data[i];
                    if (id >= 0 && id < (int) id_to_token.size()) {
                        suppress_tokens.push_back(id);
                    }
                }
            }
        }

        // auto-detect special tokens by text
        // TODO: convert scripts should provide these tokens through the KV metadata LLM_KV_TOKENIZER_...
        //       for now, we apply this workaround to find the tokens based on their text

        for (const auto & t : token_to_id) {
            auto & attr = id_to_token[t.second].attr;

            // find EOT token: "<|eot_id|>", "<|im_end|>", "<end_of_turn>", etc.
            if (special_eot_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|eot_id|>"
                        || t.first == "<|im_end|>"
                        || t.first == "<|end|>"
                        || t.first == "<end_of_turn>"
                        || t.first == "<|endoftext|>"
                        || t.first == "<|end_of_text|>" // granite
                        || t.first == "<EOT>"
                        || t.first == "_<EOT>"
                        || t.first == "[EOT]" // Kimi-K2
                        || t.first == "<｜end▁of▁sentence｜>" // DeepSeek
                        || t.first == "<end_of_utterance>" // smoldocling
                   ) {
                    special_eot_id = t.second;
                    if ((attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        attr = (llama_token_attr) (attr | LLAMA_TOKEN_ATTR_CONTROL);
                    }
                }
            }

            // find EOM token: "<|eom_id|>"
            if (special_eom_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|eom_id|>"
                        ) {
                    special_eom_id = t.second;
                    if ((attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        attr = (llama_token_attr) (attr | LLAMA_TOKEN_ATTR_CONTROL);
                    }
                }
            }

            // find FIM_PRE token: "<|fim_prefix|>", "<fim-prefix>", "<PRE>", etc.
            if (special_fim_pre_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_prefix|>"  // Qwen
                        || t.first == "<fim-prefix>"
                        || t.first == "<fim_prefix>"    // Granite
                        || t.first == "<｜fim▁begin｜>" // DeepSeek
                        || t.first == "<PRE>"
                        || t.first == "▁<PRE>"          // CodeLlama
                        || t.first == "<|code_prefix|>" // GLM-4.5
                        || t.first == "<|prefix|>"      // Falcon-H1-Tiny-Coder
                        ) {
                    special_fim_pre_id = t.second;
                    if ((attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        attr = (llama_token_attr) (attr | LLAMA_TOKEN_ATTR_CONTROL);
                    }
                }
            }

            // find FIM_SUF token: "<|fim_suffix|>", "<fim-suffix>", "<SUF>", etc.
            if (special_fim_suf_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_suffix|>" // Qwen
                        || t.first == "<fim-suffix>"
                        || t.first == "<fim_suffix>"   // Granite
                        || t.first == "<｜fim▁hole｜>" // DeepSeek
                        || t.first == "<SUF>"
                        || t.first == "▁<SUF>"         // CodeLlama
                        || t.first == "<|code_suffix|>" // GLM-4.5
                        || t.first == "<|suffix|>"      // Falcon-H1-Tiny-Coder
                        ) {
                    special_fim_suf_id = t.second;
                    if ((attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        attr = (llama_token_attr) (attr | LLAMA_TOKEN_ATTR_CONTROL);
                    }
                }
            }

            // find FIM_MID token: "<|fim_middle|>", "<fim-middle>", "<MID>", etc.
            if (special_fim_mid_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_middle|>" // Qwen
                        || t.first == "<fim-middle>"
                        || t.first == "<fim_middle>"   // Granite
                        || t.first == "<｜fim▁end｜>"  // DeepSeek
                        || t.first == "<MID>"
                        || t.first == "▁<MID>"         // CodeLlama
                        || t.first == "<|code_middle|>" // GLM-4.5
                        || t.first == "<|middle|>"      // Falcon-H1-Tiny-Coder
                        ) {
                    special_fim_mid_id = t.second;
                    if ((attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        attr = (llama_token_attr) (attr | LLAMA_TOKEN_ATTR_CONTROL);
                    }
                }
            }

            // find FIM_PAD token: "<|fim_pad|>", "<fim-pad>", "<PAD>", etc.
            if (special_fim_pad_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_pad|>" // Qwen
                        || t.first == "<fim-pad>"
                        || t.first == "<fim_pad>"   // Granite
                        || t.first == "<PAD>"
                        || t.first == "[PAD]" // Kimi-K2
                        ) {
                    special_fim_pad_id = t.second;
                    if ((attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        attr = (llama_token_attr) (attr | LLAMA_TOKEN_ATTR_CONTROL);
                    }
                }
            }

            // find FIM_REP token: "<|fim_repo|>", "<fim-repo>", "<REP>", etc.
            if (special_fim_rep_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_repo|>"  // Qwen
                        || t.first == "<|repo_name|>"
                        || t.first == "<fim-repo>"
                        || t.first == "<REPO>"
                        || t.first == "<reponame>"    // Granite
                        ) {
                    special_fim_rep_id = t.second;
                    if ((attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        attr = (llama_token_attr) (attr | LLAMA_TOKEN_ATTR_CONTROL);
                    }
                }
            }

            // find FIM_SEP token: "<|file_sep|>"
            if (special_fim_sep_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|file_sep|>" // Qwen
                        ) {
                    special_fim_sep_id = t.second;
                    if ((attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        attr = (llama_token_attr) (attr | LLAMA_TOKEN_ATTR_CONTROL);
                    }
                }
            }
        }

        // auto-detect unused tokens: e.g. control tokens with the word "unused"
        // ideally, these tokens should be marked as unused during conversion
        {
            uint32_t n_unused = 0;

            for (const auto & t : token_to_id) {
                auto & attr = id_to_token[t.second].attr;

                if ((attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                    continue;
                }

                if ((attr & LLAMA_TOKEN_ATTR_UNUSED) == 0) {
                    if (strstr(t.first.c_str(), "unused") != NULL) {
                        attr = (llama_token_attr) (attr | LLAMA_TOKEN_ATTR_UNUSED);
                    }
                }

                if (attr & LLAMA_TOKEN_ATTR_UNUSED) {
                    n_unused++;
                }
            }

            LLAMA_LOG_INFO("%s: %u unused tokens\n", __func__, n_unused);
        }

        // maintain a list of tokens that cause end-of-generation
        // this is currently determined based on the token text, which is obviously not ideal
        // ref: https://github.com/ggml-org/llama.cpp/issues/9606
        special_eog_ids.clear();

        if (special_fim_pad_id != LLAMA_TOKEN_NULL && special_eog_ids.count(special_fim_pad_id) == 0) {
            special_eog_ids.insert(special_fim_pad_id);
        }

        if (special_fim_rep_id != LLAMA_TOKEN_NULL && special_eog_ids.count(special_fim_rep_id) == 0) {
            special_eog_ids.insert(special_fim_rep_id);
        }

        if (special_fim_sep_id != LLAMA_TOKEN_NULL && special_eog_ids.count(special_fim_sep_id) == 0) {
            special_eog_ids.insert(special_fim_sep_id);
        }

        for (const auto & t : token_to_id) {
            auto & attr = id_to_token[t.second].attr;

            if (false
                    || t.first == "<|eot_id|>"
                    || t.first == "<|im_end|>"
                    || t.first == "<|end|>"
                    || t.first == "<|return|>" // o200k_harmony
                    || t.first == "<|call|>"   // o200k_harmony
                    || t.first == "<|flush|>"  // solar-open
                    || t.first == "<|calls|>"  // solar-open
                    || t.first == "<end_of_turn>"
                    || t.first == "<|endoftext|>"
                    || t.first == "</s>"      // paddleocr
                    || t.first == "<|eom_id|>"
                    || t.first == "<EOT>"
                    || t.first == "_<EOT>"
                    || t.first == "[EOT]" // Kimi-K2
                    || t.first == "[EOS]" // Kimi-K2
                    || t.first == "<|end_of_text|>"
                    || t.first == "<end_of_utterance>" // smoldocling
                    || t.first == "<eos>"            // gemma4
                    || t.first == "<turn|>"          // gemma4
                    || t.first == "<|tool_response>" // gemma4
                    || t.first == "<｜end▁of▁sentence｜>" // deepseek-ocr
                    || t.first == "[e~[" // minimax-m2/m3
               ) {
                special_eog_ids.insert(t.second);
                if ((attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                    LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                            __func__, t.second, t.first.c_str());
                    attr = (llama_token_attr) (attr | LLAMA_TOKEN_ATTR_CONTROL);
                }
            } else {
                if (attr & LLAMA_TOKEN_ATTR_CONTROL && !(attr & LLAMA_TOKEN_ATTR_UNUSED)) {
                    // token is control, but not marked as EOG -> print a debug log
                    if (special_eog_ids.count(t.second) == 0) {
                        LLAMA_LOG_DEBUG("%s: control token: %6d '%s' is not marked as EOG\n",
                                __func__, t.second, t.first.c_str());
                    }
                }
            }
        }

        // @ngxson : quick hack for gpt-oss, always render these tokens
        for (const auto & t : token_to_id) {
            auto & attr = id_to_token[t.second].attr;

            if (t.first == "<|channel|>" || t.first == "<|message|>" || t.first == "<|start|>" || t.first == "<|constrain|>") {
                LLAMA_LOG_WARN("%s: setting token '%s' (%d) attribute to USER_DEFINED (%u), old attributes: %u\n",
                        __func__, t.first.c_str(), t.second, LLAMA_TOKEN_ATTR_USER_DEFINED, attr);

                attr = LLAMA_TOKEN_ATTR_USER_DEFINED;
            }
        }

        // sanity checks
        if (special_eos_id != LLAMA_TOKEN_NULL && special_eog_ids.count(special_eos_id) == 0) {
            special_eog_ids.insert(special_eos_id);
            LLAMA_LOG_WARN("%s: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }

        if (special_eot_id != LLAMA_TOKEN_NULL && special_eog_ids.count(special_eot_id) == 0) {
            special_eog_ids.insert(special_eot_id);
            LLAMA_LOG_WARN("%s: special_eot_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }

        if (special_eom_id != LLAMA_TOKEN_NULL && special_eog_ids.count(special_eom_id) == 0) {
            special_eog_ids.insert(special_eom_id);
            LLAMA_LOG_WARN("%s: special_eom_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }

        // TODO: workaround for o200k_harmony and solar-open tokenizer: the "<|end|>" token should not be EOG
        //       we don't have a good way to detect this, so for now, if we have "<|return|>" and "<|call|>" tokens ("<|calls|>" and "<|flush|>" for solar-open),
        //       we remove the "<|end|>" token from the EOG list
        {
            bool has_return = false;
            bool has_call   = false;
            bool has_end    = false;
            bool has_flush  = false;

            llama_token end_id = LLAMA_TOKEN_NULL;

            LLAMA_LOG_INFO("%s: printing all EOG tokens:\n", __func__);
            for (auto tid : special_eog_ids) {
                if (tid < 0 || tid >= (llama_token) id_to_token.size()) {
                    LLAMA_LOG_WARN("%s: EOG token id %d is out of range (vocab size %zu), skipping\n",
                            __func__, tid, id_to_token.size());
                    continue;
                }
                auto & text = id_to_token[tid].text;

                LLAMA_LOG_INFO("%s:   - %d ('%s')\n", __func__, tid, text.c_str());

                if (text == "<|return|>") {
                    has_return = true;
                } else if (text == "<|call|>" || text == "<|calls|>") {
                    has_call = true;
                } else if (text == "<|flush|>") {
                    has_flush = true;
                } else if (text == "<|end|>") {
                    has_end = true;
                    end_id = tid;
                }
            }

            if ((has_return && has_call && has_end) || (has_call && has_flush && has_end)) {
                special_eog_ids.erase(end_id);

                auto & attr = id_to_token[end_id].attr;
                attr = LLAMA_TOKEN_ATTR_USER_DEFINED;

                LLAMA_LOG_WARN("%s: special_eog_ids contains both '<|return|>' and '<|call|>', or '<|calls|>' and '<|flush|>' tokens, removing '<|end|>' token from EOG list\n", __func__);
            }
        }

        // workaround for gemma4 and paddleocr: do not include </s> as an eog token
        {
            bool has_tool_response = false;
            bool has_s = false;

            llama_token s_id = LLAMA_TOKEN_NULL;

            for (auto tid : special_eog_ids) {
                if (tid < 0 || tid >= (llama_token) id_to_token.size()) {
                    continue;
                }
                const auto & text = id_to_token[tid].text;
                if (text == "<|tool_response>") {
                    has_tool_response = true;
                } else if (text == "</s>") {
                    has_s = true;
                    s_id = tid;
                }
            }

            if (has_tool_response && has_s) {
                special_eog_ids.erase(s_id);

                auto & attr = id_to_token[s_id].attr;
                attr = LLAMA_TOKEN_ATTR_NORMAL;

                LLAMA_LOG_WARN("%s: special_eog_ids contains '<|tool_response>', removing '</s>' token from EOG list\n", __func__);
            }
        }
    }

    // build special tokens cache
    {
        for (llama_token id = 0; id < (llama_token) n_tokens; ++id) {
            if (id_to_token[id].attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN)) {
                cache_special_tokens.push_back(id);
            }
        }

        std::sort(cache_special_tokens.begin(), cache_special_tokens.end(),
            [&] (const llama_token a, const llama_token b) {
                return id_to_token[a].text.size() > id_to_token[b].text.size();
            }
        );

        LLAMA_LOG_INFO("%s: special tokens cache size = %u\n", __func__, (uint32_t) cache_special_tokens.size());
    }

    // build token to piece cache
    {
        size_t size_cache = 0;

        std::vector<std::string> cache(n_tokens);

        for (uint32_t id = 0; id < n_tokens; ++id) {
            cache[id] = token_to_piece_for_cache(id, true);

            size_cache += cache[id].size();
        }

        std::swap(cache_token_to_piece, cache);

        LLAMA_LOG_INFO("%s: token to piece cache size = %.4f MB\n", __func__, size_cache / 1024.0 / 1024.0);
    }

    // Handle per token attributes
    //NOTE: Each model customizes per token attributes.
    //NOTE: Per token attributes are missing from the GGUF file.
    //TODO: Extract attributes from GGUF file.
    {
        auto _contains_any = [] (const std::string & str, const std::vector<std::string_view> & substrs) -> bool {
            for (const auto & substr : substrs) {
                if (str.find(substr) != std::string::npos) {
                    return true;
                }
            }
            return false;
        };

        auto _set_tokenid_attr = [&] (const llama_token id, llama_token_attr attr, bool value) {
            uint32_t current = id_to_token.at(id).attr;
            current = value ? (current | attr) : (current & ~attr);
            id_to_token[id].attr = (llama_token_attr) current;
        };

        auto _set_token_attr = [&] (const std::string & token, llama_token_attr attr, bool value) {
            _set_tokenid_attr(token_to_id.at(token), attr, value);
        };

        std::string model_name;
        std::string tokenizer_pre;
        std::string general_arch;

        ml.get_key(LLM_KV_GENERAL_NAME,  model_name,    false);
        ml.get_key(LLM_KV_TOKENIZER_PRE, tokenizer_pre, false);
        ml.get_key(LLM_KV_GENERAL_ARCHITECTURE, general_arch, false);

        // model name to lowercase
        std::transform(model_name.begin(), model_name.end(), model_name.begin(),
            [] (const std::string::value_type x) {
                return std::tolower(x);
            }
        );

        // set attributes by model/tokenizer/architecture name
        if (false
                || _contains_any(tokenizer_pre, {"jina-v2-de", "jina-v2-es", "jina-v2-code"})
                || _contains_any(general_arch, {"nomic-bert-moe", "jina-bert-v3"})
           ) {
            if (token_to_id.count("<mask>") == 0) {
                LLAMA_LOG_WARN("%s: Mask token is missing in vocab, please reconvert model!\n", __func__);
            } else {
                _set_token_attr("<mask>", LLAMA_TOKEN_ATTR_LSTRIP, true);
            }
        } else if (_contains_any(model_name, {"phi-3", "phi3"})) {
            for (auto id : cache_special_tokens) {
                _set_tokenid_attr(id, LLAMA_TOKEN_ATTR_RSTRIP, true);
            }
            for (const auto * token : {"</s>"}) {
                _set_token_attr(token, LLAMA_TOKEN_ATTR_RSTRIP, true);
            }
            for (const auto * token : {"<unk>", "<s>", "<|endoftext|>"}) {
                _set_token_attr(token, LLAMA_TOKEN_ATTR_RSTRIP, false);
            }
        } else if (_contains_any(model_name, {"modern-bert"})) {
            if (token_to_id.count("[MASK]") == 0 ) {
                LLAMA_LOG_WARN("%s: Mask token missing in vocab!\n", __func__);
            }
            else {
                _set_token_attr("[MASK]", LLAMA_TOKEN_ATTR_LSTRIP, true);
            }
        }
    }
}

enum llama_vocab_type llama_vocab::impl::get_type() const {
    return type;
}

std::string llama_vocab::impl::type_name() const{
    switch (type) {
        case LLAMA_VOCAB_TYPE_NONE:   return "no vocab";
        case LLAMA_VOCAB_TYPE_BPE:    return "BPE";
        case LLAMA_VOCAB_TYPE_TEST:   return "TEST";
        default:                      return "unknown";
    }
}

bool llama_vocab::impl::is_normal(llama_token id) const {
    GGML_ASSERT(type != LLAMA_VOCAB_TYPE_NONE);
    return id_to_token[id].attr & LLAMA_TOKEN_ATTR_NORMAL;
}

bool llama_vocab::impl::is_unknown(llama_token id) const {
    GGML_ASSERT(type != LLAMA_VOCAB_TYPE_NONE);
    return id_to_token[id].attr & LLAMA_TOKEN_ATTR_UNKNOWN;
}

bool llama_vocab::impl::is_control(llama_token id) const {
    GGML_ASSERT(type != LLAMA_VOCAB_TYPE_NONE);
    return id_to_token[id].attr & LLAMA_TOKEN_ATTR_CONTROL;
}

bool llama_vocab::impl::is_byte(llama_token id) const {
    GGML_ASSERT(type != LLAMA_VOCAB_TYPE_NONE);
    return id_to_token[id].attr & LLAMA_TOKEN_ATTR_BYTE;
}

bool llama_vocab::impl::is_user_defined(llama_token id) const {
    GGML_ASSERT(type != LLAMA_VOCAB_TYPE_NONE);
    return id_to_token[id].attr & LLAMA_TOKEN_ATTR_USER_DEFINED;
}

bool llama_vocab::impl::is_unused(llama_token id) const {
    GGML_ASSERT(type != LLAMA_VOCAB_TYPE_NONE);
    return id_to_token[id].attr & LLAMA_TOKEN_ATTR_UNUSED;
}

bool llama_vocab::impl::is_eog(llama_token id) const {
    return id != LLAMA_TOKEN_NULL && special_eog_ids.count(id) > 0;
}

uint8_t llama_vocab::impl::token_to_byte(llama_token id) const {
    GGML_ASSERT(get_type() != LLAMA_VOCAB_TYPE_NONE);
    GGML_ASSERT(is_byte(id));
    const auto & token_data = id_to_token.at(id);
    switch (get_type()) {
        case LLAMA_VOCAB_TYPE_BPE: {
            auto buf = token_data.text.substr(3, 2);
            return strtol(buf.c_str(), NULL, 16);
        }
        default:
            GGML_ABORT("fatal error");
    }
}

llama_token_attr llama_vocab::impl::token_get_attr(llama_token id) const {
    GGML_ASSERT(type != LLAMA_VOCAB_TYPE_NONE);
    return id_to_token.at(id).attr;
}

void llama_vocab::impl::init_tokenizer(enum llama_vocab_type type) {
    LLAMA_LOG_DEBUG("%s: initializing tokenizer for type %d\n", __func__, type);

    switch (type) {
        case LLAMA_VOCAB_TYPE_BPE:
            tokenizer = std::make_unique<llm_tokenizer_bpe>(vocab);
            break;
        case LLAMA_VOCAB_TYPE_TEST:
            tokenizer = std::make_unique<llm_tokenizer>();
            break;
        default:
            GGML_ABORT("unsupported vocab type");
    }
}

//
// (de-) tokenize
//

// #define PRETOKENIZERDEBUG

void llama_vocab::impl::tokenizer_st_partition(std::forward_list<fragment_buffer_variant> & buffer, bool parse_special) const {
    // for each special token
    for (const llama_token special_id : cache_special_tokens) {
        const auto & data = vocab.get_token_data(special_id);
        const auto & text = data.text;

        if (!parse_special && (data.attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_UNKNOWN))) {
            // Ignore control and unknown tokens when parse_special == false
            continue;
            // User-defined tokens are still pre-tokenized before everything else
            // ref: https://github.com/huggingface/tokenizers/blob/fdd26ba9a3f0c133427aab0423888cbde91362d7/tokenizers/src/tokenizer/mod.rs#L726
            // This is mostly relevant for neox-style tokenizers (mpt, olmo, stablelm, etc.)
        }

        // for each text fragment
        std::forward_list<fragment_buffer_variant>::iterator it = buffer.begin();
        while (it != buffer.end()) {
            auto & fragment = (*it);

            // if a fragment is text ( not yet processed )
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                const auto & raw_text = fragment.raw_text;

                auto raw_text_base_offset = fragment.offset;
                auto raw_text_base_length = fragment.length;

                // loop over the text
                while (true) {
                    // find the first occurrence of a given special token in this fragment
                    //  passing offset argument only limit the "search area" but match coordinates
                    //  are still relative to the source full raw_text
                    //  string_view begins at pos 0 for the same reason
                    auto match = std::string_view(raw_text.data(), raw_text_base_offset + raw_text_base_length).find(text, raw_text_base_offset);

                    // no occurrences found, stop processing this fragment for a given special token
                    if (match == std::string::npos) break;

#ifdef PRETOKENIZERDEBUG
                    LLAMA_LOG_WARN("FF: (%ld %ld %ld) '%s'\n", raw_text->length(), raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    auto source = std::distance(buffer.begin(), it);

                    // if match is further than base offset
                    //  then we have some text to the left of it
                    if (match > raw_text_base_offset) {
                        // left
                        const int64_t left_reminder_offset = raw_text_base_offset + 0;
                        int64_t left_reminder_length = match - raw_text_base_offset;

                        if (data.attr & LLAMA_TOKEN_ATTR_LSTRIP) {
                            while (left_reminder_length > 0 && isspace(raw_text[left_reminder_offset + left_reminder_length - 1])) {
                                left_reminder_length--;
                            }
                        }

                        if (left_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, left_reminder_offset, left_reminder_length);
                            it++;
                        }

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("FL: (%ld %ld) '%s'\n", left_reminder_offset, left_reminder_length, raw_text->substr(left_reminder_offset, left_reminder_length).c_str());
#endif
                    }

                    // special token
                    buffer.emplace_after(it, special_id);
                    it++;

                    // right
                    if (match + text.length() < raw_text_base_offset + raw_text_base_length) {
                        int64_t right_reminder_offset = match + text.length();
                        int64_t right_reminder_length = raw_text_base_length - ((match - raw_text_base_offset) + text.length());

                        if (data.attr & LLAMA_TOKEN_ATTR_RSTRIP) {
                            while (right_reminder_length > 0 && isspace(raw_text[right_reminder_offset])) {
                                right_reminder_offset++;
                                right_reminder_length--;
                            }
                        }

                        if (right_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, right_reminder_offset, right_reminder_length);
                            it++;
                        }

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("FR: (%ld %ld) '%s'\n", right_reminder_offset, right_reminder_length, raw_text->substr(right_reminder_offset, right_reminder_length).c_str());
#endif

                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source - 1)));
                        }

                        // repeat for the right side
                        raw_text_base_offset = right_reminder_offset;
                        raw_text_base_length = right_reminder_length;

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("RR: (%ld %ld) '%s'\n", raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    } else {
                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source - 1)));
                        }
                        break;
                    }
                }
            }
            it++;
        }
    }
}

// NOTE: avoid ever using this except for building the token_to_piece caches
std::string llama_vocab::impl::token_to_piece_for_cache(llama_token token, bool special) const {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache
    const int n_chars = vocab.token_to_piece(token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = vocab.token_to_piece(token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

static void llama_escape_whitespace(std::string & text) {
    replace_all(text, " ", "\xe2\x96\x81");
}

static void llama_unescape_whitespace(std::string & word) {
    replace_all(word, "\xe2\x96\x81", " ");
}

static std::string llama_decode_text(const std::string & text) {
    std::string decoded_text;

    const auto cpts = unicode_cpts_from_utf8(text);
    for (const auto cpt : cpts) {
        const auto utf8 = unicode_cpt_to_utf8(cpt);
        try {
            decoded_text += unicode_utf8_to_byte(utf8);
        } catch (const std::out_of_range & /*e*/) {
            decoded_text += "[UNK_BYTE_0x";
            for (const auto c : utf8) {
                decoded_text += format("%02x", (uint8_t) c);
            }
            decoded_text += text + "]";
        }
    }

    return decoded_text;
}

std::vector<llama_token> llama_vocab::impl::tokenize(
        const std::string & raw_text,
        bool add_special,
        bool parse_special) const {
    GGML_ASSERT(tokenizer && "Tokenizer not initialized. Call llama_vocab::init_tokenizer() first.");

    std::vector<llama_token> output;
    std::forward_list<fragment_buffer_variant> fragment_buffer;

    if (!raw_text.empty()) {
        fragment_buffer.emplace_front(raw_text, 0, raw_text.length());
        tokenizer_st_partition(fragment_buffer, parse_special);
    }

    switch (get_type()) {
        case LLAMA_VOCAB_TYPE_BPE:
            {
                // it calls some other methods that are not exist in llm_tokenizer,
                // here just cast it to bpe tokenizer object
                const llm_tokenizer_bpe * tok_bpe = static_cast<const llm_tokenizer_bpe *>(tokenizer.get());

                llm_tokenizer_bpe_session session(vocab, *tok_bpe);

                if (add_special) {
                    session.append_bos(output);
                }
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text = fragment.raw_text.substr(fragment.offset, fragment.length);

                        if (escape_whitespaces) {
                            llama_escape_whitespace(text);
                        }

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif
                        session.tokenize(text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        session.append(fragment.token, output);
                    }
                }

                if (add_special) {
                    session.append_eos(output);
                    session.check_double_bos_eos(output);
                }
            } break;
        case LLAMA_VOCAB_TYPE_TEST:
            {
                const uint32_t n_vocab = vocab.n_tokens();
                constexpr size_t chunk_size = 5;

                // reserve output to avoid repeated reallocations
                size_t n_tokens = 0;
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        n_tokens += (fragment.length + chunk_size - 1) / chunk_size;
                    } else {
                        ++n_tokens;
                    }
                }
                output.reserve(output.size() + n_tokens);

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        const auto & text = fragment.raw_text;
                        const size_t begin = fragment.offset;
                        const size_t end   = begin + fragment.length;
                        size_t pos = begin;
                        while (pos < end) {
                            const size_t n = std::min(chunk_size, end - pos);
                            uint64_t hash = 0;
                            for (size_t i = 0; i < n; ++i) {
                                hash = hash*31 + (uint8_t) text[pos + i];
                            }
                            output.push_back((llama_token)(hash % n_vocab));
                            pos += n;
                        }
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    return output;
}

int32_t llama_vocab::impl::token_to_piece(llama_token token, char * buf, int32_t length, int32_t lstrip, bool special) const {
    // ref: https://github.com/ggml-org/llama.cpp/pull/7587#discussion_r1620983843
    static const int attr_special = LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_CONTROL;
    const llama_token_attr attr = token_get_attr(token);
    if (!special && (attr & attr_special)) {
        return 0;
    }

    // copy piece chars to output text buffer
    // skip up to 'lstrip' leading spaces before copying
    auto _try_copy = [=] (const char * token, size_t size) -> int32_t {
        if (size >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            GGML_ABORT("invalid token size: %zu exceeds int32_t limit", size);
        }

        for (int32_t i = 0; i < lstrip && size && *token == ' '; ++i) {
            token++;
            size--;
        }
        if (length < (int32_t)size) {
            return -(int32_t) size;
        }
        memcpy(buf, token, size);
        return (int32_t) size;
    };

    // if we have a cache - use it
    {
        const auto & cache = cache_token_to_piece;

        if (!cache.empty()) {
            const auto & result = cache.at(token);
            return _try_copy(result.data(), result.size());
        }
    }

    if (0 <= token && token < (int32_t) id_to_token.size()) {
        const std::string & token_text = id_to_token[token].text;
        switch (get_type()) {
            case LLAMA_VOCAB_TYPE_BPE: {
                // NOTE: we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                if (attr & (attr_special | LLAMA_TOKEN_ATTR_USER_DEFINED)) {
                    return _try_copy(token_text.data(), token_text.size());
                }
                if (attr & LLAMA_TOKEN_ATTR_NORMAL) {
                    if (escape_whitespaces) {
                        // SPM-style BPE: tokens contain ▁ for spaces
                        std::string result = token_text;
                        llama_unescape_whitespace(result);
                        return _try_copy(result.data(), result.size());
                    }
                    std::string result = llama_decode_text(token_text);
                    return _try_copy(result.data(), result.size());
                }
                if (attr & LLAMA_TOKEN_ATTR_BYTE) {
                    char byte = (char) token_to_byte(token);
                    return _try_copy((char*) &byte, 1);
                }
                break;
            }
            case LLAMA_VOCAB_TYPE_TEST: {
                // tokens -> text: simply stringify the token id in hex
                std::string result = format("%x", token);
                return _try_copy(result.data(), result.size());
            }
            default:
                GGML_ABORT("fatal error");
        }
    }

    return 0;
}

const std::string & llama_vocab::impl::token_to_piece(llama_token token) const {
    return cache_token_to_piece.at(token);
}

int32_t llama_vocab::impl::detokenize(
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special) const {
    if (type == LLAMA_VOCAB_TYPE_NONE) {
        return 0;
    }

    GGML_ASSERT(tokenizer && "Tokenizer not initialized. Call llama_vocab::init_tokenizer() first.");

    int32_t avail = text_len_max;
    int32_t total = 0;

    // remove the leading space
    bool remove_space = add_space_prefix;

    if (remove_special && add_bos) {
        if (n_tokens > 0 && tokens[0] == special_bos_id) {
            remove_space = false;
            n_tokens--;
            tokens++;
        }
    }

    if (remove_special && add_eos) {
        if (n_tokens > 0 && tokens[n_tokens - 1] == special_eos_id) {
            n_tokens--;
        }
    }

    for (int32_t i = 0; i < n_tokens; ++i) {
        GGML_ASSERT(avail >= 0);
        int32_t n_chars = token_to_piece(tokens[i], text, avail, remove_space, unparse_special);
        remove_space = false;
        if (n_chars < 0) {
            avail = 0;
            total -= n_chars;
        } else if (n_chars > 0) {
            avail -= n_chars;
            text  += n_chars;
            total += n_chars;
        }
    }

    if (total > text_len_max) {
        return -total;
    }

    if (clean_spaces) {
        text -= total;  // restart text

        // first pass: characters ?!.,  //TODO: where do these characters come from?
        const int32_t total1 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total1; ++i) {
            const char x = text[i];
            if (text[i - 1] == ' ') {
                if (x == '?' || x == '!' || x == '.' || x == ',') {  // " ?", " !", " .", " ,"
                    total--;  // remove space
                }
            }
            text[total++] = x;
        }

        // second pass: strip single apostrophe between spaces
        const int32_t total2 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total2; ++i) {
            const char x = text[i];
            if (x == '\'' && i + 1 < total2 && text[i - 1] == ' ' && text[i + 1] == ' ') {  // " ' "
                total--;           // remove prev space
                text[++i] = '\0';  // remove next space
            }
            text[total++] = x;
        }

        // third pass: apostrophe contractions  //NOTE: this makes sense?
        const int32_t total3 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total3; ++i) {
            const char x = text[i];
            if (text[i - 1] == ' ') {
                if (x == '\'' && i + 1 < total3) {
                    const char x1 = text[i + 1];
                    if (x1 == 't' || x1 == 'd') {  // " 't", " 'd"
                        //total--;  // remove space
                    } else if (x1 == 's' || x1 == 'm') {  // " 's", " 'm"
                        total--;  // remove space
                    } else if (i + 2 < total3) {
                        const char x2 = text[i + 2];
                        if ((x1 == 'l' && x2 == 'l')) {  // " 'll"
                            //total--;  // remove space
                        } else if ((x1 == 'r' && x2 == 'e') || (x1 == 'v' && x2 == 'e')) {  // " 're", " 've"
                            total--;  // remove space
                        } else {
                            //total--;  // remove space
                        }
                    } else {
                        //total--;  // remove space
                    }
                }
            }
            text[total++] = x;
        }
    }

    return total <= text_len_max ? total : -total;
}

void llama_vocab::impl::print_info() const {
    LLAMA_LOG_INFO("%s: vocab type            = %s\n",     __func__, type_name().c_str());
    LLAMA_LOG_INFO("%s: n_vocab               = %u\n",     __func__, vocab.n_tokens());
    LLAMA_LOG_INFO("%s: n_merges              = %u\n",     __func__, (uint32_t) bpe_ranks.size());

    // special tokens
    if (special_bos_id  != LLAMA_TOKEN_NULL)    { LLAMA_LOG_INFO( "%s: BOS token             = %d '%s'\n", __func__, special_bos_id,     id_to_token.at(special_bos_id).text.c_str() );  }
    if (special_eos_id  != LLAMA_TOKEN_NULL)    { LLAMA_LOG_INFO( "%s: EOS token             = %d '%s'\n", __func__, special_eos_id,     id_to_token.at(special_eos_id).text.c_str() );  }
    if (special_eot_id  != LLAMA_TOKEN_NULL)    { LLAMA_LOG_INFO( "%s: EOT token             = %d '%s'\n", __func__, special_eot_id,     id_to_token.at(special_eot_id).text.c_str() );  }
    if (special_eom_id  != LLAMA_TOKEN_NULL)    { LLAMA_LOG_INFO( "%s: EOM token             = %d '%s'\n", __func__, special_eom_id,     id_to_token.at(special_eom_id).text.c_str() );  }
    if (special_unk_id  != LLAMA_TOKEN_NULL)    { LLAMA_LOG_INFO( "%s: UNK token             = %d '%s'\n", __func__, special_unk_id,     id_to_token.at(special_unk_id).text.c_str() );  }
    if (special_sep_id  != LLAMA_TOKEN_NULL)    { LLAMA_LOG_INFO( "%s: SEP token             = %d '%s'\n", __func__, special_sep_id,     id_to_token.at(special_sep_id).text.c_str() );  }
    if (special_pad_id  != LLAMA_TOKEN_NULL)    { LLAMA_LOG_INFO( "%s: PAD token             = %d '%s'\n", __func__, special_pad_id,     id_to_token.at(special_pad_id).text.c_str() );  }
    if (special_mask_id != LLAMA_TOKEN_NULL)    { LLAMA_LOG_INFO( "%s: MASK token            = %d '%s'\n", __func__, special_mask_id,    id_to_token.at(special_mask_id).text.c_str() ); }

    if (linefeed_id != LLAMA_TOKEN_NULL)        { LLAMA_LOG_INFO( "%s: LF token              = %d '%s'\n", __func__, linefeed_id,        id_to_token.at(linefeed_id).text.c_str() ); }

    if (special_fim_pre_id != LLAMA_TOKEN_NULL) { LLAMA_LOG_INFO( "%s: FIM PRE token         = %d '%s'\n", __func__, special_fim_pre_id, id_to_token.at(special_fim_pre_id).text.c_str() ); }
    if (special_fim_suf_id != LLAMA_TOKEN_NULL) { LLAMA_LOG_INFO( "%s: FIM SUF token         = %d '%s'\n", __func__, special_fim_suf_id, id_to_token.at(special_fim_suf_id).text.c_str() ); }
    if (special_fim_mid_id != LLAMA_TOKEN_NULL) { LLAMA_LOG_INFO( "%s: FIM MID token         = %d '%s'\n", __func__, special_fim_mid_id, id_to_token.at(special_fim_mid_id).text.c_str() ); }
    if (special_fim_pad_id != LLAMA_TOKEN_NULL) { LLAMA_LOG_INFO( "%s: FIM PAD token         = %d '%s'\n", __func__, special_fim_pad_id, id_to_token.at(special_fim_pad_id).text.c_str() ); }
    if (special_fim_rep_id != LLAMA_TOKEN_NULL) { LLAMA_LOG_INFO( "%s: FIM REP token         = %d '%s'\n", __func__, special_fim_rep_id, id_to_token.at(special_fim_rep_id).text.c_str() ); }
    if (special_fim_sep_id != LLAMA_TOKEN_NULL) { LLAMA_LOG_INFO( "%s: FIM SEP token         = %d '%s'\n", __func__, special_fim_sep_id, id_to_token.at(special_fim_sep_id).text.c_str() ); }

    for (const auto & id : special_eog_ids) {
        LLAMA_LOG_INFO( "%s: EOG token             = %d '%s'\n", __func__, id, id_to_token.at(id).text.c_str() );
    }

    LLAMA_LOG_INFO("%s: max token length      = %d\n", __func__, max_token_len);
}

llama_vocab::llama_vocab() : pimpl(new impl(*this)) {
}

llama_vocab::~llama_vocab() = default;

void llama_vocab::load(llama_model_loader & ml, const LLM_KV & kv) {
    pimpl->load(ml, kv);
}

std::string llama_vocab::get_tokenizer_model() const {
    return pimpl->tokenizer_model;
}

std::string llama_vocab::get_tokenizer_pre() const {
    return pimpl->tokenizer_pre;
}

enum llama_vocab_type llama_vocab::get_type() const {
    return pimpl->type;
}

enum llama_vocab_pre_type llama_vocab::get_pre_type() const {
    return pimpl->pre_type;
}

uint32_t llama_vocab::n_tokens() const {
    return (uint32_t) pimpl->id_to_token.size();
}

uint32_t llama_vocab::n_token_types() const {
    return (uint32_t) pimpl->n_token_types;
}

std::string llama_vocab::type_name() const{
    return pimpl->type_name();
}

bool llama_vocab::is_normal(llama_token id) const {
    return pimpl->is_normal(id);
}

bool llama_vocab::is_unknown(llama_token id) const {
    return pimpl->is_unknown(id);
}

bool llama_vocab::is_control(llama_token id) const {
    return pimpl->is_control(id);
}

bool llama_vocab::is_byte(llama_token id) const {
    return pimpl->is_byte(id);
}

bool llama_vocab::is_user_defined(llama_token id) const {
    return pimpl->is_user_defined(id);
}

bool llama_vocab::is_unused(llama_token id) const {
    return pimpl->is_unused(id);
}

bool llama_vocab::is_eog(llama_token id) const {
    return pimpl->is_eog(id);
}

uint8_t llama_vocab::token_to_byte(llama_token id) const {
    return pimpl->token_to_byte(id);
}

llama_token llama_vocab::byte_to_token(uint8_t ch) const {
    GGML_ASSERT(get_type() != LLAMA_VOCAB_TYPE_NONE);
    switch (get_type()) {
        case LLAMA_VOCAB_TYPE_BPE: {
            return pimpl->token_to_id.at(unicode_byte_to_utf8(ch));
        }
        case LLAMA_VOCAB_TYPE_TEST:
            // TEST tokens have no byte-level mapping
            return LLAMA_TOKEN_NULL;
        default:
            GGML_ABORT("fatal error");
    }
}

llama_token llama_vocab::text_to_token(const std::string & text) const {
    GGML_ASSERT(pimpl->type != LLAMA_VOCAB_TYPE_NONE);
    auto it = pimpl->token_to_id.find(text);
    if (it != pimpl->token_to_id.end()) {
        return (*it).second;
    }
    return LLAMA_TOKEN_NULL;
}

const llama_vocab::token_data & llama_vocab::get_token_data(llama_token id) const {
    GGML_ASSERT(pimpl->type != LLAMA_VOCAB_TYPE_NONE);
    return pimpl->id_to_token.at(id);
}

const char * llama_vocab::token_get_text(llama_token id) const {
    GGML_ASSERT(pimpl->type != LLAMA_VOCAB_TYPE_NONE);
    return pimpl->id_to_token.at(id).text.c_str();
}

float llama_vocab::token_get_score(llama_token id) const {
    GGML_ASSERT(pimpl->type != LLAMA_VOCAB_TYPE_NONE);
    return pimpl->id_to_token.at(id).score;
}

llama_token_attr llama_vocab::token_get_attr(llama_token id) const {
    return pimpl->token_get_attr(id);
}

llama_token llama_vocab::token_bos() const {
    return pimpl->special_bos_id;
}

llama_token llama_vocab::token_eos() const {
    return pimpl->special_eos_id;
}

llama_token llama_vocab::token_eot() const {
    return pimpl->special_eot_id;
}

llama_token llama_vocab::token_eom() const {
    return pimpl->special_eom_id;
}

llama_token llama_vocab::token_unk() const {
    return pimpl->special_unk_id;
}

llama_token llama_vocab::token_sep() const {
    return pimpl->special_sep_id;
}

llama_token llama_vocab::token_nl() const {
    return pimpl->linefeed_id;
}

llama_token llama_vocab::token_pad() const {
    return pimpl->special_pad_id;
}

llama_token llama_vocab::token_prefix() const {
    return pimpl->special_fim_pre_id;
}

llama_token llama_vocab::token_middle() const {
    return pimpl->special_fim_mid_id;
}

llama_token llama_vocab::token_suffix() const {
    return pimpl->special_fim_suf_id;
}

llama_token llama_vocab::token_fim_pre() const {
    return pimpl->special_fim_pre_id;
}

llama_token llama_vocab::token_fim_suf() const {
    return pimpl->special_fim_suf_id;
}

llama_token llama_vocab::token_fim_mid() const {
    return pimpl->special_fim_mid_id;
}

llama_token llama_vocab::token_fim_pad() const {
    return pimpl->special_fim_pad_id;
}

llama_token llama_vocab::token_fim_rep() const {
    return pimpl->special_fim_rep_id;
}

llama_token llama_vocab::token_fim_sep() const {
    return pimpl->special_fim_sep_id;
}

llama_token llama_vocab::token_mask() const {
    return pimpl->special_mask_id;
}

bool llama_vocab::get_add_space_prefix() const {
    return pimpl->add_space_prefix;
}

bool llama_vocab::get_add_bos() const {
    return pimpl->add_bos;
}

bool llama_vocab::get_add_eos() const {
    return pimpl->add_eos;
}

bool llama_vocab::get_add_sep() const {
    return pimpl->add_sep;
}

bool llama_vocab::get_ignore_merges() const {
    return pimpl->ignore_merges;
}

bool llama_vocab::get_clean_spaces() const {
    return pimpl->clean_spaces;
}

bool llama_vocab::get_remove_extra_whitespaces() const {
    return pimpl->remove_extra_whitespaces;
}

bool llama_vocab::get_escape_whitespaces() const {
    return pimpl->escape_whitespaces;
}

bool llama_vocab::get_treat_whitespace_as_suffix() const {
    return pimpl->treat_whitespace_as_suffix;
}

const llama_vocab::normalizer_options & llama_vocab::get_normalizer_opts() const {
    return pimpl->normalizer_opts;
}

const std::vector<llama_token> & llama_vocab::get_suppress_tokens() const {
    return pimpl->suppress_tokens;
}

int llama_vocab::max_token_len() const {
    return pimpl->max_token_len;
}

int llama_vocab::find_bpe_rank(const std::string & token_left, const std::string & token_right) const {
    GGML_ASSERT(token_left.find(' ')   == std::string::npos);
    GGML_ASSERT(token_right.find(' ')  == std::string::npos);

    auto it = pimpl->bpe_ranks.find(std::make_pair(token_left, token_right));
    if (it == pimpl->bpe_ranks.end()) {
        return -1;
    }

    return it->second;
}

std::vector<std::string> llama_vocab::get_bpe_merges() const {
    int max_rank = -1;
    for (const auto & pair : pimpl->bpe_ranks) {
        max_rank = std::max(max_rank, pair.second);
    }
    std::vector<std::string> result(max_rank + 1);

    for (const auto & pair : pimpl->bpe_ranks) {
        result[pair.second] = pair.first.first + " " + pair.first.second;
    }

    return result;
}

std::vector<char> llama_vocab::get_precompiled_charsmap() const {
    return pimpl->precompiled_charsmap;
}

int32_t llama_vocab::tokenize(
                  const char * text,
                     int32_t   text_len,
                 llama_token * tokens,
                     int32_t   n_tokens_max,
                        bool   add_special,
                        bool   parse_special) const {
    auto res = tokenize(std::string(text, text_len), add_special, parse_special);
    if (res.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        LLAMA_LOG_ERROR("%s: tokenization result size %zu exceeds int32_t limit\n", __func__, res.size());
        return std::numeric_limits<int32_t>::min();
    }

    if (n_tokens_max < (int) res.size()) {
        // LLAMA_LOG_ERROR("%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

std::vector<llama_token> llama_vocab::tokenize(
        const std::string & raw_text,
        bool add_special,
        bool parse_special) const {
    return pimpl->tokenize(raw_text, add_special, parse_special);
}

const std::string & llama_vocab::token_to_piece(llama_token token) const {
    return pimpl->token_to_piece(token);
}

int32_t llama_vocab::token_to_piece(llama_token token, char * buf, int32_t length, int32_t lstrip, bool special) const {
    return pimpl->token_to_piece(token, buf, length, lstrip, special);
}

int32_t llama_vocab::detokenize(
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special) const {
    return pimpl->detokenize(tokens, n_tokens, text, text_len_max, remove_special, unparse_special);
}

std::string llama_vocab::detokenize(const std::vector<llama_token> & tokens, bool special) const {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = detokenize(tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = detokenize(tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        GGML_ASSERT(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

void llama_vocab::print_info() const {
    pimpl->print_info();
}

//
// interface implementation
//

int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab) {
    return vocab->n_tokens();
}

// deprecated
int32_t llama_n_vocab(const struct llama_vocab * vocab) {
    return llama_vocab_n_tokens(vocab);
}

enum llama_vocab_type llama_vocab_type(const struct llama_vocab * vocab) {
    return vocab->get_type();
}

const char * llama_vocab_get_text(const struct llama_vocab * vocab, llama_token token) {
    return vocab->token_get_text(token);
}

float llama_vocab_get_score(const struct llama_vocab * vocab, llama_token token) {
    return vocab->token_get_score(token);
}

enum llama_token_attr llama_vocab_get_attr(const struct llama_vocab * vocab, llama_token token) {
    return vocab->token_get_attr(token);
}

bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token) {
    return vocab->is_eog(token);
}

bool llama_vocab_is_control(const struct llama_vocab * vocab, llama_token token) {
    return vocab->is_control(token);
}

llama_token llama_vocab_bos(const struct llama_vocab * vocab) {
    return vocab->token_bos();
}

llama_token llama_vocab_eos(const struct llama_vocab * vocab) {
    return vocab->token_eos();
}

llama_token llama_vocab_eot(const struct llama_vocab * vocab) {
    return vocab->token_eot();
}

// deprecated
llama_token llama_vocab_cls(const struct llama_vocab * vocab) {
    return vocab->token_bos();
}

llama_token llama_vocab_sep(const struct llama_vocab * vocab) {
    return vocab->token_sep();
}

llama_token llama_vocab_nl (const struct llama_vocab * vocab) {
    return vocab->token_nl();
}

llama_token llama_vocab_pad(const struct llama_vocab * vocab) {
    return vocab->token_pad();
}

bool llama_vocab_get_add_bos(const struct llama_vocab * vocab) {
    return vocab->get_add_bos();
}

bool llama_vocab_get_add_eos(const struct llama_vocab * vocab) {
    return vocab->get_add_eos();
}

bool llama_vocab_get_add_sep(const struct llama_vocab * vocab) {
    return vocab->get_add_sep();
}

const llama_token * llama_vocab_get_suppress_tokens(const struct llama_vocab * vocab, int32_t * n_suppress_tokens) {
    const std::vector<llama_token> & tokens = vocab->get_suppress_tokens();
    if (n_suppress_tokens) {
        *n_suppress_tokens = (int32_t) tokens.size();
    }
    return tokens.data();
}

llama_token llama_vocab_fim_pre(const struct llama_vocab * vocab) {
    return vocab->token_fim_pre();
}

llama_token llama_vocab_fim_suf(const struct llama_vocab * vocab) {
    return vocab->token_fim_suf();
}

llama_token llama_vocab_fim_mid(const struct llama_vocab * vocab) {
    return vocab->token_fim_mid();
}

llama_token llama_vocab_fim_pad(const struct llama_vocab * vocab) {
    return vocab->token_fim_pad();
}

llama_token llama_vocab_fim_rep(const struct llama_vocab * vocab) {
    return vocab->token_fim_rep();
}

llama_token llama_vocab_fim_sep(const struct llama_vocab * vocab) {
    return vocab->token_fim_sep();
}

llama_token llama_vocab_mask(const struct llama_vocab* vocab) {
    return vocab->token_mask();
}

// deprecated
const char * llama_token_get_text(const struct llama_vocab * vocab, llama_token token) {
    return llama_vocab_get_text(vocab, token);
}

// deprecated
float llama_token_get_score(const struct llama_vocab * vocab, llama_token token) {
    return llama_vocab_get_score(vocab, token);
}

// deprecated
enum llama_token_attr llama_token_get_attr(const struct llama_vocab * vocab, llama_token token) {
    return llama_vocab_get_attr(vocab, token);
}

// deprecated
bool llama_token_is_eog(const struct llama_vocab * vocab, llama_token token) {
    return llama_vocab_is_eog(vocab, token);
}

// deprecated
bool llama_token_is_control(const struct llama_vocab * vocab, llama_token token) {
    return llama_vocab_is_control(vocab, token);
}

// deprecated
llama_token llama_token_bos(const struct llama_vocab * vocab) {
    return llama_vocab_bos(vocab);
}

// deprecated
llama_token llama_token_eos(const struct llama_vocab * vocab) {
    return llama_vocab_eos(vocab);
}

// deprecated
llama_token llama_token_eot(const struct llama_vocab * vocab) {
    return llama_vocab_eot(vocab);
}

// deprecated
llama_token llama_token_cls(const struct llama_vocab * vocab) {
    //return llama_vocab_cls(vocab);
    return llama_vocab_bos(vocab); // avoid deprecation warning
}

// deprecated
llama_token llama_token_sep(const struct llama_vocab * vocab) {
    return llama_vocab_sep(vocab);
}

// deprecated
llama_token llama_token_nl (const struct llama_vocab * vocab) {
    return llama_vocab_nl(vocab);
}

// deprecated
llama_token llama_token_pad(const struct llama_vocab * vocab) {
    return llama_vocab_pad(vocab);
}

// deprecated
bool llama_add_bos_token(const struct llama_vocab * vocab) {
    return llama_vocab_get_add_bos(vocab);
}

// deprecated
bool llama_add_eos_token(const struct llama_vocab * vocab) {
    return llama_vocab_get_add_eos(vocab);
}

// deprecated
llama_token llama_token_fim_pre(const struct llama_vocab * vocab) {
    return llama_vocab_fim_pre(vocab);
}

// deprecated
llama_token llama_token_fim_suf(const struct llama_vocab * vocab) {
    return llama_vocab_fim_suf(vocab);
}

// deprecated
llama_token llama_token_fim_mid(const struct llama_vocab * vocab) {
    return llama_vocab_fim_mid(vocab);
}

// deprecated
llama_token llama_token_fim_pad(const struct llama_vocab * vocab) {
    return llama_vocab_fim_pad(vocab);
}

// deprecated
llama_token llama_token_fim_rep(const struct llama_vocab * vocab) {
    return llama_vocab_fim_rep(vocab);
}

// deprecated
llama_token llama_token_fim_sep(const struct llama_vocab * vocab) {
    return llama_vocab_fim_sep(vocab);
}

//
// tokenization
//

int32_t llama_tokenize(
    const struct llama_vocab * vocab,
                  const char * text,
                     int32_t   text_len,
                 llama_token * tokens,
                     int32_t   n_tokens_max,
                        bool   add_special,
                        bool   parse_special) {
    return vocab->tokenize(text, text_len, tokens, n_tokens_max, add_special, parse_special);
}

int32_t llama_token_to_piece(
    const struct llama_vocab * vocab,
                 llama_token   token,
                        char * buf,
                     int32_t   length,
                     int32_t   lstrip,
                        bool   special) {
    return vocab->token_to_piece(token, buf, length, lstrip, special);
}

int32_t llama_detokenize(
    const struct llama_vocab * vocab,
           const llama_token * tokens,
                     int32_t   n_tokens,
                        char * text,
                     int32_t   text_len_max,
                        bool   remove_special,
                        bool   unparse_special) {
    return vocab->detokenize(tokens, n_tokens, text, text_len_max, remove_special, unparse_special);
}
