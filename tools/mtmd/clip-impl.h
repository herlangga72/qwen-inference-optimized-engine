#pragma once

#include "ggml.h"
#include "gguf.h"
#include "clip.h"

#include <array>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cinttypes>
#include <string>
#include <map>
#include <sstream>
#include <vector>
#include <memory>
#include <fstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// Internal header for clip.cpp

#define MTMD_INTERNAL_HEADER

#define KEY_NAME                "general.name"
#define KEY_DESCRIPTION         "general.description"
#define KEY_PROJ_TYPE           "clip.projector_type"
#define KEY_HAS_AUDIO_ENC       "clip.has_audio_encoder"
#define KEY_HAS_VISION_ENC      "clip.has_vision_encoder"
#define KEY_HAS_GEN_AUDIO_ENC   "clip.has_gen_audio_encoder"
#define KEY_USE_GELU            "clip.use_gelu"
#define KEY_USE_SILU            "clip.use_silu"

#define KEY_N_EMBD              "clip.%s.embedding_length"
#define KEY_N_FF                "clip.%s.feed_forward_length"
#define KEY_N_BLOCK             "clip.%s.block_count"
#define KEY_PROJ_DIM            "clip.%s.projection_dim"
#define KEY_N_HEAD              "clip.%s.attention.head_count"
#define KEY_N_HEAD_KV           "clip.%s.attention.head_count_kv"
#define KEY_N_EMBD_HEAD         "clip.%s.attention.head_dim"
#define KEY_LAYER_NORM_EPS      "clip.%s.attention.layer_norm_epsilon"
#define KEY_FEATURE_LAYERS      "clip.%s.feature_layer"

// vision-specific
#define KEY_VISION_PROJ_TYPE        "clip.vision.projector_type" // for models with mixed modalities
#define KEY_IMAGE_SIZE              "clip.vision.image_size"
#define KEY_PATCH_SIZE              "clip.vision.patch_size"
#define KEY_IMAGE_MEAN              "clip.vision.image_mean"
#define KEY_IMAGE_STD               "clip.vision.image_std"
#define KEY_SPATIAL_MERGE_SIZE      "clip.vision.spatial_merge_size"

#define KEY_MM_PATCH_MERGE_TYPE    "clip.vision.mm_patch_merge_type"
#define KEY_IMAGE_GRID_PINPOINTS   "clip.vision.image_grid_pinpoints"
#define KEY_WIN_ATTN_PATTERN       "clip.vision.n_wa_pattern"
#define KEY_MINICPMV_VERSION       "clip.minicpmv_version"
#define KEY_MINICPMV_QUERY_NUM     "clip.minicpmv_query_num"
// audio-specific
#define KEY_AUDIO_PROJ_TYPE        "clip.audio.projector_type" // for models with mixed modalities
#define KEY_A_NUM_MEL_BINS         "clip.audio.num_mel_bins"
// audio generation (gen-audio)-specific
#define KEY_GEN_AUDIO_PROJ_TYPE    "clip.gen.audio.projector_type" // for models with mixed modalities
// name of the weight variant, for settings that are not in the checkpoint
#define KEY_GEN_AUDIO_VARIANT      "clip.gen.audio.model_variant"

//
// tensor name constants
//

#define TN_POS_EMBD        "%s.position_embd.weight"
#define TN_CLASS_EMBD      "v.class_embd"
#define TN_PATCH_EMBD      "v.patch_embd.weight"  // not rename tensor with ".0" postfix for backward compat
#define TN_PATCH_EMBD_1    "v.patch_embd.weight.1"
#define TN_PATCH_BIAS      "v.patch_embd.bias"
#define TN_NORM_EMBD       "v.norm_embd.%s"
#define TN_ATTN_QKV        "%s.blk.%d.attn_qkv.%s"
#define TN_ATTN_K          "%s.blk.%d.attn_k.%s"
#define TN_ATTN_Q          "%s.blk.%d.attn_q.%s"
#define TN_ATTN_V          "%s.blk.%d.attn_v.%s"
#define TN_ATTN_OUTPUT     "%s.blk.%d.attn_out.%s"
#define TN_ATTN_SINKS      "%s.blk.%d.attn_sinks"
#define TN_ATTN_K_NORM     "%s.blk.%d.attn_k_norm.%s"
#define TN_ATTN_Q_NORM     "%s.blk.%d.attn_q_norm.%s"
#define TN_FFN_DOWN        "%s.blk.%d.ffn_down.%s"
#define TN_FFN_GATE        "%s.blk.%d.ffn_gate.%s"
#define TN_FFN_UP          "%s.blk.%d.ffn_up.%s"
#define TN_FFN_GATE_INP    "%s.blk.%d.ffn_gate_inp.%s"    // MoE router (dots3note)
#define TN_FFN_GATE_EXPS   "%s.blk.%d.ffn_gate_exps.%s"
#define TN_FFN_UP_EXPS     "%s.blk.%d.ffn_up_exps.%s"
#define TN_FFN_DOWN_EXPS   "%s.blk.%d.ffn_down_exps.%s"
#define TN_FFN_EXP_PROBS_B "%s.blk.%d.exp_probs_b.%s"
#define TN_LN_1            "%s.blk.%d.ln1.%s" // layer norm
#define TN_LN_2            "%s.blk.%d.ln2.%s" // layer norm
#define TN_LS_1            "%s.blk.%d.ls1.%s"         // layer scale
#define TN_LS_2            "%s.blk.%d.ls2.%s"         // layer scale
#define TN_LS_OUT          "%s.blk.%d.out_scale.%s"      // layer out scale (gemma4)
#define TN_ATTN_POST_NORM  "%s.blk.%d.attn_post_norm.%s" // post-attn norm (gemma4)
#define TN_FFN_POST_NORM   "%s.blk.%d.ffn_post_norm.%s"  // post-FFN norm (gemma4)
#define TN_LN_PRE          "%s.pre_ln.%s"
#define TN_LN_POST         "%s.post_ln.%s"
#define TN_LLAVA_PROJ      "mm.%d.%s"
#define TN_DEEPSTACK_NORM  "v.deepstack.%d.norm.%s"     // qwen3vl deepstack
#define TN_DEEPSTACK_FC1   "v.deepstack.%d.fc1.%s"      // qwen3vl deepstack
#define TN_DEEPSTACK_FC2   "v.deepstack.%d.fc2.%s"      // qwen3vl deepstack

// align x to upper multiple of n
#define CLIP_ALIGN(x, n) ((((x) + (n) - 1) / (n)) * (n))

// forward declaration
// TODO: improve this later
struct clip_ctx;

enum projector_type {
    PROJECTOR_TYPE_QWEN3VL,
    PROJECTOR_TYPE_UNKNOWN,
};

static std::map<projector_type, std::string> PROJECTOR_TYPE_NAMES = {
    { PROJECTOR_TYPE_QWEN3VL,           "qwen3vl_merger"},
};

static projector_type clip_projector_type_from_string(const std::string & str) {
    for (const auto & pair : PROJECTOR_TYPE_NAMES) {
        if (pair.second == str) {
            return pair.first;
        }
    }
    return PROJECTOR_TYPE_UNKNOWN;
}

// RGB uint8 image
struct clip_image_u8 {
    clip_image_size get_size() const {
        return { nx, ny };
    }

    void set_size(clip_image_size size, bool is_placeholder) {
        nx = size.width;
        ny = size.height;
        if (is_placeholder) {
            buf.clear();
        } else {
            buf.resize((size_t) nx * (size_t) ny * 3);
        }
    }

    void cpy_buf(const std::vector<uint8_t> & new_buf) {
        buf = new_buf;
    }

    const std::vector<uint8_t> & get_ro_buf() const {
        if (is_placeholder()) {
            throw std::runtime_error("this clip_image_u8 is a placeholder");
        }
        return buf;
    }

    // note to contributors: NEVER add a get_rw_buf(), it is a DANGEROUS pattern. always use get_pixel / set_pixel for buffer manipulation

    bool is_placeholder() const {
        return buf.empty();
    }

    std::array<uint8_t, 3> get_pixel(int x, int y) const {
        if (is_placeholder()) {
            // return a dummy value, so that legacy code can still process image without errors
            return { 0, 0, 0 };
        }
        size_t idx = ((size_t) y * (size_t) nx + (size_t) x) * 3;
        return { buf[idx], buf[idx + 1], buf[idx + 2] };
    }

    void set_pixel(int x, int y, const std::array<uint8_t, 3> & rgb) {
        if (is_placeholder()) {
            return; // no-op
        }
        size_t idx = ((size_t) y * (size_t) nx + (size_t) x) * 3;
        buf[idx]     = rgb[0];
        buf[idx + 1] = rgb[1];
        buf[idx + 2] = rgb[2];
    }

    size_t n_elements() const {
        return n_pixels() * 3;
    }

  private:
    std::vector<uint8_t> buf;
    int nx = 0;
    int ny = 0;

    size_t n_pixels() const {
        return (size_t) nx * (size_t) ny;
    }
};

struct mtmd_serialization; // forward declaration

// For images, buf.size() == nx*ny*3
//     Memory layout: RGBRGBRGB...
// For seq, buf.size() == nx*ny*3*nt
//     Memory layout: RGBRGB...RGBRGB... (nt times)
// For audio, only one channel is used, buf.size() == nx*ny
//     nx will be n_frames and ny will be n_mel
struct clip_image_f32 {
    // marks the global view in e.g., DeepSeek-OCR Models
    bool add_viewsep = false;
    // appends a learned newline (or EOI) token after the image
    // no model uses it now (Granite4 Vision moved to anyres), kept for future models
    bool add_newline = false;
    // deepseek4v: number of leading IMAGE_PAD embeddings, aligns IMAGE_START to the LLM compressor ratio
    // depends on the chunk position, set at tokenize time (see mtmd_tokenizer::add_media)
    int32_t lead_pad = 0;

    // llava-next "anyres" tiling, used by Granite4 Vision
    // the whole grid is encoded and assembled in a single graph
    // NOTE: excluded from serialized: a deserialized image is always a placeholder, which is never encoded
    struct anyres_info {
        int grid_x = 0; // tiles per row, 0 means the image is not tiled
        int grid_y = 0; // tiles per column
        int orig_nx = 0; // size of the source image, used to drop the padding tokens
        int orig_ny = 0;

        bool is_tiled() const {
            return grid_x > 0 && grid_y > 0;
        }
    };
    anyres_info anyres;

    clip_image_size get_size() const {
        return { nx_, ny_ };
    }

    int nx() const { return nx_; }
    int ny() const { return ny_; }

    void set_size(clip_image_size size, bool is_placeholder, bool is_audio) {
        nx_ = size.width;
        ny_ = size.height;
        if (is_placeholder) {
            buf.clear();
        } else {
            if (is_audio) {
                buf.resize((size_t) nx_ * (size_t) ny_);
            } else {
                buf.resize((size_t) nx_ * (size_t) ny_ * 3);
            }
        }
    }

    void cpy_buf(const std::vector<float> & new_buf) {
        buf = new_buf;
    }

    void from_u8(const clip_image_u8 & img) {
        auto size = img.get_size();
        nx_ = size.width;
        ny_ = size.height;
        if (img.is_placeholder()) {
            buf.clear();
            return; // no-op
        }
        buf.resize(img.n_elements());
        const auto & u8_buf = img.get_ro_buf();
        for (size_t i = 0; i < img.n_elements(); ++i) {
            buf[i] = (float) u8_buf[i] / 255.0f;
        }
    }

    size_t n_elements() const {
        return n_pixels() * 3;
    }

    void normalize(const float mean[3], const float std[3]) {
        if (is_placeholder()) {
            return; // no-op
        }
        for (size_t i = 0; i < n_pixels(); ++i) {
            buf[i * 3 + 0] = (buf[i * 3 + 0] - mean[0]) / std[0];
            buf[i * 3 + 1] = (buf[i * 3 + 1] - mean[1]) / std[1];
            buf[i * 3 + 2] = (buf[i * 3 + 2] - mean[2]) / std[2];
        }
    }

    const std::vector<float> & get_ro_buf() const {
        if (is_placeholder()) {
            throw std::runtime_error("this clip_image_f32 is a placeholder");
        }
        return buf;
    }

    // note to contributors: NEVER add a get_rw_buf(), it is a DANGEROUS pattern

    bool is_placeholder() const {
        return buf.empty();
    }

    void serialize(struct mtmd_serialization & ser) const;
    void deserialize(struct mtmd_serialization & ser);

  private:
    std::vector<float> buf;
    int nx_ = 0;
    int ny_ = 0;

    size_t n_pixels() const {
        return (size_t) nx_ * (size_t) ny_;
    }
};

//
// logging
//

static void clip_log_callback_default(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct clip_logger_state {
    ggml_log_callback log_callback;
    void * log_callback_user_data;
};

extern struct clip_logger_state g_logger_state;

static void clip_log_internal_v(enum ggml_log_level level, const char * format, va_list args) {
    if (format == NULL) {
        return;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } else {
        char * buffer2 = (char *) calloc(len + 1, sizeof(char));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        free(buffer2);
    }
    va_end(args_copy);
}

static void clip_log_internal(enum ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    clip_log_internal_v(level, format, args);
    va_end(args);
}

#define LOG_TRC(...) clip_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_DBG(...) clip_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_INF(...) clip_log_internal(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_WRN(...) clip_log_internal(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_ERR(...) clip_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LOG_CNT(...) clip_log_internal(GGML_LOG_LEVEL_CONT,  __VA_ARGS__)

//
// cpp wrappers
//

struct clip_image_f32_batch {
    std::vector<clip_image_f32> entries;
    bool is_audio = false;

    clip_image_f32_batch clone() const {
        clip_image_f32_batch new_batch{
            /* entries  */ {},
            /* is_audio */ is_audio,
        };
        new_batch.entries.reserve(entries.size());
        for (const auto & entry : entries) {
            new_batch.entries.emplace_back(entry); // copy
        }
        return new_batch;
    }

    void serialize(struct mtmd_serialization & ser) const;
    void deserialize(struct mtmd_serialization & ser);
};

//
// common utils
//

#ifdef _WIN32
static std::ifstream open_ifstream_binary(const std::string & fname) {
    int wlen = MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, NULL, 0);
    if (!wlen) {
        throw std::runtime_error("failed to convert filename to UTF-16: " + fname);
    }
    std::vector<wchar_t> wfname(wlen);
    (void)MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, wfname.data(), wlen);
    return std::ifstream(wfname.data(), std::ios::binary);
}
#else
static std::ifstream open_ifstream_binary(const std::string & fname) {
    return std::ifstream(fname, std::ios::binary);
}
#endif

// in test-mtmd-impl, we include woth common.h and this file, and these functions are duplicated
// this is a quick fix to avoid compilation errors
#ifndef DIRECTORY_SEPARATOR
static std::string string_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), buf.size());
}

static void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

// remove when moving to c++20
inline bool string_starts_with(std::string_view str, std::string_view prefix) {
    return str.size() >= prefix.size() &&
           str.compare(0, prefix.size(), prefix) == 0;
}

// remove when moving to c++20
inline bool string_ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}
#endif

//
// gguf utils
//

static std::string gguf_data_to_str(enum gguf_type type, const void * data, int i) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(((const uint8_t  *)data)[i]);
        case GGUF_TYPE_INT8:    return std::to_string(((const int8_t   *)data)[i]);
        case GGUF_TYPE_UINT16:  return std::to_string(((const uint16_t *)data)[i]);
        case GGUF_TYPE_INT16:   return std::to_string(((const int16_t  *)data)[i]);
        case GGUF_TYPE_UINT32:  return std::to_string(((const uint32_t *)data)[i]);
        case GGUF_TYPE_INT32:   return std::to_string(((const int32_t  *)data)[i]);
        case GGUF_TYPE_UINT64:  return std::to_string(((const uint64_t *)data)[i]);
        case GGUF_TYPE_INT64:   return std::to_string(((const int64_t  *)data)[i]);
        case GGUF_TYPE_FLOAT32: return std::to_string(((const float    *)data)[i]);
        case GGUF_TYPE_FLOAT64: return std::to_string(((const double   *)data)[i]);
        case GGUF_TYPE_BOOL:    return ((const int8_t *)data)[i] != 0 ? "true" : "false";
        default:                return string_format("unknown type %d", type);
    }
}

static std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY:
            {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
                int arr_n = gguf_get_arr_n(ctx_gguf, i);
                const void * data = arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx_gguf, i);
                std::stringstream ss;
                ss << "[";
                for (int j = 0; j < arr_n; j++) {
                    if (arr_type == GGUF_TYPE_STRING) {
                        std::string val = gguf_get_arr_str(ctx_gguf, i, j);
                        // escape quotes
                        string_replace_all(val, "\\", "\\\\");
                        string_replace_all(val, "\"", "\\\"");
                        ss << '"' << val << '"';
                    } else if (arr_type == GGUF_TYPE_ARRAY) {
                        ss << "???";
                    } else {
                        ss << gguf_data_to_str(arr_type, data, j);
                    }
                    if (j < arr_n - 1) {
                        ss << ", ";
                    }
                }
                ss << "]";
                return ss.str();
            }
        default:
            return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}

//
// API used internally with mtmd
//

projector_type clip_get_projector_type(const struct clip_ctx * ctx);
void clip_set_debug_output_embeddings(struct clip_ctx * ctx, bool debug);
