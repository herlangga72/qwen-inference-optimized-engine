#pragma once

#include "ggml.h"
#include "clip.h"
#include "clip-impl.h"

#include <algorithm>
#include <array>
#include <vector>
#include <unordered_set>
#include <cstdint>
#include <cmath>

enum ffn_op_type {
    FFN_GELU,
    FFN_GELU_ERF,
    FFN_SILU,
    FFN_GELU_QUICK,
    FFN_RELU_SQR,
};

enum norm_type {
    NORM_TYPE_NORMAL,
    NORM_TYPE_RMS,
};

enum patch_merge_type {
    PATCH_MERGE_FLAT,
    PATCH_MERGE_SPATIAL_UNPAD,
};

// all algos are Pillow-compatible (matching PIL.Image.resize output)
enum resize_algo {
    RESIZE_ALGO_BILINEAR,
    RESIZE_ALGO_BICUBIC,
    RESIZE_ALGO_LANCZOS,
};

// Padding style for img_tool::resize
//   PAD_NONE    - no padding; direct resize to target dimensions
//   PAD_CEIL    - aspect-preserving pad (default)
//   PAD_NEAREST - aspect-preserving pad with nearest-integer rounding (Pillow byte-parity)
enum pad_style {
    PAD_NONE,
    PAD_CEIL,
    PAD_NEAREST,
};

struct clip_hparams {
    int32_t image_size = 0;
    int32_t patch_size = 0;
    int32_t n_embd = 0;
    int32_t n_ff = 0;
    int32_t projection_dim = 0;
    int32_t n_head = 0;
    int32_t n_head_kv = 0;
    // 0 = derive from n_embd; set when qkv width != n_embd
    int32_t n_embd_head = 0;
    int32_t n_layer = 0;
    int32_t n_merge = 1; // number of patch merges **per-side**

    // for preprocessor
    int32_t image_min_pixels = -1;
    int32_t image_max_pixels = -1;
    resize_algo image_resize_algo = RESIZE_ALGO_BICUBIC;
    pad_style image_resize_pad = PAD_CEIL; // padding style when resizing
    std::array<uint8_t, 3> image_pad_color = {0, 0, 0};

    // (preprocessor) for llava-uhd style models
    std::vector<clip_image_size> image_res_candidates;
    int32_t preproc_min_tiles = 0;
    int32_t preproc_max_tiles = 0;

    float image_mean[3];
    float image_std[3];

    // for models using dynamic image size, we need to have a smaller image size to warmup
    // otherwise, user will get OOM every time they load the model
    int32_t warmup_image_size = 0;
    int32_t warmup_audio_size = 3000;

    ffn_op_type ffn_op = FFN_GELU;

    patch_merge_type mm_patch_merge_type = PATCH_MERGE_FLAT;

    float eps = 1e-6;
    int32_t n_expert_used = 0;
    std::vector<int32_t> feature_layers;
    int32_t attn_window_size = 0;
    int32_t n_wa_pattern = 0;
    std::unordered_set<int32_t> wa_layer_indexes; // explicit layer indexes that use full attention (for irregular patterns like YoutuVL)

    // audio
    int32_t n_mel_bins = 0; // whisper preprocessor
    int32_t proj_stack_factor = 0; // ultravox

    // audio-to-mel preprocessor params
    int32_t audio_chunk_len   = -1; // in seconds
    int32_t audio_sample_rate = -1;
    int32_t audio_n_fft       = -1;
    int32_t audio_window_len  = -1;
    int32_t audio_hop_len     = -1;

    // threshold for the "out_eos_score" graph output
    float gen_eos_threshold = 0.0f;

    // name of the weight variant, some pipelines tune themselves on it
    std::string gen_model_variant;

    // qwen3tts code2wav
    int32_t wav_tfm_swa          = 0; // pre_transformer's KV cache size, in frames

    // legacy
    bool has_llava_projector = false;
    int minicpmv_version = 0;
    int32_t minicpmv_query_num = 0;         // MiniCPM-V query number

    // custom value provided by user, can be undefined if not set
    int32_t custom_image_min_tokens = -1;
    int32_t custom_image_max_tokens = -1;

    void set_limit_image_tokens(int n_tokens_min, int n_tokens_max) {
        const int patch_area = patch_size * patch_size * n_merge * n_merge;
        image_min_pixels = (custom_image_min_tokens > 0 ? custom_image_min_tokens : n_tokens_min) * patch_area;
        image_max_pixels = (custom_image_max_tokens > 0 ? custom_image_max_tokens : n_tokens_max) * patch_area;
        warmup_image_size = static_cast<int>(std::sqrt(image_max_pixels));
    }

    void set_warmup_n_tokens(int n_tokens) {
        int n_tok_per_side = static_cast<int>(std::sqrt(n_tokens));
        GGML_ASSERT(n_tok_per_side * n_tok_per_side == n_tokens && "n_tokens must be n*n");
        warmup_image_size = n_tok_per_side * patch_size * n_merge;
        // TODO: support warmup size for custom token numbers
    }
};

struct clip_layer {
    // layernorm 1 (or layer input norm, or pre-attention norm)
    ggml_tensor * ln_1_w = nullptr;
    ggml_tensor * ln_1_b = nullptr;

    // attention
    ggml_tensor * k_w = nullptr;
    ggml_tensor * k_b = nullptr;
    ggml_tensor * q_w = nullptr;
    ggml_tensor * q_b = nullptr;
    ggml_tensor * v_w = nullptr;
    ggml_tensor * v_b = nullptr;
    ggml_tensor * qkv_w = nullptr;
    ggml_tensor * qkv_b = nullptr;

    ggml_tensor * o_w = nullptr;
    ggml_tensor * o_b = nullptr;

    ggml_tensor * attn_sinks = nullptr;

    ggml_tensor * k_norm = nullptr;
    ggml_tensor * q_norm = nullptr;

    ggml_tensor * attn_post_norm_w = nullptr;

    ggml_tensor * ff_up_w = nullptr;
    ggml_tensor * ff_up_b = nullptr;
    ggml_tensor * ff_gate_w = nullptr;
    ggml_tensor * ff_gate_b = nullptr;
    ggml_tensor * ff_down_w = nullptr;
    ggml_tensor * ff_down_b = nullptr;

    // MoE FFN (dots3note vision pyramid blocks)
    ggml_tensor * ff_gate_inp_w  = nullptr;
    ggml_tensor * ff_gate_exps_w = nullptr;
    ggml_tensor * ff_up_exps_w   = nullptr;
    ggml_tensor * ff_down_exps_w = nullptr;
    ggml_tensor * ff_exp_probs_b = nullptr;

    // layernorm 2 (or pre-FFN norm)
    ggml_tensor * ln_2_w = nullptr;
    ggml_tensor * ln_2_b = nullptr;

    ggml_tensor * ff_post_norm_w = nullptr;

    // layer scale (no bias)
    ggml_tensor * ls_1_w   = nullptr;
    ggml_tensor * ls_2_w   = nullptr;
    ggml_tensor * ls_out_w = nullptr; // gemma4

    // qwen3vl deepstack merger
    ggml_tensor * deepstack_norm_w = nullptr;
    ggml_tensor * deepstack_norm_b = nullptr;
    ggml_tensor * deepstack_fc1_w = nullptr;
    ggml_tensor * deepstack_fc1_b = nullptr;
    ggml_tensor * deepstack_fc2_w = nullptr;
    ggml_tensor * deepstack_fc2_b = nullptr;

    bool has_deepstack() const {
        return deepstack_fc1_w != nullptr;
    }
};

// one persisted state buffer used by the gen-audio decoders
struct c2w_state_slot {
    std::string name;
    int64_t     ne0;
    int64_t     ne1;
};

struct clip_model {
    clip_modality modality = CLIP_MODALITY_VISION;
    projector_type proj_type = PROJECTOR_TYPE_UNKNOWN;
    clip_hparams hparams;

    // embeddings
    ggml_tensor * class_embedding = nullptr;
    ggml_tensor * patch_embeddings_0 = nullptr;
    ggml_tensor * patch_embeddings_1 = nullptr;  // second Conv2D kernel when we decouple Conv3D along temporal dimension (Qwen2VL)
    ggml_tensor * patch_bias = nullptr;
    ggml_tensor * position_embeddings = nullptr;
    ggml_tensor * norm_embd_w = nullptr;
    ggml_tensor * norm_embd_b = nullptr;

    ggml_tensor * pre_ln_w = nullptr;
    ggml_tensor * pre_ln_b = nullptr;

    std::vector<clip_layer> layers;

    int32_t n_deepstack_layers = 0; // used by Qwen3-VL, calculated from clip_layer

    ggml_tensor * post_ln_w;
    ggml_tensor * post_ln_b;

    // LLaVA projection
    ggml_tensor * mm_0_w = nullptr;
    ggml_tensor * mm_0_b = nullptr;

    // Yi type models with mlp+normalization projection
    ggml_tensor * mm_1_w = nullptr; // Yi type models have 0, 1, 3, 4
    ggml_tensor * mm_1_b = nullptr;

    // qwen3tts code_predictor
    ggml_tensor * gen_code_head_w     = nullptr; // per-codebook output head, merged 3D
};

const clip_hparams * clip_get_hparams(const struct clip_ctx * ctx);
