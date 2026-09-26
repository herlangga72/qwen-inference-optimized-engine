#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"
#include "clip-graph.h"
#include "models/models.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <cinttypes>
#include <limits>
#include <array>
#include <functional>
#include <float.h>

struct clip_logger_state g_logger_state = {clip_log_callback_default, NULL};

//#define CLIP_DEBUG_FUNCTIONS

#ifdef CLIP_DEBUG_FUNCTIONS
static void clip_image_write_image_to_ppm(const clip_image_u8& img, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }

    // PPM header: P6 format, width, height, and max color value
    const auto ppm_size = img.get_size();
    file << "P6\n" << ppm_size.width << " " << ppm_size.height << "\n255\n";

    // Write pixel data
    const auto & ppm_buf = img.get_ro_buf();
    for (size_t i = 0; i < ppm_buf.size(); i += 3) {
        // PPM expects binary data in RGB format, which matches our image buffer
        file.write(reinterpret_cast<const char*>(&ppm_buf[i]), 3);
    }

    file.close();
}

static void clip_image_save_to_bmp(const clip_image_u8& img, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }

    const auto bmp_size = img.get_size();
    int fileSize = 54 + 3 * bmp_size.width * bmp_size.height; // File header + info header + pixel data
    int bytesPerPixel = 3;
    int widthInBytes = bmp_size.width * bytesPerPixel;
    int paddingAmount = (4 - (widthInBytes % 4)) % 4;
    int stride = widthInBytes + paddingAmount;

    // Bitmap file header
    unsigned char fileHeader[14] = {
        'B','M',     // Signature
        0,0,0,0,    // Image file size in bytes
        0,0,0,0,    // Reserved
        54,0,0,0    // Start of pixel array
    };

    // Total file size
    fileSize = 54 + (stride * bmp_size.height);
    fileHeader[2] = (unsigned char)(fileSize);
    fileHeader[3] = (unsigned char)(fileSize >> 8);
    fileHeader[4] = (unsigned char)(fileSize >> 16);
    fileHeader[5] = (unsigned char)(fileSize >> 24);

    // Bitmap information header (BITMAPINFOHEADER)
    unsigned char infoHeader[40] = {
        40,0,0,0,   // Size of this header (40 bytes)
        0,0,0,0,    // Image width
        0,0,0,0,    // Image height
        1,0,        // Number of color planes
        24,0,       // Bits per pixel
        0,0,0,0,    // No compression
        0,0,0,0,    // Image size (can be 0 for no compression)
        0,0,0,0,    // X pixels per meter (not specified)
        0,0,0,0,    // Y pixels per meter (not specified)
        0,0,0,0,    // Total colors (color table not used)
        0,0,0,0     // Important colors (all are important)
    };

    // Width and height in the information header
    infoHeader[4] = (unsigned char)(bmp_size.width);
    infoHeader[5] = (unsigned char)(bmp_size.width >> 8);
    infoHeader[6] = (unsigned char)(bmp_size.width >> 16);
    infoHeader[7] = (unsigned char)(bmp_size.width >> 24);
    infoHeader[8] = (unsigned char)(bmp_size.height);
    infoHeader[9] = (unsigned char)(bmp_size.height >> 8);
    infoHeader[10] = (unsigned char)(bmp_size.height >> 16);
    infoHeader[11] = (unsigned char)(bmp_size.height >> 24);

    // Write file headers
    file.write(reinterpret_cast<char*>(fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<char*>(infoHeader), sizeof(infoHeader));

    // Pixel data
    std::vector<unsigned char> padding(3, 0); // Max padding size to be added to each row
    for (int y = bmp_size.height - 1; y >= 0; --y) { // BMP files are stored bottom-to-top
        for (int x = 0; x < bmp_size.width; ++x) {
            // Each pixel
            const auto px = img.get_pixel(x, y);
            unsigned char pixel[3] = {
                px[2], // BMP stores pixels in BGR format
                px[1],
                px[0]
            };
            file.write(reinterpret_cast<char*>(pixel), 3);
        }
        // Write padding for the row
        file.write(reinterpret_cast<char*>(padding.data()), paddingAmount);
    }

    file.close();
}

// debug function to convert f32 to u8
static void clip_image_convert_f32_to_u8(const clip_image_f32& src, clip_image_u8& dst) {
    dst.set_size(src.get_size(), false);
    const auto & src_buf = src.get_ro_buf();
    std::vector<uint8_t> dst_buf(src.n_elements());
    for (size_t i = 0; i < src.n_elements(); ++i) {
        dst_buf[i] = static_cast<uint8_t>(std::min(std::max(int(src_buf[i] * 255.0f), 0), 255));
    }
    dst.cpy_buf(dst_buf);
}
#endif


struct clip_ctx {
    clip_model model;

    gguf_context_ptr ctx_gguf;
    ggml_context_ptr ctx_data;

    std::vector<uint8_t> buf_compute_meta;

    std::vector<ggml_backend_t> backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_ptr buf;


    int max_nodes = 8192;
    ggml_backend_sched_ptr sched;
    clip_flash_attn_type flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;
    bool is_allocated = false;

    bool debug_output_embeddings = false;

    // for measuring memory usage
    bool no_alloc = false;
    std::map<ggml_backend_dev_t, size_t> mem_usage;
    std::map<ggml_backend_dev_t, size_t> mem_compute;

    bool support_batch = false;

    // for audio gen, reseeded only when the caller asks for another seed
    std::mt19937 rng{std::random_device{}()};
    uint32_t rng_seed = UINT32_MAX;

    clip_ctx(clip_context_params & ctx_params) {
        flash_attn_type = ctx_params.flash_attn_type;
        no_alloc = ctx_params.no_alloc;
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend_cpu) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        if (ctx_params.use_gpu) {
            if (ctx_params.device != nullptr) {
                backend = ggml_backend_dev_init(ctx_params.device, nullptr);
                if (!backend) {
                    throw std::runtime_error(string_format("%s: failed to initialize \"%s\" backend\n",
                                                           __func__, ggml_backend_dev_name(ctx_params.device)));
                }
            } else {
                backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
                backend = backend ? backend : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
            }
        }

        if (backend) {
            LOG_INF("%s: CLIP using %s backend\n", __func__, ggml_backend_name(backend));
            backend_ptrs.push_back(backend);
            backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
        } else {
            backend = backend_cpu;
            LOG_INF("%s: CLIP using CPU backend\n", __func__);
        }

        if (ctx_params.image_min_tokens > 0) {
            model.hparams.custom_image_min_tokens = ctx_params.image_min_tokens;
        }
        if (ctx_params.image_max_tokens > 0) {
            model.hparams.custom_image_max_tokens = ctx_params.image_max_tokens;
        }

        backend_ptrs.push_back(backend_cpu);
        backend_buft.push_back(ggml_backend_get_default_buffer_type(backend_cpu));

        sched.reset(
            ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), 8192, false, true)
        );

        if (ctx_params.cb_eval != nullptr) {
            ggml_backend_sched_set_eval_callback(sched.get(), ctx_params.cb_eval, ctx_params.cb_eval_user_data);
        }

        debug_output_embeddings = std::getenv("MTMD_DEBUG_EMBEDDINGS") != nullptr;
    }

    ~clip_ctx() {
        ggml_backend_free(backend);
        if (backend != backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }

    // this function is added so that we don't change too much of the existing code
    projector_type proj_type() const {
        return model.proj_type;
    }
};

//
// clip_graph
//

clip_graph::clip_graph(clip_ctx * ctx, const clip_image_f32 & img) :
        model(ctx->model),
        hparams(model.hparams),
        proj_type(ctx->proj_type()),
        img(img),
        patch_size(hparams.patch_size),
        n_patches_x(img.nx() / patch_size),
        n_patches_y(img.ny() / patch_size),
        n_patches(n_patches_x * n_patches_y),
        n_embd(hparams.n_embd),
        n_head(hparams.n_head),
        n_head_kv(hparams.n_head_kv),
        d_head(hparams.n_embd_head > 0 ? hparams.n_embd_head : (n_head > 0 ? n_embd / n_head : 0)),
        n_layer(hparams.n_layer),
        n_mmproj_embd(clip_n_mmproj_embd(ctx)),
        eps(hparams.eps),
        kq_scale(d_head > 0 ? 1.0f / sqrtf((float)d_head) : 0.0f),
        flash_attn_type(ctx->flash_attn_type) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    ctx0_ptr.reset(ggml_init(params));
    ctx0 = ctx0_ptr.get();
    gf = ggml_new_graph_custom(ctx0, ctx->max_nodes, false);
}

clip_graph::clip_graph(const clip_graph & parent) :
        model(parent.model),
        hparams(parent.hparams),
        proj_type(parent.proj_type),
        img(parent.img),
        patch_size(parent.patch_size),
        n_patches_x(parent.n_patches_x),
        n_patches_y(parent.n_patches_y),
        n_patches(parent.n_patches),
        n_embd(parent.n_embd),
        n_head(parent.n_head),
        n_head_kv(parent.n_head_kv),
        d_head(parent.d_head),
        n_layer(parent.n_layer),
        n_mmproj_embd(parent.n_mmproj_embd),
        eps(parent.eps),
        kq_scale(parent.kq_scale),
        flash_attn_type(parent.flash_attn_type) {
    // reuse from parent
    ctx0 = parent.ctx0;
    gf   = parent.gf;
}

ggml_tensor * clip_graph::build_mm(ggml_tensor * w, ggml_tensor * x) const {
    return ggml_mul_mat(ctx0, w, x);
}

void clip_graph::cb(ggml_tensor * cur, const char * name, int il) const {
    if (il >= 0) {
        ggml_format_name(cur, "%s-%d", name, il);
    } else {
        ggml_set_name(cur, name);
    }
}

// siglip2 naflex
ggml_tensor * clip_graph::resize_position_embeddings(uint32_t interpolation_mode) {
    ggml_tensor * pos_embd = model.position_embeddings;
    const int height       = img.ny() / patch_size;
    const int width        = img.nx() / patch_size;
    const uint32_t mode    = interpolation_mode;
    const int n_per_side   = (int)std::sqrt(pos_embd->ne[1]);

    GGML_ASSERT(pos_embd);

    if (height == n_per_side && width == n_per_side) {
        return pos_embd;
    }

    pos_embd = ggml_reshape_3d(ctx0, pos_embd, n_embd, n_per_side, n_per_side);  // -> (n_embd, n_per_side, n_per_side)
    pos_embd = ggml_permute(ctx0, pos_embd, 2, 0, 1, 3);                         // -> (n_per_side, n_per_side, n_embd)
    pos_embd = ggml_interpolate(ctx0, pos_embd, width, height, n_embd, 1, mode); // -> (width, height, n_embd)
    pos_embd = ggml_permute(ctx0, pos_embd, 1, 2, 0, 3);                         // -> (n_embd, width, height)
    pos_embd = ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);             // -> (n_embd, width * height)

    return pos_embd;
}

// build the input after conv2d (inp_raw --> patches)
// returns tensor with shape [n_embd, n_patches]
ggml_tensor * clip_graph::build_inp() {
    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    inp = ggml_reshape_3d(ctx0, inp, n_patches, n_embd, n_batch);
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    if (model.patch_bias) {
        inp = ggml_add(ctx0, inp, model.patch_bias);
        cb(inp, "patch_bias", -1);
    }
    return inp;
}

ggml_tensor * clip_graph::build_inp_raw(int channels) {
    ggml_tensor * inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, img.nx(), img.ny(), channels, n_batch);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);
    return inp_raw;
}

ggml_tensor * clip_graph::build_norm(
        ggml_tensor * cur,
        ggml_tensor * mw,
        ggml_tensor * mb,
        norm_type type,
        float norm_eps,
        int il) const {

    cur = type == NORM_TYPE_RMS
        ? ggml_rms_norm(ctx0, cur, norm_eps)
        : ggml_norm(ctx0, cur, norm_eps);

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        cb(cur, "norm_w", il);
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
        cb(cur, "norm_b", il);
    }

    return cur;
}

ggml_tensor * clip_graph::build_ffn(
        ggml_tensor * cur,
        ggml_tensor * up,
        ggml_tensor * up_b,
        ggml_tensor * gate,
        ggml_tensor * gate_b,
        ggml_tensor * down,
        ggml_tensor * down_b,
        ffn_op_type type_op,
        int il) const {

    ggml_tensor * tmp = up ? build_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (gate) {
        cur = build_mm(gate, cur);
        cb(cur, "ffn_gate", il);

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }
    } else {
        cur = tmp;
    }

    // we only support parallel ffn for now
    switch (type_op) {
        case FFN_SILU:
            if (gate) {
                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case FFN_GELU:
            if (gate) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
            } break;
        case FFN_GELU_ERF:
            if (gate) {
                cur = ggml_geglu_erf_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu_erf", il);
            } else {
                cur = ggml_gelu_erf(ctx0, cur);
                cb(cur, "ffn_gelu_erf", il);
            } break;
        case FFN_GELU_QUICK:
            if (gate) {
                cur = ggml_geglu_quick_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu_quick", il);
            } else {
                cur = ggml_gelu_quick(ctx0, cur);
                cb(cur, "ffn_gelu_quick", il);
            } break;
        case FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_relu_sqr", il);
            } break;
    }

    if (down) {
        cur = build_mm(down, cur);
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    return cur;
}

ggml_tensor * clip_graph::build_attn(
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_mask,
        float kq_scale,
        int il,
        ggml_tensor * sinks) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // the order is fixed without the compute flag, so an unselected branch stays out of the compute set
    ggml_build_forward_order(gf, q_cur);
    ggml_build_forward_order(gf, k_cur);
    ggml_build_forward_order(gf, v_cur);

    ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    //cb(q, "q", il);

    ggml_tensor * k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
    //cb(k, "k", il);

    ggml_tensor * cur;

    if (flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
        ggml_tensor * v = ggml_permute(ctx0, v_cur, 0, 2, 1, 3);

        k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        if (kq_mask) {
            kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F16);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, 0.0f, 0.0f);
        ggml_prec_set_acc(cur, GGML_PREC_F32);
        if (sinks != nullptr) {
            ggml_flash_attn_ext_add_sinks(cur, sinks);
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

    } else {
        ggml_tensor * v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
        v = ggml_cont(ctx0, v);

        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        // F32 may not needed for vision encoders?
        // ggml_prec_set_acc(kq, GGML_PREC_F32);

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);
        if (sinks != nullptr) {
            ggml_soft_max_add_sinks(kq, sinks);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
    }

    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_mm(wo, cur);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<clip_graph> clip_get_graph_builder(clip_ctx * ctx, const clip_image_f32_batch & imgs,
                                                            const clip_encode_params * = nullptr) {
    const clip_image_f32 & img = imgs.entries[0];
    std::unique_ptr<clip_graph> builder;

    switch (ctx->proj_type()) {
        case PROJECTOR_TYPE_QWEN3VL:
            {
                builder = std::make_unique<clip_graph_qwen3vl>(ctx, img);
            } break;
        default:
            GGML_ABORT("missing cgraph builder");
    }

    builder->img_batch = &imgs;

    // TODO [QWEN_VIDEO]: improve this in the future
    builder->n_batch = imgs.entries.size();

    return builder;
}

//
// clip_model_loader
//

struct clip_model_loader {
    ggml_context_ptr ctx_meta;
    gguf_context_ptr ctx_gguf;

    std::string fname;

    size_t model_size = 0; // in bytes

    bool has_vision    = false;
    bool has_audio     = false;
    bool has_gen_audio = false;

    mtmd_progress_callback progress_callback = nullptr;
    void * progress_callback_user_data = nullptr;

    // TODO @ngxson : we should not pass clip_ctx here, it should be clip_model
    clip_model_loader(const char * fname,
            bool skip_tensors = false,
            mtmd_progress_callback progress_cb = nullptr,
            void * progress_user_data = nullptr)
        : fname(fname),
          progress_callback(progress_cb),
          progress_callback_user_data(progress_user_data) {
        struct ggml_context * meta = nullptr;

        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };

        ctx_gguf = gguf_context_ptr(gguf_init_from_file(fname, params));
        if (!ctx_gguf.get()) {
            throw std::runtime_error(string_format("%s: failed to load CLIP model from %s. Does this file exist?\n", __func__, fname));
        }

        ctx_meta.reset(meta);

        const int n_tensors = gguf_get_n_tensors(ctx_gguf.get());

        // print gguf info
        {
            std::string name;
            get_string(KEY_NAME, name, false);
            std::string description;
            get_string(KEY_DESCRIPTION, description, false);
            LOG_INF("%s: model name:   %s\n",  __func__, name.c_str());
            LOG_INF("%s: description:  %s\n",  __func__, description.c_str());
            LOG_INF("%s: GGUF version: %d\n",  __func__, gguf_get_version(ctx_gguf.get()));
            LOG_INF("%s: alignment:    %zu\n", __func__, gguf_get_alignment(ctx_gguf.get()));
            LOG_INF("%s: n_tensors:    %d\n",  __func__, n_tensors);
            LOG_INF("%s: n_kv:         %d\n",  __func__, (int)gguf_get_n_kv(ctx_gguf.get()));
            LOG_INF("\n");
        }

        // modalities
        {
            get_bool(KEY_HAS_VISION_ENC,    has_vision,    false);
            get_bool(KEY_HAS_AUDIO_ENC,     has_audio,     false);
            get_bool(KEY_HAS_GEN_AUDIO_ENC, has_gen_audio, false);

            if (has_vision) {
                LOG_INF("%s: has vision encoder\n", __func__);
            }
            if (has_audio) {
                LOG_INF("%s: has audio encoder\n", __func__);
            }
            if (has_gen_audio) {
                LOG_INF("%s: has audio generation (gen) encoder\n", __func__);
            }
        }

        // tensors
        if (!skip_tensors) {
            for (int i = 0; i < n_tensors; ++i) {
                const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
                const size_t offset = gguf_get_tensor_offset(ctx_gguf.get(), i);
                enum ggml_type type = gguf_get_tensor_type(ctx_gguf.get(), i);
                ggml_tensor * cur = ggml_get_tensor(meta, name);
                size_t tensor_size = ggml_nbytes(cur);
                model_size += tensor_size;
                LOG_DBG("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                    __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
            }
        }
    }

    void load_hparams(clip_model & model, clip_modality modality) {
        auto & hparams = model.hparams;
        std::string log_ffn_op; // for logging

        // sanity check
        if (modality == CLIP_MODALITY_VISION) {
            GGML_ASSERT(has_vision);
        } else if (modality == CLIP_MODALITY_AUDIO) {
            GGML_ASSERT(has_audio);
        } else if (modality == CLIP_MODALITY_GEN_AUDIO) {
            GGML_ASSERT(has_gen_audio);
        }
        model.modality = modality;


        // projector type
        std::string proj_type;
        {
            // default key
            get_string(KEY_PROJ_TYPE, proj_type, false);

            // for models with mixed modalities
            if (proj_type.empty()) {
                if (modality == CLIP_MODALITY_VISION) {
                    get_string(KEY_VISION_PROJ_TYPE, proj_type, false);
                } else if (modality == CLIP_MODALITY_AUDIO) {
                    get_string(KEY_AUDIO_PROJ_TYPE, proj_type, false);
                } else if (modality == CLIP_MODALITY_GEN_AUDIO) {
                    get_string(KEY_GEN_AUDIO_PROJ_TYPE, proj_type, false);
                } else {
                    GGML_ABORT("unknown modality");
                }
            }

            model.proj_type = clip_projector_type_from_string(proj_type);

            if (model.proj_type == PROJECTOR_TYPE_UNKNOWN) {
                throw std::runtime_error(string_format("%s: unknown projector type: %s\n", __func__, proj_type.c_str()));
            }
        }

        const bool is_vision    = model.modality == CLIP_MODALITY_VISION;
        const bool is_audio     = model.modality == CLIP_MODALITY_AUDIO;
        const bool is_gen_audio = model.modality == CLIP_MODALITY_GEN_AUDIO;

        // other hparams
        {
            const char * prefix = is_vision ? "vision" : (is_audio ? "audio" : "gen.audio");
            get_u32(string_format(KEY_N_EMBD,         prefix), hparams.n_embd);
            get_u32(string_format(KEY_N_HEAD,         prefix), hparams.n_head);
            get_u32(string_format(KEY_N_EMBD_HEAD,    prefix), hparams.n_embd_head, false);
            get_u32(string_format(KEY_N_FF,           prefix), hparams.n_ff);
            get_u32(string_format(KEY_N_BLOCK,        prefix), hparams.n_layer);
            get_u32(string_format(KEY_PROJ_DIM,       prefix), hparams.projection_dim);
            get_f32(string_format(KEY_LAYER_NORM_EPS, prefix), hparams.eps);

            // n_head_kv is optional (for GQA), default to n_head
            hparams.n_head_kv = hparams.n_head;
            get_u32(string_format(KEY_N_HEAD_KV, prefix), hparams.n_head_kv, false);

            if (is_vision) {
                get_u32(KEY_IMAGE_SIZE, hparams.image_size);
                get_u32(KEY_PATCH_SIZE, hparams.patch_size);
                get_i32(KEY_MINICPMV_VERSION, hparams.minicpmv_version, false); // legacy
                get_u32(KEY_MINICPMV_QUERY_NUM, hparams.minicpmv_query_num, false);
                if (hparams.minicpmv_query_num == 0) {
                    // Fallback to hardcoded values for legacy models
                    if (hparams.minicpmv_version == 3) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 4) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 5) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 6) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 100045) {
                        hparams.minicpmv_query_num = 64;
                    } else {
                        hparams.minicpmv_query_num = 96;
                    }
                }
            } else if (is_audio) {
                get_u32(KEY_A_NUM_MEL_BINS, hparams.n_mel_bins);
                // some hparams are unused, but still need to set to avoid issues
                hparams.image_size = 0;
                hparams.patch_size = 1;

            } else if (is_gen_audio) {
                // these are unused, but still need to be set to avoid issues
                hparams.image_size = 0;
                hparams.patch_size = 1;
                get_string(KEY_GEN_AUDIO_VARIANT, hparams.gen_model_variant, false);

            } else {
                GGML_ASSERT(false && "unknown modality");
            }

            // for pinpoints, we need to convert it into a list of resolution candidates
            {
                std::vector<int> pinpoints;
                get_arr_int(KEY_IMAGE_GRID_PINPOINTS, pinpoints, false);
                if (pinpoints.size() % 2 != 0) {
                    throw std::runtime_error(string_format("%s: image_grid_pinpoints must have an even number of elements, got %zu\n", __func__, pinpoints.size()));
                }
                if (!pinpoints.empty()) {
                    for (size_t i = 0; i < pinpoints.size(); i += 2) {
                        hparams.image_res_candidates.push_back({
                            pinpoints[i],
                            pinpoints[i+1],
                        });
                    }
                }
            }

            // default warmup value
            hparams.warmup_image_size = hparams.image_size;

            {
                bool use_gelu = false;
                bool use_silu = false;
                get_bool(KEY_USE_GELU, use_gelu, false);
                get_bool(KEY_USE_SILU, use_silu, false);
                if (use_gelu && use_silu) {
                    throw std::runtime_error(string_format("%s: both use_gelu and use_silu are set to true\n", __func__));
                }
                if (use_gelu) {
                    hparams.ffn_op = FFN_GELU;
                    log_ffn_op = "gelu";
                } else if (use_silu) {
                    hparams.ffn_op = FFN_SILU;
                    log_ffn_op = "silu";
                } else {
                    hparams.ffn_op = FFN_GELU_QUICK;
                    log_ffn_op = "gelu_quick";
                }
            }

            {
                std::string mm_patch_merge_type;
                get_string(KEY_MM_PATCH_MERGE_TYPE, mm_patch_merge_type, false);
                if (mm_patch_merge_type == "spatial_unpad") {
                    hparams.mm_patch_merge_type = PATCH_MERGE_SPATIAL_UNPAD;
                }
            }

            if (is_vision) {
                std::vector<float> image_mean;
                std::vector<float> image_std;
                get_arr_f32(KEY_IMAGE_MEAN, image_mean, false);
                get_arr_f32(KEY_IMAGE_STD , image_std, false);
                if (image_mean.size() < 3 || image_std.size() < 3) {
                    throw std::runtime_error(string_format("%s: image_mean/image_std arrays must have at least 3 elements, got %zu and %zu\n", __func__, image_mean.size(), image_std.size()));
                }
                for (int i = 0; i < 3; ++i) {
                    hparams.image_mean[i] = image_mean[i];
                    hparams.image_std[i]  = image_std[i];
                }
            }

            // Load the vision/audio feature layer indices if they are explicitly provided
            // NOTE: gguf conversions should standardize the values of the vision feature layer to be non-negative, since we use -1 to mark values as unset here.
            get_arr_int(string_format(KEY_FEATURE_LAYERS, prefix), hparams.feature_layers, false);
            for (const auto & v : hparams.feature_layers) {
                if (v > (int) hparams.n_layer) {
                    throw std::runtime_error(string_format("%s: feature layer index %d is out of range (n_layer: %d)",
                                                           __func__, v, hparams.n_layer));
                }
            }

            // model-specific params
            switch (model.proj_type) {
                case PROJECTOR_TYPE_QWEN3VL:
                    {
                        hparams.n_merge = 2; // default value for Qwen 2 and 2.5
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        get_u32(KEY_WIN_ATTN_PATTERN, hparams.n_wa_pattern, false);
                        // ref: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/preprocessor_config.json
                        hparams.set_limit_image_tokens(8, 4096);
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                        const int warn_min_pixels = 1024 * hparams.n_merge * hparams.n_merge * hparams.patch_size * hparams.patch_size;
                        if (hparams.image_min_pixels < warn_min_pixels) {
                            LOG_WRN("%s: Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks\n", __func__);
                            LOG_WRN("%s: if you encounter problems with accuracy, try adding --image-min-tokens 1024\n", __func__);
                            LOG_WRN("%s: more info: https://github.com/ggml-org/llama.cpp/issues/16842\n\n", __func__);
                        }
                    } break;
                default:
                    throw std::runtime_error(string_format("%s: unknown vision projector type %s\n", __func__, proj_type.c_str()));
            }

            // sanity check
            {
                if (hparams.image_size < 0) {
                    // note: some models having hparams.image_size == 0, which means the image size is dynamic
                    throw std::runtime_error(string_format("%s: image_size (%d) cannot be negative\n", __func__, hparams.image_size));
                }
                if (hparams.image_size > 8192) {
                    // cap prevents int32 overflow in n_patches = (image_size/patch_size)^2
                    throw std::runtime_error(string_format("%s: image_size (%d) is too large (max 8192)\n", __func__, hparams.image_size));
                }
                if (hparams.patch_size <= 0 || hparams.patch_size >= 65536) {
                    throw std::runtime_error(string_format("%s: patch_size (%d) must be positive and less than 65536\n", __func__, hparams.patch_size));
                }
                if (hparams.n_embd <= 0) {
                    throw std::runtime_error(string_format("%s: n_embd (%d) must be greater than 0\n", __func__, hparams.n_embd));
                }
                if (hparams.image_max_pixels < hparams.image_min_pixels) {
                    throw std::runtime_error(string_format("%s: image_max_pixels (%d) is less than image_min_pixels (%d)\n", __func__, hparams.image_max_pixels, hparams.image_min_pixels));
                }
                if (hparams.n_merge <= 0 || hparams.n_merge >= 65536) {
                    throw std::runtime_error(string_format("%s: n_merge (%d) must be greater than 0 and less than 65536\n", __func__, hparams.n_merge));
                }
                if (hparams.attn_window_size > 4096) {
                    throw std::runtime_error(string_format("%s: attn_window_size (%d) is too large (max 4096)\n", __func__, hparams.attn_window_size));
                }
            }

            LOG_INF("%s: projector:          %s\n", __func__, proj_type.c_str());
            LOG_INF("%s: n_embd:             %d\n", __func__, hparams.n_embd);
            LOG_INF("%s: n_head:             %d\n", __func__, hparams.n_head);
            LOG_INF("%s: n_ff:               %d\n", __func__, hparams.n_ff);
            LOG_INF("%s: n_layer:            %d\n", __func__, hparams.n_layer);
            LOG_INF("%s: ffn_op:             %s\n", __func__, log_ffn_op.c_str());
            LOG_INF("%s: projection_dim:     %d\n", __func__, hparams.projection_dim);
            if (is_vision) {
                LOG_INF("\n--- vision hparams ---\n");
                LOG_INF("%s: image_size:         %d\n", __func__, hparams.image_size);
                LOG_INF("%s: patch_size:         %d\n", __func__, hparams.patch_size);
                LOG_INF("%s: has_llava_proj:     %d\n", __func__, hparams.has_llava_projector);
                LOG_INF("%s: minicpmv_version:   %d\n", __func__, hparams.minicpmv_version);
                LOG_INF("%s: n_merge:            %d\n", __func__, hparams.n_merge);
                LOG_INF("%s: n_wa_pattern: %d\n", __func__, hparams.n_wa_pattern);
                if (!hparams.wa_layer_indexes.empty()) {
                    LOG_INF("%s: wa_layer_indexes:  ", __func__);
                    for (auto & layer : hparams.wa_layer_indexes) {
                        LOG_INF("%d ", layer);
                    }
                    LOG_INF("\n");
                }
                if (hparams.image_min_pixels > 0) {
                    LOG_INF("%s: image_min_pixels:   %d%s\n", __func__, hparams.image_min_pixels, hparams.custom_image_min_tokens > 0 ? " (custom value)" : "");
                }
                if (hparams.image_max_pixels > 0) {
                    LOG_INF("%s: image_max_pixels:   %d%s\n", __func__, hparams.image_max_pixels, hparams.custom_image_max_tokens > 0 ? " (custom value)" : "");
                }
                if (hparams.preproc_max_tiles > 0) {
                    LOG_INF("%s: preproc_tiles:      %d - %d\n", __func__, hparams.preproc_min_tiles, hparams.preproc_max_tiles);
                }
            } else if (is_audio) {
                LOG_INF("\n--- audio hparams ---\n");
                LOG_INF("%s: n_mel_bins:         %d\n", __func__, hparams.n_mel_bins);
                LOG_INF("%s: proj_stack_factor:  %d\n", __func__, hparams.proj_stack_factor);
                LOG_INF("%s: audio_chunk_len:    %d\n", __func__, hparams.audio_chunk_len);
                LOG_INF("%s: audio_sample_rate:  %d\n", __func__, hparams.audio_sample_rate);
                LOG_INF("%s: audio_n_fft:        %d\n", __func__, hparams.audio_n_fft);
                LOG_INF("%s: audio_window_len:   %d\n", __func__, hparams.audio_window_len);
                LOG_INF("%s: audio_hop_len:      %d\n", __func__, hparams.audio_hop_len);

                // Validate audio hparams loaded from GGUF metadata
                if (hparams.n_mel_bins <= 0 || hparams.n_mel_bins > 256) {
                    throw std::runtime_error(string_format("%s: n_mel_bins (%d) must be in range [1, 256]\n", __func__, hparams.n_mel_bins));
                }
                if (hparams.audio_sample_rate <= 0 || hparams.audio_n_fft <= 0 || hparams.audio_hop_len <= 0 || hparams.audio_window_len <= 0) {
                    throw std::runtime_error(string_format("%s: audio hparams invalid: sample_rate=%d n_fft=%d window_len=%d hop_len=%d\n",
                        __func__, hparams.audio_sample_rate, hparams.audio_n_fft, hparams.audio_window_len, hparams.audio_hop_len));
                }
            }
            LOG_INF("\n");
            LOG_INF("%s: model size:         %.2f MiB\n", __func__, model_size / 1024.0 / 1024.0);
            LOG_INF("%s: metadata size:      %.2f MiB\n", __func__, ggml_get_mem_size(ctx_meta.get()) / 1024.0 / 1024.0);
        }
    }

    void load_tensors(clip_ctx & ctx_clip) {
        auto & model = ctx_clip.model;
        auto & hparams = model.hparams;
        std::map<std::string, size_t> tensor_offset;
        std::vector<ggml_tensor *> tensors_to_load;

        auto fin = open_ifstream_binary(fname);
        if (!fin) {
            throw std::runtime_error(string_format("%s: failed to open %s\n", __func__, fname.c_str()));
        }

        // TODO @ngxson : support both audio and video in the future
        const char * prefix = model.modality == CLIP_MODALITY_AUDIO ? "a"
                             : model.modality == CLIP_MODALITY_GEN_AUDIO ? "a.gen.code"
                             : "v";

        // get offsets
        for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
            tensor_offset[name] = gguf_get_data_offset(ctx_gguf.get()) + gguf_get_tensor_offset(ctx_gguf.get(), i);
        }

        // create data context
        struct ggml_init_params params = {
            /*.mem_size =*/ static_cast<size_t>(gguf_get_n_tensors(ctx_gguf.get()) + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ true,
        };
        ctx_clip.ctx_data.reset(ggml_init(params));
        if (!ctx_clip.ctx_data) {
            throw std::runtime_error(string_format("%s: failed to init ggml context\n", __func__));
        }

        // helper function
        std::unordered_set<std::string> loaded_tensor_names;
        auto get_tensor = [&](const std::string & name, bool required = true) {
            // Each tensor should only be loaded once; duplicates indicate a bug
            if (loaded_tensor_names.count(name)) {
                throw std::runtime_error(string_format("%s: tensor already loaded: %s\n", __func__, name.c_str()));
            }
            ggml_tensor * cur = ggml_get_tensor(ctx_meta.get(), name.c_str());
            if (!cur && required) {
                throw std::runtime_error(string_format("%s: unable to find tensor %s\n", __func__, name.c_str()));
            }
            if (cur) {
                tensors_to_load.push_back(cur);
                ggml_tensor * data_tensor = ggml_dup_tensor(ctx_clip.ctx_data.get(), cur);
                ggml_set_name(data_tensor, cur->name);
                loaded_tensor_names.insert(name);
                cur = data_tensor;
                // add to weight memory counter
                ctx_clip.mem_usage[ggml_backend_get_device(ctx_clip.backend)] += ggml_nbytes(cur);
            }
            return cur;
        };

        model.class_embedding = get_tensor(TN_CLASS_EMBD, false);

        model.pre_ln_w = get_tensor(string_format(TN_LN_PRE, prefix, "weight"), false);
        model.pre_ln_b = get_tensor(string_format(TN_LN_PRE, prefix, "bias"),   false);

        model.post_ln_w = get_tensor(string_format(TN_LN_POST, prefix, "weight"), false);
        model.post_ln_b = get_tensor(string_format(TN_LN_POST, prefix, "bias"),   false);

        model.patch_bias = get_tensor(TN_PATCH_BIAS, false);
        model.patch_embeddings_0 = get_tensor(TN_PATCH_EMBD,   false);
        model.patch_embeddings_1 = get_tensor(TN_PATCH_EMBD_1, false);

        model.norm_embd_w = get_tensor(string_format(TN_NORM_EMBD, "weight"), false);
        model.norm_embd_b = get_tensor(string_format(TN_NORM_EMBD, "bias"),   false);

        model.position_embeddings = get_tensor(string_format(TN_POS_EMBD, prefix), false);

        // layers
        const int n_layers_to_load = hparams.n_layer;
        model.layers.resize(n_layers_to_load);
        for (int il = 0; il < n_layers_to_load; ++il) {
            auto & layer = model.layers[il];
            layer.k_w    = get_tensor(string_format(TN_ATTN_K,      prefix, il, "weight"), false);
            layer.q_w    = get_tensor(string_format(TN_ATTN_Q,      prefix, il, "weight"), false);
            layer.v_w    = get_tensor(string_format(TN_ATTN_V,      prefix, il, "weight"), false);
            layer.o_w    = get_tensor(string_format(TN_ATTN_OUTPUT, prefix, il, "weight"));
            layer.qkv_w  = get_tensor(string_format(TN_ATTN_QKV,    prefix, il, "weight"), false);
            layer.k_norm = get_tensor(string_format(TN_ATTN_K_NORM, prefix, il, "weight"), false);
            layer.q_norm = get_tensor(string_format(TN_ATTN_Q_NORM, prefix, il, "weight"), false);
            layer.ln_1_w = get_tensor(string_format(TN_LN_1,        prefix, il, "weight"), false);
            layer.ln_2_w = get_tensor(string_format(TN_LN_2,        prefix, il, "weight"), false);
            layer.ls_1_w        = get_tensor(string_format(TN_LS_1,         prefix, il, "weight"), false); // no bias
            layer.ls_2_w        = get_tensor(string_format(TN_LS_2,         prefix, il, "weight"), false); // no bias
            layer.ls_out_w      = get_tensor(string_format(TN_LS_OUT,        prefix, il, "weight"), false); // no bias
            layer.attn_post_norm_w = get_tensor(string_format(TN_ATTN_POST_NORM, prefix, il, "weight"), false); // no bias
            layer.ff_post_norm_w   = get_tensor(string_format(TN_FFN_POST_NORM,  prefix, il, "weight"), false); // no bias

            layer.k_b    = get_tensor(string_format(TN_ATTN_K,      prefix, il, "bias"), false);
            layer.q_b    = get_tensor(string_format(TN_ATTN_Q,      prefix, il, "bias"), false);
            layer.v_b    = get_tensor(string_format(TN_ATTN_V,      prefix, il, "bias"), false);
            layer.o_b    = get_tensor(string_format(TN_ATTN_OUTPUT, prefix, il, "bias"), false);
            layer.qkv_b  = get_tensor(string_format(TN_ATTN_QKV,    prefix, il, "bias"), false);
            layer.ln_1_b = get_tensor(string_format(TN_LN_1,        prefix, il, "bias"), false);
            layer.ln_2_b = get_tensor(string_format(TN_LN_2,        prefix, il, "bias"), false);

            // MoE ffn (dots3note vision pyramid blocks); replaces the dense ffn when present
            layer.ff_gate_inp_w  = get_tensor(string_format(TN_FFN_GATE_INP,  prefix, il, "weight"), false);
            layer.ff_gate_exps_w = get_tensor(string_format(TN_FFN_GATE_EXPS, prefix, il, "weight"), false);
            layer.ff_up_exps_w   = get_tensor(string_format(TN_FFN_UP_EXPS,   prefix, il, "weight"), false);
            layer.ff_down_exps_w = get_tensor(string_format(TN_FFN_DOWN_EXPS, prefix, il, "weight"), false);
            layer.ff_exp_probs_b = get_tensor(string_format(TN_FFN_EXP_PROBS_B, prefix, il, "weight"), false);
            const bool is_moe = layer.ff_gate_exps_w != nullptr;

            // ffn
            layer.ff_up_w   = get_tensor(string_format(TN_FFN_UP,   prefix, il, "weight"), !is_moe);
            layer.ff_up_b   = get_tensor(string_format(TN_FFN_UP,   prefix, il, "bias"),   false);
            layer.ff_gate_w = get_tensor(string_format(TN_FFN_GATE, prefix, il, "weight"), false);
            layer.ff_gate_b = get_tensor(string_format(TN_FFN_GATE, prefix, il, "bias"),   false);
            layer.ff_down_w = get_tensor(string_format(TN_FFN_DOWN, prefix, il, "weight"), !is_moe);
            layer.ff_down_b = get_tensor(string_format(TN_FFN_DOWN, prefix, il, "bias"),   false);

            // mimovl per-head attention sink bias
            layer.attn_sinks = get_tensor(string_format(TN_ATTN_SINKS, prefix, il), false);

            // qwen3vl deepstack layer
            layer.deepstack_norm_w = get_tensor(string_format(TN_DEEPSTACK_NORM, il, "weight"), false);
            layer.deepstack_norm_b = get_tensor(string_format(TN_DEEPSTACK_NORM, il, "bias"), false);
            layer.deepstack_fc1_w  = get_tensor(string_format(TN_DEEPSTACK_FC1,  il, "weight"), false);
            layer.deepstack_fc1_b  = get_tensor(string_format(TN_DEEPSTACK_FC1,  il, "bias"), false);
            layer.deepstack_fc2_w  = get_tensor(string_format(TN_DEEPSTACK_FC2,  il, "weight"), false);
            layer.deepstack_fc2_b  = get_tensor(string_format(TN_DEEPSTACK_FC2,  il, "bias"), false);
            if (layer.has_deepstack()) {
                model.n_deepstack_layers++;
            }

        }


        switch (model.proj_type) {
            case PROJECTOR_TYPE_QWEN3VL:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            default:
                GGML_ASSERT(false && "unknown projector type");
        }

        // load data
        {
            std::vector<uint8_t> read_buf;

            // start loading event
            if (progress_callback){
                progress_callback(0.0, progress_callback_user_data);
            }

            // compute total tensor data size for progress reporting
            size_t total_data_size = 0;
            for (auto & t : tensors_to_load) {
                total_data_size += ggml_nbytes(t);
            }

            // alloc memory and offload data
            ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(ctx_clip.backend);
            ctx_clip.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_clip.ctx_data.get(), buft));
            ggml_backend_buffer_set_usage(ctx_clip.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            // read the weight from file
            if (!ctx_clip.no_alloc) {
                size_t data_loaded = 0;
                for (auto & t : tensors_to_load) {
                    ggml_tensor * cur = ggml_get_tensor(ctx_clip.ctx_data.get(), t->name);
                    GGML_ASSERT(cur && "tensor not found in ctx_data");
                    auto it_off = tensor_offset.find(t->name);
                    GGML_ASSERT(it_off != tensor_offset.end() && "no offset for tensor");
                    const size_t offset = it_off->second;
                    fin.seekg(offset, std::ios::beg);
                    if (!fin) {
                        throw std::runtime_error(string_format("%s: failed to seek for tensor %s\n", __func__, t->name));
                    }
                    size_t num_bytes = ggml_nbytes(cur);
                    if (ggml_backend_buft_is_host(buft)) {
                        // for the CPU and Metal backend, we can read directly into the tensor
                        fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
                    } else {
                        // read into a temporary buffer first, then copy to device memory
                        read_buf.resize(num_bytes);
                        fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                        ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
                    }
                    data_loaded += num_bytes;
                    if (progress_callback && total_data_size > 0) {
                        const float progress = (float)data_loaded / (float)total_data_size;
                        if (!progress_callback(progress, progress_callback_user_data)) {
                            throw std::runtime_error(string_format("%s: model loading cancelled by progress_callback\n", __func__));
                        }
                    }
                }
                LOG_DBG("%s: loaded %zu tensors from %s\n", __func__, tensors_to_load.size(), fname.c_str());
            } else {
                LOG_DBG("%s: no_alloc is set, skipping tensor data loading (%zu tensors)\n", __func__, tensors_to_load.size());
            }
            fin.close();
        }

    }

    struct support_info_op {
        ggml_tensor * op;

        // true if the op runs on the accelerated ctx_clip.backend
        bool is_accel = true;
    };

    struct support_info_graph {
        // whether the clip_ctx.backend supports flash attention
        bool fattn = true;
        ggml_tensor * fattn_op = nullptr; // for debugging

        std::vector<support_info_op> ops;
    };

    static clip_image_f32_batch get_dummy_batch(clip_ctx & ctx_clip) {
        // create a fake batch
        const auto & hparams = ctx_clip.model.hparams;
        clip_image_f32_batch batch;
        clip_image_f32 img;
        if (ctx_clip.model.modality == CLIP_MODALITY_VISION) {
            const int sz = hparams.warmup_image_size;
            img.set_size({sz, sz}, false, false);
            LOG_INF("%s: warmup with image size = %d x %d\n", __func__, sz, sz);
        } else {
            if (hparams.n_mel_bins <= 0 || hparams.n_mel_bins > 256) {
                throw std::runtime_error(string_format("%s: invalid n_mel_bins (%d), must be in [1, 256]\n", __func__, hparams.n_mel_bins));
            }
            img.set_size({hparams.warmup_audio_size, hparams.n_mel_bins}, false, false);
            LOG_INF("%s: warmup with audio size = %d\n", __func__, hparams.warmup_audio_size);
        }
        batch.entries.push_back(img);
        return batch;
    }

    static void init_ctx(clip_ctx & ctx_clip) {
        ctx_clip.buf_compute_meta.resize(ctx_clip.max_nodes * ggml_tensor_overhead() + ggml_graph_overhead());

        // check batching support
        auto batch = get_dummy_batch(ctx_clip);
        auto builder = clip_get_graph_builder(&ctx_clip, batch);
        ctx_clip.support_batch = builder->support_batch();
    }

    static void warmup(clip_ctx & ctx_clip) {
        auto batch = get_dummy_batch(ctx_clip);
        warmup(ctx_clip, batch);
    }

    static void warmup(clip_ctx & ctx_clip, const clip_image_f32_batch & batch) {
        support_info_graph info;

        if (ctx_clip.flash_attn_type == CLIP_FLASH_ATTN_TYPE_AUTO) {
            // try to enable flash attention to see if it's supported
            ctx_clip.flash_attn_type = CLIP_FLASH_ATTN_TYPE_ENABLED;
            info = reserve_compute_meta(ctx_clip, batch);
            if (!info.fattn && info.fattn_op) {
                auto op = info.fattn_op;
                LOG_WRN("%s: *****************************************************************\n", __func__);
                LOG_WRN("%s: WARNING: flash attention not supported by %s, memory usage will increase\n", __func__, ggml_backend_name(ctx_clip.backend));
                LOG_WRN("%s: op params: \n", __func__);
                static auto print_shape = [](const char * fn, const char * name, ggml_tensor * t) {
                    LOG_WRN("%s:   %s: type = %s, ne = [%d %d %d %d], nb = [%d %d %d %d]\n", fn,
                            name, ggml_type_name(t->type),
                            t->ne[0], t->ne[1], t->ne[2], t->ne[3],
                            t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
                };
                print_shape(__func__, " dst", op);
                print_shape(__func__, "src0", op->src[0]);
                print_shape(__func__, "src1", op->src[1]);
                print_shape(__func__, "src2", op->src[2]);
                LOG_WRN("%s: please report this on github as an issue\n", __func__);
                LOG_WRN("%s: *****************************************************************\n", __func__);
                ctx_clip.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
                reserve_compute_meta(ctx_clip, batch);
            }
        } else {
            info = reserve_compute_meta(ctx_clip, batch);
            if (!info.fattn && ctx_clip.flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
                LOG_WRN("%s: flash attention is not supported by the current backend; falling back to CPU (performance will be degraded)\n", __func__);
            }
        }

        ctx_clip.is_allocated = true; // mark buffers as allocated

        LOG_INF("%s: flash attention is %s\n", __func__,
            (ctx_clip.flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) ? "enabled" : "disabled");

        // print ops that are not supported by the GPU backend (if there is one)
        if (ctx_clip.backend && ctx_clip.backend != ctx_clip.backend_cpu) {
            std::vector<support_info_op> unsupported_ops;
            for (const auto & op : info.ops) {
                if (!op.is_accel) {
                    unsupported_ops.push_back(op);
                }
            }
            if (!unsupported_ops.empty()) {
                LOG_WRN("%s: *****************************************************************\n", __func__);
                LOG_WRN("%s: WARNING: the CLIP graph uses unsupported operators by the backend\n", __func__);
                LOG_WRN("%s:          the performance will be suboptimal                      \n", __func__);
                LOG_WRN("%s:          list of unsupported ops (backend=%s):\n", __func__, ggml_backend_name(ctx_clip.backend));
                for (const auto & op : unsupported_ops) {
                    LOG_WRN("%s: %16s: type = %s, ne = [%d %d %d %d]\n", __func__,
                            ggml_op_name(op.op->op),
                            ggml_type_name(op.op->type),
                            op.op->ne[0], op.op->ne[1], op.op->ne[2], op.op->ne[3]);
                }
                LOG_WRN("%s: flash attention is %s\n", __func__,
                    (ctx_clip.flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) ? "enabled" : "disabled");
                LOG_WRN("%s: please report this on github as an issue\n", __func__);
                LOG_WRN("%s: ref: https://github.com/ggml-org/llama.cpp/pull/16837#issuecomment-3461676118\n", __func__);
                LOG_WRN("%s: *****************************************************************\n", __func__);
            }
        }
    }

    // only initialize backend buffers, but do not allocate them yet
    static support_info_graph reserve_compute_meta(clip_ctx & ctx_clip, const clip_image_f32_batch & batch) {
        ggml_cgraph * gf = clip_get_graph_builder(&ctx_clip, batch)->build();
        ggml_backend_sched_reserve(ctx_clip.sched.get(), gf);

        ctx_clip.mem_compute.clear();
        for (size_t i = 0; i < ctx_clip.backend_ptrs.size(); ++i) {
            ggml_backend_t backend = ctx_clip.backend_ptrs[i];
            ggml_backend_buffer_type_t buft = ctx_clip.backend_buft[i];
            size_t size = ggml_backend_sched_get_buffer_size(ctx_clip.sched.get(), backend);
            if (size > 1) {
                LOG_INF("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                        ggml_backend_buft_name(buft),
                        size / 1024.0 / 1024.0);
            }
            ctx_clip.mem_compute[ggml_backend_get_device(backend)] += size;
        }

        const int n_splits = ggml_backend_sched_get_n_splits(ctx_clip.sched.get());
        const int n_nodes  = ggml_graph_n_nodes(gf);

        LOG_INF("%s: graph splits = %d, nodes = %d\n", __func__,  n_splits, n_nodes);

        support_info_graph res {
            /*.fattn    = */ true,
            /*.fattn_op = */ nullptr,
            /*.ops      = */ {},
        };

        // check op support
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            res.ops.push_back({node, true});
            if (!ggml_backend_supports_op(ctx_clip.backend, node)) {
                res.ops.back().is_accel = false;
                if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                    res.fattn    = false;
                    res.fattn_op = node;
                }
            }
        }

        return res;
    }

    void get_bool(const std::string & key, bool & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_bool(ctx_gguf.get(), i);
    }

    void get_i32(const std::string & key, int & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_i32(ctx_gguf.get(), i);
    }

    void get_u32(const std::string & key, int & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        const uint32_t val = gguf_get_val_u32(ctx_gguf.get(), i);
        // sanity check
        if (val > (uint32_t) INT32_MAX) {
            throw std::runtime_error(string_format("%s: value %u for key '%s' exceeds INT32_MAX\n",
                __func__, val, key.c_str()));
        }
        output = (int) val;
    }

    void get_f32(const std::string & key, float & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_f32(ctx_gguf.get(), i);
    }

    void get_arr_f32(const std::string & key, std::vector<float> & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        if (gguf_get_kv_type(ctx_gguf.get(), i) != GGUF_TYPE_ARRAY) {
            throw std::runtime_error(string_format("%s: key '%s' is not an array\n", __func__, key.c_str()));
        }
        const auto type = gguf_get_arr_type(ctx_gguf.get(), i);
        if (type != GGUF_TYPE_FLOAT32) {
            throw std::runtime_error(string_format("%s: array '%s' has type %d, expected %d (GGUF_TYPE_FLOAT32)\n", __func__, key.c_str(), type, GGUF_TYPE_FLOAT32));
        }
        const size_t n = gguf_get_arr_n(ctx_gguf.get(), i);
        if (n > (size_t) std::numeric_limits<int>::max()) {
            throw std::runtime_error(string_format("%s: array '%s' is too large (%zu elements)\n", __func__, key.c_str(), n));
        }
        output.resize(n);
        const float * values = (const float *)gguf_get_arr_data(ctx_gguf.get(), i);
        for (size_t j = 0; j < n; ++j) {
            output[j] = values[j];
        }
    }

    void get_string(const std::string & key, std::string & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = std::string(gguf_get_val_str(ctx_gguf.get(), i));
    }

    void get_arr_int(const std::string & key, std::vector<int> & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        if (gguf_get_kv_type(ctx_gguf.get(), i) != GGUF_TYPE_ARRAY) {
            throw std::runtime_error(string_format("%s: key '%s' is not an array\n", __func__, key.c_str()));
        }
        const auto type = gguf_get_arr_type(ctx_gguf.get(), i);
        if (type != GGUF_TYPE_INT32) {
            throw std::runtime_error(string_format("%s: array '%s' has type %d, expected %d (GGUF_TYPE_INT32)\n", __func__, key.c_str(), type, GGUF_TYPE_INT32));
        }
        const size_t n = gguf_get_arr_n(ctx_gguf.get(), i);
        if (n > (size_t) std::numeric_limits<int>::max()) {
            throw std::runtime_error(string_format("%s: array '%s' is too large (%zu elements)\n", __func__, key.c_str(), n));
        }
        output.resize(n);
        const int32_t * values = (const int32_t *)gguf_get_arr_data(ctx_gguf.get(), i);
        for (size_t j = 0; j < n; ++j) {
            output[j] = values[j];
        }
    }

};

struct clip_init_result clip_init(const char * fname, struct clip_context_params ctx_params) {
    clip_ctx * ctx_vision = nullptr;
    clip_ctx * ctx_audio = nullptr;
    clip_ctx * ctx_gen_audio = nullptr;

    try {
        clip_model_loader loader(fname,
            /* skip_tensors */ false,
            ctx_params.progress_callback,
            ctx_params.progress_callback_user_data);
        if (loader.has_vision) {
            ctx_vision = new clip_ctx(ctx_params);
            loader.load_hparams(ctx_vision->model, CLIP_MODALITY_VISION);
            loader.load_tensors(*ctx_vision);
            loader.init_ctx(*ctx_vision);
            if (ctx_params.warmup) {
                loader.warmup(*ctx_vision);
            }

        }

        if (loader.has_audio) {
            ctx_audio = new clip_ctx(ctx_params);
            loader.load_hparams(ctx_audio->model, CLIP_MODALITY_AUDIO);
            loader.load_tensors(*ctx_audio);
            loader.init_ctx(*ctx_audio);
            if (ctx_params.warmup) {
                loader.warmup(*ctx_audio);
            }
        }

        if (loader.has_gen_audio) {
            ctx_gen_audio = new clip_ctx(ctx_params);
            loader.load_hparams(ctx_gen_audio->model, CLIP_MODALITY_GEN_AUDIO);
            loader.load_tensors(*ctx_gen_audio);
            // TODO: fix warmup
            ctx_gen_audio->buf_compute_meta.resize(ctx_gen_audio->max_nodes * ggml_tensor_overhead() + ggml_graph_overhead());
        }

    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to load model '%s': %s\n", __func__, fname, e.what());

        delete ctx_vision;
        delete ctx_audio;
        delete ctx_gen_audio;

        return {nullptr, nullptr, nullptr};
    }

    return {ctx_vision, ctx_audio, ctx_gen_audio};
}

struct clip_cap clip_get_cap(const char * fname) {
    clip_cap res;
    clip_model_loader loader(fname, /* skip_tensors= */ true);
    res.has_vision = loader.has_vision;
    res.has_audio  = loader.has_audio;
    return res;
}

void clip_free(clip_ctx * ctx) {
    if (ctx == nullptr) {
        return;
    }
    delete ctx;
}

const char * clip_patch_merge_type(const struct clip_ctx * ctx) {
    return ctx->model.hparams.mm_patch_merge_type == PATCH_MERGE_SPATIAL_UNPAD ? "spatial_unpad" : "flat";
}

int clip_n_output_tokens_x(const clip_ctx * ctx, const clip_image_f32 * img) {
    const auto & params = ctx->model.hparams;
    const int n_total = clip_n_output_tokens(ctx, img);
    const auto & proj = ctx->proj_type();
    switch (proj) {
        case PROJECTOR_TYPE_QWEN3VL:
            return (img->nx() / params.patch_size) / 2;
        default:
            break;
    }
    return n_total;
}

int clip_n_output_tokens_y(const clip_ctx * ctx, const clip_image_f32 * img) {
    const auto & params = ctx->model.hparams;
    const auto & proj = ctx->proj_type();
    switch (proj) {
        case PROJECTOR_TYPE_QWEN3VL:
            return (img->ny() / params.patch_size) / 2;
        default:
            break;
    }
    return 1;
}

int clip_n_output_tokens(const clip_ctx * ctx, const clip_image_f32 * img) {
    const auto & params = ctx->model.hparams;

    // for models with fixed size image, the input image is already pre-processed and resized to square
    int patch_size = params.patch_size;
    int n_patches = (img->nx() / patch_size) * (img->ny() / patch_size);

    projector_type proj = ctx->proj_type();

    switch (proj) {
        case PROJECTOR_TYPE_QWEN3VL:
            {
                // dynamic size (2 conv, so double patch size)
                int x_patch = img->nx() / (params.patch_size * 2);
                int y_patch = img->ny() / (params.patch_size * 2);
                n_patches = x_patch * y_patch;
            } break;
        default:
            GGML_ABORT("unsupported projector type");
    }

    return n_patches;
}

bool clip_image_encode(struct clip_ctx * ctx, int n_threads, const clip_image_f32 * img, std::vector<float> & out_vec) {
    clip_image_f32_batch imgs;
    clip_image_f32 img_copy = *img;
    imgs.entries.push_back(std::move(img_copy));

    return clip_image_batch_encode(ctx, n_threads, &imgs, out_vec);
}

bool clip_image_batch_encode(clip_ctx * ctx, int n_threads, const clip_image_f32_batch * imgs_c_ptr, std::vector<float> & out_batch_embd) {
    clip_encode_params params;
    params.imgs = imgs_c_ptr;
    params.n_threads = n_threads;
    params.out_embd = &out_batch_embd;

    return clip_encode(ctx, &params);
}

// persisted state slots of the gen-audio decoder, per pipeline
static std::vector<c2w_state_slot> list_gen_state_slots(const clip_hparams &, const clip_model &) {
    return {};
}

bool clip_encode(struct clip_ctx * ctx, struct clip_encode_params * params) {
    const clip_image_f32_batch & imgs = *params->imgs;
    int n_batch_cur = imgs.entries.size();

    // [QWEN_VIDEO] for video models, the batch dimension is used as temporal dimension for merged frames
    if (!ctx->support_batch && n_batch_cur > clip_model_n_temporal_merge(ctx)) {
        LOG_ERR("%s: batch size %d exceeds maximum supported batch/temporal-merge size %d\n", __func__, n_batch_cur, clip_model_n_temporal_merge(ctx));
        return false;
    }

    // if buffers are not allocated, we need to do a warmup run to allocate them
    if (!ctx->is_allocated) {
        clip_model_loader::warmup(*ctx, *params->imgs);
    }

    if (params->seed != ctx->rng_seed) {
        ctx->rng_seed = params->seed;
        ctx->rng.seed(params->seed == UINT32_MAX ? std::random_device{}() : params->seed);
    }

    // build the inference graph
    ggml_backend_sched_reset(ctx->sched.get());
    ggml_cgraph * gf = clip_get_graph_builder(ctx, imgs, params)->build();
    if (!ggml_backend_sched_alloc_graph(ctx->sched.get(), gf)) {
        LOG_ERR("%s: failed to allocate compute graph\n", __func__);
        return false;
    }

    // set inputs
    const auto & model   = ctx->model;
    const auto & hparams = model.hparams;

    const int image_size_width  = imgs.entries[0].nx();
    const int image_size_height = imgs.entries[0].ny();

    const int patch_size    = hparams.patch_size;
    const int num_patches   = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int n_pos = num_patches + (model.class_embedding ? 1 : 0);

    auto get_inp_tensor = [&gf](const char * name) {
        ggml_tensor * inp = ggml_graph_get_tensor(gf, name);
        if (inp == nullptr) {
            GGML_ABORT("Failed to get tensor %s", name);
        }
        if (!(inp->flags & GGML_TENSOR_FLAG_INPUT)) {
            GGML_ABORT("Tensor %s is not an input tensor", name);
        }
        return inp;
    };

    auto set_input_f32 = [&get_inp_tensor](const char * name, const std::vector<float> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    auto set_input_i32 = [&get_inp_tensor](const char * name, std::vector<int32_t> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    // set input pixel values
    if (!imgs.is_audio) {
        size_t nelem = 0;
        for (const auto & img : imgs.entries) {
            nelem += img.nx() * img.ny() * 3;
        }
        std::vector<float> inp_raw(nelem);

        // layout of data (note: the channel dim is unrolled to better visualize the layout):
        //
        // ┌──W──┐
        // │     H │  channel = R
        // ├─────┤ │
        // │     H │  channel = G
        // ├─────┤ │
        // │     H │  channel = B
        // └─────┘ │
        //   ──────┘ x B

        // IMPORTANT: [QWEN_VIDEO] the batch dim is currently used for temporal dim in Qwen-VL models
        // All entries must have the same spatial size (enforced by can_batch_with() during merging)
        {
            const int nx = imgs.entries[0].nx();
            const int ny = imgs.entries[0].ny();
            const int n  = nx * ny;

            for (int b = 0; b < n_batch_cur; b++) {
                LOG_DBG("%s: copying image %d/%d to input buffer (nx=%d, ny=%d)\n", __func__, b+1, n_batch_cur, nx, ny);
                const auto & buf = imgs.entries[b].get_ro_buf();
                float * batch_entry = inp_raw.data() + b * (3*n);
                for (int y = 0; y < ny; y++) {
                    for (int x = 0; x < nx; x++) {
                        size_t base_src = 3*(y * nx + x);
                        size_t base_dst =    y * nx + x;
                        batch_entry[      base_dst] = buf[base_src    ];
                        batch_entry[1*n + base_dst] = buf[base_src + 1];
                        batch_entry[2*n + base_dst] = buf[base_src + 2];
                    }
                }
            }
        }
        set_input_f32("inp_raw", inp_raw);

    } else if (params->gen_process != CLIP_GEN_PROCESS_GEN_WAV) {
        // audio input. GEN_WAV is not here: it takes codes or feats, set in the switch below
        GGML_ASSERT(imgs.entries.size() == 1);

        const auto & mel_inp = imgs.entries[0];
        const auto & buf = mel_inp.get_ro_buf();
        const int n_step = mel_inp.nx();
        const int n_mel  = mel_inp.ny();
        GGML_ASSERT((size_t)n_step * n_mel == buf.size());

        set_input_f32("inp_raw", buf);
    }

    // set input per projector
    switch (ctx->model.proj_type) {
        case PROJECTOR_TYPE_QWEN3VL:
            {
                const int merge_ratio = hparams.n_merge;
                const int pw = image_size_width  / patch_size;
                const int ph = image_size_height / patch_size;
                std::vector<int> positions(n_pos * 4);
                int ptr = 0;
                for (int y = 0; y < ph; y += merge_ratio) {
                    for (int x = 0; x < pw; x += merge_ratio) {
                        for (int dy = 0; dy < 2; dy++) {
                            for (int dx = 0; dx < 2; dx++) {
                                positions[                  ptr] = y + dy;
                                positions[    num_patches + ptr] = x + dx;
                                positions[2 * num_patches + ptr] = y + dy;
                                positions[3 * num_patches + ptr] = x + dx;
                                ptr++;
                            }
                        }
                    }
                }

                set_input_i32("positions", positions);
            } break;
        default:
            GGML_ABORT("Unknown projector type");
    }

    // ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
    ggml_backend_dev_t dev = ggml_backend_get_device(ctx->backend_cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) {
        auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            ggml_backend_set_n_threads_fn(ctx->backend_cpu, params->n_threads);
        }
    }

    auto status = ggml_backend_sched_graph_compute(ctx->sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("%s: ggml_backend_sched_graph_compute failed with error %d\n", __func__, status);
        return false;
    }

    // the last node is the embedding tensor, code2wav has no out_embd
    ggml_tensor * embeddings = params->out_embd ? ggml_graph_node(gf, -1) : nullptr;

    if (embeddings != nullptr) {
        // sanity check (assuming that all images in batch have the same number of tokens, so we only check the first one)
        const int n_tokens_out = embeddings->ne[1];
        const int expected_n_tokens_out = clip_n_output_tokens(ctx, &imgs.entries[0]);
        if (n_tokens_out != expected_n_tokens_out) {
            LOG_ERR("%s: expected output %d tokens, got %d\n", __func__, expected_n_tokens_out, n_tokens_out);
            GGML_ABORT("Invalid number of output tokens");
        }

        LOG_DBG("%s: output embedding shape [%d, %d, %d]\n", __func__,
            (int)embeddings->ne[0], (int)embeddings->ne[1], (int)embeddings->ne[2]);

        // copy output to user buffer if provided
        // if output is empty, skip the copy
        auto & out_batch_embd = *params->out_embd;
        if (!out_batch_embd.empty()) {
            if (out_batch_embd.size() != (size_t)ggml_nelements(embeddings)) {
                LOG_ERR("%s: output buffer has %zu elements but expected %zu\n", __func__, out_batch_embd.size(), (size_t)ggml_nelements(embeddings));
                GGML_ABORT("Output buffer size mismatch");
            }
            ggml_backend_tensor_get(embeddings, out_batch_embd.data(), 0, ggml_nbytes(embeddings));
        } else {
            LOG_WRN("%s: output buffer is empty, skipping copy\n", __func__);
        }
    }

    //
    // for audio gen models
    //

    // optional outputs: a pipeline yields codes or feats, and not all have an eos head
    if (params->out_codes != nullptr) {
        ggml_tensor * codes = ggml_graph_get_tensor(gf, "out_codes");
        if (codes != nullptr) {
            auto & out_codes = *params->out_codes;
            out_codes.resize(ggml_nelements(codes));
            ggml_backend_tensor_get(codes, out_codes.data(), 0, ggml_nbytes(codes));
        }
    }
    if (params->out_feats != nullptr) {
        ggml_tensor * feats = ggml_graph_get_tensor(gf, "out_feats");
        if (feats != nullptr) {
            auto & out_feats = *params->out_feats;
            out_feats.resize(ggml_nelements(feats));
            ggml_backend_tensor_get(feats, out_feats.data(), 0, ggml_nbytes(feats));
        }
    }
    if (params->out_is_eos != nullptr) {
        ggml_tensor * eos = ggml_graph_get_tensor(gf, "out_eos_score");
        if (eos != nullptr) {
            GGML_ASSERT(ggml_nelements(eos) == 1);
            float score = 0.0f;
            ggml_backend_tensor_get(eos, &score, 0, sizeof(float));
            *params->out_is_eos = score > hparams.gen_eos_threshold;
        }
    }
    if (params->out_audio != nullptr) {
        ggml_tensor * audio = ggml_graph_get_tensor(gf, "out_audio");
        if (audio == nullptr) {
            GGML_ABORT("out_audio requested but graph has no \"out_audio\" tensor");
        }
        auto & out_audio = *params->out_audio;
        out_audio.resize(ggml_nelements(audio));
        ggml_backend_tensor_get(audio, out_audio.data(), 0, ggml_nbytes(audio));

        // drop the tail audio that comes from the code-0 rear padding
        const int64_t n_codes    = params->codes ? model.gen_code_head_w->ne[2] + 1 : 0;
        const int64_t n_frames_w = hparams.wav_tfm_swa;
        const int64_t n_frames   = params->codes ? (int64_t) params->codes->size() / n_codes : n_frames_w;
        if (n_frames < n_frames_w) {
            const size_t hop = out_audio.size() / n_frames_w;
            out_audio.resize((size_t) n_frames * hop);
        }
    }
    if (params->state_out != nullptr) {
        auto & state_out = *params->state_out;
        size_t total = 0;
        for (const auto & slot : list_gen_state_slots(hparams, model)) {
            total += (size_t) (slot.ne0 * slot.ne1) * sizeof(float);
        }
        state_out.resize(total);
        size_t offset = 0;
        for (const auto & slot : list_gen_state_slots(hparams, model)) {
            ggml_tensor * t = ggml_graph_get_tensor(gf, ("state_out_" + slot.name).c_str());
            if (t == nullptr) {
                GGML_ABORT("state_out requested but graph has no \"state_out_%s\" tensor", slot.name.c_str());
            }
            const size_t nb = ggml_nbytes(t);
            ggml_backend_tensor_get(t, state_out.data() + offset, 0, nb);
            offset += nb;
        }
    }

    //
    // Debug: dump final embeddings if MTMD_DEBUG_EMBEDDINGS is set
    //

    if (ctx->debug_output_embeddings && embeddings != nullptr) {
        const int64_t n_embd = embeddings->ne[0];
        const int64_t n_tokens = embeddings->ne[1];
        std::vector<float> emb_data(ggml_nelements(embeddings));
        ggml_backend_tensor_get(embeddings, emb_data.data(), 0, ggml_nbytes(embeddings));

        LOG_INF("\n=== MTMD_DEBUG_EMBEDDINGS ===\n");
        LOG_INF("Shape: [%lld, %lld]\n", (long long)n_embd, (long long)n_tokens);

        // TEMP debugging (parity validation), will be removed before merge
        // when the env var holds a path, dump the raw data: [int32 n_tokens][int32 n_embd][f32 data]
        const char * dump_path = std::getenv("MTMD_DEBUG_EMBEDDINGS");
        if (dump_path && strcmp(dump_path, "1") != 0) {
            FILE * f = fopen(dump_path, "wb");
            if (f) {
                const int32_t hdr[2] = { (int32_t)n_tokens, (int32_t)n_embd };
                fwrite(hdr, sizeof(hdr), 1, f);
                fwrite(emb_data.data(), sizeof(float), emb_data.size(), f);
                fclose(f);
            }
        }

        // Print first few values of first token
        LOG_INF("Token 0 (first 16 values): ");
        for (int i = 0; i < std::min((int64_t)16, n_embd); i++) {
            LOG_INF("%.6f ", emb_data[i]);
        }
        LOG_INF("\n");

        // Print last few values of first token
        if (n_embd > 16) {
            LOG_INF("Token 0 (last 16 values):  ");
            for (int64_t i = n_embd - 16; i < n_embd; i++) {
                LOG_INF("%.6f ", emb_data[i]);
            }
            LOG_INF("\n");
        }

        // Compute and print statistics
        float sum = 0.0f, sum_sq = 0.0f, min_val = emb_data[0], max_val = emb_data[0];
        for (size_t i = 0; i < emb_data.size(); i++) {
            sum += emb_data[i];
            sum_sq += emb_data[i] * emb_data[i];
            min_val = std::min(min_val, emb_data[i]);
            max_val = std::max(max_val, emb_data[i]);
        }
        float mean = sum / emb_data.size();
        float variance = (sum_sq / emb_data.size()) - (mean * mean);
        LOG_INF("Stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f, sum=%.6f\n",
                mean, sqrtf(variance), min_val, max_val, sum);
        LOG_INF("=== END MTMD_DEBUG_EMBEDDINGS ===\n\n");
    }

    return true;
}

int clip_n_mmproj_embd(const struct clip_ctx * ctx) {
    switch (ctx->model.proj_type) {
        case PROJECTOR_TYPE_QWEN3VL:
            // main path + deepstack paths
            return ctx->model.mm_1_b->ne[0] * (1 + ctx->model.n_deepstack_layers);
        default:
            GGML_ABORT("Unknown projector type");
    }
}

bool clip_is_llava(const struct clip_ctx * ctx) {
    return ctx->model.hparams.has_llava_projector;
}

bool clip_has_vision_encoder(const struct clip_ctx * ctx) {
    return ctx->model.modality == CLIP_MODALITY_VISION;
}

bool clip_has_audio_encoder(const struct clip_ctx * ctx) {
    return ctx->model.modality == CLIP_MODALITY_AUDIO;
}

bool clip_support_batch(const struct clip_ctx * ctx) {
    return ctx->support_batch;
}

// TODO @ngxson : this is no longer correct with mtmd_batch API
// this was only meant to be used by qwen-vl-based models, to fuse 2 input images into one (qwen-vl video support)
// this logic should be refactored in near future to distinctly handle "merge frames" and "batching"
int clip_model_n_temporal_merge(const struct clip_ctx * ctx) {
    switch (ctx->proj_type()) {
        case PROJECTOR_TYPE_QWEN3VL:
            return 2;
        default:
            return 1;
    }
}

//
// API used internally with mtmd
//

projector_type clip_get_projector_type(const struct clip_ctx * ctx) {
    return ctx->proj_type();
}

const clip_hparams * clip_get_hparams(const struct clip_ctx * ctx) {
    return &ctx->model.hparams;
}

std::map<ggml_backend_dev_t, size_t> clip_get_mem_usage(const struct clip_ctx * ctx) {
    std::map<ggml_backend_dev_t, size_t> result = ctx->mem_usage;
    for (auto & [dev, size] : ctx->mem_compute) {
        result[dev] += size;
    }
    return result;
}

//
// API for debugging
//

void clip_set_debug_output_embeddings(clip_ctx * ctx, bool enable) {
    ctx->debug_output_embeddings = enable;
}
