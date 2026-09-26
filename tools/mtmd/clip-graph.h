#pragma once

#include "ggml.h"
#include "ggml-cpp.h"
#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"

#include <vector>
#include <functional>

#define DEFAULT_INTERPOLATION_MODE (GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS)

struct clip_graph {
    const clip_model & model;
    const clip_hparams & hparams;
    projector_type proj_type;

    const clip_image_f32 & img; // for backward compat
    const clip_image_f32_batch * img_batch = nullptr;

    const int patch_size;
    const int n_patches_x;
    const int n_patches_y;
    const int n_patches;
    const int n_embd;
    const int n_head;
    const int n_head_kv;
    const int d_head;
    const int n_layer;
    const int n_mmproj_embd;
    const float eps;
    float kq_scale; // TODO: maybe move this to hparams
    const clip_flash_attn_type flash_attn_type;

    // TODO [QWEN_VIDEO]: improve this in the future
    int n_batch = 1;

    ggml_context_ptr ctx0_ptr;
    ggml_context * ctx0;
    ggml_cgraph * gf;

    clip_graph(clip_ctx * ctx, const clip_image_f32 & img);

    // build sub-graph, reuse buf from parent
    clip_graph(const clip_graph & parent);

    virtual ~clip_graph() = default;
    virtual ggml_cgraph * build() = 0;

    // wrapper around ggml_mul_mat, allow hooking (e.g. LoRA, clamping) depending on the model
    // tensor w should be the weight matrix, and tensor x should be the input
    virtual ggml_tensor * build_mm(ggml_tensor * w, ggml_tensor * x) const;
    // TODO: build_mm(w, b, x) to support bias

    virtual bool support_batch() const {
        return false;
    }

    //
    // utility functions
    //
    void cb(ggml_tensor * cur0, const char * name, int il) const;

    // siglip2 naflex
    ggml_tensor * resize_position_embeddings(uint32_t interpolation_mode = DEFAULT_INTERPOLATION_MODE);

    // build the input after conv2d (inp_raw --> patches)
    // returns tensor with shape [n_embd, n_patches]
    ggml_tensor * build_inp();

    ggml_tensor * build_inp_raw(int channels = 3);

    ggml_tensor * build_norm(
            ggml_tensor * cur,
            ggml_tensor * mw,
            ggml_tensor * mb,
            norm_type type,
            float norm_eps,
            int il) const;

    ggml_tensor * build_ffn(
            ggml_tensor * cur,
            ggml_tensor * up,
            ggml_tensor * up_b,
            ggml_tensor * gate,
            ggml_tensor * gate_b,
            ggml_tensor * down,
            ggml_tensor * down_b,
            ffn_op_type type_op,
            int il) const;

    ggml_tensor * build_attn(
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * q_cur,
            ggml_tensor * k_cur,
            ggml_tensor * v_cur,
            ggml_tensor * kq_mask,
            float kq_scale,
            int il,
            ggml_tensor * sinks = nullptr) const;
};
