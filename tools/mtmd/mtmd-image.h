#pragma once

#include "ggml.h"
#include "clip-model.h"

#include <vector>
#include <string>

#define MTMD_INTERNAL_HEADER

struct mtmd_image_preproc_out {
    std::vector<clip_image_f32> entries;
    // grid size is required for llava-uhd style models

    clip_image_f32 overview; // overview image (downscaled image)
    int grid_x = 0;
    int grid_y = 0;

    void append(const clip_hparams & hparams, const clip_image_u8 & img, bool normalized = true);
    void append(const clip_hparams & hparams, const std::vector<clip_image_u8> & imgs, bool normalized = true);
    void append(const clip_hparams & hparams, clip_image_f32 & img, bool normalized = true);

    void append_overview(const clip_hparams & hparams, const clip_image_u8 & img, bool normalized = true);
    bool has_overview() const {
        return overview.nx() > 0 || overview.ny() > 0;
    }
};

// base class, models must inherit from this class
struct mtmd_image_preprocessor {
    const clip_hparams & hparams;

    mtmd_image_preprocessor(const clip_ctx * ctx): hparams(*clip_get_hparams(ctx)) {}

    virtual ~mtmd_image_preprocessor() = default;
    virtual mtmd_image_preproc_out preprocess(const clip_image_u8 & img) const = 0;
};

// resize image to multiple of patch_size*n_merge, while preserving aspect ratio
// if image_resize_pad is true, the resized image will be padded, otherwise it will be either stretched or center-cropped depending on image_resize_pad
// this is used by models with native support for dynamic image size, for example: Qwen-VL, Pixtral, Kimi-VL, etc
struct mtmd_image_preprocessor_dyn_size : mtmd_image_preprocessor {
    mtmd_image_preprocessor_dyn_size(const clip_ctx * ctx) : mtmd_image_preprocessor(ctx) {}
    mtmd_image_preproc_out preprocess(const clip_image_u8 & img) const override;
};
