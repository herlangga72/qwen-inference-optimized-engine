#pragma once

#include "../clip-graph.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

/*
 * IMPORTANT: The mtmd module does NOT accept pull requests that are fully or predominantly AI-generated.
 * We encourage human contributors to ensure the quality and reliability of the codebase.
 */

struct clip_graph_qwen3vl : clip_graph {
    clip_graph_qwen3vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_tensor * build_inp_with_temporal_merge();
    ggml_cgraph * build() override;
};
