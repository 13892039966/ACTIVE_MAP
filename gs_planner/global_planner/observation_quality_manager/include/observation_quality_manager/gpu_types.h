#pragma once

#include <cstdint>
#include <vector>

// AdaptiveCpp expects compiler feature-test macros that GCC might miss.
// Define conservative fallbacks so #if __has_builtin(...) blocks compile.
#ifndef __has_builtin
#define __has_builtin(x) 0
#endif
#ifndef __has_feature
#define __has_feature(x) 0
#endif
#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

// ==========================================
// 1. 基础数学辅助 & 常量
// ==========================================
struct alignas(16) Float3 {
    float x, y, z, w; // w 用于对齐或填充
};

constexpr int SUB_MASK_ARRAY_SIZE = 32; // 1000 bits -> 32 uint32

// ==========================================
// 2. 核心数据结构：传入 GPU 的 Voxel
// ==========================================
struct alignas(16) GPUVoxel {
    // --- 位置与几何 (16 bytes) ---
    Float3 center;

    // --- 属性 (16 bytes) ---
    float geo_complexity;
    float tex_complexity;
    int32_t normal_bin_idx;
    float current_score;

    // --- 状态 (16 bytes) ---
    uint32_t obs_mask;
    uint32_t available_mask;  // 可用观测方向掩码（raycast后计算）
    int32_t well_observed;
    uint32_t is_frontier;     // 0 = poorly_observed voxel, 1 = frontier voxel

// --- 子体素精细掩码 (128 bytes) ---
uint32_t sub_masks[SUB_MASK_ARRAY_SIZE];

// --- 未知子体素掩码 (128 bytes) ---
uint32_t unknown_masks[SUB_MASK_ARRAY_SIZE];

    // --- 输出结果 (8 bytes) ---
    float max_possible_score;
    float padding;
};

// ==========================================
// 3. 核心数据结构：候选位置 (Candidate Position)
// ==========================================
struct alignas(16) GPUCandidatePos {
    Float3 pos;       // 候选点的世界坐标
    int32_t id;       // 对应 CPU 端 vector 的原始下标
    float padding[3]; // 填充 12 字节，凑齐 32 字节
};

// ==========================================
// 4. 辅助数据：评分表 (LUT)
// ==========================================
struct GPUScoringLUT {
    float scores[400]; // 20x20 flattened
};

// ==========================================
// 5. 3D 密集索引网格信息
// ==========================================
struct GridInfo {
    int origin_x, origin_y, origin_z;  // grid 原点 (格子索引，相对于 map_origin)
    int size_x, size_y, size_z;        // grid 尺寸 (格子数)
    float voxel_size;                  // 体素大小
    float map_origin_x, map_origin_y, map_origin_z;  // 地图原点
};

// ==========================================
// 6. 双向CSR稀疏可见性矩阵
// ==========================================
// Compressed Sparse Row (CSR) 格式，支持两种查询：
//   1. viewpoint -> targets: 给定viewpoint，查询可见的target索引
//   2. target -> viewpoints: 给定target，查询能观测到它的viewpoint索引
struct VisibilityCSR {
    // 方向1: viewpoint -> targets (哪些target可以被这个viewpoint看到)
    std::vector<int> viewpoint_to_targets;     // 所有可见target索引（拼接）
    std::vector<int> viewpoint_offsets;        // [M+1] 每个viewpoint的起始位置

    // 方向2: target -> viewpoints (哪些viewpoint可以看到这个target)
    std::vector<int> target_to_viewpoints;     // 所有能看到的viewpoint索引（拼接）
    std::vector<int> target_offsets;           // [N+1] 每个target的起始位置

    // GPU计算的target voxel结果 (available_mask 和 max_possible_score)
    std::vector<uint32_t> updated_available_masks;  // [num_targets] 每个target的available_mask
    std::vector<float> updated_max_scores;          // [num_targets] 每个target的max_possible_score

    // 查询接口：获取viewpoint可见的所有target索引
    std::vector<int> getVisibleTargets(int viewpoint_idx) const {
        if (viewpoint_idx < 0 || viewpoint_idx >= (int)viewpoint_offsets.size() - 1) {
            return {};
        }
        int start = viewpoint_offsets[viewpoint_idx];
        int end = viewpoint_offsets[viewpoint_idx + 1];
        return std::vector<int>(viewpoint_to_targets.begin() + start,
                                viewpoint_to_targets.begin() + end);
    }

    // 查询接口：获取能观测到target的所有viewpoint索引
    std::vector<int> getObservingViewpoints(int target_idx) const {
        if (target_idx < 0 || target_idx >= (int)target_offsets.size() - 1) {
            return {};
        }
        int start = target_offsets[target_idx];
        int end = target_offsets[target_idx + 1];
        return std::vector<int>(target_to_viewpoints.begin() + start,
                                target_to_viewpoints.begin() + end);
    }
};
