#pragma once

#include "gpu_types.h"
#include <cmath>

// SYCL 头文件（AdaptiveCpp 和 Intel oneAPI 都支持）
#include <sycl/sycl.hpp>

namespace gpu_raycast {

// ==========================================
// 辅助数学函数 (设备端/主机端通用)
// ==========================================
inline float dot(Float3 a, Float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Float3 sub(Float3 a, Float3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z, 0}; }
inline Float3 add(Float3 a, Float3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z, 0}; }
inline Float3 mul(Float3 a, float b) { return {a.x*b, a.y*b, a.z*b, 0}; }
inline float len_sq(Float3 a) { return dot(a, a); }

// ==========================================
// DDA 算法辅助函数 (Amanatides-Woo)
// ==========================================
// 取模函数（确保结果为正）
inline float fmod_positive(float s, float d) {
    float m = s - d * sycl::floor(s / d);
    return m < 0.0f ? m + d : m;
}

// 计算到下一个整数边界的 t 值
// 用于 DDA 算法中确定光线何时穿过体素边界
inline float intbound(float s, float ds) {
    if (sycl::fabs(ds) < 1e-10f) return 1e30f; // 避免除零，返回大值表示不会穿过
    // Avoid recursion: SYCL device code forbids recursive calls.
    float s_local = s;
    float ds_local = ds;
    if (ds_local < 0.0f) {
        s_local = -s_local;
        ds_local = -ds_local;
    }
    float frac = fmod_positive(s_local, 1.0f);
    return (1.0f - frac) / ds_local;
}

// 符号函数
inline int signum(float x) {
    return (x > 1e-10f) ? 1 : ((x < -1e-10f) ? -1 : 0);
}

// ==========================================
// 方向分箱：GPU 版本的 SphericalBinning::get_bin_index
// ==========================================
// 20个正二十面体面法向量（与 CPU 端 SphericalBinning::face_normals 一致）
// 注意：这些值由 PHI = 1.618033988749895f 计算得出并归一化
static constexpr Float3 FACE_NORMALS[20] = {
    {-0.5773503f, 0.5773503f, 0.5773503f, 0}, {0.0000000f, 0.9341723f, 0.3568221f, 0},
    {0.0000000f, 0.9341723f, -0.3568221f, 0}, {-0.5773503f, 0.5773503f, -0.5773503f, 0},
    {-0.9341723f, 0.3568221f, 0.0000000f, 0}, {0.5773503f, 0.5773503f, 0.5773503f, 0},
    {-0.3568221f, 0.0000000f, 0.9341723f, 0}, {-0.9341723f, -0.3568221f, 0.0000000f, 0},
    {-0.3568221f, 0.0000000f, -0.9341723f, 0}, {0.5773503f, 0.5773503f, -0.5773503f, 0},
    {0.5773503f, -0.5773503f, 0.5773503f, 0}, {0.0000000f, -0.9341723f, 0.3568221f, 0},
    {0.0000000f, -0.9341723f, -0.3568221f, 0}, {0.5773503f, -0.5773503f, -0.5773503f, 0},
    {0.9341723f, -0.3568221f, 0.0000000f, 0}, {0.3568221f, 0.0000000f, 0.9341723f, 0},
    {-0.5773503f, -0.5773503f, 0.5773503f, 0}, {-0.5773503f, -0.5773503f, -0.5773503f, 0},
    {0.3568221f, 0.0000000f, -0.9341723f, 0}, {0.9341723f, 0.3568221f, 0.0000000f, 0}
};

// 将观测方向映射到 20 个 bin 之一
// 返回: 0-19 的 bin 索引，-1 表示无效
inline int getBinIndex(Float3 view_dir) {
    // 归一化
    float len_sq = view_dir.x*view_dir.x + view_dir.y*view_dir.y + view_dir.z*view_dir.z;
    if (len_sq < 1e-6f) return -1; // 无效向量

    float inv_len = 1.0f / sycl::sqrt(len_sq);
    Float3 normalized = {view_dir.x*inv_len, view_dir.y*inv_len, view_dir.z*inv_len, 0};

    // 找点积最大的方向
    int best_idx = 0;
    float max_dot = -2.0f;
    for (int i = 0; i < 20; ++i) {
        float dot_product = normalized.x*FACE_NORMALS[i].x +
                           normalized.y*FACE_NORMALS[i].y +
                           normalized.z*FACE_NORMALS[i].z;
        if (dot_product > max_dot) {
            max_dot = dot_product;
            best_idx = i;
        }
    }
    return best_idx;
}

// ==========================================
// GPU 辅助函数：查询体素 (O(1) 密集索引版本)
// ==========================================
// 通过 3D 密集索引数组快速查询
// 参数:
//   gx, gy, gz: 格子索引 (相对于 map_origin)
//   d_grid: 密集索引数组 (存储 voxel 索引，-1 表示空)
//   grid_info: 网格信息
// 返回: voxel 在 d_voxels 数组中的索引，-1 表示空
inline int getVoxelIndex(
    int gx, int gy, int gz,
    const int* d_grid,
    const GridInfo& grid_info)
{
    // 转换为 grid 局部坐标
    int lx = gx - grid_info.origin_x;
    int ly = gy - grid_info.origin_y;
    int lz = gz - grid_info.origin_z;

    // 边界检查
    if (lx < 0 || lx >= grid_info.size_x ||
        ly < 0 || ly >= grid_info.size_y ||
        lz < 0 || lz >= grid_info.size_z) {
        return -1;
    }

    // O(1) 查询
    int idx = lx * grid_info.size_y * grid_info.size_z +
              ly * grid_info.size_z + lz;
    return d_grid[idx];
}

// ==========================================
// 子体素遮挡检查 (细分辨率 DDA, 0.02m)
// ==========================================
// 在粗体素内部执行完整的 DDA 遍历，检查 sub_masks
// 参数:
//   ray_start, ray_end: 光线起点和终点 (世界坐标)
//   v: 当前粗体素
//   voxel_size: 粗体素大小 (0.2m)
//   target_sx, target_sy, target_sz: 目标子体素坐标 (-1 表示不在此粗体素内)
// 返回: true = 被遮挡, false = 可通过
inline bool checkSubVoxelOcclusion(
    Float3 ray_start,
    Float3 ray_end,
    const GPUVoxel* v,
    float voxel_size,
    int target_sx, int target_sy, int target_sz)
{
    float sub_size = voxel_size / 10.0f; // 0.02m

    // 计算粗体素的原点 (左下角)
    Float3 voxel_origin = sub(v->center, {voxel_size*0.5f, voxel_size*0.5f, voxel_size*0.5f, 0});

    // 转换到子体素坐标系 (0~10 范围)
    Float3 start_s = {
        (ray_start.x - voxel_origin.x) / sub_size,
        (ray_start.y - voxel_origin.y) / sub_size,
        (ray_start.z - voxel_origin.z) / sub_size, 0
    };
    Float3 end_s = {
        (ray_end.x - voxel_origin.x) / sub_size,
        (ray_end.y - voxel_origin.y) / sub_size,
        (ray_end.z - voxel_origin.z) / sub_size, 0
    };

    // 光线方向 (子体素坐标系)
    float dx = end_s.x - start_s.x;
    float dy = end_s.y - start_s.y;
    float dz = end_s.z - start_s.z;

    // 裁剪光线到 [0, 10) 范围 (AABB 求交)
    float t_enter = 0.0f, t_exit = 1.0f;

    // X 轴裁剪
    if (sycl::fabs(dx) > 1e-10f) {
        float t0 = (0.0f - start_s.x) / dx;
        float t1 = (10.0f - start_s.x) / dx;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        t_enter = sycl::fmax(t_enter, t0);
        t_exit = sycl::fmin(t_exit, t1);
    } else if (start_s.x < 0.0f || start_s.x >= 10.0f) {
        return false; // 光线平行于 X 轴且在边界外
    }

    // Y 轴裁剪
    if (sycl::fabs(dy) > 1e-10f) {
        float t0 = (0.0f - start_s.y) / dy;
        float t1 = (10.0f - start_s.y) / dy;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        t_enter = sycl::fmax(t_enter, t0);
        t_exit = sycl::fmin(t_exit, t1);
    } else if (start_s.y < 0.0f || start_s.y >= 10.0f) {
        return false;
    }

    // Z 轴裁剪
    if (sycl::fabs(dz) > 1e-10f) {
        float t0 = (0.0f - start_s.z) / dz;
        float t1 = (10.0f - start_s.z) / dz;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        t_enter = sycl::fmax(t_enter, t0);
        t_exit = sycl::fmin(t_exit, t1);
    } else if (start_s.z < 0.0f || start_s.z >= 10.0f) {
        return false;
    }

    if (t_enter >= t_exit) return false; // 光线不经过此体素

    // 计算裁剪后的起点 (在子体素坐标系中)
    Float3 clipped_start = {
        start_s.x + t_enter * dx,
        start_s.y + t_enter * dy,
        start_s.z + t_enter * dz, 0
    };
    Float3 clipped_end = {
        start_s.x + t_exit * dx,
        start_s.y + t_exit * dy,
        start_s.z + t_exit * dz, 0
    };

    // 细分辨率 DDA 初始化
    int sx = sycl::clamp(static_cast<int>(sycl::floor(clipped_start.x)), 0, 9);
    int sy = sycl::clamp(static_cast<int>(sycl::floor(clipped_start.y)), 0, 9);
    int sz = sycl::clamp(static_cast<int>(sycl::floor(clipped_start.z)), 0, 9);
    int endSx = sycl::clamp(static_cast<int>(sycl::floor(clipped_end.x)), 0, 9);
    int endSy = sycl::clamp(static_cast<int>(sycl::floor(clipped_end.y)), 0, 9);
    int endSz = sycl::clamp(static_cast<int>(sycl::floor(clipped_end.z)), 0, 9);

    // 处理正好落在边界 10.0 的情况
    if (clipped_end.x >= 10.0f) endSx = 9;
    if (clipped_end.y >= 10.0f) endSy = 9;
    if (clipped_end.z >= 10.0f) endSz = 9;

    int stepX = signum(dx);
    int stepY = signum(dy);
    int stepZ = signum(dz);

    // 计算 tMax 和 tDelta (基于裁剪后的起点和原始方向)
    float tMaxX = intbound(clipped_start.x, dx);
    float tMaxY = intbound(clipped_start.y, dy);
    float tMaxZ = intbound(clipped_start.z, dz);

    float tDeltaX = (sycl::fabs(dx) > 1e-10f) ? sycl::fabs(1.0f / dx) : 1e30f;
    float tDeltaY = (sycl::fabs(dy) > 1e-10f) ? sycl::fabs(1.0f / dy) : 1e30f;
    float tDeltaZ = (sycl::fabs(dz) > 1e-10f) ? sycl::fabs(1.0f / dz) : 1e30f;

    // 如果方向为零，不能移动
    if (stepX == 0 && stepY == 0 && stepZ == 0) {
        // 检查当前位置
        if (sx == target_sx && sy == target_sy && sz == target_sz) {
            return false; // 目标子体素，视为可见
        }
        int bit_idx = sx * 100 + sy * 10 + sz;
        int array_idx = bit_idx / 32;
        int bit_offset = bit_idx % 32;
        bool occupied = (v->sub_masks[array_idx] >> bit_offset) & 1;
        bool unknown = (v->unknown_masks[array_idx] >> bit_offset) & 1;
        // return occupied || unknown;
        return occupied;
    }

    int max_iter = 30; // 最多遍历 30 个子体素 (对角线约 17 个)
    while (max_iter-- > 0) {
        // 检查当前子体素是否为目标子体素
        bool is_target = (sx == target_sx && sy == target_sy && sz == target_sz);

        if (!is_target) {
            // 检查 sub_mask
            int bit_idx = sx * 100 + sy * 10 + sz;
            int array_idx = bit_idx / 32;
            int bit_offset = bit_idx % 32;

            bool occupied = (v->sub_masks[array_idx] >> bit_offset) & 1;
            bool unknown = (v->unknown_masks[array_idx] >> bit_offset) & 1;
            if (occupied || unknown) {
                return true; // 被遮挡!
            }
        }

        // 到达终点
        if (sx == endSx && sy == endSy && sz == endSz) break;

        // DDA 步进
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                sx += stepX;
                tMaxX += tDeltaX;
            } else {
                sz += stepZ;
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                sy += stepY;
                tMaxY += tDeltaY;
            } else {
                sz += stepZ;
                tMaxZ += tDeltaZ;
            }
        }

        // 边界检查 (退出粗体素范围)
        if (sx < 0 || sx > 9 || sy < 0 || sy > 9 || sz < 0 || sz > 9) break;
    }

    return false; // 未被遮挡
}

// ==========================================
// 核心 Raycast 函数 (两层 DDA 算法)
// ==========================================
// 第一层: 粗分辨率 DDA (0.2m) - 遍历光线经过的粗体素
// 第二层: 细分辨率 DDA (0.02m) - 在非空粗体素内检查 sub_masks
// 返回值: 1 = 可见 (无遮挡), 0 = 被遮挡
inline int RayCast(
    Float3 start,
    Float3 end,
    const GPUVoxel* voxels,
    const int* d_grid,
    const GridInfo& grid_info)
{
    Float3 dir = sub(end, start);
    float dist_sq = len_sq(dir);

    // 1. 距离极小或极大直接剔除
    if (dist_sq < 0.01f) return 1; // 自己看自己
    // if (dist_sq > 9.0f) return 0; // 太远不可见 (>10m)
    // if (dist_sq > 9.0f) return 0; // 太远不可见 (>10m)

    float voxel_size = grid_info.voxel_size;
    Float3 map_origin = {grid_info.map_origin_x, grid_info.map_origin_y, grid_info.map_origin_z, 0};
    // printf("map_origin: (%.3f, %.3f, %.3f)\n", 
    //         map_origin.x, map_origin.y, map_origin.z);

    // 计算目标点所在的粗体素和子体素坐标
    int target_gx = static_cast<int>(sycl::floor((end.x - map_origin.x) / voxel_size));
    int target_gy = static_cast<int>(sycl::floor((end.y - map_origin.y) / voxel_size));
    int target_gz = static_cast<int>(sycl::floor((end.z - map_origin.z) / voxel_size));

    // 计算目标点在粗体素内的子体素坐标
    float sub_size = voxel_size / 10.0f;
    Float3 target_voxel_origin = {
        map_origin.x + target_gx * voxel_size,
        map_origin.y + target_gy * voxel_size,
        map_origin.z + target_gz * voxel_size, 0
    };
    int target_sx = static_cast<int>(sycl::floor((end.x - target_voxel_origin.x) / sub_size));
    int target_sy = static_cast<int>(sycl::floor((end.y - target_voxel_origin.y) / sub_size));
    int target_sz = static_cast<int>(sycl::floor((end.z - target_voxel_origin.z) / sub_size));
    target_sx = sycl::clamp(target_sx, 0, 9);
    target_sy = sycl::clamp(target_sy, 0, 9);
    target_sz = sycl::clamp(target_sz, 0, 9);

    // === 粗分辨率 DDA 初始化 (转换到体素坐标系) ===
    Float3 start_v = {
        (start.x - map_origin.x) / voxel_size,
        (start.y - map_origin.y) / voxel_size,
        (start.z - map_origin.z) / voxel_size, 0
    };
    Float3 end_v = {
        (end.x - map_origin.x) / voxel_size,
        (end.y - map_origin.y) / voxel_size,
        (end.z - map_origin.z) / voxel_size, 0
    };

    int x = static_cast<int>(sycl::floor(start_v.x));
    int y = static_cast<int>(sycl::floor(start_v.y));
    int z = static_cast<int>(sycl::floor(start_v.z));
    // 记录起点所在的粗体素，用于后续跳过自身子体素检查
    const int start_gx = x;
    const int start_gy = y;
    const int start_gz = z;
    int endX = static_cast<int>(sycl::floor(end_v.x));
    int endY = static_cast<int>(sycl::floor(end_v.y));
    int endZ = static_cast<int>(sycl::floor(end_v.z));

    int dx = endX - x;
    int dy = endY - y;
    int dz = endZ - z;

    int stepX = (dx == 0) ? 0 : (dx > 0 ? 1 : -1);
    int stepY = (dy == 0) ? 0 : (dy > 0 ? 1 : -1);
    int stepZ = (dz == 0) ? 0 : (dz > 0 ? 1 : -1);

    float tMaxX = (dx != 0) ? intbound(start_v.x, static_cast<float>(dx)) : 1e30f;
    float tMaxY = (dy != 0) ? intbound(start_v.y, static_cast<float>(dy)) : 1e30f;
    float tMaxZ = (dz != 0) ? intbound(start_v.z, static_cast<float>(dz)) : 1e30f;

    float tDeltaX = (dx != 0) ? static_cast<float>(stepX) / static_cast<float>(dx) : 1e30f;
    float tDeltaY = (dy != 0) ? static_cast<float>(stepY) / static_cast<float>(dy) : 1e30f;
    float tDeltaZ = (dz != 0) ? static_cast<float>(stepZ) / static_cast<float>(dz) : 1e30f;

    // 如果起点和终点在同一个体素
    if (stepX == 0 && stepY == 0 && stepZ == 0) {
        return 1; // 同一个体素内，视为可见
    }

    int max_iter = 1000; // 防止无限循环

    // === 粗分辨率 DDA 遍历 ===
    while (max_iter-- > 0) {
        // 查询当前粗体素
        int voxel_idx = getVoxelIndex(x, y, z, d_grid, grid_info);

        if (voxel_idx >= 0) {
            // 非空粗体素：除起点/终点体素外，进行遮挡检查
            bool is_start_coarse = (x == start_gx && y == start_gy && z == start_gz);
            bool is_target_coarse = (x == target_gx && y == target_gy && z == target_gz);

            if (!is_start_coarse && !is_target_coarse) {
                // Unknown 粗体素应该穿透（不视为障碍），允许探索未知区域
                // 只检查 occupied 体素（sub_masks）
                const GPUVoxel* v = &voxels[voxel_idx];

                // 计算当前体素中心到目标点的距离
                Float3 voxel_center = {
                    map_origin.x + (x + 0.5f) * voxel_size,
                    map_origin.y + (y + 0.5f) * voxel_size,
                    map_origin.z + (z + 0.5f) * voxel_size, 0
                };
                float dist_to_target_sq = len_sq(sub(voxel_center, end));

                constexpr float FINE_CHECK_DISTANCE_SQ = 4.0f;  // 2m^2
                // constexpr float FINE_CHECK_DISTANCE_SQ = 0.5f;  // 2m^2
                // constexpr float FINE_CHECK_DISTANCE_SQ = 0.0f;  // 2m^2

                // if (dist_to_target_sq < FINE_CHECK_DISTANCE_SQ) {
                if (dist_to_target_sq < FINE_CHECK_DISTANCE_SQ) {
                    // 距离 target 两米内：使用细分辨率检查
                    // 传递正确的目标子体素坐标，让子体素DDA能跳过目标点所在子体素的检查
                    if (checkSubVoxelOcclusion(start, end, v, voxel_size, target_sx, target_sy, target_sz)) {
                        return 0;  // 被遮挡
                    }
                } else {
                    // 远距离：粗体素存在即视为遮挡
                    return 0;
                }
            }
        }

        // 到达目标粗体素
        if (x == endX && y == endY && z == endZ) break;

        // 粗分辨率 DDA 步进
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                x += stepX;
                tMaxX += tDeltaX;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                y += stepY;
                tMaxY += tDeltaY;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        }
    }

    return 1; // 未被遮挡
}

// ==========================================
// SYCL Kernel 启动函数（在 gpu_raycast.cpp 中实现）
// ==========================================
// SYCL Kernel 异步版本：MxN 可见性检查（CSR输出）
// ==========================================
// 返回双向CSR格式，支持两种查询：
//   1. viewpoint -> targets: 给定viewpoint，查询可见的target索引
//   2. target -> viewpoints: 给定target，查询能观测到它的viewpoint索引
// 参数说明:
//   d_viewpoints: 视点数组 (M个)
//   d_target_indices: 目标体素索引数组 (N个) - 指向d_voxels数组的索引
//   d_voxels: 体素数组 - 用于构建遮挡网格，target必须是此数组的子集
//   d_viewpoint_to_targets: CSR输出 - viewpoint可见的target索引
//   d_viewpoint_offsets: CSR输出 - [M+1] 每个viewpoint的起始位置
//   d_target_to_viewpoints: CSR输出 - target被哪些viewpoint观测
//   d_target_offsets: CSR输出 - [N+1] 每个target的起始位置
//   d_viewpoint_counts: 辅助数组 - [M] 每个viewpoint的可见target数量
//   d_target_counts: 辅助数组 - [N] 每个target被观测的viewpoint数量
//   d_visibility_temp: 辅助数组 - [M*N] 临时存储visibility结果 (int: 0或1)
//   d_scoring_table: 评分表 - [400] flatten的20x20评分表，用于计算max_possible_score
//   depends_on: 输入事件列表（如H2D传输事件），kernel将等待这些事件完成
// 返回值:
//   sycl::event: kernel执行事件，可用于后续依赖或等待
sycl::event launchMxNVisibilityCheckKernelAsync(
    sycl::queue& q,
    const Float3* d_viewpoints,
    int num_viewpoints,
    const int* d_target_indices,
    int num_targets,
    GPUVoxel* d_voxels,
    int num_voxels,
    int* d_viewpoint_to_targets,
    int* d_viewpoint_offsets,
    int* d_target_to_viewpoints,
    int* d_target_offsets,
    int* d_viewpoint_counts,
    int* d_target_counts,
    int* d_visibility_temp,
    int* d_grid,
    const float* d_scoring_table,
    const GridInfo& grid_info,
    const Float3* d_cam_rows = nullptr,   // 每个viewpoint的 world->cam 旋转矩阵行（3*num_viewpoints）
    float half_horizontal_fov = 0.0f,     // 水平FOV/2
    float half_vertical_fov = 0.0f,       // 垂直FOV/2
    bool use_horizontal_fov = false,      // 是否启用水平FOV裁剪
    bool use_vertical_fov = false,        // 是否启用垂直FOV裁剪
    const std::vector<sycl::event>& depends_on = {});

} // namespace gpu_raycast
