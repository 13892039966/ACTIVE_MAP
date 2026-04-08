#pragma once

#include "gpu_types.h"
#include "observation_quality_manager.h"
#include <Eigen/Eigen>
#include <vector>
#include <unordered_map>
#include <utility>

namespace observation_quality {

// ==========================================
// GPU可见性检查函数 (MxN版本，CSR输出)
// ==========================================
// 检查M个视点到N个目标的可见性，返回双向CSR稀疏矩阵
// 参数:
//   viewpoints_eigen: M个视点位置
//   targets_eigen: N个目标位置
//   voxels: 所有体素数据（用于遮挡检查）
//   voxel_size: 体素大小
//   map_min_bd: 地图最小边界坐标（用于GPU raycast的坐标原点，必须与CPU端region_origin一致）
// 返回: VisibilityCSR 双向可见性矩阵
VisibilityCSR checkVisibilityGPU(
    const std::vector<Eigen::Vector3f>& viewpoints_eigen,
    const std::vector<Eigen::Vector3f>& targets_eigen,
    const std::vector<GPUVoxel>& voxels,
    float voxel_size,
    const Eigen::Vector3f& map_min_bd,
    const std::vector<Float3>* cam_rows = nullptr, // 可选：相机world->cam旋转矩阵行（每个viewpoint 3行）
    float horizontal_fov_rad = 0.0f,
    float vertical_fov_rad = 0.0f);

// 带相机朝向和FOV裁剪的可见性检查（在 GPU 结果基础上进行 FOV 过滤）
VisibilityCSR checkVisibilityGPUWithFOV(
    const std::vector<Eigen::Vector3f>& viewpoints_eigen,
    const std::vector<Eigen::Quaternionf>& viewpoint_orientations,
    const std::vector<Eigen::Vector3f>& targets_eigen,
    const std::vector<GPUVoxel>& voxels,
    float voxel_size,
    const Eigen::Vector3f& map_min_bd,
    float horizontal_fov_rad,
    float vertical_fov_rad);

// ==========================================
// VisibilityChecker - CPU-GPU 交互层
// ==========================================
// 功能：
//   1. 从 spatial_hash_ 提取 odom 附近的占据体素
//   2. 将 VoxelCell 转换为 GPUVoxel 格式
//   3. 调用 GPU Raycast 进行批量可见性检查
//   4. 返回可见/遮挡列表
class VisibilityChecker {
public:
    struct VisibilityLine {
        Eigen::Vector3f start;     // 视点位置
        Eigen::Vector3f end;       // 体素中心
        bool visible;              // true=可见, false=遮挡
        float distance;            // 距离
        Eigen::Vector3i voxel_idx; // 体素在 spatial_hash 中的索引
    };


private:
    // 将 VoxelCell 转换为 GPUVoxel
    static GPUVoxel convertToGPUVoxel(
        const VoxelCell& cell,
        const Eigen::Vector3i& idx,
        float voxel_size);

    // 从 bitset<1000> 提取到 uint32[32]
    static void extractSubMasks(
        const std::bitset<1000>& geometry_mask,
        uint32_t* sub_masks);
};

} // namespace observation_quality
