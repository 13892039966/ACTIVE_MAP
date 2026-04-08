#include <observation_quality_manager/visibility_checker.h>
#include <observation_quality_manager/gpu_raycast.h>
#include <observation_quality_manager/gpu_types.h>
#include <observation_quality_manager/observation_quality_manager.h>
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <vector>
#include <cstdint>
#include <random>
#include <bitset>
#include <cmath>
#include <cstdlib>
#include <string>

#include <sycl/sycl.hpp>

namespace observation_quality {

// ==========================================
// GPU可见性检查实现：MxN版本（CSR输出）
// 使用 O(1) 密集索引查询优化 + 双向CSR稀疏表示
// ==========================================
VisibilityCSR checkVisibilityGPU(
    const std::vector<Eigen::Vector3f>& viewpoints_eigen,
    const std::vector<Eigen::Vector3f>& targets_eigen,
    const std::vector<GPUVoxel>& voxels,
    float voxel_size,
    const Eigen::Vector3f& map_min_bd,
    const std::vector<Float3>* cam_rows,
    float horizontal_fov_rad,
    float vertical_fov_rad)
{
    VisibilityCSR result;

    if (viewpoints_eigen.empty() || targets_eigen.empty()) {
        return result;
    }

    int num_viewpoints = viewpoints_eigen.size();
    int num_targets = targets_eigen.size();
    int num_voxels = voxels.size();

    // GPU raycast 使用与 CPU 端一致的坐标原点
    // 必须与 GlobalPlanner::region_origin_ 一致
    const Eigen::Vector3f map_origin = map_min_bd;

    try {
        ros::Time module_start, module_end;

        // 1. 获取全局 SYCL 队列 (只初始化一次)
        module_start = ros::Time::now();
        static sycl::queue q = []() {
            auto make_queue = [](const auto& selector) {
                return sycl::queue(selector, sycl::property::queue::enable_profiling());
            };

            // Allow forcing CPU execution to keep the pipeline running on machines without SYCL GPU support.
            const char* force_cpu_env = std::getenv("OQM_FORCE_CPU");
            const bool force_cpu = force_cpu_env && std::string(force_cpu_env) == "1";

            if (force_cpu) {
                sycl::queue queue = make_queue(sycl::cpu_selector_v);
                ROS_WARN("[GPU Raycast] OQM_FORCE_CPU=1, forcing SYCL CPU device: %s",
                         queue.get_device().get_info<sycl::info::device::name>().c_str());
                return queue;
            }

            try {
                sycl::queue queue = make_queue(sycl::default_selector_v);
                ROS_INFO("[GPU Raycast] Initialized SYCL queue with device: %s (profiling enabled)",
                         queue.get_device().get_info<sycl::info::device::name>().c_str());
                return queue;
            } catch (const sycl::exception& e) {
                ROS_WARN("[GPU Raycast] default_selector failed: %s. Falling back to CPU selector.",
                         e.what());
            }

            try {
                sycl::queue queue = make_queue(sycl::cpu_selector_v);
                ROS_WARN("[GPU Raycast] Fallback to SYCL CPU device: %s",
                         queue.get_device().get_info<sycl::info::device::name>().c_str());
                return queue;
            } catch (const sycl::exception& e) {
                const bool has_filter = std::getenv("ONEAPI_DEVICE_SELECTOR") ||
                                        std::getenv("SYCL_DEVICE_FILTER");
                if (has_filter) {
                    ROS_WARN("[GPU Raycast] SYCL filter env blocks CPU fallback, clearing ONEAPI_DEVICE_SELECTOR/SYCL_DEVICE_FILTER and retrying CPU.");
                    unsetenv("ONEAPI_DEVICE_SELECTOR");
                    unsetenv("SYCL_DEVICE_FILTER");
                    sycl::queue queue = make_queue(sycl::cpu_selector_v);
                    ROS_WARN("[GPU Raycast] CPU retry succeeded after clearing filters: %s",
                             queue.get_device().get_info<sycl::info::device::name>().c_str());
                    return queue;
                }
                throw;
            }
        }();

        // ==========================================
        // 2. 计算体素的边界框 (AABB) 用于创建密集索引
        // ==========================================
        module_end = ros::Time::now();
        double t_queue = (module_end - module_start).toSec() * 1000.0;
        module_start = ros::Time::now();

        int min_gx = INT_MAX, min_gy = INT_MAX, min_gz = INT_MAX;
        int max_gx = INT_MIN, max_gy = INT_MIN, max_gz = INT_MIN;

        for (const auto& v : voxels) {
            int gx = static_cast<int>((v.center.x - map_origin.x()) / voxel_size);
            int gy = static_cast<int>((v.center.y - map_origin.y()) / voxel_size);
            int gz = static_cast<int>((v.center.z - map_origin.z()) / voxel_size);

            min_gx = std::min(min_gx, gx);
            min_gy = std::min(min_gy, gy);
            min_gz = std::min(min_gz, gz);
            max_gx = std::max(max_gx, gx);
            max_gy = std::max(max_gy, gy);
            max_gz = std::max(max_gz, gz);
        }

        // 添加 1 格子的边界余量
        min_gx -= 1; min_gy -= 1; min_gz -= 1;
        max_gx += 1; max_gy += 1; max_gz += 1;

        // 创建 GridInfo
        GridInfo grid_info;
        grid_info.origin_x = min_gx;
        grid_info.origin_y = min_gy;
        grid_info.origin_z = min_gz;
        grid_info.size_x = max_gx - min_gx + 1;
        grid_info.size_y = max_gy - min_gy + 1;
        grid_info.size_z = max_gz - min_gz + 1;
        grid_info.voxel_size = voxel_size;
        grid_info.map_origin_x = map_origin.x();
        grid_info.map_origin_y = map_origin.y();
        grid_info.map_origin_z = map_origin.z();

        int grid_total_size = grid_info.size_x * grid_info.size_y * grid_info.size_z;

        ROS_INFO("[GPU Raycast MxN] Grid size: %d x %d x %d = %d cells",
                 grid_info.size_x, grid_info.size_y, grid_info.size_z, grid_total_size);

        // ==========================================
        // 3. 将 viewpoints 和 targets 转换为 Float3
        // ==========================================
        module_end = ros::Time::now();
        double t_aabb = (module_end - module_start).toSec() * 1000.0;
        module_start = ros::Time::now();

        std::vector<Float3> viewpoints(num_viewpoints);
        for (int i = 0; i < num_viewpoints; ++i) {
            viewpoints[i] = {
                viewpoints_eigen[i].x(),
                viewpoints_eigen[i].y(),
                viewpoints_eigen[i].z(),
                0.0f
            };
        }

        // 创建 target_indices 数组: 0, 1, 2, ..., num_targets-1
        std::vector<int> target_indices(num_targets);
        for (int i = 0; i < num_targets; ++i) {
            target_indices[i] = i;
        }

        // ==========================================
        // 4. 分配设备内存（使用shared memory，自动适配设备）
        // ==========================================
        module_end = ros::Time::now();
        double t_convert = (module_end - module_start).toSec() * 1000.0;
        module_start = ros::Time::now();

        GPUVoxel* d_voxels = sycl::malloc_shared<GPUVoxel>(num_voxels, q);
        Float3* d_viewpoints = sycl::malloc_shared<Float3>(num_viewpoints, q);
        int* d_target_indices = sycl::malloc_shared<int>(num_targets, q);
        Float3* d_cam_rows = nullptr;

        // 分配CSR输出数组 (预分配最大可能大小)
        int max_entries = num_viewpoints * num_targets;  // 最坏情况：全部可见
        int* d_viewpoint_to_targets = sycl::malloc_shared<int>(max_entries, q);
        int* d_viewpoint_offsets = sycl::malloc_shared<int>(num_viewpoints + 1, q);
        int* d_target_to_viewpoints = sycl::malloc_shared<int>(max_entries, q);
        int* d_target_offsets = sycl::malloc_shared<int>(num_targets + 1, q);

        // 辅助数组
        int* d_viewpoint_counts = sycl::malloc_shared<int>(num_viewpoints, q);
        int* d_target_counts = sycl::malloc_shared<int>(num_targets, q);
        int* d_visibility_temp = sycl::malloc_shared<int>(max_entries, q);

        int* d_grid = sycl::malloc_shared<int>(grid_total_size, q);

        // 分配并初始化 scoring_table (20x20 = 400 floats)
        float* d_scoring_table = sycl::malloc_shared<float>(400, q);
        for (int i = 0; i < 20; ++i) {
            for (int j = 0; j < 20; ++j) {
                d_scoring_table[i * 20 + j] = SphericalBinning::scoring_table[i][j];
            }
        }

        // 5. 拷贝数据到设备（使用std::memcpy直接写入shared memory）
        module_end = ros::Time::now();
        double t_malloc = (module_end - module_start).toSec() * 1000.0;
        module_start = ros::Time::now();

        std::memcpy(d_voxels, voxels.data(), num_voxels * sizeof(GPUVoxel));
        std::memcpy(d_viewpoints, viewpoints.data(), num_viewpoints * sizeof(Float3));
        std::memcpy(d_target_indices, target_indices.data(), num_targets * sizeof(int));

        // 可选：拷贝 FOV 所需的相机行向量
        bool use_horizontal_fov = false;
        bool use_vertical_fov = false;
        float half_horizontal_fov = 0.0f;
        float half_vertical_fov = 0.0f;

        if (cam_rows && cam_rows->size() == static_cast<size_t>(num_viewpoints * 3)) {
            use_horizontal_fov = horizontal_fov_rad > 1e-3f && horizontal_fov_rad < static_cast<float>(M_PI) * 1.99f;
            use_vertical_fov = vertical_fov_rad > 1e-3f && vertical_fov_rad < static_cast<float>(M_PI) * 0.99f;
            if (use_horizontal_fov || use_vertical_fov) {
                half_horizontal_fov = 0.5f * horizontal_fov_rad;
                half_vertical_fov = 0.5f * vertical_fov_rad;
                d_cam_rows = sycl::malloc_shared<Float3>(num_viewpoints * 3, q);
                std::memcpy(d_cam_rows, cam_rows->data(), cam_rows->size() * sizeof(Float3));
            }
        }

        // 注意：d_scoring_table 已在分配时直接初始化

        // ==========================================
        // 6. 启动 MxN CSR kernel (索引构建 + raycast + CSR构造, 异步执行)
        // ==========================================
        module_end = ros::Time::now();
        double t_h2d = (module_end - module_start).toSec() * 1000.0;
        module_start = ros::Time::now();

        auto kernel_event = gpu_raycast::launchMxNVisibilityCheckKernelAsync(
            q,
            d_viewpoints,
            num_viewpoints,
            d_target_indices,
            num_targets,
            d_voxels,
            num_voxels,
            d_viewpoint_to_targets,
            d_viewpoint_offsets,
            d_target_to_viewpoints,
            d_target_offsets,
            d_viewpoint_counts,
            d_target_counts,
            d_visibility_temp,
            d_grid,
            d_scoring_table,
            grid_info,
            d_cam_rows,
            half_horizontal_fov,
            half_vertical_fov,
            use_horizontal_fov,
            use_vertical_fov
        );

        // 等待kernel完成
        kernel_event.wait();

        // 7. 从shared memory读取CSR结果
        module_end = ros::Time::now();
        double t_kernel = (module_end - module_start).toSec() * 1000.0;
        module_start = ros::Time::now();

        // 读取offsets以确定实际大小
        result.viewpoint_offsets.resize(num_viewpoints + 1);
        result.target_offsets.resize(num_targets + 1);
        std::memcpy(result.viewpoint_offsets.data(), d_viewpoint_offsets,
                    (num_viewpoints + 1) * sizeof(int));
        std::memcpy(result.target_offsets.data(), d_target_offsets,
                    (num_targets + 1) * sizeof(int));

        int total_viewpoint_entries = result.viewpoint_offsets[num_viewpoints];
        int total_target_entries = result.target_offsets[num_targets];

        // 读取CSR数据（只读取实际使用的部分）
        result.viewpoint_to_targets.resize(total_viewpoint_entries);
        result.target_to_viewpoints.resize(total_target_entries);
        std::memcpy(result.viewpoint_to_targets.data(), d_viewpoint_to_targets,
                    total_viewpoint_entries * sizeof(int));
        std::memcpy(result.target_to_viewpoints.data(), d_target_to_viewpoints,
                    total_target_entries * sizeof(int));

        // 读取GPU计算的target voxel结果 (available_mask 和 max_possible_score)
        result.updated_available_masks.resize(num_targets);
        result.updated_max_scores.resize(num_targets);
        for (int i = 0; i < num_targets; ++i) {
            int voxel_idx = d_target_indices[i];
            result.updated_available_masks[i] = d_voxels[voxel_idx].available_mask;
            result.updated_max_scores[i] = d_voxels[voxel_idx].max_possible_score;
        }

        // ==========================================
        // 8. 释放设备内存
        // ==========================================
        module_end = ros::Time::now();
        double t_d2h = (module_end - module_start).toSec() * 1000.0;
        module_start = ros::Time::now();

        sycl::free(d_voxels, q);
        sycl::free(d_viewpoints, q);
        sycl::free(d_target_indices, q);
        sycl::free(d_viewpoint_to_targets, q);
        sycl::free(d_viewpoint_offsets, q);
        sycl::free(d_target_to_viewpoints, q);
        sycl::free(d_target_offsets, q);
        sycl::free(d_viewpoint_counts, q);
        sycl::free(d_target_counts, q);
        sycl::free(d_visibility_temp, q);
        sycl::free(d_grid, q);
        sycl::free(d_scoring_table, q);
        if (d_cam_rows) {
            sycl::free(d_cam_rows, q);
        }

        module_end = ros::Time::now();
        double t_free = (module_end - module_start).toSec() * 1000.0;

        // 打印所有模块时间（绿色输出）
        printf("\033[32m[GPU CSR Timing] Queue:%.2fms AABB:%.2fms Convert:%.2fms Malloc:%.2fms H2D:%.2fms Kernel:%.2fms D2H:%.2fms Free:%.2fms\033[0m\n",
               t_queue, t_aabb, t_convert, t_malloc, t_h2d, t_kernel, t_d2h, t_free);

        ROS_INFO("[GPU CSR] %d viewpoints x %d targets -> %d visible entries (%.1f%% sparse)",
                 num_viewpoints, num_targets, total_viewpoint_entries,
                 100.0 * (1.0 - (float)total_viewpoint_entries / (num_viewpoints * num_targets)));

        return result;

    } catch (const sycl::exception& e) {
        ROS_ERROR("[GPU CSR] SYCL exception: %s", e.what());
        return result;  // 返回空的CSR
    }
}

// ==========================================
// 带相机朝向和FOV裁剪的 GPU 可见性检查
// 先调用原始 GPU 可见性，再在CPU侧按照FOV过滤
// ==========================================
VisibilityCSR checkVisibilityGPUWithFOV(
    const std::vector<Eigen::Vector3f>& viewpoints_eigen,
    const std::vector<Eigen::Quaternionf>& viewpoint_orientations,
    const std::vector<Eigen::Vector3f>& targets_eigen,
    const std::vector<GPUVoxel>& voxels,
    float voxel_size,
    const Eigen::Vector3f& map_min_bd,
    float horizontal_fov_rad,
    float vertical_fov_rad)
{
    int num_viewpoints = viewpoints_eigen.size();
    int num_targets = targets_eigen.size();

    bool use_horizontal = horizontal_fov_rad > 1e-3f && horizontal_fov_rad < static_cast<float>(M_PI) * 1.99f;
    bool use_vertical = vertical_fov_rad > 1e-3f && vertical_fov_rad < static_cast<float>(M_PI) * 0.99f;

    std::vector<Float3> cam_rows;
    if ((use_horizontal || use_vertical) && num_viewpoints > 0) {
        cam_rows.resize(num_viewpoints * 3);
        for (int vp_idx = 0; vp_idx < num_viewpoints; ++vp_idx) {
            Eigen::Quaternionf q = Eigen::Quaternionf::Identity();
            if (vp_idx < static_cast<int>(viewpoint_orientations.size())) {
                q = viewpoint_orientations[vp_idx];
                q.normalize();
            }
            Eigen::Matrix3f Rcw = q.conjugate().toRotationMatrix(); // world -> camera
            int base = vp_idx * 3;
            cam_rows[base + 0] = {Rcw(0, 0), Rcw(0, 1), Rcw(0, 2), 0.0f};
            cam_rows[base + 1] = {Rcw(1, 0), Rcw(1, 1), Rcw(1, 2), 0.0f};
            cam_rows[base + 2] = {Rcw(2, 0), Rcw(2, 1), Rcw(2, 2), 0.0f};
        }
    }

    VisibilityCSR raw = checkVisibilityGPU(
        viewpoints_eigen,
        targets_eigen,
        voxels,
        voxel_size,
        map_min_bd,
        cam_rows.empty() ? nullptr : &cam_rows,
        use_horizontal ? horizontal_fov_rad : 0.0f,
        use_vertical ? vertical_fov_rad : 0.0f);

    return raw;
}

// ==========================================
// 辅助函数：计算体素索引到中心位置
// ==========================================
static Eigen::Vector3f idx2pos(const Eigen::Vector3i& idx, float voxel_size, const Eigen::Vector3f& map_min_bd) {
    return map_min_bd + idx.cast<float>() * voxel_size;
}

// ==========================================
// 将 VoxelCell 转换为 GPUVoxel
// ==========================================
GPUVoxel VisibilityChecker::convertToGPUVoxel(
    const VoxelCell& cell,
    const Eigen::Vector3i& idx,
    float voxel_size)
{
    GPUVoxel gpu_voxel;

    // 1. 位置
    gpu_voxel.center.x = cell.voxel_center.x();
    gpu_voxel.center.y = cell.voxel_center.y();
    gpu_voxel.center.z = cell.voxel_center.z();
    gpu_voxel.center.w = 0.0f;

    // 2. 属性
    gpu_voxel.geo_complexity = cell.geometric_complexity;
    gpu_voxel.tex_complexity = cell.texture_complexity;
    gpu_voxel.normal_bin_idx = cell.normal_bin_idx;
    gpu_voxel.current_score = cell.observation_score;

    // 3. 状态
    gpu_voxel.obs_mask = static_cast<uint32_t>(cell.observation_direction_mask.to_ulong());
    gpu_voxel.available_mask = 0;  // 初始化为全部不可见
    gpu_voxel.well_observed = cell.well_observed ? 1 : 0;

    // 4. 子体素掩码 (从 bitset<1000> 提取到 uint32[32])
    extractSubMasks(cell.geometry_occupancy_mask, gpu_voxel.sub_masks);
    for (int i = 0; i < SUB_MASK_ARRAY_SIZE; ++i) {
        gpu_voxel.unknown_masks[i] = 0;
    }

    // 5. 输出结果
    gpu_voxel.max_possible_score = 0.0f;
    gpu_voxel.padding = 0.0f;

    return gpu_voxel;
}

// ==========================================
// 从 bitset<1000> 提取到 uint32[32]
// ==========================================
void VisibilityChecker::extractSubMasks(
    const std::bitset<1000>& geometry_mask,
    uint32_t* sub_masks)
{
    // 初始化为 0
    for (int i = 0; i < SUB_MASK_ARRAY_SIZE; ++i) {
        sub_masks[i] = 0;
    }

    // 逐位转换
    for (int bit_idx = 0; bit_idx < 1000; ++bit_idx) {
        if (geometry_mask[bit_idx]) {
            int array_idx = bit_idx / 32;
            int bit_offset = bit_idx % 32;
            sub_masks[array_idx] |= (1u << bit_offset);
        }
    }
}


} // namespace observation_quality
