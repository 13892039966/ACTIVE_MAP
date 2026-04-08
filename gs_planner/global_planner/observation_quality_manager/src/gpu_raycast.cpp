#include <observation_quality_manager/gpu_raycast.h>
#include <chrono>
#include <iostream>

namespace gpu_raycast {

// ==========================================
// SYCL Kernel: MxN 批量可见性检查（CSR输出，异步版本）
// ==========================================
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
    const Float3* d_cam_rows,
    float half_horizontal_fov,
    float half_vertical_fov,
    bool use_horizontal_fov,
    bool use_vertical_fov,
    const std::vector<sycl::event>& depends_on)
{
    int grid_total_size = grid_info.size_x * grid_info.size_y * grid_info.size_z;

    // ==========================================
    // Stage 1: 初始化grid和计数数组
    // ==========================================
    sycl::event event1 = q.submit([&](sycl::handler& h) {
        h.depends_on(depends_on);
        h.parallel_for(sycl::range<1>(grid_total_size + num_viewpoints + num_targets), [=](sycl::id<1> idx) {
            int i = idx[0];
            if (i < grid_total_size) {
                d_grid[i] = -1;
            } else if (i < grid_total_size + num_viewpoints) {
                d_viewpoint_counts[i - grid_total_size] = 0;
            } else {
                d_target_counts[i - grid_total_size - num_viewpoints] = 0;
            }
        });
    });

    // ==========================================
    // Stage 2: 构建voxel密集索引
    // ==========================================
    sycl::event event2 = q.submit([&](sycl::handler& h) {
        h.depends_on(event1);
        h.parallel_for(sycl::range<1>(num_voxels), [=](sycl::id<1> idx) {
            int i = idx[0];
            int gx = static_cast<int>((d_voxels[i].center.x - grid_info.map_origin_x) / grid_info.voxel_size);
            int gy = static_cast<int>((d_voxels[i].center.y - grid_info.map_origin_y) / grid_info.voxel_size);
            int gz = static_cast<int>((d_voxels[i].center.z - grid_info.map_origin_z) / grid_info.voxel_size);

            int lx = gx - grid_info.origin_x;
            int ly = gy - grid_info.origin_y;
            int lz = gz - grid_info.origin_z;

            if (lx >= 0 && lx < grid_info.size_x &&
                ly >= 0 && ly < grid_info.size_y &&
                lz >= 0 && lz < grid_info.size_z) {
                int grid_idx = lx * grid_info.size_y * grid_info.size_z +
                               ly * grid_info.size_z + lz;
                d_grid[grid_idx] = i;
            }
        });
    });

    // ==========================================
    // Stage 3: Raycast并原子计数
    // ==========================================
    sycl::event event3 = q.submit([&](sycl::handler& h) {
        h.depends_on(event2);
        h.parallel_for(sycl::range<1>(num_viewpoints * num_targets), [=](sycl::id<1> idx) {
            int global_idx = idx[0];
            int viewpoint_idx = global_idx / num_targets;
            int target_idx = global_idx % num_targets;

            Float3 viewpoint = d_viewpoints[viewpoint_idx];
            int voxel_idx = d_target_indices[target_idx];
            Float3 target = d_voxels[voxel_idx].center;

            int visibility = 0;

            // FOV 预裁剪，避免不在视锥内的射线
            if (use_horizontal_fov || use_vertical_fov) {
                // 相机旋转矩阵行（world -> camera），按viewpoint顺序紧密排列
                Float3 row0 = d_cam_rows[viewpoint_idx * 3 + 0];
                Float3 row1 = d_cam_rows[viewpoint_idx * 3 + 1];
                Float3 row2 = d_cam_rows[viewpoint_idx * 3 + 2];

                // world 坐标下的目标方向
                Float3 rel = {target.x - viewpoint.x, target.y - viewpoint.y, target.z - viewpoint.z, 0};
                float x_cam = row0.x * rel.x + row0.y * rel.y + row0.z * rel.z;
                float y_cam = row1.x * rel.x + row1.y * rel.y + row1.z * rel.z;
                float z_cam = row2.x * rel.x + row2.y * rel.y + row2.z * rel.z;

                if (use_horizontal_fov) {
                    float yaw = sycl::atan2(y_cam, x_cam);
                    if (sycl::fabs(yaw) > half_horizontal_fov) {
                        d_visibility_temp[global_idx] = 0;
                        return;
                    }
                }

                if (use_vertical_fov) {
                    float dist_xy = sycl::sqrt(x_cam * x_cam + y_cam * y_cam);
                    float pitch = sycl::atan2(-z_cam, dist_xy);
                    if (sycl::fabs(pitch) > half_vertical_fov) {
                        d_visibility_temp[global_idx] = 0;
                        return;
                    }
                }
            }

            visibility = RayCast(viewpoint, target, d_voxels, d_grid, grid_info);

            // 存储临时结果
            d_visibility_temp[global_idx] = visibility;

            // 原子计数（仅当可见时）
            if (visibility) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>(d_viewpoint_counts[viewpoint_idx]).fetch_add(1);
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>(d_target_counts[target_idx]).fetch_add(1);
            }
        });
    });


    event3.wait();  // 等待stage3完成
    // ==========================================
    // Stage 4: Prefix-sum生成offsets（CPU端）
    // ==========================================
    // 注意：SYCL的scan算子在不同实现中支持程度不同
    // 为了兼容性，在CPU端执行prefix-sum
    // GPU完成后，CPU读取counts，计算offsets，再写回
    auto stage4_start = std::chrono::high_resolution_clock::now();

    // 计算viewpoint offsets
    d_viewpoint_offsets[0] = 0;
    for (int i = 0; i < num_viewpoints; ++i) {
        d_viewpoint_offsets[i + 1] = d_viewpoint_offsets[i] + d_viewpoint_counts[i];
    }

    // 计算target offsets
    d_target_offsets[0] = 0;
    for (int i = 0; i < num_targets; ++i) {
        d_target_offsets[i + 1] = d_target_offsets[i] + d_target_counts[i];
    }

    // 重置counts用作写入位置计数器
    for (int i = 0; i < num_viewpoints; ++i) {
        d_viewpoint_counts[i] = 0;
    }
    for (int i = 0; i < num_targets; ++i) {
        d_target_counts[i] = 0;
    }

    auto stage4_end = std::chrono::high_resolution_clock::now();
    auto stage4_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stage4_end - stage4_start).count();

    // ==========================================
    // Stage 5: 并行写入CSR数组
    // ==========================================
    sycl::event event5 = q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(num_viewpoints * num_targets), [=](sycl::id<1> idx) {
            int global_idx = idx[0];
            int viewpoint_idx = global_idx / num_targets;
            int target_idx = global_idx % num_targets;

            int vis = d_visibility_temp[global_idx];

            if (vis) {
                // 原子获取写入位置
                int pos1 = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>(d_viewpoint_counts[viewpoint_idx]).fetch_add(1);
                int write_idx1 = d_viewpoint_offsets[viewpoint_idx] + pos1;
                d_viewpoint_to_targets[write_idx1] = target_idx;

                int pos2 = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>(d_target_counts[target_idx]).fetch_add(1);
                int write_idx2 = d_target_offsets[target_idx] + pos2;
                d_target_to_viewpoints[write_idx2] = viewpoint_idx;
            }
        });
    });

    // 等待最后一个kernel完成，然后打印所有stage的时间
    event5.wait();

    // ==========================================
    // Stage 6: 计算每个 target 的 available_mask（单向更新：可见→不可见）
    // ==========================================
    auto stage6_start = std::chrono::high_resolution_clock::now();

    sycl::event event6 = q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(num_targets), [=](sycl::id<1> idx) {
            int target_idx = idx[0];
            int voxel_idx = d_target_indices[target_idx];

            // 构建当前可见方向的mask（从能看到该target的viewpoints计算）
            uint32_t current_visible_mask = 0;

            // 获取能看到这个 target 的所有 viewpoints
            int start = d_target_offsets[target_idx];
            int end = d_target_offsets[target_idx + 1];

            // 遍历所有能看到这个 target 的 viewpoints
            for (int i = start; i < end; ++i) {
                int viewpoint_idx = d_target_to_viewpoints[i];
                Float3 viewpoint = d_viewpoints[viewpoint_idx];
                Float3 voxel_center = d_voxels[voxel_idx].center;

                // 计算观测方向: voxel_center -> viewpoint
                Float3 view_dir = {viewpoint.x - voxel_center.x,
                                   viewpoint.y - voxel_center.y,
                                   viewpoint.z - voxel_center.z, 0};

                // 获取方向索引
                int bin_idx = getBinIndex(view_dir);
                if (bin_idx >= 0 && bin_idx < 20) {
                    // 设置当前可见方向的对应位
                    current_visible_mask |= (1u << bin_idx);
                }
            }

            // 单向更新：保留原mask中的方向，加上当前新发现的可见方向（OR操作）
            // 效果：如果某个方向在当前raycast结果中可见，则在mask中标记为可见
            d_voxels[voxel_idx].available_mask |= current_visible_mask;
        });
    });

    event6.wait();

    // Stage 7: 计算 max_possible_score (仅对非frontier voxels)
    sycl::event event7 = q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(num_targets), [=](sycl::id<1> idx) {
            int target_idx = idx[0];
            int voxel_idx = d_target_indices[target_idx];

            // 跳过frontier voxels (不需要更新max_possible_score)
            if (d_voxels[voxel_idx].is_frontier) {
                return;
            }

            int normal_bin_idx = d_voxels[voxel_idx].normal_bin_idx;
            if (normal_bin_idx < 0 || normal_bin_idx >= 20) {
                d_voxels[voxel_idx].max_possible_score = 0.0f;
                return;
            }

            float score = 0.0f;
            uint32_t available_mask = d_voxels[voxel_idx].available_mask;

            // 遍历 available_mask 的每一位
            for (int i = 0; i < 20; ++i) {
                if (available_mask & (1u << i)) {
                    // 从评分表读取分数
                    score += d_scoring_table[normal_bin_idx * 20 + i];
                }
            }

            d_voxels[voxel_idx].max_possible_score = score;
        });
    });

    event7.wait();

    auto stage6_end = std::chrono::high_resolution_clock::now();
    auto stage6_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stage6_end - stage6_start).count();

    // 使用 SYCL profiling API 获取准确的GPU执行时间
    auto t1_start = event1.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto t1_end = event1.get_profiling_info<sycl::info::event_profiling::command_end>();
    // std::cout << "[GPU MxN CSR] Stage 1 (Init) duration: " << (t1_end - t1_start) / 1e6 << " ms" << std::endl;

    auto t2_start = event2.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto t2_end = event2.get_profiling_info<sycl::info::event_profiling::command_end>();
    // std::cout << "[GPU MxN CSR] Stage 2 (Build grid) duration: " << (t2_end - t2_start) / 1e6 << " ms" << std::endl;

    auto t3_start = event3.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto t3_end = event3.get_profiling_info<sycl::info::event_profiling::command_end>();
    // std::cout << "[GPU MxN CSR] Stage 3 (Raycast+Count) duration: " << (t3_end - t3_start) / 1e6 << " ms" << std::endl;

    // std::cout << "[GPU MxN CSR] Stage 4 (Prefix-sum CPU) duration: " << stage4_duration << " ms" << std::endl;

    auto t5_start = event5.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto t5_end = event5.get_profiling_info<sycl::info::event_profiling::command_end>();
    // std::cout << "[GPU MxN CSR] Stage 5 (Write CSR) duration: " << (t5_end - t5_start) / 1e6 << " ms" << std::endl;

    // std::cout << "[GPU MxN CSR] Stage 6+7 (Compute available_mask+score) duration: " << stage6_duration << " ms" << std::endl;

    return event7;
}

} // namespace gpu_raycast
