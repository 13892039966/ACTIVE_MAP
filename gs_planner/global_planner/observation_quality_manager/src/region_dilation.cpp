#include <observation_quality_manager/global_planner.h>
#include <queue>
#include <unordered_set>

// ==========================================
// Region膨胀与连通性扩展函数实现
// ==========================================

std::vector<Eigen::Vector3i> GlobalPlanner::getDilatedRegionIndices(
    const Eigen::Vector3i& center_region_idx) const {
  std::vector<Eigen::Vector3i> result;
  result.reserve(27);

  // 26-邻域 + 中心 = 27个region
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        result.push_back(center_region_idx + Eigen::Vector3i(dx, dy, dz));
      }
    }
  }
  return result;
}

void GlobalPlanner::expandConnectedFreeRegion(
    const ClusterInfo& cluster,
    Eigen::Vector3i& aabb_min,
    Eigen::Vector3i& aabb_max,
    std::vector<Eigen::Vector3i>& connected_free_voxels,
    Eigen::Vector3i& target_aabb_min,  // output for target AABB
    Eigen::Vector3i& target_aabb_max) {  // output for target AABB

  if (cluster.cluster_free_voxels.empty()) {
    ROS_WARN("GlobalPlanner::expandConnectedFreeRegion: cluster has no free voxels");
    aabb_min = aabb_max = Eigen::Vector3i::Zero();
    target_aabb_min = target_aabb_max = Eigen::Vector3i::Zero();
    connected_free_voxels.clear();
    return;
  }

  // 1. 计算 poorly_observed_voxels 的 AABB (target)
  if (!cluster.poorly_observed_voxels.empty()) {
    target_aabb_min = cluster.poorly_observed_voxels[0];
    target_aabb_max = cluster.poorly_observed_voxels[0];
    for (const auto& v : cluster.poorly_observed_voxels) {
      target_aabb_min = target_aabb_min.cwiseMin(v);
      target_aabb_max = target_aabb_max.cwiseMax(v);
    }
  } else {
    target_aabb_min = target_aabb_max = Eigen::Vector3i::Zero();
  }

  // 2. 直接使用原 FreeCluster 的 voxels（不再膨胀扩展）
  connected_free_voxels = cluster.cluster_free_voxels;

  // 3. 计算这些 voxels 的 AABB
  aabb_min = aabb_max = connected_free_voxels[0];
  for (const auto& v : connected_free_voxels) {
    aabb_min = aabb_min.cwiseMin(v);
    aabb_max = aabb_max.cwiseMax(v);
  }

  ROS_INFO("GlobalPlanner: Using %zu cluster free voxels directly (no expansion)",
           connected_free_voxels.size());
}

void GlobalPlanner::collectOccludersFromAABB(
    const Eigen::Vector3i& aabb_min,
    const Eigen::Vector3i& aabb_max,
    const std::vector<Eigen::Vector3i>& target_voxels,
    std::vector<GPUVoxel>& out_occluders,
    std::vector<Eigen::Vector3i>* out_occupied_indices) {

  ros::Time start_time = ros::Time::now();

  // 计算AABB体积
  int64_t volume = static_cast<int64_t>(aabb_max.x() - aabb_min.x() + 1) *
                   static_cast<int64_t>(aabb_max.y() - aabb_min.y() + 1) *
                   static_cast<int64_t>(aabb_max.z() - aabb_min.z() + 1);

  out_occluders.clear();

  // 构建target_set用于快速排除
  std::unordered_set<Eigen::Vector3i, VoxelHash, std::equal_to<Eigen::Vector3i>> target_set;
  target_set.reserve(target_voxels.size());
  for (const auto& v : target_voxels) {
    target_set.insert(v);
  }

  // Lambda: 将VoxelCell转换为GPUVoxel (遮挡物，is_frontier=0)
  auto convertCellToOccluderGPUVoxel = [](const VoxelCell& cell) {
    GPUVoxel gpu_voxel;
    gpu_voxel.center.x = cell.voxel_center.x();
    gpu_voxel.center.y = cell.voxel_center.y();
    gpu_voxel.center.z = cell.voxel_center.z();
    gpu_voxel.center.w = 0.0f;

    gpu_voxel.geo_complexity = cell.geometric_complexity;
    gpu_voxel.tex_complexity = cell.texture_complexity;
    gpu_voxel.normal_bin_idx = cell.normal_bin_idx;
    gpu_voxel.current_score = cell.observation_score;

    gpu_voxel.obs_mask = static_cast<uint32_t>(cell.observation_direction_mask.to_ulong());
    gpu_voxel.available_mask = static_cast<uint32_t>(cell.available_direction_mask.to_ulong());
    gpu_voxel.well_observed = cell.well_observed ? 1 : 0;
    gpu_voxel.is_frontier = 0;

    for (int i = 0; i < SUB_MASK_ARRAY_SIZE; ++i) {
      gpu_voxel.sub_masks[i] = 0;
      gpu_voxel.unknown_masks[i] = 0;
    }
    for (int bit_idx = 0; bit_idx < 1000; ++bit_idx) {
      if (cell.geometry_occupancy_mask[bit_idx]) {
        int array_idx = bit_idx / 32;
        int bit_offset = bit_idx % 32;
        gpu_voxel.sub_masks[array_idx] |= (1u << bit_offset);
      }
    }

    gpu_voxel.max_possible_score = 0.0f;
    gpu_voxel.padding = 0.0f;
    return gpu_voxel;
  };

  // Lambda: 创建unknown遮挡物GPUVoxel
  auto createUnknownOccluderGPUVoxel = [this](const Eigen::Vector3i& voxel_idx) {
    GPUVoxel gpu_voxel{};
    Eigen::Vector3f offset = ((voxel_idx.cast<float>().array() + 0.5f) * voxel_size_).matrix();
    Eigen::Vector3f center = region_origin_ + offset;
    gpu_voxel.center.x = center.x();
    gpu_voxel.center.y = center.y();
    gpu_voxel.center.z = center.z();
    gpu_voxel.center.w = 0.0f;

    gpu_voxel.geo_complexity = 0.0f;
    gpu_voxel.tex_complexity = 0.0f;
    gpu_voxel.normal_bin_idx = -1;
    gpu_voxel.current_score = 0.0f;
    gpu_voxel.obs_mask = 0;
    gpu_voxel.available_mask = 0;
    gpu_voxel.well_observed = 0;
    gpu_voxel.is_frontier = 0;

    for (int i = 0; i < SUB_MASK_ARRAY_SIZE; ++i) {
      gpu_voxel.sub_masks[i] = 0;
      gpu_voxel.unknown_masks[i] = 0xFFFFFFFFu;
    }

    gpu_voxel.max_possible_score = 0.0f;
    gpu_voxel.padding = 0.0f;
    return gpu_voxel;
  };

  size_t occupied_count = 0;
  size_t unknown_count = 0;
  size_t skipped_target_count = 0;

  for (int x = aabb_min.x(); x <= aabb_max.x(); ++x) {
    for (int y = aabb_min.y(); y <= aabb_max.y(); ++y) {
      for (int z = aabb_min.z(); z <= aabb_max.z(); ++z) {
        Eigen::Vector3i idx(x, y, z);

        // 排除targets
        if (target_set.count(idx)) {
          ++skipped_target_count;
          continue;
        }

        auto it = spatial_hash_.find(idx);
        if (it != spatial_hash_.end()) {
          out_occluders.push_back(convertCellToOccluderGPUVoxel(it->second));
          ++occupied_count;
          // 收集 occupied voxel 索引（用于后续膨胀）
          if (out_occupied_indices) {
            out_occupied_indices->push_back(idx);
          }
        } else if (treat_unknown_as_occupied_) {
          out_occluders.push_back(createUnknownOccluderGPUVoxel(idx));
          ++unknown_count;
        }
      }
    }
  }

  ros::Time end_time = ros::Time::now();
  ROS_INFO("GlobalPlanner: Collected %zu occluders from AABB in %.3f ms "
           "(volume: %ld, occupied: %zu, unknown: %zu, skipped_targets: %zu)",
           out_occluders.size(), (end_time - start_time).toSec() * 1000.0,
           volume, occupied_count, unknown_count, skipped_target_count);
}

void GlobalPlanner::dilateOccupiedVoxels(
    const std::vector<Eigen::Vector3i>& occupied_indices,
    std::unordered_set<Eigen::Vector3i, VoxelHash, std::equal_to<Eigen::Vector3i>>& free_voxels_set) {

  size_t removed_count = 0;

  // 遍历 occupied voxels，进行 26-邻域膨胀
  for (const auto& occ : occupied_indices) {
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          Eigen::Vector3i neighbor = occ + Eigen::Vector3i(dx, dy, dz);
          if (free_voxels_set.erase(neighbor)) {
            ++removed_count;
          }
        }
      }
    }
  }

  ROS_INFO("GlobalPlanner: Dilated %zu occupied voxels, removed %zu free voxels",
           occupied_indices.size(), removed_count);
}
