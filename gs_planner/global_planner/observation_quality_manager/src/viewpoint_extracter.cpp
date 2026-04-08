#include <observation_quality_manager/global_planner.h>
#include <observation_quality_manager/visibility_checker.h>
#include <observation_quality_manager/gpu_raycast.h>
#include <observation_quality_manager/gpu_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <algorithm>
#include <bitset>
#include <limits>
#include <queue>
#include <unordered_set>

std::vector<ClusterInfo> GlobalPlanner::collectAndFilterClusters(
    const std::unordered_map<Eigen::Vector3i, FreeRegion, RegionHash,
                             std::equal_to<Eigen::Vector3i>>& region_map) {

  ros::Time start_time = ros::Time::now();
  std::vector<ClusterInfo> all_clusters;

  // 遍历所有 regions
  for (const auto& [region_idx, region] : region_map) {
    if (region.isAllWellObserved()) {
      continue;
    }
    const auto& clusters = region.getClusters();
    // ROS_INFO("GlobalPlanner: Region [%d,%d,%d] has %zu clusters",
    //          region_idx.x(), region_idx.y(), region_idx.z(), clusters.size());

    // 遍历该 region 的所有 clusters
    for (const auto& cluster : clusters) {
      // 过滤条件1: 体素数量 > min_voxel_count_
      if (cluster.voxel_count <= min_voxel_count_) {
        continue;
      }

      // 过滤条件2: 必须有 poorly-observed 邻居
      if (cluster.poorly_observed_neighbors.empty()) {
        continue;
      }

      // 过滤条件3: 跳过已完成观测的 cluster (center 距离小于 1 个 cell_size)
      const auto& completed_centers = region.getCompletedClusterCenters();
      bool is_near_completed = false;
      for (const auto& completed_center : completed_centers) {
        float dist = (cluster.center - completed_center).cast<float>().norm();
        if (dist < 2.0f) {  // 小于 2 个 cell_size (voxel index 单位)
          is_near_completed = true;
          break;
        }
      }
      if (is_near_completed) {
        continue;
      }

      // 创建 ClusterInfo
      ClusterInfo info;
      info.position = voxelIdxToPosition(cluster.center);
      info.voxel_count = cluster.voxel_count;
      info.poorly_observed_count = cluster.poorly_observed_neighbors.size();
      info.voxel_idx = cluster.center;

      // 层级规划相关字段
      info.region_idx = region_idx;
      info.poorly_observed_voxels.assign(cluster.poorly_observed_neighbors.begin(),
                                          cluster.poorly_observed_neighbors.end());
      info.cluster_free_voxels.assign(cluster.free_voxels.begin(),
                                       cluster.free_voxels.end());

      all_clusters.push_back(info);
    }
  }

  // 按 poorly_observed_count 降序排序
  std::sort(all_clusters.begin(), all_clusters.end(),
            [](const ClusterInfo& a, const ClusterInfo& b) {
              return a.poorly_observed_count > b.poorly_observed_count;
            });

  // 选择前 max_clusters_ 个
  if (all_clusters.size() > static_cast<size_t>(max_clusters_)) {
    all_clusters.resize(max_clusters_);
  }

  ROS_INFO_THROTTLE(2.0, "GlobalPlanner: Collected %zu clusters after filtering",
                    all_clusters.size());

  ros::Time end_time = ros::Time::now();
  ROS_INFO("GlobalPlanner: collectAndFilterClusters completed in %.3f ms (%zu clusters)",
           (end_time - start_time).toSec() * 1000.0, all_clusters.size());

  // 可视化：发布每个 cluster 的 poorly_observed_count 文本标记
  if (poorly_observed_count_text_pub_.getNumSubscribers() > 0) {
    visualization_msgs::MarkerArray marker_array;

    // 先清除所有旧的 marker
    visualization_msgs::Marker delete_all;
    delete_all.action = visualization_msgs::Marker::DELETEALL;
    marker_array.markers.push_back(delete_all);

    // 为每个 cluster 创建文本标记
    for (size_t i = 0; i < all_clusters.size(); ++i) {
      const auto& cluster = all_clusters[i];

      visualization_msgs::Marker text_marker;
      text_marker.header.frame_id = "world";
      text_marker.header.stamp = ros::Time::now();
      text_marker.ns = "poorly_observed_count";
      text_marker.id = static_cast<int>(i);
      text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::Marker::ADD;

      text_marker.pose.position.x = cluster.position.x();
      text_marker.pose.position.y = cluster.position.y();
      text_marker.pose.position.z = cluster.position.z() + 0.3;  // 稍微上移避免与球体重叠
      text_marker.pose.orientation.w = 1.0;

      text_marker.scale.z = 0.3;  // 文字大小
      text_marker.color.r = 1.0;
      text_marker.color.g = 1.0;
      text_marker.color.b = 0.0;  // 黄色
      text_marker.color.a = 1.0;

      text_marker.text = std::to_string(cluster.poorly_observed_count);

      marker_array.markers.push_back(text_marker);
    }

    poorly_observed_count_text_pub_.publish(marker_array);
  }

  return all_clusters;
}

std::vector<Eigen::Vector3f> GlobalPlanner::sampleViewpointsFromCluster(
    const std::vector<Eigen::Vector3i>& cluster_free_voxels,
    float downsample_resolution) {

  ros::Time start_time = ros::Time::now();

  if (cluster_free_voxels.empty()) {
    ROS_WARN("GlobalPlanner: Empty cluster_free_voxels");
    return {};
  }

  // 1. 将 free voxel indices 转换为点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->points.reserve(cluster_free_voxels.size());

  for (const auto& voxel_idx : cluster_free_voxels) {
    Eigen::Vector3f position = voxelIdxToPosition(voxel_idx);
    cloud->points.emplace_back(position.x(), position.y(), position.z());
  }

  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = true;

  // 2. 使用 PCL VoxelGrid 降采样
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setInputCloud(cloud);
  voxel_filter.setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
  voxel_filter.filter(*downsampled);

  // 3. 转换回 Eigen::Vector3f，并过滤距离障碍物过近的点
  std::vector<Eigen::Vector3f> viewpoints;
  viewpoints.reserve(downsampled->points.size());

  int filtered_count = 0;

  for (const auto& point : downsampled->points) {
    Eigen::Vector3f position(point.x, point.y, point.z);

    // 检查到障碍物的距离
    if (lidar_map_interface_) {
      float distance = lidar_map_interface_->getDisToOcc(position);
      if (distance < min_distance_to_obstacle_) {
        filtered_count++;
        continue;  // 过滤掉距离障碍物过近的点
      }
    }

    viewpoints.emplace_back(position);
  }

  ROS_INFO("GlobalPlanner: Sampled %zu viewpoints from %zu free voxels (resolution: %.2fm), filtered %d too close to obstacles",
           viewpoints.size(), cluster_free_voxels.size(), downsample_resolution, filtered_count);

  ros::Time end_time = ros::Time::now();
  ROS_INFO("GlobalPlanner: sampleViewpointsFromCluster completed in %.3f ms",
           (end_time - start_time).toSec() * 1000.0);

  return viewpoints;
}

VisibilityCSR GlobalPlanner::evaluateViewpointsWithGPU(
    const std::vector<Eigen::Vector3f>& viewpoints,
    const std::vector<Eigen::Vector3i>& poorly_observed_voxels,
    const std::vector<Eigen::Vector3i>& frontier_voxels,
    std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                       std::equal_to<Eigen::Vector3i>>& spatial_hash,
    float voxel_size,
    const Eigen::Vector3f& map_min_bd,
    const Eigen::Vector3i& aabb_min,
    const Eigen::Vector3i& aabb_max) {

  // 合并两种target类型
  std::vector<Eigen::Vector3i> target_voxels;
  target_voxels.reserve(poorly_observed_voxels.size() + frontier_voxels.size());
  target_voxels.insert(target_voxels.end(), poorly_observed_voxels.begin(), poorly_observed_voxels.end());
  target_voxels.insert(target_voxels.end(), frontier_voxels.begin(), frontier_voxels.end());

  if (viewpoints.empty() || target_voxels.empty()) {
    ROS_WARN("GlobalPlanner: Empty viewpoints or target_voxels");
    return VisibilityCSR{};
  }

  ROS_INFO("GlobalPlanner: Evaluating %zu viewpoints for %zu poorly_observed + %zu frontier voxels",
           viewpoints.size(), poorly_observed_voxels.size(), frontier_voxels.size());

  // 1. 将 target voxel indices 转换为世界坐标
  std::vector<Eigen::Vector3f> targets;
  targets.reserve(target_voxels.size());
  for (const auto& voxel_idx : target_voxels) {
    Eigen::Vector3f position = voxelIdxToPosition(voxel_idx);
    targets.push_back(position);
  }

  // 构建 target 集合用于快速过滤
  std::unordered_set<Eigen::Vector3i, VoxelHash, std::equal_to<Eigen::Vector3i>> target_set;
  target_set.reserve(target_voxels.size());
  for (const auto& idx : target_voxels) {
    target_set.insert(idx);
  }

  // 构建 frontier 集合用于标记
  std::unordered_set<Eigen::Vector3i, VoxelHash, std::equal_to<Eigen::Vector3i>> frontier_set;
  frontier_set.reserve(frontier_voxels.size());
  for (const auto& idx : frontier_voxels) {
    frontier_set.insert(idx);
  }

  // Lambda: 为frontier voxel创建简化的GPUVoxel (仅需要位置信息用于可见性检查)
  auto createFrontierGPUVoxel = [this](const Eigen::Vector3i& voxel_idx) {
    GPUVoxel gpu_voxel;
    Eigen::Vector3f pos = voxelIdxToPosition(voxel_idx);
    gpu_voxel.center.x = pos.x();
    gpu_voxel.center.y = pos.y();
    gpu_voxel.center.z = pos.z();
    gpu_voxel.center.w = 0.0f;

    // Frontier不需要复杂度和观测质量信息
    gpu_voxel.geo_complexity = 0.0f;
    gpu_voxel.tex_complexity = 0.0f;
    gpu_voxel.normal_bin_idx = -1;  // 无效
    gpu_voxel.current_score = 0.0f;

    gpu_voxel.obs_mask = 0;
    gpu_voxel.available_mask = 0;
    gpu_voxel.well_observed = 0;
    gpu_voxel.is_frontier = 1;  // 标记为frontier

    for (int i = 0; i < SUB_MASK_ARRAY_SIZE; ++i) {
      gpu_voxel.sub_masks[i] = 0;  // Frontier无子体素掩码
      gpu_voxel.unknown_masks[i] = 0;
    }

    gpu_voxel.max_possible_score = 0.0f;
    gpu_voxel.padding = 0.0f;
    return gpu_voxel;
  };

  auto convertToGPUVoxel = [&frontier_set](const VoxelCell& cell, const Eigen::Vector3i& voxel_idx) {
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

    // 设置is_frontier标记：1=frontier, 0=poorly_observed
    gpu_voxel.is_frontier = (frontier_set.find(voxel_idx) != frontier_set.end()) ? 1 : 0;

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

  // 2. 提取spatial_hash中的GPUVoxel数据
  std::vector<GPUVoxel> gpu_voxels;

  // 先处理poorly_observed voxels (需要从spatial_hash查找)
  for (const auto& voxel_idx : poorly_observed_voxels) {
    auto it = spatial_hash.find(voxel_idx);
    ROS_ASSERT(it != spatial_hash.end());
    gpu_voxels.push_back(convertToGPUVoxel(it->second, voxel_idx));
  }

  // 再处理frontier voxels (直接创建GPUVoxel)
  for (const auto& voxel_idx : frontier_voxels) {
    gpu_voxels.push_back(createFrontierGPUVoxel(voxel_idx));
  }

  // 从AABB收集其他遮挡物 (排除已添加的targets)
  // 同时收集 occupied voxel 索引（用于后续 Dijkstra 搜索的膨胀）
  std::vector<GPUVoxel> aabb_occluders;
  aabb_occupied_indices_.clear();
  collectOccludersFromAABB(aabb_min, aabb_max, target_voxels, aabb_occluders, &aabb_occupied_indices_);
  gpu_voxels.insert(gpu_voxels.end(), aabb_occluders.begin(), aabb_occluders.end());

  // 可视化AABB包围盒
  visualizeAABB(aabb_min, aabb_max, "gpu_occlusion_aabb", 1.0f, 0.5f, 0.0f);

  ROS_INFO("GlobalPlanner: GPU voxels total: %zu (targets: %zu, occluders: %zu)",
           gpu_voxels.size(), target_voxels.size(), aabb_occluders.size());

  // 可视化：发布每个GPU voxel的子体素点（2cm 分辨率，10x10x10）
  if (gpu_voxel_points_pub_.getNumSubscribers() > 0) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr subvoxel_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    subvoxel_cloud->points.reserve(gpu_voxels.size() * 1000);

    const float sub_size = voxel_size / 10.0f;   // 0.02m when voxel_size=0.2
    const float half_voxel = voxel_size * 0.5f;

    for (const auto& v : gpu_voxels) {
      Eigen::Vector3f origin(v.center.x - half_voxel,
                             v.center.y - half_voxel,
                             v.center.z - half_voxel);
      for (int sx = 0; sx < 10; ++sx) {
        for (int sy = 0; sy < 10; ++sy) {
          for (int sz = 0; sz < 10; ++sz) {
            int bit_idx = sx * 100 + sy * 10 + sz;
            int array_idx = bit_idx / 32;
            int bit_offset = bit_idx % 32;
            bool occupied = (v.sub_masks[array_idx] >> bit_offset) & 1u;
            bool unknown = (v.unknown_masks[array_idx] >> bit_offset) & 1u;
            // if (!occupied && !(treat_unknown_as_occupied_ && unknown)) {
            if (!occupied ) {
              continue;
            }
            float x = origin.x() + (static_cast<float>(sx) + 0.5f) * sub_size;
            float y = origin.y() + (static_cast<float>(sy) + 0.5f) * sub_size;
            float z = origin.z() + (static_cast<float>(sz) + 0.5f) * sub_size;
            subvoxel_cloud->points.emplace_back(x, y, z);
          }
        }
      }
    }

    subvoxel_cloud->width = subvoxel_cloud->points.size();
    subvoxel_cloud->height = 1;
    subvoxel_cloud->is_dense = true;

    sensor_msgs::PointCloud2 subvoxel_msg;
    pcl::toROSMsg(*subvoxel_cloud, subvoxel_msg);
    subvoxel_msg.header.frame_id = "world";
    subvoxel_msg.header.stamp = ros::Time::now();

    gpu_voxel_points_pub_.publish(subvoxel_msg);
    ROS_INFO("GlobalPlanner: Published %zu sub-voxel points (res=%.3fm) from %zu GPU voxels",
             subvoxel_cloud->points.size(), sub_size, gpu_voxels.size());
  }

  if (gpu_voxels.empty()) {
    ROS_ERROR("GlobalPlanner: No valid GPU voxels extracted from spatial_hash");
    return VisibilityCSR{};
  }

  // 3. 调用 GPU raycast
  ros::Time start = ros::Time::now();

  VisibilityCSR csr_result = observation_quality::checkVisibilityGPU(
      viewpoints,
      targets,
      gpu_voxels,
      voxel_size,
      map_min_bd  // 传递地图原点，确保GPU和CPU使用相同的坐标系统
  );

  ros::Time end = ros::Time::now();
  ROS_INFO("GlobalPlanner: GPU raycast completed in %.3f ms (%zu viewpoints x %zu targets)",
           (end - start).toSec() * 1000.0, viewpoints.size(), targets.size());

  // 4. 计算并记录可见性统计
  int min_visible = INT_MAX;
  int max_visible = 0;
  for (size_t i = 0; i < viewpoints.size(); ++i) {
    int visible_count = csr_result.getVisibleTargets(i).size();
    min_visible = std::min(min_visible, visible_count);
    max_visible = std::max(max_visible, visible_count);
  }

  ROS_INFO("GlobalPlanner: Evaluated %zu viewpoints, visible target count range: [%d, %d]",
           viewpoints.size(), min_visible, max_visible);

  // 同步 GPU 计算出的 available_mask / max_possible_score
  // 注意：这里只更新 available_mask，不更新 observation_direction_mask
  // 因为这些是候选 viewpoints，还没有真正访问
  if (!csr_result.updated_available_masks.empty()) {
    int synced_count = 0;
    for (size_t i = 0; i < target_voxels.size(); ++i) {
      auto cell_it = spatial_hash.find(target_voxels[i]);
      ROS_ASSERT(cell_it != spatial_hash.end());

      VoxelCell& cell = cell_it->second;
      cell.available_direction_mask |= std::bitset<20>(csr_result.updated_available_masks[i]);


      if (i < csr_result.updated_max_scores.size()) {
        cell.max_possible_score = csr_result.updated_max_scores[i];
      } else {
        cell.max_possible_score = 0.0f;
      }

      synced_count++;
    }
    ROS_INFO("GlobalPlanner: Synced available_mask for %d/%zu target voxels",
             synced_count, target_voxels.size());
  }

  return csr_result;
}

SelectedViewResult GlobalPlanner::selectTopNViewpoints(
    const VisibilityCSR& csr_result,
    const std::vector<Eigen::Vector3f>& viewpoints,
    const std::vector<Eigen::Vector3i>& target_voxels,
    int num_poorly_observed,
    int top_n,
    const std::vector<std::bitset<20>>& initial_masks) {

  // ==========================================
  // 朝向网格配置 (Orientation Grid)
  // Yaw: 10 bins, [-π, π] -> 36°/bin
  // Pitch: 5 bins, [-60°, 60°] -> 24°/bin
  // ==========================================
  const int Y_BINS = 10;
  const int P_BINS = 5;
  const float yaw_step = 2.0f * M_PI / Y_BINS;          // 36° per bin
  const float pitch_min = pitch_min_;                   // configured min pitch
  const float pitch_max = pitch_max_;                   // configured max pitch
  const float pitch_range = pitch_max - pitch_min;      // 120°
  const float pitch_step = pitch_range / P_BINS;        // 24° per bin

  // FOV覆盖的bin半径
  const float yaw_radius_bins = std::max(0.0f, (h_fov_ * 0.5f) / yaw_step);
  const float pitch_radius_bins = std::max(0.0f, (v_fov_ * 0.5f) / pitch_step);

  struct TargetCache {
      std::bitset<20> current_mask;  // 已被观测的方向掩码
      int normal_bin_idx;            // 法线方向bin索引
      Eigen::Vector3f center;        // 体素中心坐标
      float current_score;           // 当前观测分数
      float target_score;            // 目标分数（达到后视为well-observed）
      float geometric_complexity;    // 几何复杂度
      float texture_complexity;      // 纹理复杂度
  };

  // 预计算的观测关系 (带朝向范围)
  struct RelDiscrete {
      uint16_t t_idx;       // Target索引
      uint8_t obs_bin;      // 观测质量方向bin
      float score;          // 基础得分
      int8_t yaw_start, yaw_end;       // Yaw范围 [start, end]
      int8_t pitch_start, pitch_end;   // Pitch范围 [start, end]
      bool yaw_wrapped;     // Yaw是否跨越±π边界
  };

  SelectedViewResult result;
  if (viewpoints.empty() || target_voxels.empty()) return result;

  int num_targets = target_voxels.size(); 
  int num_viewpoints = viewpoints.size();

  // ==========================================
  // Phase 1: 数据线性化 (Linearization)
  // ==========================================
  std::vector<TargetCache> target_cache(num_targets);

  for (int i = 0; i < num_targets; ++i) {
    auto it = spatial_hash_.find(target_voxels[i]);
    if (it != spatial_hash_.end()) {
      target_cache[i].current_mask = it->second.observation_direction_mask;
      target_cache[i].normal_bin_idx = it->second.normal_bin_idx;
      target_cache[i].center = it->second.voxel_center;
      target_cache[i].current_score = it->second.observation_score;
      target_cache[i].geometric_complexity = it->second.geometric_complexity;
      target_cache[i].texture_complexity = it->second.texture_complexity;

      // 计算目标分数：base_score + tex_weight*tex + geo_weight*geo，但不能超过max_possible_score
      float threshold_score = well_observed_base_score_ +
                             well_observed_texture_weight_ * it->second.texture_complexity +
                             well_observed_geo_weight_ * it->second.geometric_complexity;
      if (it->second.max_possible_score > 0.0f) {
        target_cache[i].target_score = std::min(it->second.max_possible_score, threshold_score);
      } else {
        target_cache[i].target_score = threshold_score;
      }
    } else {
      target_cache[i].normal_bin_idx = -1;  // 无效
      target_cache[i].current_score = 0.0f;
      target_cache[i].target_score = 1.0f;
      target_cache[i].geometric_complexity = 0.0f;
      target_cache[i].texture_complexity = 0.0f;
    }
  }

  // 如果提供了初始掩码，使用它来更新 current_mask 和 current_score
  if (!initial_masks.empty()) {
    for (int i = 0; i < num_targets && i < static_cast<int>(initial_masks.size()); ++i) {
      // 合并掩码（已有 + 初始）
      target_cache[i].current_mask |= initial_masks[i];

      // 重新计算 current_score 基于合并后的掩码
      int normal_bin = target_cache[i].normal_bin_idx;
      if (normal_bin >= 0 && normal_bin < 20) {
        float new_score = 0.0f;
        for (int bin = 0; bin < 20; ++bin) {
          if (target_cache[i].current_mask[bin]) {
            new_score += SphericalBinning::scoring_table[normal_bin][bin];
          }
        }
        target_cache[i].current_score = new_score;
      }
    }
    ROS_INFO("GlobalPlanner: Applied initial_masks to %zu targets", initial_masks.size());
  }

  // ==========================================
  // Phase 2: 几何预计算 + 朝向映射 (Pre-computation)
  // 对每个可见的 (Viewpoint, Target) 对:
  // 1. 计算相对角度 (yaw, pitch)
  // 2. 计算该Target能被哪些朝向bin看到 (基于FOV)
  // ==========================================
  std::vector<std::vector<RelDiscrete>> vp_relations(num_viewpoints);

  // OpenMP 4核并行计算
  #pragma omp parallel for schedule(dynamic, 8) num_threads(4)
  for (int vp_idx = 0; vp_idx < num_viewpoints; ++vp_idx) {
    const std::vector<int>& visible_indices = csr_result.getVisibleTargets(vp_idx);
    vp_relations[vp_idx].reserve(visible_indices.size());

    for (int t_global_idx : visible_indices) {
      if (t_global_idx < 0 || t_global_idx >= num_targets) continue;

      const auto& t_data = target_cache[t_global_idx];
      // 如果没有法向但已经有观测方向，视为已充分观测，跳过规划
      if (t_data.normal_bin_idx < 0 && t_data.current_mask.any()) {
        continue;
      }

      // 计算从视点到目标的方向向量
      Eigen::Vector3f dir = t_data.center - viewpoints[vp_idx];

      // 计算相对角度 (相机需要朝向目标)
      float yaw = std::atan2(dir.y(), dir.x());  // [-π, π]
      float dist_xy = std::sqrt(dir.x() * dir.x() + dir.y() * dir.y());
      // ROS Standard: Positive Pitch = Down
      // If target is above (dir.z > 0), pitch should be negative.
      float pitch = std::atan2(-dir.z(), dist_xy);  // 俯仰角

      // 如果pitch超出范围，跳过 (相机看不到)
      if (pitch < pitch_min || pitch > pitch_max) continue;

      // 计算观测质量得分 (从视点看target)
      Eigen::Vector3f view_dir = viewpoints[vp_idx] - t_data.center;
      int obs_bin = SphericalBinning::get_bin_index(view_dir);
      if (obs_bin < 0) continue;

      // 法向缺失时用当前观测方向作为法向分箱，避免丢弃该目标
      int normal_bin = (t_data.normal_bin_idx >= 0) ? t_data.normal_bin_idx : obs_bin;
      float score = SphericalBinning::scoring_table[normal_bin][obs_bin];

      // 构建关系数据
      RelDiscrete rel;
      rel.t_idx = (uint16_t)t_global_idx;
      rel.obs_bin = (uint8_t)obs_bin;
      rel.score = score;

      // Pitch范围 (精确FOV几何计算)
      // Bin i 的中心角度: pitch_min + (i + 0.5) * pitch_step
      // Bin i 能看到 target 当且仅当: |pitch - bin_center| <= v_fov/2
      float center_p_continuous = (pitch - pitch_min) / pitch_step;
      int p_min = static_cast<int>(std::ceil(center_p_continuous - 0.5f - pitch_radius_bins));
      int p_max = static_cast<int>(std::floor(center_p_continuous - 0.5f + pitch_radius_bins));
      rel.pitch_start = (int8_t)std::max(0, p_min);
      rel.pitch_end = (int8_t)std::min(P_BINS - 1, p_max);

      // Yaw范围 (精确FOV几何计算 + 环形wrap-around)
      float center_y_continuous = (yaw + M_PI) / yaw_step;
      int y_min = static_cast<int>(std::ceil(center_y_continuous - 0.5f - yaw_radius_bins));
      int y_max = static_cast<int>(std::floor(center_y_continuous - 0.5f + yaw_radius_bins));

      rel.yaw_wrapped = false;
      if (y_min < 0) {
        // 跨越左边界: 例如 [-1, 2] -> [9, 2] (wrapped)
        rel.yaw_start = (int8_t)(y_min + Y_BINS);
        rel.yaw_end = (int8_t)y_max;
        rel.yaw_wrapped = true;
      } else if (y_max >= Y_BINS) {
        // 跨越右边界: 例如 [8, 11] -> [8, 1] (wrapped)
        rel.yaw_start = (int8_t)y_min;
        rel.yaw_end = (int8_t)(y_max - Y_BINS);
        rel.yaw_wrapped = true;
      } else {
        // 正常情况: 无wrap-around
        rel.yaw_start = (int8_t)y_min;
        rel.yaw_end = (int8_t)y_max;
      }

      vp_relations[vp_idx].push_back(rel);
    }
  }

  // ==========================================
  // Phase 3: 懒惰贪婪投票循环 (Lazy Greedy Voting Loop)
  // 使用优先队列 + 懒惰评估加速视点选择
  // 核心思想：上一轮的高分视点大概率仍是高分，无需每轮全量计算
  // ==========================================
  std::vector<SelectedView> selected;
  std::vector<std::vector<int>> covered_targets;

  // 存储已选中的视点索引
  // 同一个位置只允许一个视点被选中
  std::unordered_set<int> selected_mask;

  int actual_n = std::min(top_n, num_viewpoints);

  // 候选视点状态结构
  struct CandidateState {
    int id;              // 视点索引
    float gain;          // 最佳增益
    int grid_idx;        // 最佳朝向格子索引 (y_idx + p_idx * Y_BINS)

    // 大顶堆: 增益高的优先
    bool operator<(const CandidateState& other) const {
      return gain < other.gain;
    }
  };

  // 记分板 (用于计算单个视点的增益)
  std::vector<float> scoreboard(Y_BINS * P_BINS);

  // Lambda: 计算单个视点的增益和最佳朝向
  // 返回 {best_gain, best_grid_idx}
  auto computeGainForViewpoint = [&](int vp_idx) -> std::pair<float, int> {
    // 清空记分板
    std::fill(scoreboard.begin(), scoreboard.end(), 0.0f);
    bool has_potential = false;

    // 遍历可见关系，进行投票
    for (const auto& rel : vp_relations[vp_idx]) {
      // ✅ 首先检查target voxel是否已达到目标分数
      if (target_cache[rel.t_idx].current_score >= target_cache[rel.t_idx].target_score) {
        continue;  // 已达到目标分数，视为well-observed，不再获得新分数
      }

      // ✅ 区分frontier和poorly_observed的检查逻辑
      bool should_skip = false;
      if (rel.t_idx >= num_poorly_observed) {
        // Frontier: 如果已被完全覆盖（所有方向都已观测），则跳过
        should_skip = target_cache[rel.t_idx].current_mask.all();
      } else {
        // Poorly observed: 只检查当前观测方向是否已被覆盖
        should_skip = target_cache[rel.t_idx].current_mask[rel.obs_bin];
      }

      if (should_skip) continue;

      has_potential = true;
      float s = rel.score;

      // 向覆盖范围内的所有bin投票
      for (int p = rel.pitch_start; p <= rel.pitch_end; ++p) {
        int row_offset = p * Y_BINS;

        if (rel.yaw_wrapped) {
          // 跨边界: [start, Y_BINS-1] 和 [0, end]
          for (int y = rel.yaw_start; y < Y_BINS; ++y) {
            scoreboard[row_offset + y] += s;
          }
          for (int y = 0; y <= rel.yaw_end; ++y) {
            scoreboard[row_offset + y] += s;
          }
        } else {
          // 正常范围: [start, end]
          for (int y = rel.yaw_start; y <= rel.yaw_end; ++y) {
            scoreboard[row_offset + y] += s;
          }
        }
      }
    }

    if (!has_potential) return {-1.0f, -1};

    // 找出记分板中分数最高的格子（覆盖增益）
    float best_gain = -1.0f;
    int best_grid_idx = -1;
    for (int k = 0; k < Y_BINS * P_BINS; ++k) {
      if (scoreboard[k] > best_gain) {
        best_gain = scoreboard[k];
        best_grid_idx = k;
      }
    }

    return {best_gain, best_grid_idx};
  };

  // Lambda: 更新选中视点覆盖的Target mask
  auto updateTargetMasks = [&](int best_vp, int best_grid_idx) {
    int best_p_idx = best_grid_idx / Y_BINS;
    int best_y_idx = best_grid_idx % Y_BINS;

    std::vector<int> newly_covered;
    for (const auto& rel : vp_relations[best_vp]) {
      // 检查best_p_idx是否在rel的pitch范围内
      bool p_overlap = (best_p_idx >= rel.pitch_start && best_p_idx <= rel.pitch_end);

      // 检查best_y_idx是否在rel的yaw范围内
      bool y_overlap = false;
      if (rel.yaw_wrapped) {
        y_overlap = (best_y_idx >= rel.yaw_start) || (best_y_idx <= rel.yaw_end);
      } else {
        y_overlap = (best_y_idx >= rel.yaw_start && best_y_idx <= rel.yaw_end);
      }

      if (p_overlap && y_overlap) {
        bool was_observed = target_cache[rel.t_idx].current_mask[rel.obs_bin];

        // ✅ 区分frontier和poorly_observed的覆盖策略
        if (rel.t_idx >= num_poorly_observed) {
          // Frontier: 被看到一次就完全覆盖（所有20个方向标记为已观测）
          if (!target_cache[rel.t_idx].current_mask.all()) {
            target_cache[rel.t_idx].current_mask.set();  // 设置所有位为1

            // 更新current_score：所有方向的分数之和
            float new_score = 0.0f;
            int normal_bin = target_cache[rel.t_idx].normal_bin_idx;
            if (normal_bin >= 0) {
              for (int i = 0; i < 20; ++i) {
                new_score += SphericalBinning::scoring_table[normal_bin][i];
              }
            }
            target_cache[rel.t_idx].current_score = new_score;

            newly_covered.push_back(rel.t_idx);
          }
        } else {
          // Poorly observed: 只标记当前观测方向
          if (!was_observed) {
            target_cache[rel.t_idx].current_mask[rel.obs_bin] = true;

            // 更新current_score：增加当前观测方向的分数
            int normal_bin = target_cache[rel.t_idx].normal_bin_idx;
            if (normal_bin >= 0) {
              target_cache[rel.t_idx].current_score += SphericalBinning::scoring_table[normal_bin][rel.obs_bin];
            }

            newly_covered.push_back(rel.t_idx);
          }
        }
      }
    }

    return newly_covered;
  };

  // ==========================================
  // 3.1 初始化：计算所有视点的初始增益
  // ==========================================
  std::priority_queue<CandidateState> pq;
  // #pragma omp parallel for schedule(dynamic) num_threads(4)
  for (int vp_idx = 0; vp_idx < num_viewpoints; ++vp_idx) {
    auto [gain, grid_idx] = computeGainForViewpoint(vp_idx);
    if (gain > 1e-4f) {
      pq.push({vp_idx, gain, grid_idx});
    }
  }

  // ==========================================
  // 3.2 懒惰循环：每轮只重算必要的视点
  // ==========================================
  int total_evaluations = num_viewpoints;  // 统计总评估次数（调试用）

  for (int iter = 0; iter < actual_n && !pq.empty(); ++iter) {
    // 懒惰评估：反复取队首，重算其分数，直到找到真正的最大值
    while (!pq.empty()) {
      CandidateState top = pq.top();
      pq.pop();

      // 跳过已选中的视点位置
      if (selected_mask.count(top.id) > 0) continue;

      // 重新评估队首视点（因为 target_cache 的 mask 可能已更新）
      auto [new_gain, new_grid_idx] = computeGainForViewpoint(top.id);
      total_evaluations++;

      // 如果新分数无效，跳过
      if (new_gain < 1e-4f) continue;

      // 懒惰检查：如果新分数仍然 >= 队列第二名，说明它确实是当前最大值
      if (pq.empty() || new_gain >= pq.top().gain) {
        // 选中这个视点！
        int best_grid_idx = new_grid_idx;
        int best_p_idx = best_grid_idx / Y_BINS;
        int best_y_idx = best_grid_idx % Y_BINS;

        float final_yaw = -M_PI + (best_y_idx + 0.5f) * yaw_step;
        float final_pitch = pitch_min + (best_p_idx + 0.5f) * pitch_step;

        // 标记这个视点位置为已选中
        selected_mask.insert(top.id);
        selected.push_back({viewpoints[top.id], final_yaw, final_pitch, new_gain, top.id});

        // 更新已覆盖的Target mask，并记录本次覆盖的targets
        covered_targets.push_back(updateTargetMasks(top.id, best_grid_idx));
        break;
      } else {
        // 分数下降了，重新插入队列等待下次比较
        pq.push({top.id, new_gain, new_grid_idx});
      }
    }
  }

  ROS_INFO("GlobalPlanner: Lazy evaluation stats - initial: %d, total: %d, ratio: %.2f",
            num_viewpoints, total_evaluations, (float)total_evaluations / num_viewpoints);
  ROS_INFO("selected viewpoints number: %zu", selected.size());

  result.views = selected;

  // Build filtered CSR using原始 viewpoint 下标，确保可视化用的索引与 selected_views_.viewpoint_idx 对齐
  result.filtered_csr.viewpoint_offsets.assign(num_viewpoints + 1, 0);

  std::vector<std::vector<int>> vp_to_targets(num_viewpoints);
  std::vector<std::vector<int>> target_to_viewers(target_voxels.size());

  // 将每个已选视点覆盖的 target 记录到原始 viewpoint_idx 下标对应的槽位
  for (size_t i = 0; i < result.views.size(); ++i) {
    int orig_vp_idx = result.views[i].viewpoint_idx;
    if (orig_vp_idx < 0 || orig_vp_idx >= static_cast<int>(num_viewpoints)) {
      continue;
    }
    if (i < covered_targets.size()) {
      const auto& covered = covered_targets[i];
      vp_to_targets[orig_vp_idx].insert(
          vp_to_targets[orig_vp_idx].end(),
          covered.begin(), covered.end());
      for (int t_idx : covered) {
        if (t_idx >= 0 && t_idx < static_cast<int>(target_to_viewers.size())) {
          target_to_viewers[t_idx].push_back(orig_vp_idx);
        }
      }
    }
  }

  // viewpoint -> targets CSR
  size_t vp_total = 0;
  for (size_t vp = 0; vp < num_viewpoints; ++vp) {
    result.filtered_csr.viewpoint_offsets[vp] = static_cast<int>(vp_total);
    const auto& covered = vp_to_targets[vp];
    result.filtered_csr.viewpoint_to_targets.insert(
        result.filtered_csr.viewpoint_to_targets.end(),
        covered.begin(), covered.end());
    vp_total += covered.size();
  }
  result.filtered_csr.viewpoint_offsets[num_viewpoints] = static_cast<int>(vp_total);

  // target -> viewpoints CSR
  result.filtered_csr.target_offsets.assign(target_voxels.size() + 1, 0);
  size_t target_total = 0;
  for (size_t t = 0; t < target_to_viewers.size(); ++t) {
    result.filtered_csr.target_offsets[t] = static_cast<int>(target_total);
    const auto& viewers = target_to_viewers[t];
    result.filtered_csr.target_to_viewpoints.insert(
        result.filtered_csr.target_to_viewpoints.end(),
        viewers.begin(), viewers.end());
    target_total += viewers.size();
  }
  result.filtered_csr.target_offsets[target_to_viewers.size()] = static_cast<int>(target_total);

  // 保存最终的 target 掩码状态
  result.final_target_masks.resize(target_cache.size());
  for (size_t t = 0; t < target_cache.size(); ++t) {
    result.final_target_masks[t] = target_cache[t].current_mask;
  }

  return result;
}
