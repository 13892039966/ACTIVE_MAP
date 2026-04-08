#include <observation_quality_manager/visualization_utils.h>
#include <observation_quality_manager/observation_quality_manager.h>
#include <observation_quality_manager/global_planner.h>
#include <gpu_visibility_checker/spherical_binning.h>
#include <algorithm>

namespace observation_quality {

Eigen::Vector3i VisualizationUtils::jetColorMap(float value, float min_val,
                                                 float max_val) {
  // Clamp value to [min_val, max_val]
  value = std::max(min_val, std::min(max_val, value));

  // Normalize to [0, 1]
  float normalized = (value - min_val) / (max_val - min_val);

  // Jet colormap implementation
  float r, g, b;

  if (normalized < 0.25f) {
    r = 0.0f;
    g = normalized * 4.0f;
    b = 1.0f;
  } else if (normalized < 0.5f) {
    r = 0.0f;
    g = 1.0f;
    b = 1.0f - (normalized - 0.25f) * 4.0f;
  } else if (normalized < 0.75f) {
    r = (normalized - 0.5f) * 4.0f;
    g = 1.0f;
    b = 0.0f;
  } else {
    r = 1.0f;
    g = 1.0f - (normalized - 0.75f) * 4.0f;
    b = 0.0f;
  }

  // Convert to 0-255 range
  return Eigen::Vector3i(static_cast<int>(r * 255),
                          static_cast<int>(g * 255),
                          static_cast<int>(b * 255));
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
VisualizationUtils::generateGeometryComplexityCloud(
    const std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                              std::equal_to<Eigen::Vector3i>> &spatial_hash,
    float voxel_size) {

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  cloud->points.reserve(spatial_hash.size());

  for (const auto &[idx, cell] : spatial_hash) {
    // Skip voxels with insufficient points
    if (cell.geo_cnt < 3) {
      continue;
    }

    pcl::PointXYZRGB pt;

    // Set position (voxel center)
    pt.x = cell.voxel_center.x();
    pt.y = cell.voxel_center.y();
    pt.z = cell.voxel_center.z();

    // Map geometric complexity to color
    Eigen::Vector3i rgb = jetColorMap(cell.geometric_complexity, 0.0f, 0.15f);
    pt.r = static_cast<uint8_t>(rgb[0]);
    pt.g = static_cast<uint8_t>(rgb[1]);
    pt.b = static_cast<uint8_t>(rgb[2]);

    cloud->points.push_back(pt);
  }

  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = true;

  return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
VisualizationUtils::generateTextureComplexityCloud(
    const std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                              std::equal_to<Eigen::Vector3i>> &spatial_hash,
    float voxel_size) {

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  cloud->points.reserve(spatial_hash.size());

  for (const auto &[idx, cell] : spatial_hash) {
    // Skip voxels with insufficient texture samples (need at least 2 for variance)
    if (cell.tex_cnt < 2) {
      continue;
    }

    pcl::PointXYZRGB pt;

    // Set position (voxel center)
    pt.x = cell.voxel_center.x();
    pt.y = cell.voxel_center.y();
    pt.z = cell.voxel_center.z();

    // Map texture complexity to color
    // Use [0.0, 1.0] range so 0.5 maps to red (middle of jet colormap)
    Eigen::Vector3i rgb = jetColorMap(cell.texture_complexity, 0.0f, 0.2f);
    pt.r = static_cast<uint8_t>(rgb[0]);
    pt.g = static_cast<uint8_t>(rgb[1]);
    pt.b = static_cast<uint8_t>(rgb[2]);

    cloud->points.push_back(pt);
  }

  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = true;

  return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
VisualizationUtils::generateObservationScoreCloud(
    const std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                              std::equal_to<Eigen::Vector3i>> &spatial_hash,
    float voxel_size) {

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  cloud->points.reserve(spatial_hash.size());

  for (const auto &[idx, cell] : spatial_hash) {
    // Skip voxels with no observations
    if (cell.observation_score == 0.0f) {
      continue;
    }

    pcl::PointXYZRGB pt;

    // Set position (voxel center)
    pt.x = cell.voxel_center.x();
    pt.y = cell.voxel_center.y();
    pt.z = cell.voxel_center.z();

    // Map observation score to color
    Eigen::Vector3i rgb = jetColorMap(cell.observation_score, 0.0f, 3.0f);
    pt.r = static_cast<uint8_t>(rgb[0]);
    pt.g = static_cast<uint8_t>(rgb[1]);
    pt.b = static_cast<uint8_t>(rgb[2]);

    cloud->points.push_back(pt);
  }

  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = true;

  return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
VisualizationUtils::generateWellObservedCloud(
    const std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                              std::equal_to<Eigen::Vector3i>> &spatial_hash,
    float voxel_size) {

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  cloud->points.reserve(spatial_hash.size());

  for (const auto &[idx, cell] : spatial_hash) {
    // Skip voxels without sufficient data (geo_cnt < 3 or tex_cnt < 2)
    // if (cell.geo_cnt < 3 || cell.tex_cnt < 2) {
    //   continue;
    // }

    pcl::PointXYZRGB pt;

    // Set position (voxel center)
    pt.x = cell.voxel_center.x();
    pt.y = cell.voxel_center.y();
    pt.z = cell.voxel_center.z();

    // Color based on well_observed status
    if (cell.well_observed) {
      // Green for well observed
      pt.r = 0;
      pt.g = 255;
      pt.b = 0;
    } else {
      // Red for not well observed
      pt.r = 255;
      pt.g = 0;
      pt.b = 0;
    }

    cloud->points.push_back(pt);
  }

  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = true;

  return cloud;
}

// ==========================================
// 新增：可见性连线可视化
// ==========================================
visualization_msgs::Marker VisualizationUtils::generateVisibilityLinesMarker(
    const std::vector<VisibilityLine>& visibility_lines,
    const std::string& frame_id)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = ros::Time::now();
  marker.ns = "visibility_lines";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.action = visualization_msgs::Marker::ADD;

  // 设置线条宽度
  marker.scale.x = 0.02; // 2cm 线宽

  // 预分配空间
  marker.points.reserve(visibility_lines.size() * 2);
  marker.colors.reserve(visibility_lines.size() * 2);

  for (const auto& line : visibility_lines) {
    // 添加起点
    geometry_msgs::Point start_pt;
    start_pt.x = line.start.x();
    start_pt.y = line.start.y();
    start_pt.z = line.start.z();
    marker.points.push_back(start_pt);

    // 添加终点
    geometry_msgs::Point end_pt;
    end_pt.x = line.end.x();
    end_pt.y = line.end.y();
    end_pt.z = line.end.z();
    marker.points.push_back(end_pt);

    // 设置颜色
    std_msgs::ColorRGBA color;
    if (line.visible) {
      // 绿色 = 可见
      color.r = 0.0f;
      color.g = 1.0f;
      color.b = 0.0f;
      color.a = 0.6f;
    } else {
      // 红色 = 遮挡
      color.r = 1.0f;
      color.g = 0.0f;
      color.b = 0.0f;
      color.a = 0.8f;
    }

    // LINE_LIST 需要为每个顶点设置颜色
    marker.colors.push_back(color);
    marker.colors.push_back(color);
  }

  return marker;
}

// ==========================================
// 自由空间聚类中心可视化
// ==========================================
visualization_msgs::Marker VisualizationUtils::generateFreeClusterCentersMarker(
    const std::unordered_map<Eigen::Vector3i, FreeRegion, RegionHash,
                              std::equal_to<Eigen::Vector3i>>& region_map,
    const std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                              std::equal_to<Eigen::Vector3i>>& spatial_hash,
    float voxel_size,
    const Eigen::Vector3f& region_origin,
    const std::string& frame_id)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = ros::Time::now();
  marker.ns = "free_cluster_centers";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::SPHERE_LIST;
  marker.action = visualization_msgs::Marker::ADD;

  // 设置球体大小 (直径 = voxel_size * 2.0)
  marker.scale.x = voxel_size * 2.0f;
  marker.scale.y = voxel_size * 2.0f;
  marker.scale.z = voxel_size * 2.0f;

  // 收集 size > 30 的聚类位置和属性
  struct ClusterInfo {
    Eigen::Vector3f position;
    bool has_poorly_observed_neighbor;
  };
  std::vector<ClusterInfo> large_clusters;

  // 遍历所有 region，收集聚类信息（聚类已在 geometry 更新时计算）
  for (const auto& [region_idx, region] : region_map) {
    // 跳过空 region
    if (region.freeVoxelCount() == 0) continue;

    // 收集 size > 30 的聚类信息
    for (const auto& cluster : region.getClusters()) {
      // 只保留 size > 30 的聚类
      if (cluster.voxel_count > 30) {
        // 将体素索引转换为世界坐标 (with origin offset)
        Eigen::Vector3f offset = ((cluster.center.cast<float>().array() + 0.5f) * voxel_size).matrix();
        Eigen::Vector3f position = region_origin + offset;
        large_clusters.push_back({position, cluster.has_poorly_observed_neighbor});
      }
    }
  }

  // 如果没有聚类，返回空marker
  if (large_clusters.empty()) {
    return marker;
  }

  // 预分配空间
  marker.points.reserve(large_clusters.size());
  marker.colors.reserve(large_clusters.size());

  // 为每个聚类添加球体
  for (const auto& cluster : large_clusters) {
    // 添加位置
    geometry_msgs::Point pt;
    pt.x = cluster.position.x();
    pt.y = cluster.position.y();
    pt.z = cluster.position.z();
    marker.points.push_back(pt);

    // 根据 has_poorly_observed_neighbor 设置颜色
    std_msgs::ColorRGBA color;
    if (cluster.has_poorly_observed_neighbor) {
      // 红色：有 poorly-observed 邻居
      color.r = 1.0f;
      color.g = 0.0f;
      color.b = 0.0f;
    } else {
      // 绿色：全是 well-observed 邻居
      color.r = 0.0f;
      color.g = 1.0f;
      color.b = 0.0f;
    }
    color.a = 0.8f;
    marker.colors.push_back(color);
  }

  return marker;
}

// ==========================================
// 自由空间点云可视化
// ==========================================
pcl::PointCloud<pcl::PointXYZRGB>::Ptr
VisualizationUtils::generateFreeSpaceCloud(
    const std::unordered_map<Eigen::Vector3i, FreeRegion, RegionHash,
                              std::equal_to<Eigen::Vector3i>>& region_map,
    float voxel_size,
    const Eigen::Vector3f& region_origin)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  // 预估点云大小（优化内存分配）
  size_t total_voxels = 0;
  for (const auto& [region_idx, region] : region_map) {
    total_voxels += region.freeVoxelCount();
  }
  cloud->points.reserve(total_voxels);

  // 遍历所有 region，收集所有 free voxels
  for (const auto& [region_idx, region] : region_map) {
    region.forEachFreeVoxel([&](const Eigen::Vector3i& voxel_idx,
                                FreeRegion::VoxelState state) {
      (void)state;
      pcl::PointXYZRGB pt;

      // 计算体素中心位置（从体素索引转换到世界坐标，with origin offset）
      Eigen::Vector3f offset = ((voxel_idx.cast<float>().array() + 0.5f) * voxel_size).matrix();
      Eigen::Vector3f voxel_center = region_origin + offset;
      pt.x = voxel_center.x();
      pt.y = voxel_center.y();
      pt.z = voxel_center.z();

      // 设置颜色为青色 (Cyan) - 方便Debug时与其他点云区分
      pt.r = 0;
      pt.g = 255;
      pt.b = 255;

      cloud->points.push_back(pt);
    });
  }

  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = true;

  return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
VisualizationUtils::generateFrontierCloud(
    const std::unordered_map<Eigen::Vector3i, FreeRegion, RegionHash,
                              std::equal_to<Eigen::Vector3i>>& region_map,
    float voxel_size,
    const Eigen::Vector3f& region_origin)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  // 粗略预估大小：frontier 至多等于 free 数
  size_t total_voxels = 0;
  for (const auto& [region_idx, region] : region_map) {
    total_voxels += region.freeVoxelCount();
  }
  cloud->points.reserve(total_voxels);

  for (const auto& [region_idx, region] : region_map) {
    region.forEachFreeVoxel([&](const Eigen::Vector3i& voxel_idx,
                                FreeRegion::VoxelState state) {
      if (state != FreeRegion::VoxelState::FRONTIER) return;

      pcl::PointXYZRGB pt;
      Eigen::Vector3f offset = ((voxel_idx.cast<float>().array() + 0.5f) * voxel_size).matrix();
      Eigen::Vector3f voxel_center = region_origin + offset;
      pt.x = voxel_center.x();
      pt.y = voxel_center.y();
      pt.z = voxel_center.z();

      // 橙色标记 frontier
      pt.r = 255;
      pt.g = 165;
      pt.b = 0;

      cloud->points.push_back(pt);
    });
  }

  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = true;
  return cloud;
}

// ==========================================
// 未观测但可观测方向可视化
// ==========================================
visualization_msgs::Marker VisualizationUtils::generateUnobservedDirectionsMarker(
    const ClusterInfo& cluster,
    const std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                              std::equal_to<Eigen::Vector3i>>& spatial_hash,
    float voxel_size,
    const std::string& frame_id)
{
  // 初始化SphericalBinning以访问20个方向向量
  gpu_visibility_checker::SphericalBinning::init();

  visualization_msgs::Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = ros::Time::now();
  marker.ns = "unobserved_directions";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.action = visualization_msgs::Marker::ADD;

  // 设置线条宽度 (比可见性连线稍粗)
  marker.scale.x = 0.03; // 3cm 线宽

  // 计算线段长度
  const float line_length = voxel_size * 2.0f;

  // 遍历cluster中的所有poorly_observed_voxels
  for (const auto& voxel_idx : cluster.poorly_observed_voxels) {
    // 查找体素
    auto it = spatial_hash.find(voxel_idx);
    if (it == spatial_hash.end()) {
      continue; // 体素不存在，跳过
    }

    const VoxelCell& voxel = it->second;
    const Eigen::Vector3f& voxel_center = voxel.voxel_center;

    // 遍历20个方向
    for (int dir_idx = 0; dir_idx < 20; ++dir_idx) {
      // 检查是否为未观测但可观测的方向
      // available_direction_mask[dir_idx] == 1 AND observation_direction_mask[dir_idx] == 0
      bool is_available = voxel.available_direction_mask[dir_idx];
      bool is_observed = voxel.observation_direction_mask[dir_idx];

      if (is_available && !is_observed) {
        // 获取该方向的单位向量
        const Eigen::Vector3f& direction =
            gpu_visibility_checker::SphericalBinning::get_face_normal(dir_idx);

        // 计算线段的起点和终点
        Eigen::Vector3f line_end = voxel_center + direction * line_length;

        // 添加起点 (体素中心)
        geometry_msgs::Point start_pt;
        start_pt.x = voxel_center.x();
        start_pt.y = voxel_center.y();
        start_pt.z = voxel_center.z();
        marker.points.push_back(start_pt);

        // 添加终点
        geometry_msgs::Point end_pt;
        end_pt.x = line_end.x();
        end_pt.y = line_end.y();
        end_pt.z = line_end.z();
        marker.points.push_back(end_pt);

        // 设置颜色：黄色表示未观测但可观测的机会
        std_msgs::ColorRGBA color;
        color.r = 1.0f;
        color.g = 1.0f;
        color.b = 0.0f;
        color.a = 0.8f;

        // LINE_LIST 需要为每个顶点设置颜色
        marker.colors.push_back(color);
        marker.colors.push_back(color);
      }
    }
  }

  return marker;
}

// ==========================================
// Voxel Score 文字可视化
// ==========================================
void VisualizationUtils::generateVoxelScoreTextMarkers(
    const std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                              std::equal_to<Eigen::Vector3i>> &spatial_hash,
    float voxel_size,
    float base_score,
    float tex_weight,
    float geo_weight,
    visualization_msgs::MarkerArray &score_markers,
    const std::string& frame_id)
{
  score_markers.markers.clear();

  // Add DELETEALL marker to clear old markers
  visualization_msgs::Marker delete_all;
  delete_all.action = visualization_msgs::Marker::DELETEALL;
  score_markers.markers.push_back(delete_all);

  int marker_id = 0;
  char text_buf[64];

  float text_size = voxel_size * 0.5f;  // Half of voxel_size

  for (const auto &[idx, cell] : spatial_hash) {
    // Calculate target_score
    float threshold_score = base_score + tex_weight * cell.texture_complexity +
                            geo_weight * cell.geometric_complexity;
    float target_score = (cell.max_possible_score > 0.0f)
        ? std::min(cell.max_possible_score, threshold_score)
        : threshold_score;
    if (cell.voxel_center.x() < 3.0f)
      continue;

    // Create base marker template
    visualization_msgs::Marker base_marker;
    base_marker.header.frame_id = frame_id;
    base_marker.header.stamp = ros::Time::now();
    base_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    base_marker.action = visualization_msgs::Marker::ADD;
    base_marker.pose.position.x = cell.voxel_center.x();
    base_marker.pose.position.y = cell.voxel_center.y();
    base_marker.pose.position.z = cell.voxel_center.z();
    base_marker.pose.orientation.w = 1.0;
    base_marker.scale.z = text_size;
    base_marker.color.a = 1.0f;

    // observation_score (white, center)
    visualization_msgs::Marker obs_marker = base_marker;
    obs_marker.ns = "observation_score";
    obs_marker.id = marker_id;
    obs_marker.color.r = 1.0f;
    obs_marker.color.g = 1.0f;
    obs_marker.color.b = 1.0f;
    snprintf(text_buf, sizeof(text_buf), "%.1f", cell.observation_score);
    obs_marker.text = text_buf;
    score_markers.markers.push_back(obs_marker);

    // max_possible_score (cyan, offset +Y)
    visualization_msgs::Marker max_marker = base_marker;
    max_marker.ns = "max_possible_score";
    max_marker.id = marker_id;
    max_marker.pose.position.y += voxel_size * 0.35f;
    max_marker.color.r = 0.0f;
    max_marker.color.g = 1.0f;
    max_marker.color.b = 1.0f;
    snprintf(text_buf, sizeof(text_buf), "%.1f", cell.max_possible_score);
    max_marker.text = text_buf;
    score_markers.markers.push_back(max_marker);

    // target_score (yellow, offset -Y)
    visualization_msgs::Marker target_marker = base_marker;
    target_marker.ns = "target_score";
    target_marker.id = marker_id;
    target_marker.pose.position.y -= voxel_size * 0.35f;
    target_marker.color.r = 1.0f;
    target_marker.color.g = 1.0f;
    target_marker.color.b = 0.0f;
    snprintf(text_buf, sizeof(text_buf), "%.1f", target_score);
    target_marker.text = text_buf;
    score_markers.markers.push_back(target_marker);

    // observation_direction_count (magenta, offset +Z)
    visualization_msgs::Marker dir_count_marker = base_marker;
    dir_count_marker.ns = "obs_direction_count";
    dir_count_marker.id = marker_id;
    dir_count_marker.pose.position.z += voxel_size * 0.35f;
    dir_count_marker.color.r = 1.0f;
    dir_count_marker.color.g = 0.0f;
    dir_count_marker.color.b = 1.0f;
    snprintf(text_buf, sizeof(text_buf), "%zu", cell.observation_direction_mask.count());
    dir_count_marker.text = text_buf;
    score_markers.markers.push_back(dir_count_marker);

    ++marker_id;
  }
}

} // namespace observation_quality
