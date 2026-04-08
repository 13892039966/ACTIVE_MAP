#include <observation_quality_manager/observation_quality_manager.h>
#include <observation_quality_manager/global_planner.h>
#include <observation_quality_manager/visualization_utils.h>
#include <observation_quality_manager/visibility_checker.h>
#include <array>
#include <bitset>
#include <queue>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <omp.h>

namespace {

const std::array<Eigen::Vector3i, 6> kNeighborOffsets6 = {
  Eigen::Vector3i(1, 0, 0), Eigen::Vector3i(-1, 0, 0),
  Eigen::Vector3i(0, 1, 0), Eigen::Vector3i(0, -1, 0),
  Eigen::Vector3i(0, 0, 1), Eigen::Vector3i(0, 0, -1)
};

using VoxelSet = std::unordered_set<Eigen::Vector3i, VoxelHash, std::equal_to<Eigen::Vector3i>>;

GPUVoxel convertCellToGPUVoxel(const VoxelCell& cell) {
  GPUVoxel gpu_voxel{};
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
}

GPUVoxel makeUnknownOccluderVoxel(const Eigen::Vector3i& idx, float voxel_size,
                                   const Eigen::Vector3f& region_origin) {
  GPUVoxel gpu_voxel{};
  Eigen::Vector3f offset = ((idx.cast<float>().array() + 0.5f) * voxel_size).matrix();
  Eigen::Vector3f center = region_origin + offset;
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

  for (int i = 0; i < SUB_MASK_ARRAY_SIZE; ++i) {
    gpu_voxel.sub_masks[i] = 0;
    gpu_voxel.unknown_masks[i] = 0xFFFFFFFFu; // treat unknown as fully unknown
  }

  gpu_voxel.max_possible_score = 0.0f;
  gpu_voxel.padding = 0.0f;
  return gpu_voxel;
}

} // namespace

// ==========================================
// FreeRegion 实现
// ==========================================
void FreeRegion::addCompletedClusterCenter(const Eigen::Vector3i& center) {
  completed_cluster_centers_.push_back(center);
}

const std::vector<Eigen::Vector3i>& FreeRegion::getCompletedClusterCenters() const {
  return completed_cluster_centers_;
}

void FreeRegion::computeClusterCenters(const std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                                                                  std::equal_to<Eigen::Vector3i>>& spatial_hash,
                                        const FreeRegion::WatershedDebugParams* debug_params) {
  clusters_.clear();

  // 定义哈希集合类型，方便复用
  using VoxelSet = std::unordered_set<Eigen::Vector3i, VoxelHash, std::equal_to<Eigen::Vector3i>>;

  // 1. 收集所有 Free Voxel（不再排除障碍物膨胀层）
  VoxelSet free_voxels;
  const size_t free_count = freeVoxelCount();
  free_voxels.reserve(free_count);

  forEachFreeVoxel([&](const Eigen::Vector3i& voxel, VoxelState /*state*/) {
    free_voxels.insert(voxel);
  });

  // 调试信息
  ROS_INFO_THROTTLE(5.0, "[FreeRegion] Total free voxels: %zu", free_count);

  // 6邻域定义 (用于 BFS 连通性)
  static const Eigen::Vector3i neighbors_6[6] = {
    {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}
  };

  // 2. BFS 构建聚类
  // 所有 free voxel 都参与聚类
  while (!free_voxels.empty()) {
    Eigen::Vector3i start_voxel = *free_voxels.begin();

    // 初始化 Cluster 数据
    VoxelSet component;
    std::vector<Eigen::Vector3i> component_vec; // 用于快速遍历求 Closest Center
    std::queue<Eigen::Vector3i> q;
    VoxelSet cluster_poorly_observed_set; // 当前 Cluster 关联的 poorly observed 障碍物
    Eigen::Vector3f centroid_sum = Eigen::Vector3f::Zero(); // 累加坐标求中心

    // 启动 BFS
    free_voxels.erase(free_voxels.begin());
    q.push(start_voxel);
    component.insert(start_voxel);
    component_vec.push_back(start_voxel);
    centroid_sum += start_voxel.cast<float>();

    while (!q.empty()) {
      Eigen::Vector3i current = q.front();
      q.pop();

      // Check A: 6邻域 扩展连通分量
      for (const auto& offset : neighbors_6) {
        Eigen::Vector3i neighbor = current + offset;
        // 如果邻居在 free_voxels 中，说明未访问且是 free voxel
        if (free_voxels.erase(neighbor)) {
          q.push(neighbor);
          component.insert(neighbor);
          component_vec.push_back(neighbor);
          centroid_sum += neighbor.cast<float>();
        }
      }

      // Check B: 检查当前 Voxel 的 26 邻域是否有 poorly_observed 的障碍物
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            if (dx == 0 && dy == 0 && dz == 0) continue;

            Eigen::Vector3i n = current + Eigen::Vector3i(dx, dy, dz);

            // 直接在 spatial_hash 中查找障碍物
            auto it = spatial_hash.find(n);
            if (it != spatial_hash.end() && !it->second.well_observed) {
              cluster_poorly_observed_set.insert(n);
            }
          }
        }
      }
    }

    if (component.empty()) continue;

    // 3. 计算中心点
    // 3.1 几何中心
    Eigen::Vector3f centroid = centroid_sum / static_cast<float>(component.size());

    // 3.2 找到距离几何中心最近的实际 Voxel
    // 因为已经维护了 component_vec，这里遍历比遍历 unordered_set 快
    Eigen::Vector3i center = component_vec[0];
    float min_dist_sq = (center.cast<float>() - centroid).squaredNorm();

    for (const auto& voxel : component_vec) {
      float dist_sq = (voxel.cast<float>() - centroid).squaredNorm();
      if (dist_sq < min_dist_sq) {
        min_dist_sq = dist_sq;
        center = voxel;
      }
    }

    // 4. 保存结果
    clusters_.push_back({
      center,
      static_cast<int>(component.size()),
      !cluster_poorly_observed_set.empty(),
      std::move(cluster_poorly_observed_set),
      std::move(component)
    });
  }

  // ===== 分水岭二次划分 =====
  // 对大型cluster进行二次划分，将跨门的房间分开
  std::vector<FreeCluster> final_clusters;
  final_clusters.reserve(clusters_.size());

  // 辅助函数：为单个cluster生成外边界线段
  auto generateClusterBoundary = [this](FreeCluster& cluster) {
    if (cluster.free_voxels.empty()) return;

    // 投影到XY平面并生成边界
    auto proj_data = projectClusterToXY(cluster);
    if (proj_data.binary_image.empty()) return;

    // 扫描二值图像边缘
    auto toVoxelCoord = [&](float lx, float ly) -> Eigen::Vector2f {
      return Eigen::Vector2f(proj_data.aabb_min.x() + lx, proj_data.aabb_min.y() + ly);
    };

    for (int y = 0; y < proj_data.binary_image.rows; ++y) {
      for (int x = 0; x < proj_data.binary_image.cols; ++x) {
        if (proj_data.binary_image.at<uchar>(y, x) == 0) continue;

        // 检查右边是否为边界
        if (x + 1 >= proj_data.binary_image.cols || proj_data.binary_image.at<uchar>(y, x + 1) == 0) {
          Eigen::Vector2f p1 = toVoxelCoord(x + 1.0f, y);
          Eigen::Vector2f p2 = toVoxelCoord(x + 1.0f, y + 1.0f);
          cluster.boundary_segments_2d.push_back({p1, p2});
        }
        // 检查下边是否为边界
        if (y + 1 >= proj_data.binary_image.rows || proj_data.binary_image.at<uchar>(y + 1, x) == 0) {
          Eigen::Vector2f p1 = toVoxelCoord(x, y + 1.0f);
          Eigen::Vector2f p2 = toVoxelCoord(x + 1.0f, y + 1.0f);
          cluster.boundary_segments_2d.push_back({p1, p2});
        }
        // 检查左边是否为边界
        if (x == 0 || proj_data.binary_image.at<uchar>(y, x - 1) == 0) {
          Eigen::Vector2f p1 = toVoxelCoord(x, y);
          Eigen::Vector2f p2 = toVoxelCoord(x, y + 1.0f);
          cluster.boundary_segments_2d.push_back({p1, p2});
        }
        // 检查上边是否为边界
        if (y == 0 || proj_data.binary_image.at<uchar>(y - 1, x) == 0) {
          Eigen::Vector2f p1 = toVoxelCoord(x, y);
          Eigen::Vector2f p2 = toVoxelCoord(x + 1.0f, y);
          cluster.boundary_segments_2d.push_back({p1, p2});
        }
      }
    }
  };

  for (auto& cluster : clusters_) {
    // 跳过小型cluster，但仍生成边界用于可视化
    if (cluster.voxel_count < kMinClusterSizeForSubdivision) {
      generateClusterBoundary(cluster);
      final_clusters.push_back(std::move(cluster));
      continue;
    }

    // Step 1: 投影到XY平面
    auto proj_data = projectClusterToXY(cluster);
    if (proj_data.binary_image.empty()) {
      generateClusterBoundary(cluster);
      final_clusters.push_back(std::move(cluster));
      continue;
    }

    // Step 2: 计算ESDF (距离变换)
    cv::Mat esdf_input = proj_data.binary_image.clone();
    // 标记四周边缘为0（障碍物），防止边界问题
    if (esdf_input.rows > 2 && esdf_input.cols > 2) {
      esdf_input.row(0) = 0;
      esdf_input.row(esdf_input.rows - 1) = 0;
      esdf_input.col(0) = 0;
      esdf_input.col(esdf_input.cols - 1) = 0;
    }
    cv::distanceTransform(esdf_input, proj_data.distance_transform, cv::DIST_L2, cv::DIST_MASK_PRECISE);

    // Step 3: 爬山法分水岭分割 (不需要预先生成seeds)
    auto labels = applyWatershed(proj_data.binary_image,
                                  proj_data.distance_transform);
    ROS_INFO("label size: %zu x %zu", labels.cols, labels.rows);

    // 调试：统计唯一区域数量
    std::set<int> unique_labels;
    for (int y = 0; y < labels.rows; ++y) {
      for (int x = 0; x < labels.cols; ++x) {
        int label = labels.at<int>(y, x);
        if (label > 0) {
          unique_labels.insert(label);
        }
      }
    }
    ROS_INFO("Unique regions found: %zu", unique_labels.size());

    if (labels.empty()) {
      generateClusterBoundary(cluster);
      final_clusters.push_back(std::move(cluster));
      continue;
    }

    // Step 4: 先创建子clusters，再基于实际结果进行可视化
    auto sub_clusters = subdivideCluster(cluster, labels, proj_data);

    // 可视化：基于sub_clusters的实际结果
    if (debug_params && debug_params->isValid()) {

      Eigen::Vector3i odom_voxel_idx = ((debug_params->odom_position - debug_params->region_origin) /
                                        debug_params->voxel_size).array().floor().cast<int>();
      auto odom_key = std::make_pair(odom_voxel_idx.x() - proj_data.aabb_min.x(),
                                     odom_voxel_idx.y() - proj_data.aabb_min.y());

      if (proj_data.xy_to_voxels.count(odom_key)) {
        // 收集 (local_x, local_y) -> region_id 映射
        std::map<std::pair<int, int>, int> local_xy_to_region;
        int max_region_id = -1;
        for (const auto& sub : sub_clusters) {
          for (const auto& [voxel, region_id] : sub.voxel_to_region_id) {
            int local_x = voxel.x() - proj_data.aabb_min.x();
            int local_y = voxel.y() - proj_data.aabb_min.y();
            auto key = std::make_pair(local_x, local_y);
            if (local_xy_to_region.find(key) == local_xy_to_region.end()) {
              local_xy_to_region[key] = region_id;
            }
            max_region_id = std::max(max_region_id, region_id);
          }
        }

        // HSV转RGB辅助函数
        auto hsvToBgr = [](float hue) -> cv::Vec3b {
          cv::Mat3b hsv(1, 1, cv::Vec3b(hue / 2, 200, 200));
          cv::Mat3b bgr;
          cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
          return bgr.at<cv::Vec3b>(0, 0);
        };

        // 生成颜色
        std::vector<cv::Vec3b> colors;
        colors.push_back(cv::Vec3b(255, 255, 255));  // 白色：障碍物
        colors.push_back(cv::Vec3b(128, 128, 128));   // 灰色：未分配区域
        for (int i = 0; i <= max_region_id; ++i) {
          colors.push_back(hsvToBgr(360.0f * i / std::max(1, max_region_id + 1)));
        }

        // 填充图像
        cv::Mat labels_rgb = cv::Mat::zeros(labels.size(), CV_8UC3);
        for (int y = 0; y < labels.rows; ++y) {
          for (int x = 0; x < labels.cols; ++x) {
            auto it = local_xy_to_region.find(std::make_pair(x, y));
            if (it != local_xy_to_region.end()) {
              labels_rgb.at<cv::Vec3b>(y, x) = colors[it->second + 2];
            } else {
              labels_rgb.at<cv::Vec3b>(y, x) = (proj_data.binary_image.at<uchar>(y, x) > 0)
                                                ? colors[1] : colors[0];
            }
          }
        }

        cv_bridge::CvImage labels_cv_img;
        labels_cv_img.header.stamp = ros::Time::now();
        labels_cv_img.encoding = sensor_msgs::image_encodings::BGR8;
        labels_cv_img.image = labels_rgb;
        debug_params->labels_pub->publish(labels_cv_img.toImageMsg());
      }
    }

    for (auto& sub : sub_clusters) {
      final_clusters.push_back(std::move(sub));
    }
  }

  clusters_ = std::move(final_clusters);
  // ROS_INFO("[FreeRegion] After watershed: %zu clusters", clusters_.size());
}
// ==========================================
// ObservationQualityManager 实现
// ==========================================

void ObservationQualityManager::init(ros::NodeHandle &nh,
                                     LIOInterface::Ptr &lio_interface,
                                     TopoGraph::Ptr graph) {
  nh_ = nh;
  graph_ = graph;
  lidar_map_interface_ = lio_interface;

  // Initialize map boundaries from lidar interface
  map_min_bd_ =
      lidar_map_interface_->lp_->global_map_min_boundary_.cast<float>();
  map_max_bd_ =
      lidar_map_interface_->lp_->global_map_max_boundary_.cast<float>();

  // Initialize region origin (use map lower boundary as coordinate system origin)
  region_origin_ = map_min_bd_;
  ROS_INFO("ObservationQualityManager: region_origin = [%.3f, %.3f, %.3f]",
           region_origin_.x(), region_origin_.y(), region_origin_.z());

  // Initialize SphericalBinning for observation direction tracking
  SphericalBinning::init();

  // Read voxel size from ROS parameters
  nh_.param("observation_quality/voxel_size", voxel_size_, 0.2f);
  ROS_INFO("ObservationQualityManager: voxel_size = %.3f", voxel_size_);

  // Read max ray length from ROS parameters
  nh_.param("lidar_perception/max_ray_length", max_ray_length_, 15.0f);
  ROS_INFO("ObservationQualityManager: max_ray_length = %.3f", max_ray_length_);

  // Camera FOV (用于可见性裁剪)
  float horizontal_fov_deg = 90.0f;
  float vertical_fov_deg = 60.0f;
  nh_.param("observation_quality/horizontal_fov_deg", horizontal_fov_deg, horizontal_fov_deg);
  nh_.param("observation_quality/vertical_fov_deg", vertical_fov_deg, vertical_fov_deg);
  horizontal_fov_rad_ = horizontal_fov_deg * static_cast<float>(M_PI) / 180.0f;
  vertical_fov_rad_ = vertical_fov_deg * static_cast<float>(M_PI) / 180.0f;
  ROS_INFO("ObservationQualityManager: FOV h=%.1f deg v=%.1f deg", horizontal_fov_deg, vertical_fov_deg);

  // Read well_observed threshold coefficients
  nh_.param("observation_quality/well_observed_base_score", well_observed_base_score_, 0.5f);
  nh_.param("observation_quality/well_observed_texture_weight", well_observed_texture_weight_, 0.5f);
  nh_.param("observation_quality/well_observed_geo_weight", well_observed_geo_weight_, 0.5f);
  ROS_INFO("ObservationQualityManager: well_observed thresholds - base=%.2f tex_weight=%.2f geo_weight=%.2f",
           well_observed_base_score_, well_observed_texture_weight_, well_observed_geo_weight_);

  raycast_vis_pub_ = nh_.advertise<visualization_msgs::Marker>("/oqm/raycast_poor_lines", 1);
  cluster_boundary_vis_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/oqm/watershed_regions", 1);
  raycast_viewpoints_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/oqm/raycast_viewpoints", 1);
  raycast_viewpoints_axis_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/oqm/raycast_viewpoints_axis", 10);
  gpu_voxels_vis_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/oqm/gpu_voxels", 1);

  // 分水岭调试图像发布器
  watershed_labels_pub_ = nh_.advertise<sensor_msgs::Image>("/oqm/watershed_labels", 1);
  watershed_distance_pub_ = nh_.advertise<sensor_msgs::Image>("/oqm/watershed_distance", 1);
  watershed_binary_pub_ = nh_.advertise<sensor_msgs::Image>("/oqm/watershed_binary", 1);

  // Initialize GlobalPlanner
  global_planner_ = std::make_shared<GlobalPlanner>(nh_, voxel_size_, spatial_hash_, map_min_bd_, lidar_map_interface_, graph_, region_map_);
  ROS_INFO("ObservationQualityManager: GlobalPlanner initialized");
}

void ObservationQualityManager::pos2idx(const PointType &pt,
                                        Eigen::Vector3i &idx) {
  // Convert point position to voxel index (with origin offset)
  Eigen::Vector3f pos = pt.getVector3fMap();
  idx = ((pos - region_origin_) / voxel_size_).array().floor().cast<int>();
}

void ObservationQualityManager::pos2idx(const Eigen::Vector3f &pt,
                                        Eigen::Vector3i &idx) {
  // Convert vector position to voxel index (with origin offset)
  idx = ((pt - region_origin_) / voxel_size_).array().floor().cast<int>();
}



void ObservationQualityManager::idx2pos(const Eigen::Vector3i &idx,
                                        PointType &pt) const{
  // Convert voxel index to voxel center position (with origin offset)
  Eigen::Vector3f offset = ((idx.cast<float>().array() + 0.5f) * voxel_size_).matrix();
  Eigen::Vector3f center = region_origin_ + offset;
  pt.x = center.x();
  pt.y = center.y();
  pt.z = center.z();
}

bool ObservationQualityManager::isInBox(const PointType &pt) const {
  return lidar_map_interface_->IsInBox(pt);
}

bool ObservationQualityManager::isInBox(const Eigen::Vector3f &pt) const {
  return lidar_map_interface_->IsInBox(pt);
}

void ObservationQualityManager::addPointCloudOnlyGeometry(
    const pcl::PointCloud<PointType>::Ptr &cloud,
    const Eigen::Vector3f &odom_position) {
  // 按体素索引对点云分组（用于几何更新，仅 box 内的点）
  std::unordered_map<Eigen::Vector3i, std::vector<Eigen::Vector3f>, VoxelHash,
                     std::equal_to<Eigen::Vector3i>>
      voxel_groups;

  // 用于 raycast 的体素（包括 box 外的点，因为需要标记 box 内对应的自由空间）
  std::unordered_map<Eigen::Vector3i, Eigen::Vector3f, VoxelHash,
                     std::equal_to<Eigen::Vector3i>>
      raycast_voxels;

  for (const auto &pt : cloud->points) {
    // 计算点到 odom 位置的距离
    Eigen::Vector3f pt_pos = pt.getVector3fMap();
    if (!std::isfinite(pt_pos.x()) || !std::isfinite(pt_pos.y()) ||
        !std::isfinite(pt_pos.z())) {
      continue;
    }
    float distance = (pt_pos - odom_position).norm();
    Eigen::Vector3i idx;

    // raycast的时候截断处理
    if (distance >= max_ray_length_) {
      Eigen::Vector3f pt_truncation =
          odom_position +
          (pt_pos - odom_position).normalized() * max_ray_length_;
      pos2idx(pt_truncation, idx);
      if ((idx.array().abs() > 1000000).any()) {
        continue;
      }
      if (raycast_voxels.find(idx) == raycast_voxels.end()) {
        raycast_voxels[idx] = pt_pos;
      }
      continue;
    }

    // 所有有效点都用于 raycast（每个体素只保留一个代表点）
    pos2idx(pt_pos, idx);
    if ((idx.array().abs() > 1000000).any()) {
      continue;
    }
    if (raycast_voxels.find(idx) == raycast_voxels.end()) {
      raycast_voxels[idx] = pt_pos;
    }

    // box 内的点用于几何更新
    if (isInBox(pt_pos)) {
      voxel_groups[idx].push_back(pt_pos);
    }
  }

  // 批量更新每个体素（仅 box 内）
  for (auto &[idx, points] : voxel_groups) {
    auto &cell = spatial_hash_[idx];

    // 如果是新体素，初始化体素中心
    if (cell.geo_cnt == 0) {
      PointType center_pt;
      idx2pos(idx, center_pt);
      cell.voxel_center = center_pt.getVector3fMap();
    }

    // 批量添加点并更新几何信息
    cell.addGeometryPoints(points, voxel_size_, odom_position,
                          well_observed_base_score_, well_observed_texture_weight_, well_observed_geo_weight_); // 已经设置好哈希表了
    addOccupiedVoxelToRegion(idx); // 只需要设置状态表
  }
  ros::Time start = ros::Time::now();

  // Raycast: 对每个体素只投射一次，记录自由空间
  // 使用 raycast_voxels（包括 box 外的点），因为 box 外的点可以用于标记 box 内的自由空间
  // 将修改的 regions 累积到成员变量中，延迟到 planGlobalTSPPath 时批量更新聚类
  VoxelSet unknown_to_free;
  VoxelSet frontier_to_free;
  for (const auto &[idx, pt_pos] : raycast_voxels) {
    raycastFreeSpace(odom_position, pt_pos, unknown_to_free, frontier_to_free);
  }
  processNewlyFreedVoxels(unknown_to_free, frontier_to_free);
  ros::Time end = ros::Time::now();
  // ROS_INFO("Raycast time: %.3f ms", (end - start).toSec() * 1000.0);
}

void ObservationQualityManager::addPointCloudWithTexture(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
    const Eigen::Vector3f &odom_position) {
  // 按体素索引对点云分组
  std::unordered_map<Eigen::Vector3i,
                     std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>,
                     VoxelHash, std::equal_to<Eigen::Vector3i>>
      voxel_groups;

  for (const auto &pt : cloud->points) {
    // 计算点到 odom 位置的距离
    Eigen::Vector3f pt_pos(pt.x, pt.y, pt.z);
    if (!std::isfinite(pt_pos.x()) || !std::isfinite(pt_pos.y()) ||
        !std::isfinite(pt_pos.z())) {
      continue;
    }
    if (!isInBox(pt_pos))
      continue;
    float distance = (pt_pos - odom_position).norm();

    // 只处理距离小于 max_ray_length 的点
    if (distance >= max_ray_length_) {
      continue;
    }

    Eigen::Vector3i idx;
    pos2idx(pt_pos, idx);
    if ((idx.array().abs() > 1000000).any()) {
      continue;
    }

    // 提取RGB颜色并归一化到[0,1]
    Eigen::Vector3f color(pt.r / 255.0f, pt.g / 255.0f, pt.b / 255.0f);
    voxel_groups[idx].emplace_back(pt_pos, color);
  }

  // 批量更新每个体素
  for (auto &[idx, points_with_colors] : voxel_groups) {
    auto &cell = spatial_hash_[idx];

    // 如果是新体素，初始化体素中心
    if (cell.geo_cnt == 0 && cell.tex_cnt == 0) {
      PointType center_pt;
      idx2pos(idx, center_pt);
      cell.voxel_center = center_pt.getVector3fMap();
    }

    // 批量添加点并更新纹理复杂度
    cell.addTexturePoints(points_with_colors, voxel_size_, odom_position,
                         well_observed_base_score_, well_observed_texture_weight_, well_observed_geo_weight_);
  }
}

Eigen::Vector3i ObservationQualityManager::voxelToRegion(
    const Eigen::Vector3i& voxel_idx) const {
  // 每个region是 6m / 0.2m = 30 个体素
  constexpr int VOXELS_PER_REGION = 30;

  // 使用floor division处理负数索引
  auto floor_div = [](int a, int b) -> int {
    return (a >= 0) ? (a / b) : ((a - b + 1) / b);
  };

  return Eigen::Vector3i(
    floor_div(voxel_idx.x(), VOXELS_PER_REGION),
    floor_div(voxel_idx.y(), VOXELS_PER_REGION),
    floor_div(voxel_idx.z(), VOXELS_PER_REGION)
  );
}

bool ObservationQualityManager::tryGetVoxelState(
    const Eigen::Vector3i& voxel_idx,
    FreeRegion::VoxelState& state) const {
  Eigen::Vector3i region_idx = voxelToRegion(voxel_idx);
  auto region_it = region_map_.find(region_idx);
  if (region_it == region_map_.end()) {
    return false;
  }
  return region_it->second.getVoxelState(voxel_idx, state);
}

bool ObservationQualityManager::hasUnknownNeighbor(
    const Eigen::Vector3i& voxel_idx) const {
  for (const auto& offset : kNeighborOffsets6) {
    Eigen::Vector3i neighbor = voxel_idx + offset;
    PointType nbr_pos;
    idx2pos(neighbor, nbr_pos);
    if(!isInBox(nbr_pos)) {
      continue;
    }
    FreeRegion::VoxelState neighbor_state;
    if (tryGetVoxelState(neighbor, neighbor_state)) {
      continue; // 已知free/frontier/occupied
    }
    if (spatial_hash_.find(neighbor) != spatial_hash_.end()) {
      continue; // 已知占据
    }
    return true;
  }
  return false;
}


void ObservationQualityManager::addFreeVoxelToRegion(
    const Eigen::Vector3i& voxel_idx,
    FreeRegion::VoxelState state) {
  // 计算region索引
  Eigen::Vector3i region_idx = voxelToRegion(voxel_idx);

  // 获取或创建region
  FreeRegion& region = region_map_[region_idx];

  // 添加体素（O(1)操作）
  region.addFreeVoxel(voxel_idx, state);
}

void ObservationQualityManager::addOccupiedVoxelToRegion(
    const Eigen::Vector3i& voxel_idx) {
  // 判断现在是不是occ, 如果是就不用管了
  FreeRegion::VoxelState current_state;
  if (tryGetVoxelState(voxel_idx, current_state) && current_state == FreeRegion::VoxelState::OCCUPIED) {
    return;
  }
  // 判断周围是否有frontier, 如果有的话判断frontier是否还是frontier ->更新状态
  Eigen::Vector3i region_idx = voxelToRegion(voxel_idx);
  FreeRegion& region = region_map_[region_idx];
  region.addOccupiedVoxel(voxel_idx);

  // 占据信息更新后，周围的 frontier 可能变为内部 free
  for (const auto& offset : kNeighborOffsets6) {
    Eigen::Vector3i neighbor = voxel_idx + offset;
    Eigen::Vector3i n_region_idx = voxelToRegion(neighbor);
    auto region_it = region_map_.find(n_region_idx);
    if (region_it == region_map_.end()) {
      continue;
    }
    FreeRegion::VoxelState neighbor_state;
    if (!region_it->second.getVoxelState(neighbor, neighbor_state)) {
      continue;
    }
    if (neighbor_state != FreeRegion::VoxelState::FRONTIER) {
      continue;
    }
    if (!hasUnknownNeighbor(neighbor)) {
      region_it->second.addFreeVoxel(neighbor, FreeRegion::VoxelState::FREE);
    }
  }
}

void ObservationQualityManager::raycastFreeSpace(
    const Eigen::Vector3f& start, const Eigen::Vector3f& end,
    std::unordered_set<Eigen::Vector3i, VoxelHash,
                       std::equal_to<Eigen::Vector3i>>& unknown_to_free,
    std::unordered_set<Eigen::Vector3i, VoxelHash,
                       std::equal_to<Eigen::Vector3i>>& frontier_to_free) {
  // DDA算法 (Amanatides-Woo) - 反向 raycast
  // 从障碍物点往回 ray 到 odom
  // start = odom, end = obstacle
  // 参考: Fast-Planner raycast.cpp

  // 检查距离并截断
  Eigen::Vector3f direction = end - start;
  float dist_sq = direction.squaredNorm();
  if (dist_sq < 0.01f) return; // 距离太小

  // 如果超过最大距离，截断到最大距离
  Eigen::Vector3f actual_end = end;
  if (dist_sq > max_ray_length_ * max_ray_length_) {
    float dist = std::sqrt(dist_sq);
    actual_end = start + direction * (max_ray_length_ / dist);
  }

  // 反向 raycast: 从 actual_end (障碍物) 到 start (odom)
  Eigen::Vector3f ray_start = actual_end;  // 障碍物位置
  Eigen::Vector3f ray_end = start;         // odom 位置

  // 转换到体素坐标系 (with origin offset)
  Eigen::Vector3f start_v = (ray_start - region_origin_) / voxel_size_;
  Eigen::Vector3f end_v = (ray_end - region_origin_) / voxel_size_;

  // 体素索引（整数部分）
  int x = static_cast<int>(std::floor(start_v.x()));
  int y = static_cast<int>(std::floor(start_v.y()));
  int z = static_cast<int>(std::floor(start_v.z()));

  int endX = static_cast<int>(std::floor(end_v.x()));
  int endY = static_cast<int>(std::floor(end_v.y()));
  int endZ = static_cast<int>(std::floor(end_v.z()));

  // 方向（整数索引差）
  int dx = endX - x;
  int dy = endY - y;
  int dz = endZ - z;

  // 步进方向
  int stepX = (dx == 0) ? 0 : (dx > 0 ? 1 : -1);
  int stepY = (dy == 0) ? 0 : (dy > 0 ? 1 : -1);
  int stepZ = (dz == 0) ? 0 : (dz > 0 ? 1 : -1);

  // 避免无限循环
  if (stepX == 0 && stepY == 0 && stepZ == 0) return;

  // intbound函数（使用std::function支持递归）
  std::function<float(float, float)> intbound = [&](float s, float ds) -> float {
    if (ds < 0.0f) {
      return intbound(-s, -ds);
    } else {
      s = fmod(fmod(s, 1.0f) + 1.0f, 1.0f); // mod(s, 1)
      return (1.0f - s) / ds;
    }
  };

  // 初始化tMax和tDelta
  float tMaxX = (dx != 0) ? intbound(start_v.x(), static_cast<float>(dx)) : 1e30f;
  float tMaxY = (dy != 0) ? intbound(start_v.y(), static_cast<float>(dy)) : 1e30f;
  float tMaxZ = (dz != 0) ? intbound(start_v.z(), static_cast<float>(dz)) : 1e30f;

  float tDeltaX = (dx != 0) ? static_cast<float>(stepX) / static_cast<float>(dx) : 1e30f;
  float tDeltaY = (dy != 0) ? static_cast<float>(stepY) / static_cast<float>(dy) : 1e30f;
  float tDeltaZ = (dz != 0) ? static_cast<float>(stepZ) / static_cast<float>(dz) : 1e30f;

  auto isOccupied = [&](const Eigen::Vector3i& voxel_idx) -> bool {
    if (spatial_hash_.find(voxel_idx) != spatial_hash_.end()) {
      return true;
    }
    FreeRegion::VoxelState state;
    if (tryGetVoxelState(voxel_idx, state)) {
      return state == FreeRegion::VoxelState::OCCUPIED;
    }
    return false;
  };

  auto markFree = [&](const Eigen::Vector3i& voxel_idx) {
    // Box检查：跳过box外的体素
    PointType voxel_pos;
    idx2pos(voxel_idx, voxel_pos);
    if (!isInBox(voxel_pos)) {
      return;
    }

    if (isOccupied(voxel_idx)) {
      return;
    }
    FreeRegion::VoxelState existing_state;
    bool is_known = tryGetVoxelState(voxel_idx, existing_state);
    if (is_known && existing_state == FreeRegion::VoxelState::FREE) {
      return; // 已经是 free 了，跳过
    }

    if (!is_known) {
      unknown_to_free.insert(voxel_idx);
    } else if (existing_state == FreeRegion::VoxelState::FRONTIER) {
      frontier_to_free.insert(voxel_idx);
    }

    addFreeVoxelToRegion(voxel_idx, FreeRegion::VoxelState::FREE);
  };

  int max_iter = 200; // 防止无限循环

  // DDA遍历
  while (max_iter-- > 0) {
    Eigen::Vector3i current(x, y, z);
    markFree(current);
    // 检查是否到达终点
    if (x == endX && y == endY && z == endZ) break;

    // DDA步进
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
}

void ObservationQualityManager::processNewlyFreedVoxels(
    const std::unordered_set<Eigen::Vector3i, VoxelHash,
                             std::equal_to<Eigen::Vector3i>>& unknown_to_free,
    const std::unordered_set<Eigen::Vector3i, VoxelHash,
                             std::equal_to<Eigen::Vector3i>>& frontier_to_free) {
  VoxelSet newly_free = unknown_to_free;
  newly_free.insert(frontier_to_free.begin(), frontier_to_free.end());

  // 对每个新 free 体素，决定自身是否是 frontier，同时刷新邻居的 frontier 状态
  for (const auto& voxel_idx : newly_free) {
    // Box检查：跳过box外的体素
    PointType voxel_pos;
    idx2pos(voxel_idx, voxel_pos);
    if (!isInBox(voxel_pos)) {
      continue;
    }

    const bool is_frontier = hasUnknownNeighbor(voxel_idx);
    addFreeVoxelToRegion(voxel_idx, is_frontier ? FreeRegion::VoxelState::FRONTIER
                                                : FreeRegion::VoxelState::FREE);

    // 邻居如果原来是 frontier，但周围未知被填满，则转成 free
    for (const auto& offset : kNeighborOffsets6) {
      Eigen::Vector3i neighbor = voxel_idx + offset;

      // Box检查：跳过box外的邻居
      PointType neighbor_pos;
      idx2pos(neighbor, neighbor_pos);
      if (!isInBox(neighbor_pos)) {
        continue;
      }

      Eigen::Vector3i region_idx = voxelToRegion(neighbor);
      auto region_it = region_map_.find(region_idx);
      if (region_it == region_map_.end()) {
        continue;
      }

      FreeRegion::VoxelState neighbor_state;
      if (!region_it->second.getVoxelState(neighbor, neighbor_state)) {
        continue;
      }
      if (neighbor_state != FreeRegion::VoxelState::FRONTIER) {
        continue;
      }

      if (!hasUnknownNeighbor(neighbor)) {
        region_it->second.addFreeVoxel(neighbor, FreeRegion::VoxelState::FREE);
      }
    }
  }
}

void ObservationQualityManager::updateClustersForRegions(
    const std::unordered_set<Eigen::Vector3i, RegionHash,
                             std::equal_to<Eigen::Vector3i>>& modified_regions) {
  // 只更新发生变化的 regions 的聚类
  // 将 unordered_set 转换为 vector 以便使用 OpenMP 并行化
  std::vector<Eigen::Vector3i> region_indices(modified_regions.begin(), modified_regions.end());

  #pragma omp parallel for schedule(dynamic) num_threads(4)
  for (size_t i = 0; i < region_indices.size(); ++i) {
    const auto& region_idx = region_indices[i];
    auto it = region_map_.find(region_idx);
    if (it != region_map_.end()) {
      ROS_INFO("region idx: %d, %d, %d", region_idx.x(), region_idx.y(), region_idx.z());
      // 计算该 region 的聚类中心
      // spatial_hash_ 只读访问，线程安全
      // it->second 每个线程访问不同的 region线程安全 
      // 准备调试可视化参数（如果odom数据可用）
      FreeRegion::WatershedDebugParams debug_params;
      if (!odom_poses.empty()) {
        debug_params.odom_position = odom_poses.back().position;
        debug_params.region_origin = region_origin_;
        debug_params.voxel_size = voxel_size_;
        debug_params.labels_pub = &watershed_labels_pub_;
        debug_params.distance_pub = &watershed_distance_pub_;
        debug_params.binary_pub = &watershed_binary_pub_;
        it->second.computeClusterCenters(spatial_hash_, &debug_params);
      } else {
        it->second.computeClusterCenters(spatial_hash_);
      }
    }
  }

  ROS_INFO_THROTTLE(2.0, "Updated clusters for %zu modified regions", modified_regions.size());

  // // 发布cluster边界可视化
  publishClusterBoundaryVisualization();
}

void ObservationQualityManager::raycastPoorlyObservedToOdoms(
    const std::vector<Eigen::Vector3f>& additional_viewpoints,
    const std::vector<Eigen::Quaternionf>& additional_orientations,
    const Eigen::Vector3i* target_region_for_additional) {

  // 如果有目标region且不在modified_regions_set中，临时添加它
  bool need_remove_target_region = false;
  if (target_region_for_additional != nullptr &&
      !additional_viewpoints.empty() &&
      modified_regions_set.find(*target_region_for_additional) == modified_regions_set.end()) {
    // 检查该region是否存在于region_map_中
    if (region_map_.find(*target_region_for_additional) != region_map_.end()) {
      modified_regions_set.insert(*target_region_for_additional);
      need_remove_target_region = true;
    }
  }

  if (modified_regions_set.empty()) {
    return;
  }

  constexpr int VOXELS_PER_REGION = 30;

  for (const auto& region_idx : modified_regions_set) {
    auto region_it = region_map_.find(region_idx);
    if (region_it == region_map_.end()) {
      ROS_WARN_THROTTLE(2.0,
                        "ObservationQualityManager: Region (%d, %d, %d) not found for GPU raycast",
                        region_idx.x(), region_idx.y(), region_idx.z());
      continue;
    }

    FreeRegion& region = region_it->second;

    // 收集目标：观测质量差的占据体素
    std::vector<Eigen::Vector3i> target_voxels;
    std::vector<Eigen::Vector3f> target_positions;
    std::vector<GPUVoxel> gpu_voxels;
    std::vector<GPUVoxel> occluder_voxels;
    const size_t occupied_count = region.occupiedVoxelCount();
    target_voxels.reserve(occupied_count);
    target_positions.reserve(occupied_count);
    gpu_voxels.reserve(occupied_count);
    occluder_voxels.reserve(occupied_count);

    std::unordered_set<Eigen::Vector3i, VoxelHash, std::equal_to<Eigen::Vector3i>> added_voxels;
    region.forEachOccupiedVoxel([&](const Eigen::Vector3i& occ_idx) {
      auto cell_it = spatial_hash_.find(occ_idx);
      assert(cell_it != spatial_hash_.end());
      // if (!cell_it->second.well_observed) {
        target_voxels.push_back(occ_idx);
        target_positions.push_back(cell_it->second.voxel_center);
        gpu_voxels.push_back(convertCellToGPUVoxel(cell_it->second)); // targets first
        added_voxels.insert(occ_idx);
      // } 
      // else {
      //   if (added_voxels.find(occ_idx) != added_voxels.end()) {
      //     return;
      //   }
      //   occluder_voxels.push_back(convertCellToGPUVoxel(cell_it->second));
      //   added_voxels.insert(occ_idx);
      // }
    });

    if (target_voxels.empty()) {
      continue;
    }

    // 将已知占据（非目标）追加为遮挡，保持 target 在前
    gpu_voxels.insert(gpu_voxels.end(), occluder_voxels.begin(), occluder_voxels.end());

    // 将 unknown 体素视为占据体素，作为遮挡加入 GPU 列表
    Eigen::Vector3i region_min = region_idx * VOXELS_PER_REGION;
    Eigen::Vector3i region_max =
        region_min + Eigen::Vector3i(VOXELS_PER_REGION - 1,
                                     VOXELS_PER_REGION - 1,
                                     VOXELS_PER_REGION - 1);

    for (int x = region_min.x(); x <= region_max.x(); ++x) {
      for (int y = region_min.y(); y <= region_max.y(); ++y) {
        for (int z = region_min.z(); z <= region_max.z(); ++z) {
          Eigen::Vector3i idx(x, y, z);
          if (added_voxels.find(idx) != added_voxels.end()) {
            continue;
          }
          FreeRegion::VoxelState state;
          if (region.getVoxelState(idx, state)) {
            if (FreeRegion::isFreeState(state)) {
              continue; // 已知自由/前沿
            } else {
              continue; // 已知占据已处理
            }
          }
          // gpu_voxels.push_back(makeUnknownOccluderVoxel(idx, voxel_size_, region_origin_));
          // added_voxels.insert(idx);
        }
      }
    }

    // 可视化：发布gpu_voxels点云（0.2m分辨率）
    // 只有当前odom属于这个region时才发布
    bool should_visualize = false;
    if (!odom_poses.empty() && gpu_voxels_vis_pub_.getNumSubscribers() > 0) {
      // 计算当前odom位置所属的region
      Eigen::Vector3i odom_voxel_idx;
      pos2idx(odom_poses.back().position, odom_voxel_idx);
      Eigen::Vector3i odom_region_idx = voxelToRegion(odom_voxel_idx);

      // 检查odom是否在当前region内
      if (odom_region_idx == region_idx) {
        should_visualize = true;
      }
    }

    if (should_visualize) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
      cloud->points.reserve(gpu_voxels.size());

      for (const auto& gpu_voxel : gpu_voxels) {
        pcl::PointXYZRGB pt;
        pt.x = gpu_voxel.center.x;
        pt.y = gpu_voxel.center.y;
        pt.z = gpu_voxel.center.z;

        // 根据类型设置颜色
        if (gpu_voxel.is_frontier) {
          // frontier: 绿色
          pt.r = 0; pt.g = 255; pt.b = 0;
        } else if (gpu_voxel.well_observed) {
          // well_observed: 蓝色
          pt.r = 0; pt.g = 0; pt.b = 255;
        } else {
          // poorly_observed: 红色
          pt.r = 255; pt.g = 0; pt.b = 0;
        }

        cloud->points.push_back(pt);
      }

      cloud->width = cloud->points.size();
      cloud->height = 1;
      cloud->is_dense = true;

      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(*cloud, cloud_msg);
      cloud_msg.header.frame_id = "world";
      cloud_msg.header.stamp = ros::Time::now();
      gpu_voxels_vis_pub_.publish(cloud_msg);

      ROS_INFO("Published GPU voxels: %zu points for region (%d, %d, %d)",
               cloud->points.size(), region_idx.x(), region_idx.y(), region_idx.z());
    }

    if (gpu_voxels.empty()) {
      ROS_WARN("ObservationQualityManager: No GPU voxels prepared for region (%d, %d, %d)",
               region_idx.x(), region_idx.y(), region_idx.z());
      continue;
    }

    // 准备视点（从 region 的 observation_odom_idx 获取）
    std::vector<Eigen::Vector3f> viewpoints;
    std::vector<Eigen::Quaternionf> orientations;

    // 预估容量
    size_t additional_count = 0;
    if (target_region_for_additional != nullptr &&
        region_idx == *target_region_for_additional) {
      additional_count = additional_viewpoints.size();
    }
    viewpoints.reserve(region.observation_odom_idx.size() + additional_count);
    orientations.reserve(region.observation_odom_idx.size() + additional_count);

    // 添加历史odom位置
    for (int odom_idx : region.observation_odom_idx) {
      Eigen::Quaternionf q = odom_poses[odom_idx].orientation;
      q.normalize();
      viewpoints.push_back(odom_poses[odom_idx].position);
      orientations.push_back(q);
    }

    // 只对目标region添加额外的viewpoints
    if (target_region_for_additional != nullptr &&
        region_idx == *target_region_for_additional) {
      for (size_t i = 0; i < additional_viewpoints.size(); ++i) {
        viewpoints.push_back(additional_viewpoints[i]);
        if (i < additional_orientations.size()) {
          Eigen::Quaternionf q = additional_orientations[i];
          q.normalize();
          orientations.push_back(q);
        } else {
          orientations.push_back(Eigen::Quaternionf::Identity());
        }
      }
    }

    if (viewpoints.empty()) {
      continue;
    }

    // 可视化：发布viewpoints点云
    if (raycast_viewpoints_pub_.getNumSubscribers() > 0) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
      cloud->points.reserve(viewpoints.size());

      for (const auto& viewpoint : viewpoints) {
        cloud->points.emplace_back(viewpoint.x(), viewpoint.y(), viewpoint.z());
      }

      cloud->width = cloud->points.size();
      cloud->height = 1;
      cloud->is_dense = true;

      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(*cloud, cloud_msg);
      cloud_msg.header.frame_id = "world";
      cloud_msg.header.stamp = ros::Time::now();
      raycast_viewpoints_pub_.publish(cloud_msg);

      ROS_INFO("Published raycast viewpoints: %zu points for region (%d, %d, %d)",
               cloud->points.size(), region_idx.x(), region_idx.y(), region_idx.z());
    }

    VisibilityCSR csr_result = observation_quality::checkVisibilityGPUWithFOV(
        viewpoints,
        orientations,
        target_positions,
        gpu_voxels,
        voxel_size_,
        region_origin_,  // 使用与CPU端一致的地图原点
        horizontal_fov_rad_,
        vertical_fov_rad_);

    // Debug可视化：使用axis显示viewpoint和orientation
    if (raycast_viewpoints_axis_pub_.getNumSubscribers() > 0) {
      visualization_msgs::MarkerArray marker_array;
      visualization_msgs::Marker del_marker;

      int marker_id = 0;
      del_marker.action = visualization_msgs::Marker::DELETEALL;
      marker_array.markers.push_back(del_marker);

      // 计算历史odom数量，用于区分普通viewpoints和additional_viewpoints
      size_t num_odom_viewpoints = region.observation_odom_idx.size();

      for (size_t i = 0; i < viewpoints.size(); ++i) {
        const Eigen::Vector3f& vp = viewpoints[i];
        const Eigen::Quaternionf& quat = orientations[i];

        // 将四元数转换为旋转矩阵
        Eigen::Matrix3f rot = quat.toRotationMatrix();

        // 判断是否为additional_viewpoints
        bool is_additional = (i >= num_odom_viewpoints);

        // 创建三个axis marker
        visualization_msgs::Marker x_axis, y_axis, z_axis;

        // 通用设置 - X轴
        x_axis.header.frame_id = "world";
        x_axis.header.stamp = ros::Time::now();
        x_axis.ns = "viewpoint_axes";
        x_axis.id = marker_id++;
        x_axis.type = visualization_msgs::Marker::ARROW;
        x_axis.action = visualization_msgs::Marker::ADD;
        x_axis.pose.orientation.w = 1.0;

        // 通用设置 - Y轴
        y_axis.header.frame_id = "world";
        y_axis.header.stamp = ros::Time::now();
        y_axis.ns = "viewpoint_axes";
        y_axis.id = marker_id++;
        y_axis.type = visualization_msgs::Marker::ARROW;
        y_axis.action = visualization_msgs::Marker::ADD;
        y_axis.pose.orientation.w = 1.0;

        // 通用设置 - Z轴
        z_axis.header.frame_id = "world";
        z_axis.header.stamp = ros::Time::now();
        z_axis.ns = "viewpoint_axes";
        z_axis.id = marker_id++;
        z_axis.type = visualization_msgs::Marker::ARROW;
        z_axis.action = visualization_msgs::Marker::ADD;
        z_axis.pose.orientation.w = 1.0;

        // 设置起点为viewpoint位置
        geometry_msgs::Point start;
        start.x = vp.x();
        start.y = vp.y();
        start.z = vp.z();

        // 根据是否为additional_viewpoint设置不同的样式
        float axis_length = 0.5f;  // 0.5米长的轴
        float line_width = is_additional ? 0.1f : 0.05f;  // additional使用双倍粗细

        // X轴 (普通:红色 / additional:金色)
        x_axis.points.push_back(start);
        geometry_msgs::Point end_x;
        Eigen::Vector3f x_dir = rot.col(0) * axis_length;
        end_x.x = vp.x() + x_dir.x();
        end_x.y = vp.y() + x_dir.y();
        end_x.z = vp.z() + x_dir.z();
        x_axis.points.push_back(end_x);
        if (is_additional) {
          // 金色
          x_axis.color.r = 1.0f; x_axis.color.g = 0.84f; x_axis.color.b = 0.0f; x_axis.color.a = 1.0f;
        } else {
          // 红色
          x_axis.color.r = 1.0f; x_axis.color.g = 0.0f; x_axis.color.b = 0.0f; x_axis.color.a = 1.0f;
        }
        x_axis.scale.x = line_width;
        marker_array.markers.push_back(x_axis);

        // Y轴 (绿色 - 两种类型相同)
        y_axis.points.push_back(start);
        geometry_msgs::Point end_y;
        Eigen::Vector3f y_dir = rot.col(1) * axis_length;
        end_y.x = vp.x() + y_dir.x();
        end_y.y = vp.y() + y_dir.y();
        end_y.z = vp.z() + y_dir.z();
        y_axis.points.push_back(end_y);
        y_axis.color.r = 0.0f; y_axis.color.g = 1.0f; y_axis.color.b = 0.0f; y_axis.color.a = 1.0f;
        y_axis.scale.x = line_width;
        marker_array.markers.push_back(y_axis);

        // Z轴 (蓝色 - 两种类型相同)
        z_axis.points.push_back(start);
        geometry_msgs::Point end_z;
        Eigen::Vector3f z_dir = rot.col(2) * axis_length;
        end_z.x = vp.x() + z_dir.x();
        end_z.y = vp.y() + z_dir.y();
        end_z.z = vp.z() + z_dir.z();
        z_axis.points.push_back(end_z);
        z_axis.color.r = 0.0f; z_axis.color.g = 0.0f; z_axis.color.b = 1.0f; z_axis.color.a = 1.0f;
        z_axis.scale.x = line_width;
        marker_array.markers.push_back(z_axis);

        // 添加索引文本marker
        visualization_msgs::Marker text_marker;
        text_marker.header.frame_id = "world";
        text_marker.header.stamp = ros::Time::now();
        text_marker.ns = "viewpoint_axes";
        text_marker.id = marker_id++;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;

        // 设置文本位置（在viewpoint上方0.3米）
        text_marker.pose.position.x = vp.x();
        text_marker.pose.position.y = vp.y();
        text_marker.pose.position.z = vp.z() + 0.3f;
        text_marker.pose.orientation.w = 1.0;

        // 设置文本内容（显示索引）
        text_marker.text = std::to_string(i);

        // 设置文本样式
        text_marker.scale.z = is_additional ? 0.15f : 0.1f;  // additional使用更大的字体
        if (is_additional) {
          // additional使用金色
          text_marker.color.r = 1.0f; text_marker.color.g = 0.84f; text_marker.color.b = 0.0f;
        } else {
          // 普通使用白色
          text_marker.color.r = 1.0f; text_marker.color.g = 1.0f; text_marker.color.b = 1.0f;
        }
        text_marker.color.a = 1.0f;

        marker_array.markers.push_back(text_marker);
      }

      // 发布MarkerArray，一次性发送所有marker
      if (!marker_array.markers.empty()) {
        raycast_viewpoints_axis_pub_.publish(marker_array);
        size_t num_additional = (viewpoints.size() > num_odom_viewpoints) ?
                                (viewpoints.size() - num_odom_viewpoints) : 0;
        ROS_INFO("Published raycast viewpoint axes: %zu markers (%zu odom + %zu additional) for region (%d, %d, %d)",
                 marker_array.markers.size(), num_odom_viewpoints, num_additional,
                 region_idx.x(), region_idx.y(), region_idx.z());
      }
    }

    if (csr_result.viewpoint_offsets.empty()) {
      ROS_WARN_THROTTLE(2.0,
                        "ObservationQualityManager: GPU raycast returned empty result for region (%d, %d, %d)",
                        region_idx.x(), region_idx.y(), region_idx.z());
      continue;
    }

    // 可视化：将所有additional_viewpoints与其可见的targets连线
    if (raycast_vis_pub_.getNumSubscribers() > 0 &&
        csr_result.viewpoint_offsets.size() >= 2) {
      size_t csr_viewpoint_count = csr_result.viewpoint_offsets.size() - 1;
      size_t num_odom_viewpoints = region.observation_odom_idx.size();

      visualization_msgs::Marker line_marker;
      line_marker.header.frame_id = "world";
      line_marker.header.stamp = ros::Time::now();
      line_marker.ns = "additional_viewpoints_visibility";
      line_marker.id = 0;
      line_marker.type = visualization_msgs::Marker::LINE_LIST;
      line_marker.action = visualization_msgs::Marker::ADD;
      line_marker.scale.x = 0.03;  // 线宽
      line_marker.color.r = 1.0;
      line_marker.color.g = 1.0;
      line_marker.color.b = 0.0;
      line_marker.color.a = 0.9;
      line_marker.pose.orientation.w = 1.0;

      // 遍历所有additional_viewpoints
      for (size_t vp_idx = num_odom_viewpoints; vp_idx < viewpoints.size(); ++vp_idx) {
        if (vp_idx >= csr_viewpoint_count) {
          break;  // 超出CSR大小
        }

        int start = csr_result.viewpoint_offsets[vp_idx];
        int end = csr_result.viewpoint_offsets[vp_idx + 1];

        geometry_msgs::Point vp_point;
        vp_point.x = viewpoints[vp_idx].x();
        vp_point.y = viewpoints[vp_idx].y();
        vp_point.z = viewpoints[vp_idx].z();

        for (int offset = start; offset < end; ++offset) {
          int target_idx = csr_result.viewpoint_to_targets[offset];
          if (target_idx < 0 ||
              target_idx >= static_cast<int>(target_positions.size())) {
            continue;
          }

          geometry_msgs::Point target_point;
          target_point.x = target_positions[target_idx].x();
          target_point.y = target_positions[target_idx].y();
          target_point.z = target_positions[target_idx].z();

          line_marker.points.push_back(vp_point);
          line_marker.points.push_back(target_point);
        }
      }

      if (!line_marker.points.empty()) {
        raycast_vis_pub_.publish(line_marker);
        ROS_INFO("Published visibility lines for %zu additional viewpoints",
                 viewpoints.size() - num_odom_viewpoints);
      }
    }

    // 更新可见 target 的观测方向掩码
    int updated_count = 0;
    size_t viewpoint_count = csr_result.viewpoint_offsets.size() > 1
                                 ? csr_result.viewpoint_offsets.size() - 1
                                 : 0;

    for (size_t vp_idx = 0; vp_idx < viewpoint_count; ++vp_idx) {
      int start = csr_result.viewpoint_offsets[vp_idx];
      int end = csr_result.viewpoint_offsets[vp_idx + 1];
      for (int offset = start; offset < end; ++offset) {
        int target_idx = csr_result.viewpoint_to_targets[offset];
        if (target_idx < 0 || target_idx >= static_cast<int>(target_voxels.size())) {
          continue;
        }

        auto cell_it = spatial_hash_.find(target_voxels[target_idx]);
        if (cell_it == spatial_hash_.end()) {
          continue;
        }

        VoxelCell& cell = cell_it->second;
        Eigen::Vector3f view_dir = viewpoints[vp_idx] - cell.voxel_center;
        int obs_bin_idx = SphericalBinning::get_bin_index(view_dir);
        if (obs_bin_idx < 0) {
          continue;
        }

        // Only update if this direction hasn't been observed yet
        if (!cell.observation_direction_mask[obs_bin_idx]) {
          cell.observation_direction_mask[obs_bin_idx] = true;
          // cell.updateObservationScore(well_observed_base_score_, well_observed_texture_weight_, well_observed_geo_weight_);  // Recalculates score, then calls updateWellObserved()
          updated_count++;
        }
      }
    }

    ROS_INFO_THROTTLE(2.0,
                      "ObservationQualityManager: %d/%zu poorly-observed voxels visible from %zu odoms in region (%d, %d, %d)",
                      updated_count, target_voxels.size(), viewpoints.size(),
                      region_idx.x(), region_idx.y(), region_idx.z());
  }
}
void ObservationQualityManager::updatePendingClusters() {
  if (modified_regions_set.empty()) {
    ROS_INFO("updatePendingClusters: No modified regions to update.");
    return;
  }

  ros::Time cluster_start = ros::Time::now();
  updateClustersForRegions(modified_regions_set);
  ROS_INFO("Batch cluster update: %zu regions in %.3f ms",
           modified_regions_set.size(),
           (ros::Time::now() - cluster_start).toSec() * 1000.0);
}
