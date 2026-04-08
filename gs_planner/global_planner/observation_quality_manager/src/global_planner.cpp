#include <observation_quality_manager/global_planner.h>
#include <observation_quality_manager/visualization_utils.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <fstream>

GlobalPlanner::GlobalPlanner(ros::NodeHandle& nh, float voxel_size,
                             std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                                                std::equal_to<Eigen::Vector3i>>& spatial_hash,
                             const Eigen::Vector3f& map_min_bd,
                             LIOInterface::Ptr lidar_map_interface,
                             TopoGraph::Ptr graph,
                             std::unordered_map<Eigen::Vector3i, FreeRegion, RegionHash,
                                                std::equal_to<Eigen::Vector3i>>& region_map)
    : nh_(nh), voxel_size_(voxel_size), spatial_hash_(spatial_hash),
      map_min_bd_(map_min_bd), region_origin_(map_min_bd),
      lidar_map_interface_(lidar_map_interface),
      graph_(graph), region_map_ptr_(&region_map) {

  // 读取参数
  nh_.param("global_planning/tsp_dir", tsp_dir_, std::string("/tmp"));
  nh_.param("global_planning/max_clusters", max_clusters_, 20);
  nh_.param("global_planning/min_voxel_count", min_voxel_count_, 15);
  nh_.param("global_planning/viewpoint_sample_resolution", viewpoint_sample_resolution_, 0.4f);
  nh_.param("global_planning/top_n_viewpoints", top_n_viewpoints_, 50);
  nh_.param("global_planning/min_distance_to_obstacle", min_distance_to_obstacle_, 0.4f);
  nh_.param("global_planning/treat_unknown_as_occupied", treat_unknown_as_occupied_, false);

  // Read well_observed threshold coefficients (与 OQM 保持一致)
  nh_.param("observation_quality/well_observed_base_score", well_observed_base_score_, 0.5f);
  nh_.param("observation_quality/well_observed_texture_weight", well_observed_texture_weight_, 0.5f);
  nh_.param("observation_quality/well_observed_geo_weight", well_observed_geo_weight_, 0.5f);

  // 相机FOV参数 (弧度)
  float h_fov_deg, v_fov_deg;
  nh_.param("global_planning/horizontal_fov_deg", h_fov_deg, 60.0f);
  nh_.param("global_planning/vertical_fov_deg", v_fov_deg, 45.0f);
  h_fov_ = h_fov_deg * M_PI / 180.0f;
  v_fov_ = v_fov_deg * M_PI / 180.0f;
  nh_.param("MIN_PITCH", pitch_min_, -60.0f * static_cast<float>(M_PI) / 180.0f);
  nh_.param("MAX_PITCH", pitch_max_, 60.0f * static_cast<float>(M_PI) / 180.0f);

  // 创建可视化发布器
  path_vis_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "/global_planner/tsp_path", 10);
  viewpoint_vis_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "/global_planner/viewpoints", 10);
  viewpoint_visibility_lines_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "/global_planner/viewpoint_visibility_lines", 10);
  frontier_vis_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      "/global_planner/frontiers", 1);
  unobserved_directions_pub_ = nh_.advertise<visualization_msgs::Marker>(
      "/global_planner/unobserved_directions", 1);
  target_aabb_pub_ = nh_.advertise<visualization_msgs::Marker>(
      "/global_planner/target_aabb", 1);
  forced_cluster_free_voxels_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      "/global_planner/forced_cluster_free_voxels", 1);
  completed_centers_vis_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "/global_planner/completed_centers", 1);
  cluster_reachability_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "/global_planner/cluster_reachability", 1);
  planning_count_pub_ = nh_.advertise<visualization_msgs::Marker>(
      "/global_planner/planning_count", 1);
  poorly_observed_count_text_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "/global_planner/poorly_observed_count_text", 1);
  filtered_target_voxels_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      "/global_planner/filtered_target_voxels", 1);
  filtered_observation_directions_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "/global_planner/filtered_observation_directions", 1);
  well_observed_voxels_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      "/global_planner/well_observed_voxels", 1);
  sampled_viewpoints_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      "/global_planner/sampled_viewpoints", 1);
  gpu_voxel_points_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      "/global_planner/gpu_voxel_subpoints", 1);
  gpu_voxel_centers_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      "/global_planner/gpu_voxel_centers", 1);

  ROS_INFO("GlobalPlanner initialized:");
  ROS_INFO("  tsp_dir: %s", tsp_dir_.c_str());
  ROS_INFO("  max_clusters: %d", max_clusters_);
  ROS_INFO("  min_voxel_count: %d", min_voxel_count_);
  ROS_INFO("  viewpoint_sample_resolution: %.2f m", viewpoint_sample_resolution_);
  ROS_INFO("  top_n_viewpoints: %d", top_n_viewpoints_);
  ROS_INFO("  min_distance_to_obstacle: %.2f m", min_distance_to_obstacle_);
  ROS_INFO("  horizontal_fov: %.1f deg", h_fov_deg);
  ROS_INFO("  vertical_fov: %.1f deg", v_fov_deg);
  ROS_INFO("  pitch range: [%.1f, %.1f] deg",
           pitch_min_ * 180.0f / M_PI, pitch_max_ * 180.0f / M_PI);
  ROS_INFO("  treat_unknown_as_occupied: %s", treat_unknown_as_occupied_ ? "true" : "false");

  // 创建TSP参数文件 (类似 FastExplorationManager)
  std::ofstream par_file(tsp_dir_ + "/global_tsp.par");
  par_file << "PROBLEM_FILE = " << tsp_dir_ << "/global_tsp.tsp\n";
  par_file << "GAIN23 = NO\n";
  par_file << "MOVE_TYPE = 2\n";
  par_file << "OUTPUT_TOUR_FILE = " << tsp_dir_ << "/global_tsp.txt\n";
  par_file << "RUNS = 10\n";
  par_file.close();
}

// ==========================================
// 检查前一个cluster观测是否完成的辅助函数
// ==========================================
int GlobalPlanner::checkPreviousClusterCompletion(
    const std::vector<ClusterInfo>& selected_clusters) {

  // Early exit: no previous state
  if (!has_previous_planning_state_) {
    return -1;  // First planning call or state was cleared
  }

  // Early exit: empty previous free voxels
  assert(!previous_cluster_free_voxels_.empty() && "Previous cluster free voxels should not be empty");

  ROS_INFO("Checking for cluster overlap with %zu previous free voxels in region [%d,%d,%d]",
           previous_cluster_free_voxels_.size(),
           previous_target_region_idx_.x(),
           previous_target_region_idx_.y(),
           previous_target_region_idx_.z());

  // Build lookup set from previous cluster's free voxels (O(N))
  std::unordered_set<Eigen::Vector3i, VoxelHash, std::equal_to<Eigen::Vector3i>>
      previous_free_voxels_set(previous_cluster_free_voxels_.begin(),
                               previous_cluster_free_voxels_.end());

  // Find cluster with maximum overlap based on cluster_free_voxels
  int best_cluster_idx = -1;
  int max_overlap = 0;

  for (size_t i = 0; i < selected_clusters.size(); ++i) {
    // OPTIMIZATION: Skip clusters not in the previous target region
    if (selected_clusters[i].region_idx != previous_target_region_idx_) {
      continue;
    }

    // Count overlap using cluster_free_voxels
    int overlap = 0;
    for (const auto& v : selected_clusters[i].cluster_free_voxels) {
      if (previous_free_voxels_set.count(v)) {
        ++overlap;
      }
    }

    if (overlap > max_overlap) {
      max_overlap = overlap;
      best_cluster_idx = static_cast<int>(i);
    }
  }

  // Return result
  if (best_cluster_idx >= 0 && max_overlap > 0) {
    ROS_INFO("Found cluster %d in same region [%d,%d,%d] with %d overlapping free voxels",
             best_cluster_idx,
             selected_clusters[best_cluster_idx].region_idx.x(),
             selected_clusters[best_cluster_idx].region_idx.y(),
             selected_clusters[best_cluster_idx].region_idx.z(),
             max_overlap);

    // Publish forced cluster's free voxels as point cloud for visualization
    if (forced_cluster_free_voxels_pub_.getNumSubscribers() > 0) {
      const std::vector<Eigen::Vector3i>& free_voxels = selected_clusters[best_cluster_idx].cluster_free_voxels;
      if (!free_voxels.empty()) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        cloud->points.reserve(free_voxels.size());

        for (const auto& voxel_idx : free_voxels) {
          Eigen::Vector3f position = voxelIdxToPosition(voxel_idx);
          cloud->points.emplace_back(position.x(), position.y(), position.z());
        }

        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = true;

        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header.frame_id = "world";
        cloud_msg.header.stamp = ros::Time::now();
        forced_cluster_free_voxels_pub_.publish(cloud_msg);

        ROS_INFO("Published forced cluster free voxels: %zu points", cloud->points.size());
      }
    }

    return best_cluster_idx;
  }

  // No overlap - cluster observation completed
  ROS_INFO("Cluster observation completed in region [%d,%d,%d], marking center [%d,%d,%d] as completed",
           previous_target_region_idx_.x(),
           previous_target_region_idx_.y(),
           previous_target_region_idx_.z(),
           previous_cluster_center_voxel_idx_.x(),
           previous_cluster_center_voxel_idx_.y(),
           previous_cluster_center_voxel_idx_.z());

  // Mark the previous cluster center as completed in the region
  auto it = region_map_ptr_->find(previous_target_region_idx_);
  if (it != region_map_ptr_->end()) {
    it->second.addCompletedClusterCenter(previous_cluster_center_voxel_idx_);
  }

  has_previous_planning_state_ = false;
  previous_global_path_.clear();
  return -1;
}

// ==========================================
// 检查是否存在 forced cluster 并进行可视化
// ==========================================
bool GlobalPlanner::checkForcedClusterAndVisualize(int& forced_cluster_idx) {
  // Check for forced cluster (continuation mode)
  forced_cluster_idx = checkPreviousClusterCompletion(selected_clusters_);
  bool has_forced_cluster = (forced_cluster_idx >= 0);

  if (!has_forced_cluster) {
    return false;
  }

  // Visualize forced cluster AABB
  ROS_INFO("GlobalPlanner: Continuation mode - forcing cluster %d, planning from forced cluster center",
           forced_cluster_idx);

  const ClusterInfo& forced_cluster = selected_clusters_[forced_cluster_idx];

  if (forced_cluster.poorly_observed_voxels.empty()) {
    ROS_WARN("Forced cluster %d has no poorly_observed_voxels, skipping AABB visualization",
             forced_cluster_idx);
    return true;
  }

  // Calculate AABB from forced cluster's poorly_observed_voxels
  Eigen::Vector3i aabb_min_voxel = forced_cluster.poorly_observed_voxels[0];
  Eigen::Vector3i aabb_max_voxel = forced_cluster.poorly_observed_voxels[0];
  for (const auto& v : forced_cluster.poorly_observed_voxels) {
    aabb_min_voxel = aabb_min_voxel.cwiseMin(v);
    aabb_max_voxel = aabb_max_voxel.cwiseMax(v);
  }

  // Visualize target AABB as a light blue box
  visualization_msgs::Marker aabb_marker;
  aabb_marker.header.frame_id = "world";
  aabb_marker.header.stamp = ros::Time::now();
  aabb_marker.ns = "target_aabb";
  aabb_marker.id = 0;
  aabb_marker.type = visualization_msgs::Marker::CUBE;
  aabb_marker.action = visualization_msgs::Marker::ADD;

  // Calculate AABB center and size in world coordinates (with region_origin offset)
  Eigen::Vector3f aabb_center_voxel =
      (aabb_min_voxel.cast<float>() + aabb_max_voxel.cast<float>()) * 0.5f;
  Eigen::Vector3f aabb_center_world = region_origin_ + aabb_center_voxel * voxel_size_;
  Eigen::Vector3i aabb_size_voxel = aabb_max_voxel - aabb_min_voxel + Eigen::Vector3i::Ones();
  Eigen::Vector3f aabb_size_world = aabb_size_voxel.cast<float>() * voxel_size_;

  aabb_marker.pose.position.x = aabb_center_world.x();
  aabb_marker.pose.position.y = aabb_center_world.y();
  aabb_marker.pose.position.z = aabb_center_world.z();
  aabb_marker.pose.orientation.w = 1.0;

  aabb_marker.scale.x = aabb_size_world.x();
  aabb_marker.scale.y = aabb_size_world.y();
  aabb_marker.scale.z = aabb_size_world.z();

  // Light blue color with transparency
  aabb_marker.color.r = 0.5f;
  aabb_marker.color.g = 0.8f;
  aabb_marker.color.b = 1.0f;
  aabb_marker.color.a = 0.3f;

  target_aabb_pub_.publish(aabb_marker);
  ROS_INFO("Published target AABB visualization: center [%.2f, %.2f, %.2f], size [%.2f, %.2f, %.2f]",
           aabb_center_world.x(), aabb_center_world.y(), aabb_center_world.z(),
           aabb_size_world.x(), aabb_size_world.y(), aabb_size_world.z());

  return true;
}

// ==========================================
// 有 forced cluster 时计算 TSP 顺序
// ==========================================
std::vector<int> GlobalPlanner::computeClusterOrderWithForcedCluster(
    int forced_cluster_idx) {

  // Handle single cluster case
  if (selected_clusters_.size() == 1) {
    // 保存本轮顺序
    previous_cluster_order_positions_.clear();
    previous_cluster_order_positions_.push_back(selected_clusters_[0].position);
    has_previous_cluster_order_ = true;
    return {0};
  }

  // ===== CONTINUATION MODE: SOP starts from forced_cluster center =====
  // Build distance matrix with forced_cluster as starting point (node 0)
  // Node 0 = forced_start position, Node 1 = forced_cluster, Node 2..N = other clusters

  const Eigen::Vector3f& forced_start = selected_clusters_[forced_cluster_idx].position;

  // Create temporary cluster list: [forced_cluster, ...other_clusters]
  // Track index mapping: reordered_to_original_idx[i] = original index of reordered_clusters[i]
  std::vector<ClusterInfo> reordered_clusters;
  std::vector<int> reordered_to_original_idx;

  reordered_clusters.push_back(selected_clusters_[forced_cluster_idx]);
  reordered_to_original_idx.push_back(forced_cluster_idx);

  for (size_t i = 0; i < selected_clusters_.size(); ++i) {
    if ((int)i != forced_cluster_idx) {
      reordered_clusters.push_back(selected_clusters_[i]);
      reordered_to_original_idx.push_back(i);
    }
  }

  Eigen::MatrixXd cost_mat;
  std::vector<bool> reachable;
  if (!buildDistanceMatrix(forced_start, reordered_clusters, cost_mat, reachable)) {
    ROS_ERROR("GlobalPlanner: Failed to build distance matrix from forced cluster");
    return {};
  }

  // Filter unreachable clusters from reordered_clusters
  // reachable[i] indicates if reordered_clusters[i] is reachable
  std::vector<ClusterInfo> reachable_reordered;
  std::vector<int> reordered_indices;  // Indices in reordered_clusters
  for (size_t i = 0; i < reordered_clusters.size(); ++i) {
    if (reachable[i]) {
      reachable_reordered.push_back(reordered_clusters[i]);
      reordered_indices.push_back(i);
    }
  }

  if (reachable_reordered.empty()) {
    ROS_ERROR("GlobalPlanner: All clusters unreachable from forced cluster");
    return {};
  }

  ROS_INFO("GlobalPlanner: %zu/%zu clusters reachable from forced cluster",
           reachable_reordered.size(), reordered_clusters.size());

  // Extract sub-matrix if needed
  if (reachable_reordered.size() < reordered_clusters.size()) {
    int new_dim = reachable_reordered.size() + 1;
    Eigen::MatrixXd new_cost_mat(new_dim, new_dim);
    new_cost_mat.setZero();

    // Node 0 (forced_start) remains
    new_cost_mat(0, 0) = cost_mat(0, 0);

    // Reachable clusters
    for (size_t i = 0; i < reordered_indices.size(); ++i) {
      int old_idx = reordered_indices[i] + 1;  // +1 because node 0 is forced_start
      int new_idx = i + 1;
      new_cost_mat(0, new_idx) = cost_mat(0, old_idx);
      new_cost_mat(new_idx, 0) = cost_mat(old_idx, 0);
    }

    for (size_t i = 0; i < reordered_indices.size(); ++i) {
      for (size_t j = 0; j < reordered_indices.size(); ++j) {
        int old_i = reordered_indices[i] + 1;
        int old_j = reordered_indices[j] + 1;
        new_cost_mat(i + 1, j + 1) = cost_mat(old_i, old_j);
      }
    }

    cost_mat = new_cost_mat;
  }

  // ===== Convert to SOP cost matrix with precedence constraints =====
  int N = cost_mat.rows();
  Eigen::MatrixXi sop_cost_matrix(N, N);

  // Copy costs (scale to integers)
  const double SCALE = 1000.0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      sop_cost_matrix(i, j) = static_cast<int>(cost_mat(i, j) * SCALE);
    }
  }

  // ===== Set precedence constraints =====
  // Node 0 = start position, Node 1 = forced_cluster
  // forced_cluster (Node 1) must come before all other clusters (Node 2..N-1)
  // Precedence: sop_cost_matrix(j, i) = -1 means node i must be visited before node j
  for (int j = 2; j < N; ++j) {
    sop_cost_matrix(j, 1) = -1;  // Node 1 must come before Node j
  }

  // ===== Apply previous order constraints if available =====
  if (has_previous_cluster_order_ && !previous_cluster_order_positions_.empty()) {
    ROS_INFO("GlobalPlanner: Applying previous cluster order constraints (%zu clusters)",
             previous_cluster_order_positions_.size());

    // Create position to current reachable index mapping
    // reachable_reordered contains the clusters that are currently reachable
    // We need to match previous positions to current clusters
    const float POSITION_MATCH_THRESHOLD = 0.5f;  // 0.5m threshold for position matching

    std::vector<int> prev_to_current_node;  // Maps previous order index to current SOP node index
    for (const auto& prev_pos : previous_cluster_order_positions_) {
      int matched_node = -1;
      float min_dist = POSITION_MATCH_THRESHOLD;

      // Search in reachable_reordered (indices 0 to reachable_reordered.size()-1)
      // These correspond to SOP nodes 1 to N-1
      for (size_t i = 0; i < reachable_reordered.size(); ++i) {
        float dist = (reachable_reordered[i].position - prev_pos).norm();
        if (dist < min_dist) {
          min_dist = dist;
          matched_node = static_cast<int>(i + 1);  // SOP node index (+1 for start node)
        }
      }

      if (matched_node >= 0) {
        prev_to_current_node.push_back(matched_node);
      }
    }

    ROS_INFO("GlobalPlanner: Matched %zu/%zu previous clusters to current clusters",
             prev_to_current_node.size(), previous_cluster_order_positions_.size());

    // Set CHAIN constraints only (adjacent pairs)
    // This is O(n) instead of O(n²), which is much more efficient for the solver
    // prev_to_current_node[i] must come before prev_to_current_node[i+1]
    // IMPORTANT: If all clusters are matched (forming a complete chain), skip the last constraint
    // to avoid over-constraining the solution space, which may cause SOP to hang
    size_t num_constraints = prev_to_current_node.size() > 1 ? prev_to_current_node.size() - 1 : 0;
    bool all_matched = (prev_to_current_node.size() == reachable_reordered.size());
    if (all_matched && num_constraints > 0) {
      num_constraints--;  // Skip the last constraint to leave some freedom
      ROS_INFO("GlobalPlanner: All clusters matched, reducing constraints to avoid over-constraining");
    }

    for (size_t i = 0; i < num_constraints; ++i) {
      int node_before = prev_to_current_node[i];
      int node_after = prev_to_current_node[i + 1];
      // Node node_before must be visited before node_after
      sop_cost_matrix(node_after, node_before) = -1;
    }

    ROS_INFO("GlobalPlanner: Set %zu chain constraints for order preservation",
             num_constraints);
  }

  // ===== Solve SOP =====
  std::vector<int> sop_path;
  int total_cost = solveSOP(sop_cost_matrix, sop_path);
  ROS_INFO("GlobalPlanner: SOP solved with cost %d, path size %zu", total_cost, sop_path.size());

  if (sop_path.empty()) {
    ROS_ERROR("GlobalPlanner: SOP solver failed");
    return {};
  }

  // ===== Map SOP path back to original cluster indices =====
  // SOP path format: [node0, node1, node2, ...]
  // Node 0 = start position (not a cluster)
  // Node 1..N-1 = reachable_reordered[0..N-2]
  std::vector<int> cluster_order;
  std::vector<Eigen::Vector3f> order_positions;  // For saving to previous_cluster_order_positions_

  for (int sop_node : sop_path) {
    if (sop_node == 0) {
      continue;  // Skip start position node
    }

    // Map: SOP node → reachable_reordered index → reordered_clusters index → original index
    int reachable_idx = sop_node - 1;
    if (reachable_idx >= 0 && reachable_idx < static_cast<int>(reordered_indices.size())) {
      int reordered_idx = reordered_indices[reachable_idx];
      if (reordered_idx >= 0 && reordered_idx < static_cast<int>(reordered_to_original_idx.size())) {
        int original_idx = reordered_to_original_idx[reordered_idx];
        cluster_order.push_back(original_idx);
        order_positions.push_back(selected_clusters_[original_idx].position);
      }
    }
  }

  // ===== Save current order for next planning cycle =====
  previous_cluster_order_positions_ = order_positions;
  has_previous_cluster_order_ = true;

  ROS_INFO("GlobalPlanner: TSP returned cluster order with %zu clusters", cluster_order.size());

  return cluster_order;
}

// ==========================================
// 无 forced cluster 时计算正常的 TSP 顺序
// ==========================================
std::vector<int> GlobalPlanner::computeClusterOrderNormal(
    const Eigen::Vector3f& odom_position) {

  // Handle single cluster case
  if (selected_clusters_.size() == 1) {
    Eigen::MatrixXd cost_mat;
    std::vector<bool> reachable;
    if (!buildDistanceMatrix(odom_position, selected_clusters_, cost_mat, reachable)) {
      ROS_ERROR("GlobalPlanner: Failed to build distance matrix for single cluster");
      return {};
    }

    // 可视化 cluster 可达性
    // visualizeClusterReachability(selected_clusters_, reachable);

    if (!reachable[0]) {
      ROS_WARN("GlobalPlanner: Single cluster is unreachable");
      return {};
    }

    // 保存本轮顺序
    previous_cluster_order_positions_.clear();
    previous_cluster_order_positions_.push_back(selected_clusters_[0].position);
    has_previous_cluster_order_ = true;
    return {0};
  }

  // ===== NORMAL MODE: TSP starts from odom =====
  Eigen::MatrixXd cost_mat;
  std::vector<bool> reachable;
  if (!buildDistanceMatrix(odom_position, selected_clusters_, cost_mat, reachable)) {
    ROS_ERROR("GlobalPlanner: Failed to build distance matrix");
    return {};
  }

  // 可视化 cluster 可达性
  // visualizeClusterReachability(selected_clusters_, reachable);

  // Filter unreachable clusters
  std::vector<int> reachable_indices;
  for (size_t i = 0; i < selected_clusters_.size(); ++i) {
    if (reachable[i]) {
      reachable_indices.push_back(i);
    }
  }

  if (reachable_indices.empty()) {
    ROS_ERROR("GlobalPlanner: All clusters unreachable");
    return {};
  }

  ROS_INFO("GlobalPlanner: %zu/%zu clusters reachable",
           reachable_indices.size(), selected_clusters_.size());

  // Extract sub-matrix for reachable clusters
  Eigen::MatrixXd filtered_cost_mat;
  if (reachable_indices.size() < selected_clusters_.size()) {
    int new_dim = reachable_indices.size() + 1;
    filtered_cost_mat.resize(new_dim, new_dim);
    filtered_cost_mat.setZero();

    // Node 0 (odom) remains
    filtered_cost_mat(0, 0) = cost_mat(0, 0);

    // Reachable clusters
    for (size_t i = 0; i < reachable_indices.size(); ++i) {
      int old_idx = reachable_indices[i] + 1;  // +1 because node 0 is odom
      int new_idx = i + 1;
      filtered_cost_mat(0, new_idx) = cost_mat(0, old_idx);
      filtered_cost_mat(new_idx, 0) = cost_mat(old_idx, 0);
    }

    for (size_t i = 0; i < reachable_indices.size(); ++i) {
      for (size_t j = 0; j < reachable_indices.size(); ++j) {
        int old_i = reachable_indices[i] + 1;
        int old_j = reachable_indices[j] + 1;
        filtered_cost_mat(i + 1, j + 1) = cost_mat(old_i, old_j);
      }
    }
  } else {
    filtered_cost_mat = cost_mat;
  }

  // ===== Solve TSP =====
  std::vector<int> tsp_path;
  solveTSP(filtered_cost_mat, tsp_path);
  ROS_INFO("GlobalPlanner: TSP solved, path size %zu", tsp_path.size());

  if (tsp_path.empty()) {
    ROS_ERROR("GlobalPlanner: TSP solver failed");
    return {};
  }

  // ===== Map TSP path back to original cluster indices =====
  std::vector<int> cluster_order;
  std::vector<Eigen::Vector3f> order_positions;

  for (int tsp_node : tsp_path) {
    if (tsp_node == 0) {
      continue;  // Skip odom node
    }

    int reachable_idx = tsp_node - 1;
    if (reachable_idx >= 0 && reachable_idx < static_cast<int>(reachable_indices.size())) {
      int original_idx = reachable_indices[reachable_idx];
      cluster_order.push_back(original_idx);
      order_positions.push_back(selected_clusters_[original_idx].position);
    }
  }

  // ===== Save current order for next planning cycle =====
  previous_cluster_order_positions_ = order_positions;
  has_previous_cluster_order_ = true;

  ROS_INFO("GlobalPlanner: TSP returned cluster order with %zu clusters", cluster_order.size());

  return cluster_order;
}


// ==========================================
// 根据 cluster 方向的观测得分过滤 target voxels
// ==========================================
std::vector<Eigen::Vector3i> GlobalPlanner::filterTargetVoxelsByClusterObservation(
    const ClusterInfo& cluster,
    const Eigen::Vector3i& aabb_min,
    const Eigen::Vector3i& aabb_max) {

  std::vector<Eigen::Vector3i> filtered_voxels;
  float total_cluster_observation_score = 0.0f;
  int total_cluster_observations = 0;
  int filtered_out_count = 0;

  // 存储被过滤掉的voxel的观测方向，用于可视化
  // pair<voxel_center, observation_direction>
  std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> filtered_observation_directions;
  // 存储已充分观测的voxel位置，用于点云可视化
  std::vector<Eigen::Vector3f> well_observed_voxel_positions;

  for (const auto& idx : cluster.poorly_observed_voxels) {
    auto it = spatial_hash_.find(idx);
    if (it == spatial_hash_.end()) {
      // Voxel 不存在，保守起见加入 target
      filtered_voxels.push_back(idx);
      continue;
    }

    const VoxelCell& cell = it->second;

    // 跳过尚未计算法向量的 voxel，保守起见加入 target
    if (cell.normal_bin_idx < 0) {
      filtered_voxels.push_back(idx);
      continue;
    }

    // 计算从当前 cluster 方向的观测得分
    float voxel_cluster_score = 0.0f;

    // 遍历所有已经观测的方向
    for (int dir = 0; dir < 20; ++dir) {
      if (!cell.observation_direction_mask[dir]) continue;

      // 获取观测方向向量（从 voxel 指向相机）
      const Eigen::Vector3f& view_dir = SphericalBinning::get_face_normal(dir);

      // 检查该观测方向是否指向当前 cluster 的 AABB
      // 方法：从 voxel 中心沿 view_dir 方向移动一段距离，检查是否进入 AABB
      Eigen::Vector3f test_pos = cell.voxel_center + view_dir * (voxel_size_ * 2.0f);

      // 将 test_pos 转换为 voxel 索引
      Eigen::Vector3i test_idx = ((test_pos - map_min_bd_).array() / voxel_size_).cast<int>();

      // 检查是否在 AABB 内
      bool is_towards_cluster = (test_idx.x() >= aabb_min.x() && test_idx.x() <= aabb_max.x() &&
                                 test_idx.y() >= aabb_min.y() && test_idx.y() <= aabb_max.y() &&
                                 test_idx.z() >= aabb_min.z() && test_idx.z() <= aabb_max.z());

      // 如果观测方向指向当前 cluster，计算并累加得分
      if (is_towards_cluster) {
        float score = SphericalBinning::scoring_table[cell.normal_bin_idx][dir];
        voxel_cluster_score += score;
        total_cluster_observations++;
      }
    }

    // 计算该 voxel 的阈值分数
    float threshold_score = well_observed_base_score_ +
                           well_observed_texture_weight_ * cell.texture_complexity +
                           well_observed_geo_weight_ * cell.geometric_complexity;

    // 如果从当前 cluster 方向的观测得分已经达到阈值，则过滤掉
    if (voxel_cluster_score >= threshold_score) {
      filtered_out_count++;
      total_cluster_observation_score += voxel_cluster_score;

      // 记录已充分观测的voxel位置
      well_observed_voxel_positions.push_back(cell.voxel_center);

      // 记录该voxel的观测方向用于可视化
      for (int dir = 0; dir < 20; ++dir) {
        if (!cell.observation_direction_mask[dir]) continue;

        const Eigen::Vector3f& view_dir = SphericalBinning::get_face_normal(dir);
        Eigen::Vector3f test_pos = cell.voxel_center + view_dir * (voxel_size_ * 2.0f);
        Eigen::Vector3i test_idx = ((test_pos - map_min_bd_).array() / voxel_size_).cast<int>();

        bool is_towards_cluster = (test_idx.x() >= aabb_min.x() && test_idx.x() <= aabb_max.x() &&
                                   test_idx.y() >= aabb_min.y() && test_idx.y() <= aabb_max.y() &&
                                   test_idx.z() >= aabb_min.z() && test_idx.z() <= aabb_max.z());

        if (is_towards_cluster) {
          filtered_observation_directions.push_back({cell.voxel_center, view_dir});
        }
      }
    } else {
      // 仍未充分观测，加入 target
      filtered_voxels.push_back(idx);
    }
  }

  ROS_INFO("Cluster observation filtering: %d voxels filtered out (score >= threshold), %zu voxels remaining as targets",
           filtered_out_count, filtered_voxels.size());
  ROS_INFO("Cluster observation score: %.2f (%d observations towards cluster AABB)",
           total_cluster_observation_score, total_cluster_observations);

  // 可视化：发布过滤后需要观测的 target voxels 点云
  if (filtered_target_voxels_pub_.getNumSubscribers() > 0 && !filtered_voxels.empty()) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->points.reserve(filtered_voxels.size());

    for (const auto& voxel_idx : filtered_voxels) {
      // 将 voxel 索引转换为世界坐标位置
      Eigen::Vector3f position = voxelIdxToPosition(voxel_idx);
      cloud->points.emplace_back(position.x(), position.y(), position.z());
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = "world";
    cloud_msg.header.stamp = ros::Time::now();
    filtered_target_voxels_pub_.publish(cloud_msg);

    ROS_INFO("Published filtered target voxels: %zu points", cloud->points.size());
  }

  // 可视化：发布被过滤voxel的观测方向箭头
  if (filtered_observation_directions_pub_.getNumSubscribers() > 0 && !filtered_observation_directions.empty()) {
    visualization_msgs::MarkerArray marker_array;
    int arrow_id = 0;

    for (const auto& arrow_data : filtered_observation_directions) {
      const Eigen::Vector3f& voxel_center = arrow_data.first;
      const Eigen::Vector3f& view_dir = arrow_data.second;

      visualization_msgs::Marker arrow_marker;
      arrow_marker.header.frame_id = "world";
      arrow_marker.header.stamp = ros::Time::now();
      arrow_marker.ns = "filtered_observation_directions";
      arrow_marker.id = arrow_id++;
      arrow_marker.type = visualization_msgs::Marker::ARROW;
      arrow_marker.action = visualization_msgs::Marker::ADD;
      arrow_marker.pose.orientation.w = 1.0;  // 单位四元数，避免未初始化警告

      // 箭头起点：voxel中心
      geometry_msgs::Point start_point;
      start_point.x = voxel_center.x();
      start_point.y = voxel_center.y();
      start_point.z = voxel_center.z();

      // 箭头终点：沿观测方向延伸0.15米
      geometry_msgs::Point end_point;
      end_point.x = voxel_center.x() + view_dir.x() * 0.15f;
      end_point.y = voxel_center.y() + view_dir.y() * 0.15f;
      end_point.z = voxel_center.z() + view_dir.z() * 0.15f;

      arrow_marker.points.push_back(start_point);
      arrow_marker.points.push_back(end_point);

      // 设置箭头尺寸
      arrow_marker.scale.x = 0.02;  // shaft diameter
      arrow_marker.scale.y = 0.04;  // head diameter
      arrow_marker.scale.z = 0.05;  // head length

      // 设置颜色（绿色表示已被充分观测）
      arrow_marker.color.r = 0.0;
      arrow_marker.color.g = 1.0;
      arrow_marker.color.b = 0.0;
      arrow_marker.color.a = 0.8;

      marker_array.markers.push_back(arrow_marker);
    }

    filtered_observation_directions_pub_.publish(marker_array);

    ROS_INFO("Published filtered observation direction arrows: %zu arrows", filtered_observation_directions.size());
  }

  // 可视化：发布已充分观测的voxel点云
  if (well_observed_voxels_pub_.getNumSubscribers() > 0 && !well_observed_voxel_positions.empty()) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->points.reserve(well_observed_voxel_positions.size());

    for (const auto& pos : well_observed_voxel_positions) {
      cloud->points.emplace_back(pos.x(), pos.y(), pos.z());
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = "world";
    cloud_msg.header.stamp = ros::Time::now();
    well_observed_voxels_pub_.publish(cloud_msg);

    ROS_INFO("Published well observed voxels: %zu points", cloud->points.size());
  }

  return filtered_voxels;
}


bool GlobalPlanner::attemptClusterPlanning(
    const ClusterInfo& cluster,
    size_t attempt_idx,
    size_t total_attempts,
    const Eigen::Vector3f& odom_position,
    const std::vector<int>& cluster_indices_to_try) {

    // local_observation_map.clear();
    // cluster.poorly_observed_voxels

  // Lambda: Mark region as observed (reduces repetition)
  auto markRegionObserved = [&](const char* reason) {
    if (region_map_ptr_) {
      auto it = region_map_ptr_->find(cluster.region_idx);
      if (it != region_map_ptr_->end()) {
        ROS_WARN("GlobalPlanner: Marked region [%d,%d,%d] as all_well_observed (%s)",
                 cluster.region_idx.x(), cluster.region_idx.y(),
                 cluster.region_idx.z(), reason);
      }
    }
  };

  // Clear previous attempt results
  selected_views_.clear();
  path_to_first_viewpoint_.clear();

  // Use first_cluster for compatibility with existing code
  const ClusterInfo& first_cluster = cluster;

  // Visualize unobserved directions
  visualization_msgs::Marker unobserved_marker =
      observation_quality::VisualizationUtils::generateUnobservedDirectionsMarker(
          first_cluster, spatial_hash_, voxel_size_, "world");
  unobserved_directions_pub_.publish(unobserved_marker);
  ROS_INFO("Published unobserved directions visualization (%zu lines)",
           unobserved_marker.points.size() / 2);

  // BFS expand connected free region
  Eigen::Vector3i aabb_min, aabb_max;
  Eigen::Vector3i target_aabb_min, target_aabb_max;
  expandConnectedFreeRegion(first_cluster, aabb_min, aabb_max,
                            connected_free_voxels_,
                            target_aabb_min, target_aabb_max);
  aabb_min -= Eigen::Vector3i(1, 1, 1);
  aabb_max += Eigen::Vector3i(1, 1, 1);

  // Sample viewpoints from connected free region
  std::vector<Eigen::Vector3f> sampled_viewpoints =
      sampleViewpointsFromCluster(connected_free_voxels_,
                                   viewpoint_sample_resolution_);


  if (sampled_viewpoints.empty()) {
    ROS_WARN("GlobalPlanner: No viewpoints sampled from cluster in region [%d, %d, %d]",
             first_cluster.region_idx.x(), first_cluster.region_idx.y(), first_cluster.region_idx.z());
    markRegionObserved("no viewpoints sampled");
    return false;
  }

  std::vector<Eigen::Vector3i> frontier_voxels;

  // 先根据 cluster 方向的观测得分过滤 poorly_observed_voxels
  std::vector<Eigen::Vector3i> filtered_poorly_observed =
      filterTargetVoxelsByClusterObservation(first_cluster, aabb_min, aabb_max);

  // 检查过滤后是否还有需要观测的 voxels
  if (filtered_poorly_observed.empty()) {
    ROS_INFO("GlobalPlanner: All voxels well-observed from cluster in region [%d, %d, %d], skipping planning",
             first_cluster.region_idx.x(), first_cluster.region_idx.y(), first_cluster.region_idx.z());
    markRegionObserved("all voxels well-observed");
    return false;
  }

  // GPU evaluation (get full CSR visibility matrix)
  // 使用过滤后的 voxels 进行评估，提高效率
  VisibilityCSR csr_result =
      evaluateViewpointsWithGPU(sampled_viewpoints,
                                 filtered_poorly_observed,
                                 frontier_voxels,
                                 spatial_hash_,
                                 voxel_size_,
                                 map_min_bd_,
                                 aabb_min, aabb_max);
  if (csr_result.viewpoint_offsets.empty()) {
    ROS_WARN("GlobalPlanner: GPU evaluation returned empty CSR result for region [%d, %d, %d]",
             first_cluster.region_idx.x(), first_cluster.region_idx.y(), first_cluster.region_idx.z());
    markRegionObserved("GPU evaluation failed");
    return false;
  }

  // Merge target_voxels for viewpoint selection and visualization
  std::vector<Eigen::Vector3i> all_target_voxels;
  all_target_voxels.reserve(filtered_poorly_observed.size() + frontier_voxels.size());
  all_target_voxels.insert(all_target_voxels.end(),
                           filtered_poorly_observed.begin(),
                           filtered_poorly_observed.end());
  all_target_voxels.insert(all_target_voxels.end(),
                           frontier_voxels.begin(),
                           frontier_voxels.end());

  // Select top N viewpoints (based on CSR visibility data, greedy algorithm and back-voting for orientation)
  ros::Time start_time_score = ros::Time::now();
  SelectedViewResult selection_result = selectTopNViewpoints(
      csr_result, sampled_viewpoints, all_target_voxels,
      static_cast<int>(filtered_poorly_observed.size()),
      top_n_viewpoints_);
  selected_views_ = selection_result.views;
  ros::Time end_time_score = ros::Time::now();
  ROS_INFO("GlobalPlanner: Scoring completed in %.3f ms",
           (end_time_score - start_time_score).toSec() * 1000.0);

  // Publish viewpoint visibility lines (using FOV-filtered results from selection)
  publishViewpointVisibilityLines(selected_views_,
                                  all_target_voxels,
                                  selection_result.filtered_csr);

  if (selected_views_.empty()) {
    markRegionObserved("no feasible viewpoints");
    ROS_WARN("GlobalPlanner: No viewpoints selected after scoring");
    return false;
  }

  // Extract positions for TSP
  std::vector<Eigen::Vector3f> selected_positions;
  selected_positions.reserve(selected_views_.size());
  for (const auto& view : selected_views_) {
    selected_positions.push_back(view.position);
  }

  // Get viewpoints region index
  Eigen::Vector3i viewpoints_region_idx = first_cluster.region_idx;

  // Determine next_cluster (if exists)
  const Eigen::Vector3f* next_cluster_position = nullptr;
  const Eigen::Vector3i* next_cluster_region_idx = nullptr;

  if (attempt_idx + 1 < cluster_indices_to_try.size()) {
    int next_cluster_idx = cluster_indices_to_try[attempt_idx + 1];
    next_cluster_position = &selected_clusters_[next_cluster_idx].position;
    next_cluster_region_idx = &selected_clusters_[next_cluster_idx].region_idx;
  }

  // Build distance matrix for viewpoints
  Eigen::MatrixXd fine_cost_mat;
  std::vector<int> fine_indices;

  buildDistanceMatrixForViewpoints(
      odom_position, selected_positions, viewpoints_region_idx,
      next_cluster_position, next_cluster_region_idx,
      fine_cost_mat, fine_indices);

  if (fine_indices.empty()) {
    ROS_WARN("GlobalPlanner: No reachable viewpoints for cluster in region [%d, %d, %d]",
             first_cluster.region_idx.x(), first_cluster.region_idx.y(), first_cluster.region_idx.z());
    markRegionObserved("no reachable viewpoints");
    return false;
  }

  // ===== Success: construct path =====
  // Build global path: odom → viewpoints → subsequent clusters (if any)
  global_path_.clear();
  for (int idx : fine_indices) {
    if (idx == 0) {
      global_path_.push_back(odom_position);
    } else if (idx <= (int)selected_positions.size()) {
      global_path_.push_back(selected_positions[idx - 1]);
    } else {
      // next_cluster case
      if (next_cluster_position) {
        global_path_.push_back(*next_cluster_position);
      }
    }
  }

  // Add remaining clusters to path (from attempt+2 onwards)
  for (size_t i = attempt_idx + 2; i < cluster_indices_to_try.size(); ++i) {
    int cluster_idx = cluster_indices_to_try[i];
    global_path_.push_back(selected_clusters_[cluster_idx].position);
  }

  // Reorder selected_views_ to match TSP order
  std::vector<SelectedView> reordered_views;
  reordered_views.reserve(selected_views_.size());
  for (int idx : fine_indices) {
    if (idx > 0 && idx <= (int)selected_views_.size()) {
      reordered_views.push_back(selected_views_[idx - 1]);
    }
  }
  selected_views_ = reordered_views;

  // Build path segments (each segment + corresponding target viewpoint)
  buildPathSegmentsFromCompletePathWithViews();

  ROS_INFO("GlobalPlanner: Hierarchical planning succeeded for cluster %zu/%zu (region [%d, %d, %d]), path updated with %zu viewpoints",
           attempt_idx + 1, total_attempts,
           first_cluster.region_idx.x(), first_cluster.region_idx.y(), first_cluster.region_idx.z(),
           selected_views_.size());

  // Save cluster's free voxels for next planning cycle
  previous_cluster_free_voxels_ = first_cluster.cluster_free_voxels;
  previous_cluster_center_voxel_idx_ = first_cluster.voxel_idx;
  previous_target_region_idx_ = first_cluster.region_idx;
  has_previous_planning_state_ = true;

  // Save next cluster info for path continuity (if exists)
  if (attempt_idx + 1 < cluster_indices_to_try.size()) {
    int next_cluster_idx = cluster_indices_to_try[attempt_idx + 1];
    previous_next_cluster_position_ = selected_clusters_[next_cluster_idx].position;
    previous_next_cluster_region_idx_ = selected_clusters_[next_cluster_idx].region_idx;
    has_next_cluster_ = true;

    // Save complete global_path_ for continuation mode visualization
    previous_global_path_ = global_path_;

    // Save path_segments_with_views_ for continuation mode
    previous_path_segments_with_views_ = path_segments_with_views_;

    ROS_INFO("Saved next cluster: position [%.2f,%.2f,%.2f], region [%d,%d,%d]",
             previous_next_cluster_position_.x(), previous_next_cluster_position_.y(), previous_next_cluster_position_.z(),
             previous_next_cluster_region_idx_.x(), previous_next_cluster_region_idx_.y(), previous_next_cluster_region_idx_.z());
    ROS_INFO("Saved global_path_ with %zu waypoints for continuation mode visualization",
             previous_global_path_.size());
  } else {
    has_next_cluster_ = false;
    previous_global_path_.clear();
    ROS_INFO("No next cluster (this was the last cluster)");
  }

  ROS_INFO("Saved previous cluster state: %zu free voxels, region [%d,%d,%d]",
           previous_cluster_free_voxels_.size(),
           first_cluster.region_idx.x(), first_cluster.region_idx.y(), first_cluster.region_idx.z());

  return true;
}

// ==========================================
// Continuation mode 下的特殊规划函数
// ==========================================
bool GlobalPlanner::attemptClusterPlanningContinuation(
    const ClusterInfo& cluster,
    const Eigen::Vector3f& odom_position,
    const Eigen::Vector3f& target_end_position,
    const std::vector<PathSegmentWithView>& unpassed_viewpoints) {


  // Lambda: Mark region as observed (reduces repetition)
  auto markRegionObserved = [&](const char* reason) {
    if (region_map_ptr_) {
      auto it = region_map_ptr_->find(cluster.region_idx);
      if (it != region_map_ptr_->end()) {
        // it->second.setAllWellObserved(true);
        ROS_WARN("GlobalPlanner: Marked region [%d,%d,%d] as all_well_observed (%s)",
                 cluster.region_idx.x(), cluster.region_idx.y(),
                 cluster.region_idx.z(), reason);
      }
    }
  };

  // Clear previous attempt results
  selected_views_.clear();
  path_to_first_viewpoint_.clear();

  // Use first_cluster for compatibility with existing code
  const ClusterInfo& first_cluster = cluster;

  ROS_INFO("GlobalPlanner: attemptClusterPlanningContinuation - target_end_position [%.2f, %.2f, %.2f]",
           target_end_position.x(), target_end_position.y(), target_end_position.z());

  // Visualize unobserved directions
  visualization_msgs::Marker unobserved_marker =
      observation_quality::VisualizationUtils::generateUnobservedDirectionsMarker(
          first_cluster, spatial_hash_, voxel_size_, "world");
  unobserved_directions_pub_.publish(unobserved_marker);
  ROS_INFO("Published unobserved directions visualization (%zu lines)",
           unobserved_marker.points.size() / 2);

  // BFS expand connected free region
  Eigen::Vector3i aabb_min, aabb_max;
  Eigen::Vector3i target_aabb_min, target_aabb_max;
  expandConnectedFreeRegion(first_cluster, aabb_min, aabb_max,
                            connected_free_voxels_,
                            target_aabb_min, target_aabb_max);

  // 定义 frontier_voxels（当前为空，与原代码保持一致）
  std::vector<Eigen::Vector3i> frontier_voxels;

  // 根据 cluster 方向的观测得分过滤 target voxels
  std::vector<Eigen::Vector3i> all_target_voxels =
      filterTargetVoxelsByClusterObservation(first_cluster, aabb_min, aabb_max);

  // ========== Phase 1: Evaluate unpassed viewpoints ==========
  std::vector<SelectedView> selected_from_unpassed;
  std::vector<std::bitset<20>> covered_masks;  // 用于传递给 Phase 2
  VisibilityCSR unpassed_filtered_csr;  // 保存 filtered CSR (包含新覆盖的目标)

  if (!unpassed_viewpoints.empty()) {
    ROS_INFO("GlobalPlanner: Evaluating %zu unpassed viewpoints", unpassed_viewpoints.size());

    // 1.1 提取位置（过滤掉不在 connected_free_voxels_ 中的点）
    std::vector<Eigen::Vector3f> unpassed_positions;
    for (const auto& seg : unpassed_viewpoints) {
      // 将位置转换为 voxel 索引
      Eigen::Vector3i voxel_idx(
          static_cast<int>(std::floor((seg.target_view.position.x() - region_origin_.x()) / voxel_size_)),
          static_cast<int>(std::floor((seg.target_view.position.y() - region_origin_.y()) / voxel_size_)),
          static_cast<int>(std::floor((seg.target_view.position.z() - region_origin_.z()) / voxel_size_)));

      // 检查是否在 connected_free_voxels_ 中
      if (std::find(connected_free_voxels_.begin(), connected_free_voxels_.end(), voxel_idx)
          != connected_free_voxels_.end()) {
        unpassed_positions.push_back(seg.target_view.position);
      }
    }

    // 1.2 GPU 评估 unpassed viewpoints 对当前 cluster 的可见性
    aabb_min -= Eigen::Vector3i(1,1,1);
    aabb_max += Eigen::Vector3i(1,1,1);
    VisibilityCSR unpassed_csr = evaluateViewpointsWithGPU(
        unpassed_positions, all_target_voxels,
        frontier_voxels, spatial_hash_, voxel_size_, map_min_bd_,
        aabb_min, aabb_max);

    if (!unpassed_csr.viewpoint_offsets.empty()) {
      // 1.3 从 unpassed 中选择所有有效的 viewpoints
      SelectedViewResult unpassed_result = selectTopNViewpoints(
          unpassed_csr, unpassed_positions, all_target_voxels,
          static_cast<int>(all_target_voxels.size()),
          static_cast<int>(unpassed_positions.size()));  // 选择所有有效的

      selected_from_unpassed = unpassed_result.views;
      covered_masks = unpassed_result.final_target_masks;
      unpassed_filtered_csr = unpassed_result.filtered_csr;  // 保存用于可视化
    }

    ROS_INFO("GlobalPlanner: Selected %zu viewpoints from unpassed", selected_from_unpassed.size());
  }

  // ========== Phase 2: Sample new viewpoints if needed ==========
  std::vector<SelectedView> selected_from_new;
  int remaining_quota = top_n_viewpoints_ - static_cast<int>(selected_from_unpassed.size());
  int max_new = std::min(remaining_quota, top_n_viewpoints_ );
  VisibilityCSR new_filtered_csr;  // 保存 filtered CSR (包含新覆盖的目标)

  if (max_new > 0) {
    ROS_INFO("GlobalPlanner: Sampling new viewpoints, max_new = %d", max_new);

    // 2.1 采样新 viewpoints
    std::vector<Eigen::Vector3f> new_sampled = sampleViewpointsFromCluster(
        connected_free_voxels_, viewpoint_sample_resolution_);

    // 2.2 过滤与已选 unpassed 位置过近的点
    if (!selected_from_unpassed.empty()) {
      std::vector<Eigen::Vector3f> unpassed_pos;
      for (const auto& v : selected_from_unpassed) {
        unpassed_pos.push_back(v.position);
      }
      float min_dist_sq = viewpoint_sample_resolution_ * viewpoint_sample_resolution_;
      new_sampled.erase(
          std::remove_if(new_sampled.begin(), new_sampled.end(),
              [&](const Eigen::Vector3f& c) {
                for (const auto& s : unpassed_pos) {
                  if ((c - s).squaredNorm() < min_dist_sq) return true;
                }
                return false;
              }),
          new_sampled.end());
      ROS_INFO("GlobalPlanner: After filtering nearby, %zu new viewpoints remain", new_sampled.size());
    }

    // 可视化：发布所有采样的viewpoints点云
    if (sampled_viewpoints_pub_.getNumSubscribers() > 0 && !new_sampled.empty()) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
      cloud->points.reserve(new_sampled.size());

      for (const auto& viewpoint : new_sampled) {
        cloud->points.emplace_back(viewpoint.x(), viewpoint.y(), viewpoint.z());
      }

      cloud->width = cloud->points.size();
      cloud->height = 1;
      cloud->is_dense = true;

      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(*cloud, cloud_msg);
      cloud_msg.header.frame_id = "world";
      cloud_msg.header.stamp = ros::Time::now();
      sampled_viewpoints_pub_.publish(cloud_msg);

      ROS_INFO("Published sampled viewpoints: %zu points", cloud->points.size());
    }

    // 2.3 GPU 评估新采样的 viewpoints
    if (!new_sampled.empty()) {
      VisibilityCSR new_csr = evaluateViewpointsWithGPU(
          new_sampled, all_target_voxels,
          frontier_voxels, spatial_hash_, voxel_size_, map_min_bd_,
          aabb_min, aabb_max);

      if (!new_csr.viewpoint_offsets.empty()) {
        // 2.4 选择补充视点，传入已覆盖状态
        SelectedViewResult new_result = selectTopNViewpoints(
            new_csr, new_sampled, all_target_voxels,
            static_cast<int>(all_target_voxels.size()),
            max_new,
            covered_masks);  // 使用 Phase 1 的覆盖状态

        selected_from_new = new_result.views;
        new_filtered_csr = new_result.filtered_csr;  // 保存用于可视化
      }
      ROS_INFO("GlobalPlanner: Selected %zu new viewpoints", selected_from_new.size());
    }
  }

  // ========== Phase 3: Merge selected views ==========
  selected_views_.clear();
  selected_views_.insert(selected_views_.end(),
                         selected_from_unpassed.begin(),
                         selected_from_unpassed.end());
  selected_views_.insert(selected_views_.end(),
                         selected_from_new.begin(),
                         selected_from_new.end());

  ROS_INFO("GlobalPlanner: Total selected viewpoints: %zu (unpassed: %zu, new: %zu)",
           selected_views_.size(), selected_from_unpassed.size(), selected_from_new.size());

  // Publish merged viewpoint visibility lines (分别发布两个阶段的结果，使用 filtered CSR)
  
  if (!selected_from_unpassed.empty() && !unpassed_filtered_csr.viewpoint_offsets.empty()) {
    publishViewpointVisibilityLines(selected_from_unpassed, all_target_voxels, unpassed_filtered_csr, true);  // 清空历史
  }
  if (!selected_from_new.empty() && !new_filtered_csr.viewpoint_offsets.empty()) {
    publishViewpointVisibilityLines(selected_from_new, all_target_voxels, new_filtered_csr, false);  // 不清空，追加
  }
  // Debug: 发布odom - target的可视化
  // {
  //   std::vector<Eigen::Vector3f> odom_viewpoints = {odom_position};
  //   VisibilityCSR odom_csr = evaluateViewpointsWithGPU(
  //       odom_viewpoints, all_target_voxels,
  //       frontier_voxels, spatial_hash_, voxel_size_, map_min_bd_,
  //       aabb_min, aabb_max);

  //   if (!odom_csr.viewpoint_offsets.empty()) {
  //     // 创建一个 SelectedView 用于可视化 (odom 没有特定的朝向，使用 0)
  //     SelectedView odom_view;
  //     odom_view.position = odom_position;
  //     odom_view.yaw = 0.0f;
  //     odom_view.pitch = 0.0f;
  //     odom_view.score = 0.0f;
  //     odom_view.viewpoint_idx = 0;

  //     std::vector<SelectedView> odom_views = {odom_view};
  //     publishViewpointVisibilityLines(odom_views, all_target_voxels, odom_csr, true);

  //     ROS_INFO("GlobalPlanner: Published odom-target visibility lines (%d visible targets)",
  //              odom_csr.getVisibleTargets(0).size());
  //   }
  // }

  // 检查是否有足够的 viewpoints
  if (selected_views_.empty()) {
    ROS_WARN("GlobalPlanner: No valid viewpoints found from unpassed or new sampling");
    markRegionObserved("no feasible viewpoints");
    return false;
  }

  // Extract positions for TSP
  std::vector<Eigen::Vector3f> selected_positions;
  selected_positions.reserve(selected_views_.size());
  for (const auto& view : selected_views_) {
    selected_positions.push_back(view.position);
  }

  // Get viewpoints region index
  Eigen::Vector3i viewpoints_region_idx = first_cluster.region_idx;

  // Use target_end_position as next_cluster_position
  const Eigen::Vector3f* next_cluster_position = &previous_next_cluster_position_;
  Eigen::Vector3i target_region_idx = positionToRegionIdx(previous_next_cluster_position_);
  const Eigen::Vector3i* next_cluster_region_idx = &target_region_idx;

  // Build distance matrix for viewpoints
  Eigen::MatrixXd fine_cost_mat;
  std::vector<int> fine_indices;

  buildDistanceMatrixForViewpoints(
      odom_position, selected_positions, viewpoints_region_idx,
      next_cluster_position, next_cluster_region_idx,
      fine_cost_mat, fine_indices);

  if (fine_indices.empty()) {
    ROS_WARN("GlobalPlanner: No reachable viewpoints for cluster in region [%d, %d, %d]",
             first_cluster.region_idx.x(), first_cluster.region_idx.y(), first_cluster.region_idx.z());
    markRegionObserved("no reachable viewpoints");
    return false;
  }

  // ===== Success: construct path =====
  // Build global path: odom → viewpoints → target_end_position
  global_path_.clear();
  for (int idx : fine_indices) {
    if (idx == 0) {
      global_path_.push_back(odom_position);
    } else if (idx <= (int)selected_positions.size()) {
      global_path_.push_back(selected_positions[idx - 1]);
    } else {
      global_path_.push_back(previous_next_cluster_position_);
    }
  }

  // TODO: In continuation mode, we don't add remaining clusters
  // The path ends at target_end_position

  // Reorder selected_views_ to match TSP order
  std::vector<SelectedView> reordered_views;
  reordered_views.reserve(selected_views_.size());
  for (int idx : fine_indices) {
    if (idx > 0 && idx <= (int)selected_views_.size()) {
      reordered_views.push_back(selected_views_[idx - 1]);
    }
  }
  selected_views_ = reordered_views;

  // Build path segments (each segment + corresponding target viewpoint)
  buildPathSegmentsFromCompletePathWithViews();

  ROS_INFO("GlobalPlanner: Continuation mode planning succeeded (region [%d, %d, %d]), path updated with %zu viewpoints, ending at [%.2f, %.2f, %.2f]",
           first_cluster.region_idx.x(), first_cluster.region_idx.y(), first_cluster.region_idx.z(),
           selected_views_.size(),
           previous_next_cluster_position_.x(), previous_next_cluster_position_.y(), previous_next_cluster_position_.z());

  // Save cluster's free voxels for next planning cycle
  previous_cluster_free_voxels_ = first_cluster.cluster_free_voxels;
  previous_cluster_center_voxel_idx_ = first_cluster.voxel_idx;
  previous_target_region_idx_ = first_cluster.region_idx;
  has_previous_planning_state_ = true;

  // Save it for potential future continuation
  previous_next_cluster_region_idx_ = target_region_idx;
  has_next_cluster_ = true;

  // Save complete global_path_ for continuation mode visualization
  previous_global_path_ = global_path_;

  // Save path_segments_with_views_ for continuation mode
  previous_path_segments_with_views_ = path_segments_with_views_;

  ROS_INFO("Saved continuation state: target_end [%.2f,%.2f,%.2f], region [%d,%d,%d]",
           previous_next_cluster_position_.x(), previous_next_cluster_position_.y(), previous_next_cluster_position_.z(),
           target_region_idx.x(), target_region_idx.y(), target_region_idx.z());

  return true;
}

void GlobalPlanner::clearPlanningState() {
  global_path_.clear();
  has_previous_planning_state_ = false;
  previous_global_path_.clear();
  previous_cluster_free_voxels_.clear();
  previous_path_segments_with_views_.clear();
}

bool GlobalPlanner::planGlobalTSPPath(
    const Eigen::Vector3f& odom_position,
    float odom_yaw,
    float odom_pitch,
    std::unordered_map<Eigen::Vector3i, FreeRegion, RegionHash,
                       std::equal_to<Eigen::Vector3i>>& region_map,
    const std::vector<PathSegmentWithView>& unpassed_viewpoints) {

  ros::Time start_time = ros::Time::now();

  // Frontier visualization
  if (frontier_vis_pub_.getNumSubscribers() > 0) {
    auto frontier_cloud = observation_quality::VisualizationUtils::generateFrontierCloud(
        region_map, voxel_size_, region_origin_);
    if (!frontier_cloud->points.empty()) {
      sensor_msgs::PointCloud2 frontier_msg;
      pcl::toROSMsg(*frontier_cloud, frontier_msg);
      frontier_msg.header.frame_id = "world";
      frontier_msg.header.stamp = ros::Time::now();
      frontier_vis_pub_.publish(frontier_msg);
    }
  }

  // Phase 1: Collect and filter clusters, compute TSP-ordered cluster indices
  // 1.1 Collect and filter clusters
  selected_clusters_ = collectAndFilterClusters(region_map);

  // 可视化已完成观测的cluster centers (竖直圆柱体)
  if (completed_centers_vis_pub_.getNumSubscribers() > 0) {
    visualization_msgs::MarkerArray marker_array;
    // 清除旧的markers
    visualization_msgs::Marker delete_marker;
    delete_marker.header.frame_id = "world";
    delete_marker.header.stamp = ros::Time::now();
    delete_marker.ns = "completed_centers";
    delete_marker.action = visualization_msgs::Marker::DELETEALL;
    visualization_msgs::MarkerArray delete_array;
    delete_array.markers.push_back(delete_marker);
    int marker_id = 0;
    for (const auto& [region_idx, region] : region_map) {
      const auto& completed_centers = region.getCompletedClusterCenters();
      for (const auto& center : completed_centers) {
        Eigen::Vector3f world_pos = voxelIdxToPosition(center);
        visualization_msgs::Marker cylinder;
        cylinder.header.frame_id = "world";
        cylinder.header.stamp = ros::Time::now();
        cylinder.ns = "completed_centers";
        cylinder.id = marker_id++;
        cylinder.type = visualization_msgs::Marker::CYLINDER;
        cylinder.action = visualization_msgs::Marker::ADD;
        cylinder.pose.position.x = world_pos.x();
        cylinder.pose.position.y = world_pos.y();
        cylinder.pose.position.z = world_pos.z() + 0.5;  // 圆柱中心在底部上方0.5m
        cylinder.pose.orientation.w = 1.0;
        cylinder.scale.x = 0.2;  // 直径 = 2 * 半径 0.1
        cylinder.scale.y = 0.2;
        cylinder.scale.z = 1.0;  // 高度1米
        cylinder.color.r = 0.5;
        cylinder.color.g = 0.0;
        cylinder.color.b = 0.5;  // 紫色
        cylinder.color.a = 0.8;
        marker_array.markers.push_back(cylinder);
      }
    }

    completed_centers_vis_pub_.publish(delete_array);
    if (!marker_array.markers.empty()) {
      completed_centers_vis_pub_.publish(marker_array);
    }
  }

  if (selected_clusters_.empty()) {
    ROS_WARN("GlobalPlanner: No valid clusters found");
    clearPlanningState();
    publishVisualization();
    return false;
  }

  // 1.2 Check for forced cluster (continuation mode)
  int forced_cluster_idx = -1;
  bool has_forced_cluster = checkForcedClusterAndVisualize(forced_cluster_idx);

  // In continuation mode, previous_next_cluster_position_ contains the previous TSP's second cluster center
  if (has_forced_cluster && has_next_cluster_) {
    ROS_INFO("Continuation mode: previous second cluster at [%.2f, %.2f, %.2f]",
             previous_next_cluster_position_.x(), previous_next_cluster_position_.y(),
             previous_next_cluster_position_.z());
  }

  // 1.3 Build distance matrix and solve TSP based on mode
  std::vector<int> cluster_indices_to_try;
  if (has_forced_cluster) {
    cluster_indices_to_try = computeClusterOrderWithForcedCluster(forced_cluster_idx);
  } else {
    cluster_indices_to_try = computeClusterOrderNormal(odom_position);
  }

  // 增加cluster规划次数计数器并发布可视化
  visualization_msgs::Marker count_marker;
  count_marker.header.frame_id = "world";
  count_marker.header.stamp = ros::Time::now();
  count_marker.ns = "planning_count";
  count_marker.id = 0;
  count_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  count_marker.action = visualization_msgs::Marker::ADD;
  count_marker.pose.position.x = odom_position.x();
  count_marker.pose.position.y = odom_position.y();
  count_marker.pose.position.z = odom_position.z();  // 在odom位置上方1米显示
  count_marker.pose.orientation.w = 1.0;
  // Visualize cluster order with red line connecting cluster centers
  visualizeGlobalPath(cluster_indices_to_try, odom_position);

  if (cluster_indices_to_try.empty()) {
    ROS_WARN("GlobalPlanner: No reachable clusters found");
    clearPlanningState();
    publishVisualization();
    return false;
  }

  ROS_INFO("GlobalPlanner: Found %zu valid clusters", selected_clusters_.size());

  // Phase 2: Attempt hierarchical planning for each cluster in TSP order
  selected_views_.clear();
  path_to_first_viewpoint_.clear();
  bool hierarchical_success = false;

  ROS_INFO("GlobalPlanner: Attempting hierarchical planning for %zu clusters in order...",
           cluster_indices_to_try.size());

  for (size_t attempt = 0; attempt < cluster_indices_to_try.size(); ++attempt) {
    int current_cluster_idx = cluster_indices_to_try[attempt];
    const ClusterInfo& current_cluster = selected_clusters_[current_cluster_idx];

    ROS_INFO("GlobalPlanner: Attempting cluster %zu/%zu (region [%d, %d, %d])...",
             attempt + 1, cluster_indices_to_try.size(),
             current_cluster.region_idx.x(), current_cluster.region_idx.y(),
             current_cluster.region_idx.z());

    // Continuation mode: attempt == 0 && has_forced_cluster && has_next_cluster_
    // if (attempt == 0 && has_forced_cluster && has_next_cluster_) {
    if (true) {
      cluster_indices_to_try.resize(min((int)cluster_indices_to_try.size(), 2));  // 截断后续clusters，确保只规划当前cluster
      // 发布红色计数标记（continuation模式）
      count_marker.scale.z = 0.5;  // 字体高度
      count_marker.color.r = 1.0f;
      count_marker.color.g = 0.0f;
      count_marker.color.b = 0.0f;  // 红色
      count_marker.color.a = 1.0f;
      cluster_planning_count_ ++;
      count_marker.text = std::to_string(cluster_planning_count_);
      planning_count_pub_.publish(count_marker);

      if (attemptClusterPlanningContinuation(
              current_cluster,
              odom_position,
              previous_next_cluster_position_,
              unpassed_viewpoints)) {
        hierarchical_success = true;
        break;
      }
      // Continuation failed - mark this cluster as completed to avoid re-planning
      auto it = region_map_ptr_->find(current_cluster.region_idx);
      if (it != region_map_ptr_->end()) {
        it->second.addCompletedClusterCenter(current_cluster.voxel_idx);
        ROS_INFO("Continuation failed for cluster in region [%d,%d,%d], marking center [%d,%d,%d] as completed",
                 current_cluster.region_idx.x(), current_cluster.region_idx.y(), current_cluster.region_idx.z(),
                 current_cluster.voxel_idx.x(), current_cluster.voxel_idx.y(), current_cluster.voxel_idx.z());
      }
    } else {
      // 发布蓝色计数标记（正常规划模式）
      count_marker.scale.z = 0.5;  // ���体高度
      count_marker.color.r = 0.0f;
      count_marker.color.g = 0.0f;
      count_marker.color.b = 1.0f;  // 蓝色
      count_marker.color.a = 1.0f;
      cluster_planning_count_ = 0;

      count_marker.text = std::to_string(cluster_planning_count_);
      planning_count_pub_.publish(count_marker);
      if (attemptClusterPlanning(current_cluster, attempt,
                                  cluster_indices_to_try.size(),
                                  odom_position, cluster_indices_to_try)) {
        hierarchical_success = true;
        break;
      }
    }
  }

  // Phase 3: Handle result
  if (!hierarchical_success) {
    ROS_WARN("GlobalPlanner: All %zu clusters failed hierarchical planning, no reachable viewpoints found",
             cluster_indices_to_try.size());
    global_path_.clear();
    selected_views_.clear();
    path_to_first_viewpoint_.clear();
    clearPlanningState();
    publishVisualization();

    ros::Time end_time = ros::Time::now();
    ROS_INFO("GlobalPlanner: TSP planning completed in %.3f ms (no reachable viewpoints)",
             (end_time - start_time).toSec() * 1000.0);
    return false;
  }

  publishVisualization();

  ros::Time end_time = ros::Time::now();
  ROS_INFO("GlobalPlanner: TSP planning completed in %.3f ms, path length: %zu",
           (end_time - start_time).toSec() * 1000.0, global_path_.size());

  return true;
}

bool GlobalPlanner::filterClustersAndSolveTSP(
    const Eigen::Vector3f& odom_position,
    Eigen::MatrixXd& cost_mat,
    const std::vector<bool>& reachable,
    std::vector<int>& indices) {

  // 过滤不可达的 clusters
  std::vector<ClusterInfo> reachable_clusters;
  std::vector<int> original_indices;  // 映射回原始索引
  reachable_clusters.reserve(selected_clusters_.size());
  original_indices.reserve(selected_clusters_.size());
  for (size_t i = 0; i < selected_clusters_.size(); ++i) {
    if (reachable[i]) {
      reachable_clusters.push_back(selected_clusters_[i]);
      original_indices.push_back(i);
    }
  }

  if (reachable_clusters.empty()) {
    ROS_WARN("GlobalPlanner: All clusters are unreachable");
    global_path_.clear();
    publishVisualization();
    return false;
  }

  ROS_INFO("GlobalPlanner: %zu/%zu clusters are reachable",
           reachable_clusters.size(), selected_clusters_.size());

  // 如果有不可达的 cluster 被过滤，从原矩阵中提取子矩阵（避免重复计算）
  if (reachable_clusters.size() < selected_clusters_.size()) {
    // 提取子矩阵：保留 odom (index 0) 和可达 clusters
    int new_dim = reachable_clusters.size() + 1;
    Eigen::MatrixXd new_cost_mat(new_dim, new_dim);
    new_cost_mat.setZero();

    // odom -> odom
    new_cost_mat(0, 0) = cost_mat(0, 0);

    // odom <-> 可达 clusters
    for (size_t i = 0; i < original_indices.size(); ++i) {
      int old_idx = original_indices[i] + 1;  // 原矩阵中的索引 (+1 因为 odom 在 0)
      int new_idx = i + 1;                     // 新矩阵中的索引
      new_cost_mat(0, new_idx) = cost_mat(0, old_idx);
      new_cost_mat(new_idx, 0) = cost_mat(old_idx, 0);
    }

    // 可达 clusters 之间
    for (size_t i = 0; i < original_indices.size(); ++i) {
      for (size_t j = 0; j < original_indices.size(); ++j) {
        int old_i = original_indices[i] + 1;
        int old_j = original_indices[j] + 1;
        new_cost_mat(i + 1, j + 1) = cost_mat(old_i, old_j);
      }
    }

    int old_dim = cost_mat.rows();
    cost_mat = new_cost_mat;
    selected_clusters_ = reachable_clusters;

    ROS_INFO("GlobalPlanner: Extracted %d x %d sub-matrix from original %d x %d matrix",
             new_dim, new_dim, old_dim, old_dim);
  }

  // 调用TSP求解器
  solveTSP(cost_mat, indices);

  if (indices.empty()) {
    ROS_ERROR("GlobalPlanner: TSP solver failed");
    global_path_.clear();
    return false;
  }

  // 构建粗粒度路径
  global_path_.clear();
  for (int idx : indices) {
    if (idx == 0) {
      global_path_.push_back(odom_position);
    } else {
      global_path_.push_back(selected_clusters_[idx - 1].position);
    }
  }

  return true;
}

// ==========================================
// 工具函数实现
// ==========================================
Eigen::Vector3f GlobalPlanner::voxelIdxToPosition(const Eigen::Vector3i& idx) const {
  // 体素索引转世界坐标: 体素中心 = region_origin + (idx + 0.5) * voxel_size
  Eigen::Vector3f offset = ((idx.cast<float>().array() + 0.5f) * voxel_size_).matrix();
  return region_origin_ + offset;
}

void GlobalPlanner::updateVoxelMasksForViewpoint(
    int viewpoint_idx,
    const VisibilityCSR& csr_result,
    const std::vector<Eigen::Vector3i>& target_voxels) {

  // 获取该 viewpoint 能看到的所有 target 索引
  const std::vector<int>& visible_targets = csr_result.getVisibleTargets(viewpoint_idx);

  if (visible_targets.empty()) {
    ROS_WARN("GlobalPlanner: No visible targets for viewpoint %d", viewpoint_idx);
    return;
  }

  // 获取该 viewpoint 的位置（从 CSR 对应的原始 viewpoints 数组）
  // 由于我们没有直接保存 viewpoint 位置，需要从 selected_views_ 中查找
  // 但实际上，调用此函数时我们已经知道 viewpoint 的位置
  // 这里通过 target 反推 viewpoint 位置会比较复杂
  // 更好的做法是传入 viewpoint 位置

  // 为了简化，我们使用 first_view 的位置（因为此函数当前只用于第一个viewpoint）
  const Eigen::Vector3f& vp_pos = selected_views_[0].position;

  int updated_count = 0;
  for (int target_idx : visible_targets) {
    if (target_idx < 0 || target_idx >= (int)target_voxels.size()) continue;

    const Eigen::Vector3i& voxel_idx = target_voxels[target_idx];
    auto it = spatial_hash_.find(voxel_idx);
    if (it == spatial_hash_.end()) continue;

    VoxelCell& cell = it->second;

    // 计算观测方向：从 voxel 中心指向 viewpoint
    Eigen::Vector3f view_dir = vp_pos - cell.voxel_center;
    int obs_bin_idx = SphericalBinning::get_bin_index(view_dir);

    if (obs_bin_idx >= 0 && !cell.observation_direction_mask[obs_bin_idx]) {
      cell.observation_direction_mask[obs_bin_idx] = true;
      cell.updateWellObserved(well_observed_base_score_, well_observed_texture_weight_, well_observed_geo_weight_);
      updated_count++;
    }
  }

  ROS_INFO("GlobalPlanner: Updated observation masks for %d voxels (viewpoint %d saw %zu targets)",
           updated_count, viewpoint_idx, visible_targets.size());
}

bool GlobalPlanner::areRegionsAdjacent(const Eigen::Vector3i& region_a,
                                      const Eigen::Vector3i& region_b) const {
  // 6-邻域: 曼哈顿距离 == 1 (仅面相邻)
  return (std::abs(region_a.x() - region_b.x()) +
          std::abs(region_a.y() - region_b.y()) +
          std::abs(region_a.z() - region_b.z())) == 1;
}

Eigen::Vector3i GlobalPlanner::positionToRegionIdx(const Eigen::Vector3f& position) const {
  // 每个region是 6m / 0.2m = 30 个体素
  constexpr int VOXELS_PER_REGION = 30;  // Fixed: was 15

  // 先转换为体素索引 (with origin offset)
  Eigen::Vector3i voxel_idx(
      static_cast<int>(std::floor((position.x() - region_origin_.x()) / voxel_size_)),
      static_cast<int>(std::floor((position.y() - region_origin_.y()) / voxel_size_)),
      static_cast<int>(std::floor((position.z() - region_origin_.z()) / voxel_size_)));

  // 使用floor division处理负数索引 (与 ObservationQualityManager::voxelToRegion 一致)
  auto floor_div = [](int a, int b) -> int {
    return (a >= 0) ? (a / b) : ((a - b + 1) / b);
  };

  return Eigen::Vector3i(
      floor_div(voxel_idx.x(), VOXELS_PER_REGION),
      floor_div(voxel_idx.y(), VOXELS_PER_REGION),
      floor_div(voxel_idx.z(), VOXELS_PER_REGION));
}

void GlobalPlanner::buildPathSegmentsFromCompletePathWithViews() {
  path_segments_with_views_.clear();

  // 边界检查
  if (complete_viewpoint_path_.empty() || selected_views_.empty()) {
    ROS_WARN("GlobalPlanner: Cannot build path segments - empty path or views");
    return;
  }

  // 构建 free voxels 查找集合（用于快速碰撞检查）
  std::unordered_set<Eigen::Vector3i, VoxelHash, std::equal_to<Eigen::Vector3i>>
      free_voxels_set;
  if (!connected_free_voxels_.empty()) {
    free_voxels_set.insert(connected_free_voxels_.begin(),
                           connected_free_voxels_.end());
    ROS_INFO("GlobalPlanner: Built free voxels lookup set with %zu voxels for path simplification",
             free_voxels_set.size());
  } else {
    ROS_WARN("GlobalPlanner: connected_free_voxels_ is empty, path simplification disabled");
  }

  // Step 1: 为每个view找到complete_viewpoint_path_中最近点的索引
  // 使用 pair<path_idx, view_idx> 以便后续按路径顺序排序
  std::vector<std::pair<int, size_t>> index_pairs;
  index_pairs.reserve(selected_views_.size());

  for (size_t view_idx = 0; view_idx < selected_views_.size(); ++view_idx) {
    const auto& view = selected_views_[view_idx];
    int closest_idx = -1;
    float min_dist_sq = std::numeric_limits<float>::max();

    for (size_t i = 0; i < complete_viewpoint_path_.size(); ++i) {
      float dist_sq = (complete_viewpoint_path_[i] - view.position).squaredNorm();
      if (dist_sq < min_dist_sq) {
        min_dist_sq = dist_sq;
        closest_idx = i;
      }
    }

    if (closest_idx >= 0) {
      index_pairs.emplace_back(closest_idx, view_idx);
    } else {
      ROS_WARN("GlobalPlanner: Could not find closest point for viewpoint %zu [%.2f, %.2f, %.2f]",
               view_idx, view.position.x(), view.position.y(), view.position.z());
    }
  }

  if (index_pairs.empty()) {
    ROS_WARN("GlobalPlanner: No valid viewpoint indices found");
    return;
  }

  // Step 2: 按路径索引排序（关键修复！TSP顺序 != 路径遍历顺序）
  std::sort(index_pairs.begin(), index_pairs.end());

  // Step 3: 按排序后的顺序构建路径段
  int start_idx = 0;
  for (const auto& pair : index_pairs) {
    int end_idx = pair.first;
    size_t view_idx = pair.second;

    // 确保索引有效（排序后应该不会出现 end_idx < start_idx）
    if (end_idx < start_idx) {
      ROS_WARN("GlobalPlanner: Unexpected index order [%d, %d] for view %zu", start_idx, end_idx, view_idx);
      continue;
    }

    // 提取路径段（包含端点）
    std::vector<Eigen::Vector3f> segment_path;
    segment_path.reserve(end_idx - start_idx + 1);
    for (int j = start_idx; j <= end_idx; ++j) {
      segment_path.push_back(complete_viewpoint_path_[j]);
    }

    // 路径简化：使用 Visibility-Based Shortcutting
    std::vector<Eigen::Vector3f> simplified_path;
    if (!free_voxels_set.empty() && segment_path.size() > 2) {
      const Eigen::Vector3f& target_viewpoint = selected_views_[view_idx].position;
      simplified_path = simplifyPathVisibilityBased(
          segment_path, target_viewpoint, free_voxels_set, 2.0);

      ROS_DEBUG("GlobalPlanner: Segment %zu simplified: %zu -> %zu points",
                view_idx, segment_path.size(), simplified_path.size());
    } else {
      // 无法简化（free_voxels_set 为空或路径太短）
      simplified_path = segment_path;
      // 但仍需确保最后一个点是精确的 viewpoint
      if (!simplified_path.empty()) {
        simplified_path.back() = selected_views_[view_idx].position;
      }
    }

    // 创建PathSegmentWithView，使用简化后的路径
    PathSegmentWithView segment(simplified_path, selected_views_[view_idx]);
    path_segments_with_views_.push_back(segment);

    // 下一段从当前viewpoint开始
    start_idx = end_idx;

    ROS_DEBUG("GlobalPlanner: Segment: path_size=%zu, view_idx=%zu, target=[%.2f, %.2f, %.2f]",
              simplified_path.size(), view_idx,
              selected_views_[view_idx].position.x(),
              selected_views_[view_idx].position.y(),
              selected_views_[view_idx].position.z());
  }

  // 统计简化效果
  size_t total_original_points = complete_viewpoint_path_.size();
  size_t total_simplified_points = 0;
  for (const auto& seg : path_segments_with_views_) {
    total_simplified_points += seg.path.size();
  }

  double reduction_percentage = 0.0;
  if (total_original_points > 0) {
    reduction_percentage = 100.0 * (1.0 - static_cast<double>(total_simplified_points) /
                                           static_cast<double>(total_original_points));
  }

  ROS_INFO("GlobalPlanner: Built %zu path segments with views. "
           "Path simplified: %zu -> %zu waypoints (%.1f%% reduction)",
           path_segments_with_views_.size(),
           total_original_points, total_simplified_points,
           reduction_percentage);

  // 保存 free_voxels_set 供 replanFirstSegment 使用
  connected_free_voxels_set_ = std::move(free_voxels_set);
}

// ==========================================
// 路径简化函数实现 - Visibility-Based Shortcutting
// ==========================================

bool GlobalPlanner::isSegmentInFreeSpace(
    const Eigen::Vector3f& start,
    const Eigen::Vector3f& end,
    const std::unordered_set<Eigen::Vector3i, VoxelHash,
                             std::equal_to<Eigen::Vector3i>>& free_voxels_set) const {

  Eigen::Vector3f diff = end - start;
  float length = diff.norm();

  if (length < 1e-6f) return true;  // 重合点，认为安全

  // 采样间隔：voxel_size 的一半（0.1m），确保不漏检
  const float sample_interval = voxel_size_ * 0.5f;
  int num_samples = std::max(2, static_cast<int>(std::ceil(length / sample_interval)));

  Eigen::Vector3f step = diff / static_cast<float>(num_samples - 1);

  for (int i = 0; i < num_samples; ++i) {
    Eigen::Vector3f sample_pos = start + step * static_cast<float>(i);

    // 世界坐标转栅格索引 (with region_origin offset)
    Eigen::Vector3i voxel_idx(
        static_cast<int>(std::floor((sample_pos.x() - region_origin_.x()) / voxel_size_)),
        static_cast<int>(std::floor((sample_pos.y() - region_origin_.y()) / voxel_size_)),
        static_cast<int>(std::floor((sample_pos.z() - region_origin_.z()) / voxel_size_)));

    // 检查该 voxel 是否在安全集合中
    if (free_voxels_set.find(voxel_idx) == free_voxels_set.end()) {
      return false;  // 不在安全区域
    }
  }

  return true;
}

std::vector<Eigen::Vector3f> GlobalPlanner::simplifyPathVisibilityBased(
    const std::vector<Eigen::Vector3f>& raw_path,
    const Eigen::Vector3f& end_point,
    const std::unordered_set<Eigen::Vector3i, VoxelHash,
                             std::equal_to<Eigen::Vector3i>>& free_voxels_set,
    double max_segment_length) const {

  // 边界情况：空路径或单点
  if (raw_path.empty()) {
    return {end_point};
  }
  if (raw_path.size() == 1) {
    return {raw_path[0], end_point};
  }

  std::vector<Eigen::Vector3f> simplified;
  simplified.reserve(raw_path.size() / 5);  // 预估简化到 1/5
  simplified.push_back(raw_path[0]);

  int i = 0;
  while (i < static_cast<int>(raw_path.size()) - 1) {
    int farthest = i + 1;

    // 尝试跳过尽可能多的点
    for (int j = i + 2; j < static_cast<int>(raw_path.size()); ++j) {
      float distance = (raw_path[j] - raw_path[i]).norm();

      // 距离限制
      if (distance > max_segment_length) {
        break;
      }

      // 碰撞检查
      if (isSegmentInFreeSpace(raw_path[i], raw_path[j], free_voxels_set)) {
        farthest = j;
      } else {
        break;  // 碰撞，停止尝试更远的点
      }
    }

    simplified.push_back(raw_path[farthest]);
    i = farthest;
  }

  // 关键：替换最后一个点为精确的 viewpoint 位置
  if (!simplified.empty()) {
    simplified.back() = end_point;
  } else {
    simplified.push_back(end_point);
  }

  return simplified;
}

// ==========================================
// replanFirstSegment - 重规划第一段路径
// ==========================================
void GlobalPlanner::replanFirstSegment(const Eigen::Vector3f& new_start_position) {
  if (path_segments_with_views_.empty()) {
    ROS_WARN("GlobalPlanner::replanFirstSegment: No path segments to replan");
    return;
  }

  // 获取第一段的目标位置
  const Eigen::Vector3f& target_position = path_segments_with_views_[0].target_view.position;

  // 获取 region index（用于 API 兼容性，实际未使用）
  Eigen::Vector3i dummy_region_idx = Eigen::Vector3i::Zero();

  // 使用 batchGridAStar 重新规划从 new_start_position 到 target_position 的路径
  std::vector<Eigen::Vector3f> starts = {new_start_position};
  std::vector<Eigen::Vector3f> ends = {target_position};
  std::vector<GridAStarResult> results;

  batchGridAStar(dummy_region_idx, starts, ends, results);

  if (results[0].success && !results[0].path.empty()) {
    // 应用路径简化（与 buildPathSegmentsFromCompletePathWithViews 中相同的逻辑）
    std::vector<Eigen::Vector3f> simplified_path;
    if (!connected_free_voxels_set_.empty() && results[0].path.size() > 2) {
      simplified_path = simplifyPathVisibilityBased(
          results[0].path, target_position, connected_free_voxels_set_, 2.0);
    } else {
      simplified_path = results[0].path;
      // 确保最后一个点是精确的 viewpoint
      if (!simplified_path.empty()) {
        simplified_path.back() = target_position;
      }
    }

    // 更新第一段路径
    path_segments_with_views_[0].path = simplified_path;

    ROS_INFO("GlobalPlanner::replanFirstSegment: Replanned first segment from [%.2f, %.2f, %.2f] "
             "to [%.2f, %.2f, %.2f], path size: %zu -> %zu",
             new_start_position.x(), new_start_position.y(), new_start_position.z(),
             target_position.x(), target_position.y(), target_position.z(),
             results[0].path.size(), simplified_path.size());
  } else {
    ROS_WARN("GlobalPlanner::replanFirstSegment: A* failed, keeping original path");
  }
}

// ==========================================
// updatePreviousPathSegmentArrivalTimes - 更新viewpoint到达时间
// ==========================================
void GlobalPlanner::updatePreviousPathSegmentArrivalTimes(
    ros::Time start_time, const std::vector<double>& offsets) {
  for (size_t i = 0; i < previous_path_segments_with_views_.size() && i < offsets.size(); ++i) {
    previous_path_segments_with_views_[i].arrival_time = start_time + ros::Duration(offsets[i]);
  }
  ROS_INFO("GlobalPlanner: Updated arrival times for %zu viewpoints (start_time=%.2f)",
           std::min(previous_path_segments_with_views_.size(), offsets.size()),
           start_time.toSec());
}

// ==========================================
// getPassedAndUnpassedViewpoints - 获取已经过/未经过的viewpoints
// ==========================================
void GlobalPlanner::getPassedAndUnpassedViewpoints(
    std::vector<PathSegmentWithView>& passed,
    std::vector<PathSegmentWithView>& unpassed) const {
  passed.clear();
  unpassed.clear();

  ros::Time now = ros::Time::now() + ros::Duration(0.5);
  for (const auto& segment : previous_path_segments_with_views_) {
    // arrival_time > 0 表示已设置有效时间
    if (segment.arrival_time.toSec() > 0 && now >= segment.arrival_time) {
      passed.push_back(segment);
    } else {
      unpassed.push_back(segment);
    }
  }
}
