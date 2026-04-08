#include <observation_quality_manager/global_planner.h>
#include <lkh_tsp_solver/lkh_interface.h>
#include <fstream>
#include <omp.h>
#include <set>
#include <algorithm>
#include <queue>
#include <limits>

// Forward declarations
struct AStarNode;
static inline double getDiagonalHeuristic(const Eigen::Vector3i& a, const Eigen::Vector3i& b);
template<typename IsOccupied, typename ToWorld>
static bool gridAStarGlobal(
    const Eigen::Vector3i& start_grid,
    const Eigen::Vector3i& goal_grid,
    std::vector<Eigen::Vector3f>& path,
    IsOccupied is_occupied,
    ToWorld to_world,
    const LIOInterface::Ptr& lidar_map_interface);

bool GlobalPlanner::buildDistanceMatrix(
    const Eigen::Vector3f& odom_position,
    const std::vector<ClusterInfo>& clusters,
    Eigen::MatrixXd& cost_mat,
    std::vector<bool>& reachable) {

  if (!region_map_ptr_) {
    ROS_ERROR("GlobalPlanner: region_map_ptr is null");
    return false;
  }

  ros::Time start_time = ros::Time::now();

  int num_clusters = clusters.size();
  int dim = num_clusters + 1;  // +1 for odom (index 0)
  cost_mat.resize(dim, dim);
  cost_mat.setZero();
  reachable.assign(num_clusters, true);

  constexpr int VOXELS_PER_REGION = 30;  // 6m / 0.2m

  // Collision check lambda - directly queries global structures
  auto is_occupied = [&](const Eigen::Vector3i& voxel_idx) -> bool {
    // Check spatial_hash for obstacles
    if (spatial_hash_.count(voxel_idx)) return true;

    // Check region_map for free voxels
    auto floor_div = [](int a, int b) -> int {
      return (a >= 0) ? (a / b) : ((a - b + 1) / b);
    };
    Eigen::Vector3i region_idx(
      floor_div(voxel_idx.x(), VOXELS_PER_REGION),
      floor_div(voxel_idx.y(), VOXELS_PER_REGION),
      floor_div(voxel_idx.z(), VOXELS_PER_REGION));

    auto it = region_map_ptr_->find(region_idx);
    if (it != region_map_ptr_->end()) {
      FreeRegion::VoxelState state;
      if (it->second.getVoxelState(voxel_idx, state)) {
        if (state == FreeRegion::VoxelState::FREE ||
            state == FreeRegion::VoxelState::FRONTIER) {
          return false;  // Free
        }
      }
    }
    return true;  // Unknown = blocked
  };

  // World coordinate conversion lambda (with region_origin offset)
  auto to_world = [&](const Eigen::Vector3i& grid_pos) -> Eigen::Vector3f {
    Eigen::Vector3f offset = ((grid_pos.cast<float>().array() + 0.5f) * voxel_size_).matrix();
    return region_origin_ + offset;
  };

  // Grid coordinate conversion lambda (with region_origin offset)
  auto to_grid = [&](const Eigen::Vector3f& pos) -> Eigen::Vector3i {
    return Eigen::Vector3i(
      static_cast<int>(std::floor((pos.x() - region_origin_.x()) / voxel_size_)),
      static_cast<int>(std::floor((pos.y() - region_origin_.y()) / voxel_size_)),
      static_cast<int>(std::floor((pos.z() - region_origin_.z()) / voxel_size_)));
  };

  // Prepare cluster positions
  std::vector<Eigen::Vector3f> cluster_positions;
  cluster_positions.reserve(num_clusters);
  for (const auto& cluster : clusters) {
    cluster_positions.push_back(cluster.position);
  }

  // 1. Check if odom position is valid
  Eigen::Vector3i odom_grid = to_grid(odom_position);
  if (is_occupied(odom_grid)) {
    ROS_ERROR("GlobalPlanner: odom position is occupied, cannot compute distance matrix");
    return false;
  }

  // 2. Check if clusters are occupied and mark unreachable ones
  for (int i = 0; i < num_clusters; ++i) {
    Eigen::Vector3i cluster_grid = to_grid(cluster_positions[i]);
    if (is_occupied(cluster_grid)) {
      reachable[i] = false;
      ROS_DEBUG("GlobalPlanner: Cluster %d is occupied (unreachable)", i);
    }
  }

  // 3. Compute odom to all clusters distances (OpenMP parallel)
  omp_set_num_threads(4);
  auto lidar_map_interface_ptr = lidar_map_interface_;

  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < num_clusters; ++i) {
    if (!reachable[i]) {
      cost_mat(0, i + 1) = 1000.0;  // Unreachable: use large penalty value
      continue;
    }

    Eigen::Vector3i start_grid = to_grid(odom_position);
    Eigen::Vector3i goal_grid = to_grid(cluster_positions[i]);

    // Run Grid A*
    std::vector<Eigen::Vector3f> path;
    bool found = gridAStarGlobal(start_grid, goal_grid, path,
                                  is_occupied, to_world, lidar_map_interface_ptr);

    if (found && !path.empty()) {
      // Calculate path length
      double path_length = 0.0;
      for (size_t j = 1; j < path.size(); ++j) {
        path_length += (path[j] - path[j - 1]).norm();
      }
      cost_mat(0, i + 1) = path_length;
    } else {
      reachable[i] = false;
      cost_mat(0, i + 1) = 1000.0;  // Unreachable: use large penalty value
      ROS_DEBUG("GlobalPlanner: Cluster %d is unreachable (Grid A* failed)", i);
    }
  }

  // 4. Compute distances between clusters (OpenMP parallel)
  int num_pairs = num_clusters * (num_clusters - 1) / 2;
  std::vector<std::tuple<int, int, double>> pair_distances(num_pairs);

  #pragma omp parallel for schedule(dynamic)
  for (int pair_idx = 0; pair_idx < num_pairs; ++pair_idx) {
    // Recover (i, j) from linear index, where i < j
    int i = 0, j = 0;
    int k = pair_idx;
    for (i = 0; i < num_clusters - 1; ++i) {
      int remaining = num_clusters - 1 - i;
      if (k < remaining) {
        j = i + 1 + k;
        break;
      }
      k -= remaining;
    }

    double dist = 1000.0;  // Default unreachable: use large penalty value
    if (reachable[i] && reachable[j]) {
      Eigen::Vector3i start_grid = to_grid(cluster_positions[i]);
      Eigen::Vector3i goal_grid = to_grid(cluster_positions[j]);

      // Run Grid A*
      std::vector<Eigen::Vector3f> path;
      if (gridAStarGlobal(start_grid, goal_grid, path,
                          is_occupied, to_world, lidar_map_interface_ptr)) {
        // Calculate path length
        dist = 0.0;
        for (size_t idx = 1; idx < path.size(); ++idx) {
          dist += (path[idx] - path[idx - 1]).norm();
        }
      }
    }
    pair_distances[pair_idx] = std::make_tuple(i, j, dist);
  }

  // Write back to distance matrix
  for (const auto& [i, j, dist] : pair_distances) {
    cost_mat(i + 1, j + 1) = dist;
    cost_mat(j + 1, i + 1) = dist;  // Symmetric matrix
  }

  // 5. mat[i][0]: cluster i to odom distance (set to large value to avoid returning to start)
  // Reference fast_exploration_manager trick: 2e3 - distance * 0.2
  for (int i = 0; i < num_clusters; ++i) {
    if (!reachable[i]) {
      cost_mat(i + 1, 0) = 2e3;  // Unreachable cluster set to fixed large value
    } else {
      float euclidean_dist = (cluster_positions[i] - odom_position).norm();
      cost_mat(i + 1, 0) = 2e3 - euclidean_dist * 0.2;
    }
  }

  ros::Time end_time = ros::Time::now();
  int unreachable_count = std::count(reachable.begin(), reachable.end(), false);
  ROS_INFO("GlobalPlanner: Distance matrix built using Grid A* in %.3f ms, dimension: %d x %d, unreachable: %d",
           (end_time - start_time).toSec() * 1000.0, dim, dim, unreachable_count);

  return true;
}

void GlobalPlanner::solveTSP(Eigen::MatrixXd& cost_mat, std::vector<int>& indices) {
  ros::Time start_time = ros::Time::now();

  // 参考 FastExplorationManager::solveLHK 实现
  int dimension = cost_mat.rows();
  if (dimension < 3) {
    ROS_WARN("GlobalPlanner: TSP dimension < 3, skipping");
    return;
  }

  // 1. 写入 TSP 问题文件
  std::ofstream prob_file(tsp_dir_ + "/global_tsp.tsp");

  // TSPLIB 格式头部
  std::string prob_spec =
      "NAME : global_tsp\nTYPE : ATSP\nDIMENSION : " + std::to_string(dimension) +
      "\nEDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n";

  prob_file << prob_spec;

  // 写入距离矩阵 (缩放到整数, scale = 100)
  const int scale = 100;
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = static_cast<int>(cost_mat(i, j) * scale);
      prob_file << int_cost << " ";
    }
    prob_file << "\n";
  }

  prob_file << "EOF";
  prob_file.close();

  // 2. 调用 LKH TSP 求解器
  std::string par_file_path = tsp_dir_ + "/global_tsp.par";
  solveTSPLKH(par_file_path.c_str());

  // 3. 读取结果文件
  std::ifstream res_file(tsp_dir_ + "/global_tsp.txt");
  if (!res_file.is_open()) {
    ROS_ERROR("GlobalPlanner: Failed to open TSP result file");
    return;
  }

  std::string line;

  // 跳到 TOUR_SECTION
  while (std::getline(res_file, line)) {
    if (line.compare("TOUR_SECTION") == 0) {
      break;
    }
  }

  // 读取路径索引 (ATSP格式)
  indices.clear();
  while (std::getline(res_file, line)) {
    int id = std::stoi(line);
    if (id == -1) {
      break;
    }
    indices.push_back(id - 1);  // LKH 索引从1开始,转换为从0开始
  }

  res_file.close();

  ros::Time end_time = ros::Time::now();
  ROS_INFO("GlobalPlanner: LKH solver completed in %.3f ms, dimension: %d",
           (end_time - start_time).toSec() * 1000.0, dimension);

  ROS_INFO("GlobalPlanner: TSP solution found with %zu waypoints", indices.size());
}

// ==========================================
// Local Grid Structure
// ==========================================
struct LocalGrid {
  std::vector<uint8_t> data;  // 0=free, 1=occupied, 2=unknown
  Eigen::Vector3i origin;     // Grid origin in voxel coordinates
  Eigen::Vector3i size;       // Grid dimensions
  float voxel_size;

  LocalGrid(const Eigen::Vector3i& o, const Eigen::Vector3i& s, float vs)
    : origin(o), size(s), voxel_size(vs) {
    data.resize(s.x() * s.y() * s.z(), 2);  // Initialize as unknown
  }

  inline int toIndex(int x, int y, int z) const {
    return x * size.y() * size.z() + y * size.z() + z;
  }

  inline uint8_t get(int x, int y, int z) const {
    if (x < 0 || x >= size.x() || y < 0 || y >= size.y() || z < 0 || z >= size.z()) {
      return 1;  // Out of bounds = occupied
    }
    return data[toIndex(x, y, z)];
  }

  inline void set(int x, int y, int z, uint8_t value) {
    if (x >= 0 && x < size.x() && y >= 0 && y < size.y() && z >= 0 && z < size.z()) {
      data[toIndex(x, y, z)] = value;
    }
  }

  inline Eigen::Vector3i worldToGrid(const Eigen::Vector3f& pos) const {
    Eigen::Vector3i voxel_idx(
      static_cast<int>(std::floor(pos.x() / voxel_size)),
      static_cast<int>(std::floor(pos.y() / voxel_size)),
      static_cast<int>(std::floor(pos.z() / voxel_size)));
    return voxel_idx - origin;
  }

  inline Eigen::Vector3f gridToWorld(const Eigen::Vector3i& grid_pos) const {
    Eigen::Vector3i voxel_idx = grid_pos + origin;
    return (voxel_idx.cast<float>().array() + 0.5f) * voxel_size;
  }
};

// ==========================================
// A* Node Structure
// ==========================================
struct AStarNode {
  Eigen::Vector3i pos;
  double g_cost;  // Cost from start
  double f_cost;  // g_cost + heuristic
  AStarNode* parent;

  AStarNode(const Eigen::Vector3i& p, double g, double f, AStarNode* par = nullptr)
    : pos(p), g_cost(g), f_cost(f), parent(par) {}

  struct Compare {
    bool operator()(AStarNode* a, AStarNode* b) const {
      return a->f_cost > b->f_cost;  // Min-heap
    }
  };
};

// ==========================================
// A* Helper Functions
// ==========================================
static inline double getDiagonalHeuristic(const Eigen::Vector3i& a, const Eigen::Vector3i& b) {
  Eigen::Vector3i diff = (a - b).cwiseAbs();
  int dx = diff.x(), dy = diff.y(), dz = diff.z();
  int dmin = std::min({dx, dy, dz});
  int dmid = std::max({std::min(dx, dy), std::min(dy, dz), std::min(dz, dx)});
  int dmax = std::max({dx, dy, dz});
  // Diagonal heuristic: sqrt(3) * dmin + sqrt(2) * (dmid - dmin) + (dmax - dmid)
  return 1.732050808 * dmin + 1.414213562 * (dmid - dmin) + (dmax - dmid);
}

// ==========================================
// Global Grid A* - uses lambdas for collision check
// ==========================================
template<typename IsOccupied, typename ToWorld>
static bool gridAStarGlobal(
    const Eigen::Vector3i& start_grid,
    const Eigen::Vector3i& goal_grid,
    std::vector<Eigen::Vector3f>& path,
    IsOccupied is_occupied,
    ToWorld to_world,
    const LIOInterface::Ptr& lidar_map_interface) {

  // 26-neighborhood (including 3D diagonals)
  static const int dx26[26] = {1,-1,0,0,0,0, 1,1,-1,-1,1,1,-1,-1,0,0,0,0, 1,1,1,1,-1,-1,-1,-1};
  static const int dy26[26] = {0,0,1,-1,0,0, 1,-1,1,-1,0,0,0,0,1,1,-1,-1, 1,1,-1,-1,1,1,-1,-1};
  static const int dz26[26] = {0,0,0,0,1,-1, 0,0,0,0,1,-1,1,-1,1,-1,1,-1, 1,-1,1,-1,1,-1,1,-1};
  static const double cost26[26] = {
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  // Face neighbors
    1.414213562, 1.414213562, 1.414213562, 1.414213562,  // Edge neighbors (xy)
    1.414213562, 1.414213562, 1.414213562, 1.414213562,  // Edge neighbors (xz)
    1.414213562, 1.414213562, 1.414213562, 1.414213562,  // Edge neighbors (yz)
    1.732050808, 1.732050808, 1.732050808, 1.732050808,  // Corner neighbors
    1.732050808, 1.732050808, 1.732050808, 1.732050808   // Corner neighbors
  };

  // Use spatial hash for node keys
  auto make_key = [](const Eigen::Vector3i& p) -> int64_t {
    return static_cast<int64_t>(p.x()) * 73856093LL +
           static_cast<int64_t>(p.y()) * 19349663LL +
           static_cast<int64_t>(p.z()) * 83492791LL;
  };

  // Priority queue and node storage
  std::priority_queue<AStarNode*, std::vector<AStarNode*>, AStarNode::Compare> open_set;
  std::unordered_map<int64_t, AStarNode*> all_nodes;
  std::unordered_set<int64_t> closed_set;

  const bool enforce_box = static_cast<bool>(lidar_map_interface);
  auto is_inside_box = [&](const Eigen::Vector3i& grid_pos) {
    if (!enforce_box) return true;
    Eigen::Vector3f world_pos = to_world(grid_pos);
    return lidar_map_interface->IsInBox(world_pos);
  };

  if (enforce_box) {
    if (!is_inside_box(start_grid) || !is_inside_box(goal_grid)) {
      return false;
    }
  }

  AStarNode* start_node = new AStarNode(start_grid, 0.0,
    getDiagonalHeuristic(start_grid, goal_grid));
  open_set.push(start_node);
  all_nodes[make_key(start_grid)] = start_node;

  bool found = false;
  AStarNode* goal_node = nullptr;

  while (!open_set.empty()) {
    AStarNode* current = open_set.top();
    open_set.pop();

    int64_t current_key = make_key(current->pos);
    if (closed_set.count(current_key)) continue;
    closed_set.insert(current_key);

    // Goal check
    if (current->pos == goal_grid) {
      found = true;
      goal_node = current;
      break;
    }

    // Expand neighbors
    for (int i = 0; i < 26; ++i) {
      Eigen::Vector3i neighbor_pos(
        current->pos.x() + dx26[i],
        current->pos.y() + dy26[i],
        current->pos.z() + dz26[i]);

      if (enforce_box && !is_inside_box(neighbor_pos)) continue;

      int64_t neighbor_key = make_key(neighbor_pos);
      if (closed_set.count(neighbor_key)) continue;

      // Use lambda for collision check
      if (is_occupied(neighbor_pos)) continue;

      double tentative_g = current->g_cost + cost26[i];
      double h = getDiagonalHeuristic(neighbor_pos, goal_grid);
      double f = tentative_g + h;

      auto it = all_nodes.find(neighbor_key);
      if (it == all_nodes.end() || tentative_g < it->second->g_cost) {
        AStarNode* neighbor_node = new AStarNode(neighbor_pos, tentative_g, f, current);
        all_nodes[neighbor_key] = neighbor_node;
        open_set.push(neighbor_node);
      }
    }
  }

  // Reconstruct path
  if (found && goal_node) {
    std::vector<Eigen::Vector3i> grid_path;
    AStarNode* node = goal_node;
    while (node) {
      grid_path.push_back(node->pos);
      node = node->parent;
    }
    std::reverse(grid_path.begin(), grid_path.end());

    // Convert to world coordinates using lambda
    path.reserve(grid_path.size());
    for (const auto& gp : grid_path) {
      path.push_back(to_world(gp));
    }
  }

  // Cleanup
  for (auto& pair : all_nodes) {
    delete pair.second;
  }

  return found;
}

// ==========================================
// batchGridAStar Implementation (Global A*)
// ==========================================
void GlobalPlanner::batchGridAStar(
    const Eigen::Vector3i& /* region_idx */,  // unused, kept for API compatibility
    const std::vector<Eigen::Vector3f>& start_positions,
    const std::vector<Eigen::Vector3f>& end_positions,
    std::vector<GridAStarResult>& results) {

  results.resize(start_positions.size());

  if (start_positions.empty() || !region_map_ptr_) {
    ROS_ERROR("GlobalPlanner::batchGridAStar: Invalid input");
    return;
  }

  constexpr int VOXELS_PER_REGION = 30;  // 6m / 0.2m

  // Collision check lambda - directly queries global structures
  auto is_occupied = [&](const Eigen::Vector3i& voxel_idx) -> bool {
    // Check spatial_hash for obstacles
    if (spatial_hash_.count(voxel_idx)) return true;

    // Check region_map for free voxels
    auto floor_div = [](int a, int b) -> int {
      return (a >= 0) ? (a / b) : ((a - b + 1) / b);
    };
    Eigen::Vector3i region_idx(
      floor_div(voxel_idx.x(), VOXELS_PER_REGION),
      floor_div(voxel_idx.y(), VOXELS_PER_REGION),
      floor_div(voxel_idx.z(), VOXELS_PER_REGION));

    auto it = region_map_ptr_->find(region_idx);
    if (it != region_map_ptr_->end()) {
      FreeRegion::VoxelState state;
      if (it->second.getVoxelState(voxel_idx, state)) {
        if (state == FreeRegion::VoxelState::FREE ||
            state == FreeRegion::VoxelState::FRONTIER) {
          return false;  // Free
        }
      }
    }
    return true;  // Unknown = blocked
  };

  // World coordinate conversion lambda (with region_origin offset)
  auto to_world = [&](const Eigen::Vector3i& grid_pos) -> Eigen::Vector3f {
    Eigen::Vector3f offset = ((grid_pos.cast<float>().array() + 0.5f) * voxel_size_).matrix();
    return region_origin_ + offset;
  };

  // Grid coordinate conversion lambda (with region_origin offset)
  auto to_grid = [&](const Eigen::Vector3f& pos) -> Eigen::Vector3i {
    return Eigen::Vector3i(
      static_cast<int>(std::floor((pos.x() - region_origin_.x()) / voxel_size_)),
      static_cast<int>(std::floor((pos.y() - region_origin_.y()) / voxel_size_)),
      static_cast<int>(std::floor((pos.z() - region_origin_.z()) / voxel_size_)));
  };

  // Parallel A* Search (OpenMP)
  omp_set_num_threads(4);
  const bool enforce_box_constraint = static_cast<bool>(lidar_map_interface_);
  auto lidar_map_interface_ptr = lidar_map_interface_;

  #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < start_positions.size(); ++i) {
    GridAStarResult& result = results[i];

    Eigen::Vector3i start_grid = to_grid(start_positions[i]);
    Eigen::Vector3i goal_grid = to_grid(end_positions[i]);

    if (enforce_box_constraint) {
      if (!lidar_map_interface_ptr->IsInBox(start_positions[i])) {
        result.success = false;
        result.distance = 1000.0;
        ROS_DEBUG("GlobalPlanner::batchGridAStar: Start position is outside allowed box for pair %zu", i);
        continue;
      }
      if (!lidar_map_interface_ptr->IsInBox(end_positions[i])) {
        result.success = false;
        result.distance = 1000.0;
        ROS_DEBUG("GlobalPlanner::batchGridAStar: Goal position is outside allowed box for pair %zu", i);
        continue;
      }
    }

    // Check if start/goal are occupied
    if (is_occupied(start_grid)) {
      result.success = false;
      result.distance = 1000.0;
      ROS_DEBUG("GlobalPlanner::batchGridAStar: Start position is occupied for pair %zu", i);
      continue;
    }

    if (is_occupied(goal_grid)) {
      result.success = false;
      result.distance = 1000.0;
      ROS_DEBUG("GlobalPlanner::batchGridAStar: Goal position is occupied for pair %zu", i);
      continue;
    }

    // Run Global A*
    std::vector<Eigen::Vector3f> path;
    bool found = gridAStarGlobal(start_grid, goal_grid, path,
                                  is_occupied, to_world, lidar_map_interface_ptr);

    if (found && !path.empty()) {
      result.success = true;

      // 确保路径至少包含起点和终点（原始世界坐标）
      // 当起点和终点在同一栅格时，A*只返回1个点
      if (path.size() == 1) {
        result.path.clear();
        result.path.push_back(start_positions[i]);
        result.path.push_back(end_positions[i]);
        result.distance = (end_positions[i] - start_positions[i]).norm();
      } else {
        // 正常情况：替换首尾为精确世界坐标
        result.path = path;
        result.path.front() = start_positions[i];
        result.path.back() = end_positions[i];

        // Calculate distance
        result.distance = 0.0;
        for (size_t j = 1; j < result.path.size(); ++j) {
          result.distance += (result.path[j] - result.path[j - 1]).norm();
        }
      }
    } else {
      result.success = false;
      result.distance = 1000.0;
      ROS_DEBUG("GlobalPlanner::batchGridAStar: A* failed for pair %zu", i);
    }
  }

  ROS_DEBUG("GlobalPlanner::batchGridAStar: Processed %zu pairs", start_positions.size());
}

void GlobalPlanner::batchTopoAStar(
    const std::vector<Eigen::Vector3f>& start_positions,
    const std::vector<Eigen::Vector3f>& end_positions,
    std::vector<TopoAStarResult>& results) {

  ROS_DEBUG("GlobalPlanner: Running batchTopoAStar for %zu pairs", start_positions.size());
  results.resize(start_positions.size());

  if (start_positions.size() != end_positions.size() || start_positions.empty()) {
    ROS_WARN("GlobalPlanner: Invalid batchTopoAStar input sizes");
    return;
  }

  // 创建所有临时节点并批量插入
  std::vector<TopoNode::Ptr> temp_nodes;
  temp_nodes.reserve(start_positions.size() + end_positions.size());

  // 生成所有start和end的临时节点
  for (size_t i = 0; i < start_positions.size(); ++i) {
    TopoNode::Ptr start_node = std::make_shared<TopoNode>();
    start_node->center_ = start_positions[i];
    temp_nodes.push_back(start_node);

    TopoNode::Ptr end_node = std::make_shared<TopoNode>();
    end_node->center_ = end_positions[i];
    temp_nodes.push_back(end_node);
  }

  // 批量插入拓扑地图 (only_raycast=true 加速)
  graph_->insertNodes(temp_nodes, true);

  // 并行执行A*搜索
  omp_set_num_threads(4);
  #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < start_positions.size(); ++i) {
    TopoAStarResult& result = results[i];

    const TopoNode::Ptr& start_node = temp_nodes[2 * i];
    const TopoNode::Ptr& end_node = temp_nodes[2 * i + 1];

    std::vector<TopoNode::Ptr> path;
    if (graph_->graphSearch(start_node, end_node, path, 3e-4)) {
      result.distance = graph_->getPathLength(path);
      result.success = true;

      // 提取路径点
      for (const auto& node : path) {
        result.path.push_back(node->center_);
      }

      // 确保路径至少包含起点和终点
      if (result.path.size() == 1) {
        // 只有1个点时，添加起点和终点
        result.path.clear();
        result.path.push_back(start_positions[i]);
        result.path.push_back(end_positions[i]);
        result.distance = (end_positions[i] - start_positions[i]).norm();
      } else if (result.path.size() >= 2) {
        // 替换首尾为精确的原始坐标
        result.path.front() = start_positions[i];
        result.path.back() = end_positions[i];
      }
    } else {
      result.distance = 1000.0;  // 不可达：使用固定大惩罚值
      result.success = false;
      ROS_DEBUG("GlobalPlanner: TopoAStar failed for pair %zu", i);
    }
  }

  // 移除临时节点，避免内存泄漏
  graph_->removeNodes(temp_nodes);
}

std::vector<Eigen::Vector3f> GlobalPlanner::buildDistanceMatrixForViewpoints(
    const Eigen::Vector3f& odom_position,
    const std::vector<Eigen::Vector3f>& viewpoints,
    const Eigen::Vector3i& viewpoints_region_idx,
    const Eigen::Vector3f* next_cluster_position,
    const Eigen::Vector3i* next_cluster_region_idx,
    Eigen::MatrixXd& cost_mat,
    std::vector<int>& tsp_indices) {

  ros::Time func_start_time = ros::Time::now();
  ROS_DEBUG("GlobalPlanner: Building merged distance matrix...");
  path_to_first_viewpoint_.clear();

  int num_viewpoints = viewpoints.size();
  bool has_next_cluster = (next_cluster_position != nullptr && next_cluster_region_idx != nullptr);
  int dim = 1 + num_viewpoints + (has_next_cluster ? 1 : 0);  // odom + viewpoints + cluster2

  cost_mat.resize(dim, dim);
  cost_mat.setZero();

  // ------------------------------------------
  // 1. odom -> viewpoints 距离计算，并检查viewpoint可达性
  // ------------------------------------------
  // 计算odom所在的region
  Eigen::Vector3i odom_region = positionToRegionIdx(odom_position);
  // bool odom_adjacent_to_viewpoints = false;
  bool odom_adjacent_to_viewpoints = true;
  // bool odom_adjacent_to_viewpoints = areRegionsAdjacent(odom_region, viewpoints_region_idx);

  std::vector<Eigen::Vector3f> odom_to_viewpoints_starts(num_viewpoints, odom_position);
  std::vector<GridAStarResult> odom_to_viewpoints_grid_results;
  std::vector<TopoAStarResult> odom_to_viewpoints_topo_results;
  std::vector<bool> viewpoint_reachable(num_viewpoints, true);  // 记录viewpoint是否可达

  if (odom_adjacent_to_viewpoints) {
    // 使用栅格A*
    batchGridAStar(
        viewpoints_region_idx,
        odom_to_viewpoints_starts,
        viewpoints,
        odom_to_viewpoints_grid_results);

    for (int i = 0; i < num_viewpoints; ++i) {
      if (odom_to_viewpoints_grid_results[i].success) {
        cost_mat(0, i + 1) = odom_to_viewpoints_grid_results[i].distance;
      } else {
        cost_mat(0, i + 1) = 1000.0;  // 不可达：使用固定大惩罚值
        viewpoint_reachable[i] = false;  // 标记为不可达
      }
    }

  } else {
    // 使用拓扑A*
    batchTopoAStar(
        odom_to_viewpoints_starts,
        viewpoints,
        odom_to_viewpoints_topo_results);

    for (int i = 0; i < num_viewpoints; ++i) {
      if (odom_to_viewpoints_topo_results[i].success) {
        cost_mat(0, i + 1) = odom_to_viewpoints_topo_results[i].distance;
      } else {
        cost_mat(0, i + 1) = 1000.0;  // 不可达：使用固定大惩罚值
        viewpoint_reachable[i] = false;  // 标记为不可达
      }
    }
  }

  // 统计可达的viewpoint数量
  int num_reachable_viewpoints = std::count(viewpoint_reachable.begin(), viewpoint_reachable.end(), true);

  // 如果没有可达的viewpoint，直接返回
  if (num_reachable_viewpoints == 0) {
    ROS_WARN("GlobalPlanner: No reachable viewpoints found");
    tsp_indices.clear();
    complete_viewpoint_path_.clear();
    return std::vector<Eigen::Vector3f>();
  }

  // 如果只有一个可达的viewpoint，TSP退化为直接路径
  if (num_reachable_viewpoints == 1) {
    tsp_indices.push_back(0);  // odom
    int reachable_idx = -1;
    for (int i = 0; i < num_viewpoints; ++i) {
      if (viewpoint_reachable[i]) {
        tsp_indices.push_back(i + 1);  // 唯一可达的viewpoint
        reachable_idx = i;
        if (has_next_cluster) {
          tsp_indices.push_back(num_viewpoints + 1);  // cluster2
        }
        break;
      }
    }

    // 填充 path_to_first_viewpoint_，避免上层读取到空路径
    if (reachable_idx >= 0) {
      if (odom_adjacent_to_viewpoints &&
          reachable_idx < static_cast<int>(odom_to_viewpoints_grid_results.size())) {
        path_to_first_viewpoint_ =
            odom_to_viewpoints_grid_results[reachable_idx].path;
      } else if (!odom_adjacent_to_viewpoints &&
                 reachable_idx < static_cast<int>(odom_to_viewpoints_topo_results.size())) {
        path_to_first_viewpoint_ =
            odom_to_viewpoints_topo_results[reachable_idx].path;
      }

      // 若A*路径为空，至少返回一条直接的直线路径
      if (path_to_first_viewpoint_.empty()) {
        path_to_first_viewpoint_.push_back(odom_position);
        path_to_first_viewpoint_.push_back(viewpoints[reachable_idx]);
      }
    }

    complete_viewpoint_path_ = path_to_first_viewpoint_;
    return path_to_first_viewpoint_;
  }

  // ------------------------------------------
  // 1.5. 过滤不可达的viewpoints（避免大矩阵传给LKH）
  // ------------------------------------------
  std::vector<Eigen::Vector3f> reachable_viewpoints;
  std::vector<int> original_viewpoint_indices;  // 映射回原始索引
  std::vector<GridAStarResult> reachable_grid_results;
  std::vector<TopoAStarResult> reachable_topo_results;

  reachable_viewpoints.reserve(num_reachable_viewpoints);
  original_viewpoint_indices.reserve(num_reachable_viewpoints);

  for (int i = 0; i < num_viewpoints; ++i) {
    if (viewpoint_reachable[i]) {
      reachable_viewpoints.push_back(viewpoints[i]);
      original_viewpoint_indices.push_back(i);
    }
  }

  // 如果有不可达的viewpoint被过滤，需要重新构建cost_mat子矩阵
  if (reachable_viewpoints.size() < (size_t)num_viewpoints) {
    ROS_INFO("GlobalPlanner: Filtering %zu/%d unreachable viewpoints",
             num_viewpoints - reachable_viewpoints.size(), num_viewpoints);

    // 提取子矩阵：保留 odom (index 0) 和可达 viewpoints
    int new_dim = reachable_viewpoints.size() + 1 + (has_next_cluster ? 1 : 0);
    Eigen::MatrixXd new_cost_mat(new_dim, new_dim);
    new_cost_mat.setZero();

    // odom -> odom
    new_cost_mat(0, 0) = cost_mat(0, 0);

    // odom <-> 可达 viewpoints
    for (size_t i = 0; i < original_viewpoint_indices.size(); ++i) {
      int old_idx = original_viewpoint_indices[i] + 1;  // 原矩阵中的索引 (+1 因为 odom 在 0)
      int new_idx = i + 1;                               // 新矩阵中的索引
      new_cost_mat(0, new_idx) = cost_mat(0, old_idx);
      new_cost_mat(new_idx, 0) = cost_mat(old_idx, 0);
    }

    // odom -> next_cluster (如果存在)
    if (has_next_cluster) {
      int old_cluster_idx = num_viewpoints + 1;
      int new_cluster_idx = reachable_viewpoints.size() + 1;
      new_cost_mat(0, new_cluster_idx) = cost_mat(0, old_cluster_idx);
      new_cost_mat(new_cluster_idx, 0) = cost_mat(old_cluster_idx, 0);
    }

    // 提取对应的A*路径结果
    if (odom_adjacent_to_viewpoints) {
      reachable_grid_results.reserve(reachable_viewpoints.size());
      for (int orig_idx : original_viewpoint_indices) {
        reachable_grid_results.push_back(odom_to_viewpoints_grid_results[orig_idx]);
      }
    } else {
      reachable_topo_results.reserve(reachable_viewpoints.size());
      for (int orig_idx : original_viewpoint_indices) {
        reachable_topo_results.push_back(odom_to_viewpoints_topo_results[orig_idx]);
      }
    }

    int old_dim = cost_mat.rows();
    cost_mat = new_cost_mat;
    num_viewpoints = reachable_viewpoints.size();

    // 替换结果向量
    odom_to_viewpoints_grid_results = reachable_grid_results;
    odom_to_viewpoints_topo_results = reachable_topo_results;

    ROS_INFO("GlobalPlanner: Extracted %d x %d sub-matrix from original %d x %d matrix",
             new_dim, new_dim, old_dim, old_dim);
  }

  // 更新dim以匹配新的矩阵大小
  dim = 1 + num_viewpoints + (has_next_cluster ? 1 : 0);

  // ------------------------------------------
  // 2. viewpoints 之间的距离计算 (使用逐源Dijkstra)
  // 注意: 此时 num_viewpoints 已经是过滤后的可达viewpoint数量
  // ------------------------------------------
  ros::Time dijkstra_start = ros::Time::now();

  // 使用过滤后的viewpoints
  const std::vector<Eigen::Vector3f>& vps_to_use =
      reachable_viewpoints.empty() ? viewpoints : reachable_viewpoints;

  // 2.1 构建 safe_free_voxels 集合（删除膨胀区域）
  std::unordered_set<Eigen::Vector3i, VoxelHash, std::equal_to<Eigen::Vector3i>> safe_free_voxels(
      connected_free_voxels_.begin(), connected_free_voxels_.end());

  size_t original_size = safe_free_voxels.size();
  dilateOccupiedVoxels(aabb_occupied_indices_, safe_free_voxels);
  ROS_INFO("GlobalPlanner: Safe free voxels: %zu (original: %zu, removed: %zu)",
           safe_free_voxels.size(), original_size, original_size - safe_free_voxels.size());

  // 2.2 构建 voxel_to_index 和 index_to_voxel 映射
  std::unordered_map<Eigen::Vector3i, int, VoxelHash, std::equal_to<Eigen::Vector3i>> voxel_to_index;
  std::vector<Eigen::Vector3i> index_to_voxel;
  index_to_voxel.reserve(safe_free_voxels.size());

  int voxel_idx = 0;
  for (const auto& v : safe_free_voxels) {
    voxel_to_index[v] = voxel_idx;
    index_to_voxel.push_back(v);
    ++voxel_idx;
  }

  // 2.3 将 viewpoints 映射到最近的 safe voxel
  std::vector<int> viewpoint_voxel_indices(num_viewpoints, -1);
  for (int i = 0; i < num_viewpoints; ++i) {
    Eigen::Vector3i vp_voxel(
        static_cast<int>(std::floor((vps_to_use[i].x() - region_origin_.x()) / voxel_size_)),
        static_cast<int>(std::floor((vps_to_use[i].y() - region_origin_.y()) / voxel_size_)),
        static_cast<int>(std::floor((vps_to_use[i].z() - region_origin_.z()) / voxel_size_)));

    auto it = voxel_to_index.find(vp_voxel);
    if (it != voxel_to_index.end()) {
      viewpoint_voxel_indices[i] = it->second;
    } else {
      // BFS 找最近的 safe voxel
      std::queue<Eigen::Vector3i> bfs_queue;
      std::unordered_set<Eigen::Vector3i, VoxelHash, std::equal_to<Eigen::Vector3i>> bfs_visited;
      bfs_queue.push(vp_voxel);
      bfs_visited.insert(vp_voxel);

      static const Eigen::Vector3i nbr6[6] = {
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

      while (!bfs_queue.empty() && bfs_visited.size() < 1000) {
        Eigen::Vector3i curr = bfs_queue.front();
        bfs_queue.pop();

        auto found_it = voxel_to_index.find(curr);
        if (found_it != voxel_to_index.end()) {
          viewpoint_voxel_indices[i] = found_it->second;
          break;
        }

        for (const auto& n : nbr6) {
          Eigen::Vector3i next = curr + n;
          if (bfs_visited.count(next) == 0) {
            bfs_visited.insert(next);
            bfs_queue.push(next);
          }
        }
      }
    }
  }

  // 2.4 26-邻域偏移和代价
  static const int dx26[26] = {1,-1,0,0,0,0, 1,1,-1,-1,1,1,-1,-1,0,0,0,0, 1,1,1,1,-1,-1,-1,-1};
  static const int dy26[26] = {0,0,1,-1,0,0, 1,-1,1,-1,0,0,0,0,1,1,-1,-1, 1,1,-1,-1,1,1,-1,-1};
  static const int dz26[26] = {0,0,0,0,1,-1, 0,0,0,0,1,-1,1,-1,1,-1,1,-1, 1,-1,1,-1,1,-1,1,-1};
  static const double cost26[26] = {
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.414213562, 1.414213562, 1.414213562, 1.414213562,
      1.414213562, 1.414213562, 1.414213562, 1.414213562,
      1.414213562, 1.414213562, 1.414213562, 1.414213562,
      1.732050808, 1.732050808, 1.732050808, 1.732050808,
      1.732050808, 1.732050808, 1.732050808, 1.732050808};

  // 2.5 存储距离和路径
  std::vector<std::vector<double>> all_distances(num_viewpoints, std::vector<double>(num_viewpoints, 1000.0));
  std::vector<std::vector<int>> all_predecessors(num_viewpoints);

  // 2.6 逐源 Dijkstra（OpenMP 并行）
  int num_voxels = static_cast<int>(index_to_voxel.size());

  omp_set_num_threads(4);
  #pragma omp parallel for schedule(dynamic)
  for (int src = 0; src < num_viewpoints; ++src) {
    int src_voxel = viewpoint_voxel_indices[src];
    if (src_voxel < 0) continue;

    // 收集目标 voxel 集合
    std::unordered_set<int> remaining_targets;
    for (int dst = src + 1; dst < num_viewpoints; ++dst) {
      int dst_voxel = viewpoint_voxel_indices[dst];
      if (dst_voxel >= 0) {
        remaining_targets.insert(dst_voxel);
      }
    }

    if (remaining_targets.empty()) continue;

    // Dijkstra 初始化
    std::vector<double> dist(num_voxels, std::numeric_limits<double>::infinity());
    std::vector<int> pred(num_voxels, -1);
    dist[src_voxel] = 0.0;

    using PQNode = std::pair<double, int>;
    std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> pq;
    pq.push({0.0, src_voxel});

    // Dijkstra 主循环（带 early termination）
    while (!pq.empty() && !remaining_targets.empty()) {
      auto [d, u] = pq.top();
      pq.pop();

      if (d > dist[u]) continue;

      // 检查是否到达目标
      if (remaining_targets.erase(u)) {
        // 找到对应的 viewpoint 索引
        for (int dst = src + 1; dst < num_viewpoints; ++dst) {
          if (viewpoint_voxel_indices[dst] == u) {
            all_distances[src][dst] = d * voxel_size_;
            all_distances[dst][src] = d * voxel_size_;
          }
        }
      }

      // 26-邻域扩展
      const Eigen::Vector3i& curr_voxel = index_to_voxel[u];
      for (int i = 0; i < 26; ++i) {
        Eigen::Vector3i neighbor_voxel(
            curr_voxel.x() + dx26[i],
            curr_voxel.y() + dy26[i],
            curr_voxel.z() + dz26[i]);

        auto it = voxel_to_index.find(neighbor_voxel);
        if (it == voxel_to_index.end()) continue;

        int v = it->second;
        double new_dist = d + cost26[i];
        if (new_dist < dist[v]) {
          dist[v] = new_dist;
          pred[v] = u;
          pq.push({new_dist, v});
        }
      }
    }

    // 存储 predecessors（用于路径重建）
    #pragma omp critical
    {
      all_predecessors[src] = std::move(pred);
    }
  }

  // 2.7 填充距离矩阵
  for (int i = 0; i < num_viewpoints; ++i) {
    for (int j = i + 1; j < num_viewpoints; ++j) {
      cost_mat(i + 1, j + 1) = all_distances[i][j];
      cost_mat(j + 1, i + 1) = all_distances[i][j];
    }
  }

  // 2.8 路径重建（存储到 vp_pair_grid_results 用于后续兼容）
  int num_viewpoint_pairs = num_viewpoints * (num_viewpoints - 1) / 2;
  std::vector<GridAStarResult> vp_pair_grid_results(num_viewpoint_pairs);

  int pair_idx = 0;
  for (int i = 0; i < num_viewpoints; ++i) {
    for (int j = i + 1; j < num_viewpoints; ++j) {
      GridAStarResult& result = vp_pair_grid_results[pair_idx];
      result.distance = all_distances[i][j];
      result.success = (all_distances[i][j] < 999.0);

      if (result.success && !all_predecessors[i].empty()) {
        // 重建路径
        int dst_voxel = viewpoint_voxel_indices[j];
        std::vector<Eigen::Vector3i> grid_path;

        int curr = dst_voxel;
        while (curr >= 0 && curr < num_voxels) {
          grid_path.push_back(index_to_voxel[curr]);
          curr = all_predecessors[i][curr];
        }
        std::reverse(grid_path.begin(), grid_path.end());

        // 转换为世界坐标 (with region_origin offset)
        result.path.reserve(grid_path.size());
        for (const auto& g : grid_path) {
          Eigen::Vector3f offset = ((g.cast<float>().array() + 0.5f) * voxel_size_).matrix();
          result.path.push_back(region_origin_ + offset);
        }

        // 替换首尾为精确坐标
        if (!result.path.empty()) {
          result.path.front() = vps_to_use[i];
          result.path.back() = vps_to_use[j];
        }
      }

      ++pair_idx;
    }
  }

  ros::Time dijkstra_end = ros::Time::now();
  ROS_INFO("GlobalPlanner: Dijkstra completed in %.3f ms (%d viewpoints, %d pairs, %d safe voxels)",
           (dijkstra_end - dijkstra_start).toSec() * 1000.0,
           num_viewpoints, num_viewpoint_pairs, num_voxels);

  // ------------------------------------------
  // 3. viewpoints -> next_cluster (如果存在)
  // 注意: 此时 num_viewpoints 已经是过滤后的可达viewpoint数量
  // ------------------------------------------
  if (has_next_cluster) {
    int cluster2_idx = num_viewpoints + 1;
    bool viewpoints_adjacent_to_cluster2 = areRegionsAdjacent(viewpoints_region_idx, *next_cluster_region_idx);

    std::vector<GridAStarResult> vp_to_cluster2_grid_results;
    std::vector<TopoAStarResult> vp_to_cluster2_topo_results;

    // 使用过滤后的viewpoints（如果有过滤）
    const std::vector<Eigen::Vector3f>& vps_to_use =
        reachable_viewpoints.empty() ? viewpoints : reachable_viewpoints;

    if (viewpoints_adjacent_to_cluster2) {
      // 使用栅格A*
      batchGridAStar(
          viewpoints_region_idx,
          vps_to_use,
          std::vector<Eigen::Vector3f>(num_viewpoints, *next_cluster_position),
          vp_to_cluster2_grid_results);

      for (int i = 0; i < num_viewpoints; ++i) {
        // 所有viewpoints此时都是可达的（已过滤）
        if (vp_to_cluster2_grid_results[i].success) {
          cost_mat(i + 1, cluster2_idx) = vp_to_cluster2_grid_results[i].distance;
          cost_mat(cluster2_idx, i + 1) = vp_to_cluster2_grid_results[i].distance;  // 对称
        } else {
          cost_mat(i + 1, cluster2_idx) = 1000.0;
          cost_mat(cluster2_idx, i + 1) = 1000.0;
        }
      }
    } else {
      // 使用拓扑A*
      batchTopoAStar(
          vps_to_use,
          std::vector<Eigen::Vector3f>(num_viewpoints, *next_cluster_position),
          vp_to_cluster2_topo_results);

      for (int i = 0; i < num_viewpoints; ++i) {
        // 所有viewpoints此时都是可达的（已过滤）
        if (vp_to_cluster2_topo_results[i].success) {
          cost_mat(i + 1, cluster2_idx) = vp_to_cluster2_topo_results[i].distance;
          cost_mat(cluster2_idx, i + 1) = vp_to_cluster2_topo_results[i].distance;
        } else {
          cost_mat(i + 1, cluster2_idx) = 1000.0;
          cost_mat(cluster2_idx, i + 1) = 1000.0;
        }
      }
    }
  }

  // ------------------------------------------
  // 4. 所有节点回 odom 的距离 (TSP trick)
  // 注意: 此时 num_viewpoints 已经是过滤后的可达viewpoint数量
  // ------------------------------------------
  // 使用过滤后的viewpoints（如果有过滤）
  const std::vector<Eigen::Vector3f>& vps_for_trick =
      reachable_viewpoints.empty() ? viewpoints : reachable_viewpoints;

  if (has_next_cluster) {
    int cluster2_idx = num_viewpoints + 1;
    // 多cluster情况：使用trick让cluster2成为终点
    for (int i = 1; i < dim; ++i) {
      if (i == cluster2_idx) {
        cost_mat(i, 0) = 0.0;  // cluster2 -> odom: 设为0，成为终点
      } else {
        Eigen::Vector3f from_pos = (i <= num_viewpoints) ? vps_for_trick[i - 1] : *next_cluster_position;
        float euclidean_dist = (odom_position - from_pos).norm();
        cost_mat(i, 0) = 99999.0 - euclidean_dist * 0.2;  // 参考 fast_exploration_manager
      }
    }
  } else {
    // 单cluster情况：设置为0，让路径自然结束
    for (int i = 1; i < dim; ++i) {
      cost_mat(i, 0) = 0.0;
    }
  }

  // ------------------------------------------
  // 5. 调用TSP求解
  // ------------------------------------------
  solveTSP(cost_mat, tsp_indices);

  // ------------------------------------------
  // 6. 提取到第一个viewpoint的路径
  // ------------------------------------------
  std::vector<Eigen::Vector3f> empty_path;

  if (tsp_indices.empty()) {
    ROS_WARN("GlobalPlanner: TSP solver failed in merged distance matrix builder");
    complete_viewpoint_path_.clear();
    return empty_path;
  }

  // 找到第一个viewpoint的索引
  int first_viewpoint_tsp_idx = -1;
  for (int idx : tsp_indices) {
    if (idx > 0 && idx <= num_viewpoints) {  // 1~num_viewpoints 是viewpoints
      first_viewpoint_tsp_idx = idx;
      break;
    }
  }

  if (first_viewpoint_tsp_idx == -1) {
    ROS_WARN("GlobalPlanner: No valid first viewpoint found in TSP result");
    complete_viewpoint_path_.clear();
    return empty_path;
  }

  int first_viewpoint_real_idx = first_viewpoint_tsp_idx - 1;

  // 获取到第一个viewpoint的路径
  if (odom_adjacent_to_viewpoints) {
    // 栅格路径
    path_to_first_viewpoint_ = odom_to_viewpoints_grid_results[first_viewpoint_real_idx].path;
    ROS_INFO("GlobalPlanner: Using GridA* for path_to_first_viewpoint");
  } else {
    // 拓扑路径
    path_to_first_viewpoint_ = odom_to_viewpoints_topo_results[first_viewpoint_real_idx].path;
    ROS_INFO("GlobalPlanner: Using TopoA* for path_to_first_viewpoint");
  }

  // 打印路径的起点和终点
  if (!path_to_first_viewpoint_.empty()) {
    const auto& start_pt = path_to_first_viewpoint_.front();
    const auto& end_pt = path_to_first_viewpoint_.back();
    ROS_INFO("GlobalPlanner: path_to_first_viewpoint_ size=%zu, start=[%f, %f, %f], end=[%f, %f, %f]",
             path_to_first_viewpoint_.size(),
             start_pt.x(), start_pt.y(), start_pt.z(),
             end_pt.x(), end_pt.y(), end_pt.z());
  } else {
    ROS_WARN("GlobalPlanner: path_to_first_viewpoint_ is EMPTY!");
  }

  // ------------------------------------------
  // 7. 构建完整的路径（经过所有viewpoints，TSP顺序，不包括next_cluster）
  // ------------------------------------------
  complete_viewpoint_path_.clear();

  // Lambda helper: 获取两个TSP节点之间的路径段
  auto getPathSegment = [&](int from_tsp_idx, int to_tsp_idx) -> std::vector<Eigen::Vector3f> {
    // Case 1: odom → viewpoint
    if (from_tsp_idx == 0 && to_tsp_idx >= 1 && to_tsp_idx <= num_viewpoints) {
      int vp_idx = to_tsp_idx - 1;  // 转换为0-based索引
      if (odom_adjacent_to_viewpoints) {
        if (vp_idx < static_cast<int>(odom_to_viewpoints_grid_results.size())) {
          return odom_to_viewpoints_grid_results[vp_idx].path;
        }
      } else {
        if (vp_idx < static_cast<int>(odom_to_viewpoints_topo_results.size())) {
          return odom_to_viewpoints_topo_results[vp_idx].path;
        }
      }
    }

    // Case 2: viewpoint → viewpoint
    if (from_tsp_idx >= 1 && from_tsp_idx <= num_viewpoints &&
        to_tsp_idx >= 1 && to_tsp_idx <= num_viewpoints) {
      int vp_i = from_tsp_idx - 1;  // 转换为0-based索引
      int vp_j = to_tsp_idx - 1;
      bool need_reverse = (vp_i > vp_j);
      if (need_reverse) {
        std::swap(vp_i, vp_j);
      }

      // 计算pair索引
      int pair_idx = vp_i * (num_viewpoints - 1) - vp_i * (vp_i - 1) / 2 + (vp_j - vp_i - 1);

      if (pair_idx >= 0 && pair_idx < static_cast<int>(vp_pair_grid_results.size())) {
        std::vector<Eigen::Vector3f> path = vp_pair_grid_results[pair_idx].path;

        // 如果交换了索引，需要反转路径
        if (need_reverse && !path.empty()) {
          std::reverse(path.begin(), path.end());
        }
        return path;
      }
    }

    // Invalid case (e.g., involving next_cluster)
    return std::vector<Eigen::Vector3f>();
  };

  // Lambda helper: 追加路径段，避免重复端点
  auto appendPathSegment = [&](const std::vector<Eigen::Vector3f>& segment, bool skip_first) {
    if (segment.empty()) return;

    int start_idx = skip_first ? 1 : 0;
    for (int i = start_idx; i < static_cast<int>(segment.size()); ++i) {
      complete_viewpoint_path_.push_back(segment[i]);
    }
  };

  // 找到TSP tour中最后一个viewpoint的位置（排除next_cluster）
  int last_viewpoint_position = -1;
  for (int i = static_cast<int>(tsp_indices.size()) - 1; i >= 0; --i) {
    int idx = tsp_indices[i];
    if (idx > 0 && idx <= num_viewpoints) {
      last_viewpoint_position = i;
      break;
    }
  }

  if (last_viewpoint_position == -1) {
    ROS_WARN("GlobalPlanner: No valid viewpoints in TSP tour for complete path");
    return path_to_first_viewpoint_;  // Fallback
  }

  // 构建完整路径：拼接各段A*路径
  for (int i = 0; i < last_viewpoint_position; ++i) {
    int from_idx = tsp_indices[i];
    int to_idx = tsp_indices[i + 1];

    // 跳过涉及next_cluster的段
    if (has_next_cluster) {
      int cluster_idx = num_viewpoints + 1;
      if (from_idx == cluster_idx || to_idx == cluster_idx) {
        continue;
      }
    }

    // 获取路径段
    std::vector<Eigen::Vector3f> segment = getPathSegment(from_idx, to_idx);

    if (segment.empty()) {
      ROS_WARN("GlobalPlanner: Empty segment from TSP idx %d to %d", from_idx, to_idx);
      continue;
    }

    // 追加路径段（如果不是第一段，跳过首点以避免重复）
    bool skip_first = !complete_viewpoint_path_.empty();
    appendPathSegment(segment, skip_first);
  }

  // 打印完整路径信息
  if (!complete_viewpoint_path_.empty()) {
    const auto& start_pt = complete_viewpoint_path_.front();
    const auto& end_pt = complete_viewpoint_path_.back();
    ROS_INFO("GlobalPlanner: complete_viewpoint_path_ size=%zu, start=[%.2f,%.2f,%.2f], end=[%.2f,%.2f,%.2f]",
             complete_viewpoint_path_.size(),
             start_pt.x(), start_pt.y(), start_pt.z(),
             end_pt.x(), end_pt.y(), end_pt.z());
  } else {
    ROS_WARN("GlobalPlanner: complete_viewpoint_path_ is EMPTY!");
  }

  // Visualize the complete path points
  visualizeCompleteViewpointPath();

  // ------------------------------------------
  // 8. 重映射TSP索引到原始视点索引（如果有过滤）
  // ------------------------------------------
  if (!original_viewpoint_indices.empty()) {
    int original_num_vps = static_cast<int>(viewpoints.size());
    for (int& idx : tsp_indices) {
      // 跳过 odom (idx=0)
      if (idx > 0 && idx <= num_viewpoints) {
        // 视点索引：从过滤后索引映射回原始索引
        int filtered_vp_idx = idx - 1;  // 转换为0-based
        idx = original_viewpoint_indices[filtered_vp_idx] + 1;  // 映射回原始索引(1-based)
      } else if (has_next_cluster && idx == num_viewpoints + 1) {
        // next_cluster索引：从过滤后的N+1映射回原始的N'+1
        idx = original_num_vps + 1;
      }
    }
    ROS_INFO("GlobalPlanner: Remapped TSP indices to original viewpoint indices (filtered=%d, original=%d)",
             num_viewpoints, original_num_vps);
  }

  ros::Time func_end_time = ros::Time::now();
  ROS_INFO("GlobalPlanner: buildDistanceMatrixForViewpoints completed in %.3f ms (%d viewpoints)",
           (func_end_time - func_start_time).toSec() * 1000.0, num_viewpoints);

  return path_to_first_viewpoint_;
}
