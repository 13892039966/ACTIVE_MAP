#pragma once

#include <Eigen/Eigen>
#include <bitset>
#include <memory>
#include <ros/ros.h>
#include <unordered_map>
#include <vector>
#include <observation_quality_manager/observation_quality_manager.h>
#include <observation_quality_manager/gpu_types.h>
#include <visualization_msgs/MarkerArray.h>
#include <pointcloud_topo/graph.h>
#include <sop_solver/sop_solver_interface.h>


// ==========================================
// ClusterInfo - 存储聚类信息
// ==========================================
struct ClusterInfo {
  Eigen::Vector3f position;       // 聚类中心的世界坐标
  int voxel_count;                // 体素数量
  int poorly_observed_count;      // 邻近的 poorly-observed 体素数量
  Eigen::Vector3i voxel_idx;      // 原始体素索引 (用于调试)

  // 层级规划相关字段
  Eigen::Vector3i region_idx;     // 所属 region 的索引
  std::vector<Eigen::Vector3i> poorly_observed_voxels;  // poorly-observed 体素索引列表
  std::vector<Eigen::Vector3i> cluster_free_voxels;     // 该 cluster 的所有 free voxels
};

// ==========================================
// SelectedView - 带朝向的视点
// ==========================================
struct SelectedView {
  Eigen::Vector3f position;  // 视点位置 (世界坐标)
  float yaw;                 // 偏航角 [-π, π], 0 = +X方向
  float pitch;               // 俯仰角 [pitch_min, pitch_max], 0 = 水平
  float score;               // 该视点的累计得分 (调试用)
  int viewpoint_idx;         // 在采样列表中的原始索引，用于追踪可见性
};

struct SelectedViewResult {
  std::vector<SelectedView> views;
  VisibilityCSR filtered_csr;
  std::vector<std::bitset<20>> final_target_masks;  // 选择结束后的 target 掩码状态
};

// 路径段结构体：包含到达某个viewpoint的路径和目标ViewPose
struct PathSegmentWithView {
  std::vector<Eigen::Vector3f> path;  // 到达这个viewpoint的A*路径段
  SelectedView target_view;            // 目标viewpoint的信息（位置、朝向、得分）
  ros::Time arrival_time;              // 到达这个viewpoint的绝对时间戳

  PathSegmentWithView() = default;

  PathSegmentWithView(const std::vector<Eigen::Vector3f>& p, const SelectedView& v)
      : path(p), target_view(v), arrival_time(ros::Time(0)) {}

  PathSegmentWithView(const std::vector<Eigen::Vector3f>& p, const SelectedView& v, ros::Time t)
      : path(p), target_view(v), arrival_time(t) {}
};

// ==========================================
// GridAStarResult - 栅格地图A*搜索结果
// ==========================================
struct GridAStarResult {
  std::vector<Eigen::Vector3f> path;  // 路径点序列 (从起点到终点)
  double distance;                     // 路径长度
  bool success;                        // 是否找到路径

  GridAStarResult() : distance(0.0), success(false) {}
};

// ==========================================
// TopoAStarResult - 拓扑地图A*搜索结果
// ==========================================
struct TopoAStarResult {
  std::vector<Eigen::Vector3f> path;  // 路径点序列 (从起点到终点)
  double distance;                     // 路径长度
  bool success;                        // 是否找到路径

  TopoAStarResult() : distance(0.0), success(false) {}
};

// ==========================================
// GlobalPlanner - TSP全局路径规划器
// ==========================================
class GlobalPlanner {
public:
  typedef std::shared_ptr<GlobalPlanner> Ptr;
  struct ClusterObservationInfo{
    unordered_map<Eigen::Vector3i, bool, VoxelHash, std::equal_to<Eigen::Vector3i>> local_observation_map;
  };
  // 构造函数
  GlobalPlanner(ros::NodeHandle& nh, float voxel_size,
                std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                                   std::equal_to<Eigen::Vector3i>>& spatial_hash,
                const Eigen::Vector3f& map_min_bd,
                LIOInterface::Ptr lidar_map_interface,
                TopoGraph::Ptr graph,
                std::unordered_map<Eigen::Vector3i, FreeRegion, RegionHash,
                                   std::equal_to<Eigen::Vector3i>>& region_map);

  // 主要规划函数: 从odom位置规划经过cluster中心点的TSP路径
  // 返回: true=规划成功, false=无可用cluster或规划失败
  bool planGlobalTSPPath(
      const Eigen::Vector3f& odom_position,
      float odom_yaw,    // 当前机器人的偏航角 [-π, π]
      float odom_pitch,  // 当前机器人的俯仰角
      std::unordered_map<Eigen::Vector3i, FreeRegion, RegionHash,
                         std::equal_to<Eigen::Vector3i>>& region_map,
      const std::vector<PathSegmentWithView>& unpassed_viewpoints = std::vector<PathSegmentWithView>());

  // 获取规划结果路径
  const std::vector<Eigen::Vector3f>& getPath() const { return global_path_; }

  // 获取选中的clusters
  const std::vector<ClusterInfo>& getSelectedClusters() const {
    return selected_clusters_;
  }

  // 获取选中的视点 (带朝向)
  const std::vector<SelectedView>& getSelectedViews() const {
    return selected_views_;
  }

  // 获取到第一个viewpoint的路径 (TSP矩阵计算时记录)
  const std::vector<Eigen::Vector3f>& getPathToFirstViewpoint() const {
    return path_to_first_viewpoint_;
  }

  // 获取完整路径（经过所有viewpoints，TSP顺序）
  const std::vector<Eigen::Vector3f>& getCompleteViewpointPath() const {
    return complete_viewpoint_path_;
  }

  // 获取分段路径（每段路径+对应的目标viewpoint）
  const std::vector<PathSegmentWithView>& getPathSegmentsWithViews() const {
    return path_segments_with_views_;
  }

  // 更新previous_path_segments_with_views_中每个viewpoint的到达时间
  // start_time: 轨迹开始的绝对时间
  // offsets: 每个viewpoint相对于start_time的时间偏移量(秒)
  void updatePreviousPathSegmentArrivalTimes(ros::Time start_time,
                                             const std::vector<double>& offsets);

  // 获取已经经过和未经过的viewpoints
  // passed: 当前时间 >= arrival_time 的viewpoints
  // unpassed: 当前时间 < arrival_time 或 arrival_time == 0 的viewpoints
  void getPassedAndUnpassedViewpoints(
      std::vector<PathSegmentWithView>& passed,
      std::vector<PathSegmentWithView>& unpassed) const;

  // 使用新的起点重规划第一段路径（补偿全局规划到轨迹优化的延迟）
  void replanFirstSegment(const Eigen::Vector3f& new_start_position);

  // 前一次规划状态跟踪（供外部访问）
  Eigen::Vector3i previous_target_region_idx_;   // 前一次规划的目标 region 索引
  bool has_previous_planning_state_ = false;     // 是否有前一次规划状态

private:
  // 收集并过滤 clusters
  // 过滤条件: voxel_count > min_voxel_count_ && has_poorly_observed_neighbor
  // 选择: 按 poorly_observed_count 降序排序,选前 max_clusters_ 个
  std::vector<ClusterInfo> collectAndFilterClusters(
      const std::unordered_map<Eigen::Vector3i, FreeRegion, RegionHash,
                               std::equal_to<Eigen::Vector3i>>& region_map);

  // 构建距离矩阵 (使用拓扑地图 A* 路径距离)
  // mat[0][i]: odom 到第 i 个 cluster 的距离
  // mat[i][j]: 第 i 个 cluster 到第 j 个 cluster 的距离
  // reachable[i]: 第 i 个 cluster 是否可达
  // 返回: true=成功, false=失败（如 odom_node 无邻居）
  bool buildDistanceMatrix(
      const Eigen::Vector3f& odom_position,
      const std::vector<ClusterInfo>& clusters,
      Eigen::MatrixXd& cost_mat,
      std::vector<bool>& reachable);

  // 过滤不可达clusters，提取子矩阵并求解TSP
  // 更新 selected_clusters_、cost_mat、global_path_ 和 indices
  bool filterClustersAndSolveTSP(
      const Eigen::Vector3f& odom_position,
      Eigen::MatrixXd& cost_mat,
      const std::vector<bool>& reachable,
      std::vector<int>& indices);

  // 调用 LKH TSP 求解器 (参考 FastExplorationManager::solveLHK)
  void solveTSP(Eigen::MatrixXd& cost_mat, std::vector<int>& indices);

  // 体素索引转世界坐标
  Eigen::Vector3f voxelIdxToPosition(const Eigen::Vector3i& idx) const;

  // 发布可视化 (MarkerArray)
  void publishVisualization();

  // 发布viewpoint->target的可见性线段
  void publishViewpointVisibilityLines(
      const std::vector<SelectedView>& selected_views,
      const std::vector<Eigen::Vector3i>& target_voxels,
      const VisibilityCSR& csr_result,
      bool clear_history = true);

  // 可视化AABB包围盒 (用于调试遮挡物收集范围)
  void visualizeAABB(const Eigen::Vector3i& aabb_min,
                     const Eigen::Vector3i& aabb_max,
                     const std::string& ns = "aabb",
                     float r = 1.0f, float g = 0.5f, float b = 0.0f);

  // 可视化完整视点路径点
  void visualizeCompleteViewpointPath();

  // 可视化全局TSP路径（红色折线）
  // 如果提供了 cluster_indices，则可视化从当前位置到各个cluster中心的顺序连线
  void visualizeGlobalPath(const std::vector<int>& cluster_indices = {},
                           const Eigen::Vector3f& start_pos = Eigen::Vector3f::Zero());

  // 可视化 cluster 可达性 (绿=可达, 红=不可达)
  void visualizeClusterReachability(const std::vector<ClusterInfo>& clusters,
                                    const std::vector<bool>& reachable);

  // ==========================================
  // 层级规划相关函数
  // ==========================================
  // 从 cluster 的 free voxels 中降采样视点
  std::vector<Eigen::Vector3f> sampleViewpointsFromCluster(
      const std::vector<Eigen::Vector3i>& cluster_free_voxels,
      float downsample_resolution);

  // 使用 GPU raycast 评分视点，返回完整的 CSR 可见性矩阵
  // poorly_observed_voxels: 需要改善观测质量的occupied voxels
  // frontier_voxels: 需要探索的frontier voxels (不会更新max_possible_score)
  // aabb_min, aabb_max: 遮挡物收集的AABB范围 (由expandConnectedFreeRegion计算)
  VisibilityCSR evaluateViewpointsWithGPU(
      const std::vector<Eigen::Vector3f>& viewpoints,
      const std::vector<Eigen::Vector3i>& poorly_observed_voxels,
      const std::vector<Eigen::Vector3i>& frontier_voxels,
      std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                         std::equal_to<Eigen::Vector3i>>& spatial_hash,
      float voxel_size,
      const Eigen::Vector3f& map_min_bd,
      const Eigen::Vector3i& aabb_min,
      const Eigen::Vector3i& aabb_max);

  // 选择 Top N 视点（基于 CSR 可见性矩阵、贪婪算法和反向投票确定朝向）
  // num_poorly_observed: 前num_poorly_observed个targets是poorly_observed voxels (需要多角度观测)
  //                      后面的是frontier voxels (只需要被看到一次)
  // initial_masks: 可选的初始 target 掩码，用于从已有覆盖状态继续选择
  SelectedViewResult selectTopNViewpoints(
      const VisibilityCSR& csr_result,
      const std::vector<Eigen::Vector3f>& viewpoints,
      const std::vector<Eigen::Vector3i>& target_voxels,
      int num_poorly_observed,
      int top_n,
      const std::vector<std::bitset<20>>& initial_masks = {});

  // ==========================================
  // 距离矩阵构建 (合并单cluster和多cluster情况)
  // ==========================================
  // 构建viewpoints的TSP距离矩阵，根据region相邻性选择栅格A*或拓扑A*
  // 节点索引: 0=odom, 1~N=viewpoints, N+1=next_cluster(如果存在)
  // 返回: 到TSP第一个viewpoint的路径 (用于局部规划)
  std::vector<Eigen::Vector3f> buildDistanceMatrixForViewpoints(
      const Eigen::Vector3f& odom_position,
      const std::vector<Eigen::Vector3f>& viewpoints,
      const Eigen::Vector3i& viewpoints_region_idx,
      const Eigen::Vector3f* next_cluster_position,  // nullptr = 单cluster情况
      const Eigen::Vector3i* next_cluster_region_idx,  // nullptr = 单cluster情况
      Eigen::MatrixXd& cost_mat,
      std::vector<int>& tsp_indices);

  // ==========================================
  // 区域相邻性判断
  // ==========================================
  // 判断两个region是否为26-邻域 (各维度差值 <= 1)
  bool areRegionsAdjacent(const Eigen::Vector3i& region_a,
                          const Eigen::Vector3i& region_b) const;

  // 将世界坐标转换为region索引
  Eigen::Vector3i positionToRegionIdx(const Eigen::Vector3f& position) const;

  // 更新指定viewpoint可见的所有voxel的observation_direction_mask
  void updateVoxelMasksForViewpoint(
      int viewpoint_idx,
      const VisibilityCSR& csr_result,
      const std::vector<Eigen::Vector3i>& target_voxels);

  // 从complete_viewpoint_path_和selected_views_构建分段路径
  void buildPathSegmentsFromCompletePathWithViews();

  // ==========================================
  // 路径简化函数（Visibility-Based Shortcutting）
  // ==========================================
  // 使用 Visibility-Based Shortcutting 简化所有 path segments
  // 保护每个 segment 的首尾点（起点和接近 viewpoint 的终点）
  void simplifyPathSegments();

  // 检查两点之间的直线段是否无碰撞
  // 采样检查沿线的栅格点是否在 spatial_hash_ 中标记为 occupied
  bool isSegmentCollisionFree(const Eigen::Vector3f& start,
                               const Eigen::Vector3f& end) const;

  // 路径简化：Visibility-Based Shortcutting（使用膨胀后的 free cluster）
  // 保证简化后的路径在 free_voxels_set 范围内，且最后一个点是 end_point
  std::vector<Eigen::Vector3f> simplifyPathVisibilityBased(
      const std::vector<Eigen::Vector3f>& raw_path,
      const Eigen::Vector3f& end_point,
      const std::unordered_set<Eigen::Vector3i, VoxelHash,
                               std::equal_to<Eigen::Vector3i>>& free_voxels_set,
      double max_segment_length = 1.5) const;

  // 检查两点之间的线段是否完全在 free_voxels_set 内
  bool isSegmentInFreeSpace(
      const Eigen::Vector3f& start,
      const Eigen::Vector3f& end,
      const std::unordered_set<Eigen::Vector3i, VoxelHash,
                               std::equal_to<Eigen::Vector3i>>& free_voxels_set) const;

  // ==========================================
  // Region膨胀与连通性扩展函数
  // ==========================================
  // 获取中心region的26-邻域region索引列表 (共27个region)
  std::vector<Eigen::Vector3i> getDilatedRegionIndices(
      const Eigen::Vector3i& center_region_idx) const;

  // BFS扩展连通free区域，基于poorly_observed_voxels的AABB进行膨胀
  // 通过引用参数返回结果，避免内存拷贝
  void expandConnectedFreeRegion(
      const ClusterInfo& cluster,
      Eigen::Vector3i& aabb_min,
      Eigen::Vector3i& aabb_max,
      std::vector<Eigen::Vector3i>& connected_free_voxels,
      Eigen::Vector3i& target_aabb_min,  // NEW: output for target AABB (required, not optional)
      Eigen::Vector3i& target_aabb_max); // NEW: output for target AABB (required, not optional)

  // 从AABB范围内收集遮挡物voxels (排除targets)
  // target_voxels: 需要排除的targets
  // out_occluders: 输出非target的occupied/unknown voxels
  // out_occupied_indices: [可选] 输出 spatial_hash_ 中存在的 occupied voxel 索引（用于膨胀）
  void collectOccludersFromAABB(
      const Eigen::Vector3i& aabb_min,
      const Eigen::Vector3i& aabb_max,
      const std::vector<Eigen::Vector3i>& target_voxels,
      std::vector<GPUVoxel>& out_occluders,
      std::vector<Eigen::Vector3i>* out_occupied_indices = nullptr);

  // 对已收集的 occupied voxels 进行 26-邻域膨胀，直接从 free_voxels_set 中删除危险区域
  void dilateOccupiedVoxels(
      const std::vector<Eigen::Vector3i>& occupied_indices,
      std::unordered_set<Eigen::Vector3i, VoxelHash, std::equal_to<Eigen::Vector3i>>& free_voxels_set);

  // ==========================================
  // 批量A*搜索接口
  // ==========================================
  // 批量栅格A*搜索 (相邻region内的路径)
  // 输入: start_positions[i] → end_positions[i]
  // 输出: results[i] 对应的路径和距离
  // 暂时实现: 返回欧式距离和直线路径
  void batchGridAStar(
      const Eigen::Vector3i& region_idx,
      const std::vector<Eigen::Vector3f>& start_positions,
      const std::vector<Eigen::Vector3f>& end_positions,
      std::vector<GridAStarResult>& results);

  // 批量拓扑A*搜索 (跨region的路径)
  // 流程: insert nodes → 并行graphSearch → 提取路径 → delete nodes
  void batchTopoAStar(
      const std::vector<Eigen::Vector3f>& start_positions,
      const std::vector<Eigen::Vector3f>& end_positions,
      std::vector<TopoAStarResult>& results);

  // ROS相关
  ros::NodeHandle nh_;
  ros::Publisher path_vis_pub_;
  ros::Publisher viewpoint_vis_pub_;  // 发布viewpoint和cluster的可视化
  ros::Publisher viewpoint_visibility_lines_pub_;  // 发布viewpoint可见性连线
  ros::Publisher frontier_vis_pub_;
  ros::Publisher unobserved_directions_pub_;
  ros::Publisher target_aabb_pub_;  // 发布目标AABB可视化
  ros::Publisher forced_cluster_free_voxels_pub_;  // 发布forced cluster的free voxels点云
  ros::Publisher completed_centers_vis_pub_;  // 发布已完成观测的cluster centers可视化
  ros::Publisher cluster_reachability_pub_;   // 发布cluster可达性可视化 (绿=可达, 红=不可达)
  ros::Publisher planning_count_pub_;  // 发布cluster规划次数计数器
  ros::Publisher poorly_observed_count_text_pub_;  // 发布cluster的poorly_observed_count文本标记
  ros::Publisher filtered_target_voxels_pub_;  // 发布过滤后需要观测的target voxels点云
  ros::Publisher filtered_observation_directions_pub_;  // 发布被过滤voxel的观测方向箭头 (MarkerArray)
  ros::Publisher well_observed_voxels_pub_;  // 发布已经充分观测的voxel点云
  ros::Publisher sampled_viewpoints_pub_;  // 发布从free region采样的所有viewpoints点云
  ros::Publisher gpu_voxel_points_pub_;  // 发布GPU voxels的子体素点云 (2cm 分辨率)
  ros::Publisher gpu_voxel_centers_pub_;  // 发布GPU voxels的中心点云 (20cm 分辨率)

  // 参数
  float voxel_size_;
  std::string tsp_dir_;
  int max_clusters_;          // 最多选择多少个 clusters (默认20)
  int min_voxel_count_;       // 最小体素数量阈值 (默认30)
  float viewpoint_sample_resolution_;  // 视点采样分辨率 (默认0.5m)
  int top_n_viewpoints_;               // 选择的最佳视点数量 (默认5)
  float min_distance_to_obstacle_;     // 视点到障碍物的最小安全距离 (默认0.4m)

  // 相机FOV参数 (用于朝向选择)
  float h_fov_;  // 水平视场角 (弧度, 默认60°)
  float v_fov_;  // 垂直视场角 (弧度, 默认45°)
  float pitch_min_;  // Pitch lower bound (rad)
  float pitch_max_;  // Pitch upper bound (rad)
  // 是否将 unknown voxel 作为"完全占据"的遮挡体素参与 GPU 可见性评估
  // 目的：让 GlobalPlanner 的可见性模型与 OQM 的 raycastPoorlyObservedToOdoms 保持一致，避免选出"看起来可见但实际被unknown挡住"的viewpoint。
  bool treat_unknown_as_occupied_ = false;

  // well_observed 阈值系数 (与 OQM 保持一致)
  float well_observed_base_score_;       // 基础分数 (默认0.5)
  float well_observed_texture_weight_;   // 纹理复杂度权重 (默认0.5)
  float well_observed_geo_weight_;       // 几何复杂度权重 (默认0.5)


  // GPU raycast相关
  std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                     std::equal_to<Eigen::Vector3i>>& spatial_hash_;
  Eigen::Vector3f map_min_bd_;  // 地图原点
  Eigen::Vector3f region_origin_;  // Region coordinate system origin (same as map_min_bd_)
  LIOInterface::Ptr lidar_map_interface_;  // 用于障碍物距离查询

  // 拓扑地图 (用于 A* 路径规划)
  TopoGraph::Ptr graph_;
  std::unordered_map<Eigen::Vector3i, FreeRegion, RegionHash,
                     std::equal_to<Eigen::Vector3i>>* region_map_ptr_ = nullptr;

  // 规划结果
  std::vector<Eigen::Vector3f> global_path_;        // TSP路径 (世界坐标)
  std::vector<ClusterInfo> selected_clusters_;      // 选中的clusters

  // 层级规划相关
  std::vector<SelectedView> selected_views_;  // 选中的视点列表 (带朝向)
  int local_path_endpoint_;                   // 局部路径终点索引 (用于可视化)
  std::vector<Eigen::Vector3f> path_to_first_viewpoint_;  // 到第一个viewpoint的路径 (用于局部规划)
  std::vector<Eigen::Vector3f> complete_viewpoint_path_;  // 完整路径（经过所有viewpoints，TSP顺序）
  std::vector<PathSegmentWithView> path_segments_with_views_;  // 分段路径+目标视点（每段对应一个viewpoint）

  // Dijkstra 搜索相关（用于膨胀 occupied）
  std::vector<Eigen::Vector3i> aabb_occupied_indices_;  // AABB 内的 occupied voxel 索引
  std::vector<Eigen::Vector3i> connected_free_voxels_;  // BFS 扩展后的连通 free voxels
  // 用于路径简化的 free voxels 集合（在 buildPathSegmentsFromCompletePathWithViews 中构建，供 replanFirstSegment 使用）
  std::unordered_set<Eigen::Vector3i, VoxelHash, std::equal_to<Eigen::Vector3i>> connected_free_voxels_set_;

  // 前一次规划状态跟踪（用于检查上一个cluster的观测是否完成）
  std::vector<Eigen::Vector3i> previous_cluster_free_voxels_;  // 前一次规划的cluster的free voxels
  Eigen::Vector3i previous_cluster_center_voxel_idx_;          // 前一次规划的cluster的center voxel index

  // 上一轮TSP的下一个cluster信息（用于路径连续性）
  Eigen::Vector3f previous_next_cluster_position_;      // 上一轮TSP的下一个cluster位置
  Eigen::Vector3i previous_next_cluster_region_idx_;    // 上一轮TSP的下一个cluster的region索引
  bool has_next_cluster_ = false;                       // 是否有下一个cluster

  // 上一轮完整的global_path_（用于continuation mode可视化）
  std::vector<Eigen::Vector3f> previous_global_path_;   // 上一轮完整的TSP路径

  // 上一轮的 path_segments_with_views_（用于 continuation mode）
  std::vector<PathSegmentWithView> previous_path_segments_with_views_;

  // 上一轮的cluster访问顺序（用于SOP优先级��束）
  // 存储的是cluster的position（因为cluster索引可能变化，但位置是稳定的标识符）
  std::vector<Eigen::Vector3f> previous_cluster_order_positions_;
  bool has_previous_cluster_order_ = false;

  // 当前cluster规划次数计数器（用于可视化）
  int cluster_planning_count_ = 0;

  // 检查前一个cluster观测是否完成的辅助函数
  // 返回值: 强制选择的cluster索引 (-1表示无需强制选择)
  int checkPreviousClusterCompletion(
      const std::vector<ClusterInfo>& selected_clusters);

  // 检查是否存在 forced cluster 并进行可视化
  // 返回值: true 表示存在 forced cluster，false 表示不存在
  // forced_cluster_idx: 输出 forced cluster 的索引 (仅当返回 true 时有效)
  bool checkForcedClusterAndVisualize(int& forced_cluster_idx);

  // 有 forced cluster 时计算 TSP 顺序
  // 返回值: 排序后的 cluster 索引列表
  std::vector<int> computeClusterOrderWithForcedCluster(
      int forced_cluster_idx);

  // 无 forced cluster 时计算正常的 TSP 顺序
  // 返回值: 排序后的 cluster 索引列表
  std::vector<int> computeClusterOrderNormal(
      const Eigen::Vector3f& odom_position);

  // 根据 cluster 方向的观测得分过滤 target voxels
  // 返回: 过滤后的 target voxel 列表（排除已充分观测的 voxels）
  std::vector<Eigen::Vector3i> filterTargetVoxelsByClusterObservation(
      const ClusterInfo& cluster,
      const Eigen::Vector3i& aabb_min,
      const Eigen::Vector3i& aabb_max);

  // 尝试对单个cluster进行层级规划
  // 返回值: true表示成功, false表示尝试下一个cluster
  bool attemptClusterPlanning(
      const ClusterInfo& cluster,
      size_t attempt_idx,
      size_t total_attempts,
      const Eigen::Vector3f& odom_position,
      const std::vector<int>& cluster_indices_to_try);

  // Continuation mode 下的特殊规划函数
  // 要求结束点为上一轮第二个 cluster 的中心点
  bool attemptClusterPlanningContinuation(
      const ClusterInfo& cluster,
      const Eigen::Vector3f& odom_position,
      const Eigen::Vector3f& target_end_position,
      const std::vector<PathSegmentWithView>& unpassed_viewpoints = std::vector<PathSegmentWithView>());

  // 清除规划状态 (用于失败情况)
  void clearPlanningState();

  // Region 常量
  static constexpr float REGION_SIZE = FreeRegion::REGION_SIZE;  // 6.0m
};
