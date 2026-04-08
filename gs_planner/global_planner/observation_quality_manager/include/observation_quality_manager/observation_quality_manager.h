#pragma once
#include "lidar_map/ikd_Tree.h"
#include <bits/stdc++.h>
#include <Eigen/Geometry>
#include <lidar_map/lidar_map.h>
#include <pointcloud_topo/graph.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <unordered_map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
using namespace fast_planner;

// 前向声明,避免循环依赖
class GlobalPlanner;

// ==========================================
// SphericalBinning - 正二十面体方向分箱
// ==========================================
class SphericalBinning {
private:
  // 黄金比例常数
  static constexpr float PHI = 1.618033988749895f; // (1 + sqrt(5)) / 2

  // 存储20个面的归一化中心向量
  inline static std::vector<Eigen::Vector3f> face_normals;

public:
  inline static std::array<std::array<float, 20>, 20> scoring_table;
  // 初始化：在程序启动时运行一次
  static void init() {
    if (!face_normals.empty()) return; // 已初始化

    // 1. 定义12个顶点 (0, ±1, ±PHI) 的循环排列
    Eigen::Vector3f verts[12] = {
      Eigen::Vector3f(-1,  PHI, 0), Eigen::Vector3f( 1,  PHI, 0),
      Eigen::Vector3f(-1, -PHI, 0), Eigen::Vector3f( 1, -PHI, 0),
      Eigen::Vector3f( 0, -1,  PHI), Eigen::Vector3f( 0,  1,  PHI),
      Eigen::Vector3f( 0, -1, -PHI), Eigen::Vector3f( 0,  1, -PHI),
      Eigen::Vector3f( PHI, 0, -1), Eigen::Vector3f( PHI, 0,  1),
      Eigen::Vector3f(-PHI, 0, -1), Eigen::Vector3f(-PHI, 0,  1)
    };

    // 2. 定义20个面的顶点索引 (标准拓扑结构)
    int indices[20][3] = {
      {0,11,5}, {0,5,1}, {0,1,7}, {0,7,10}, {0,10,11},
      {1,5,9}, {5,11,4}, {11,10,2}, {10,7,6}, {7,1,8},
      {3,9,4}, {3,4,2}, {3,2,6}, {3,6,8}, {3,8,9},
      {4,9,5}, {2,4,11}, {6,2,10}, {8,6,7}, {9,8,1}
    };

    // 3. 计算每个面的中心点并归一化 -> 得到20个方向向量
    face_normals.reserve(20);
    for (int i = 0; i < 20; ++i) {
      Eigen::Vector3f normal = Eigen::Vector3f::Zero();
      for (int k = 0; k < 3; ++k) {
        normal += verts[indices[i][k]];
      }
      // 归一化
      normal.normalize();
      face_normals.push_back(normal);
    }

    // 4. 预计算 20x20 评分表
    for (int i = 0; i < 20; ++i) {
      for (int j = 0; j < 20; ++j) {
        if (i == j) {
          // 法向量自己的方向评分固定为1
          scoring_table[i][j] = 1.0f;
        } else {
          // 其他方向评分为点积的绝对值
          scoring_table[i][j] = std::abs(face_normals[i].dot(face_normals[j]));
        }
      }
    }
  }

  // 核心函数：输入观测向量，返回 0-19 的 Bin 索引
  static int get_bin_index(const Eigen::Vector3f& view_dir) {
    // 数值稳定性处理：输入向量归一化
    float len_sq = view_dir.squaredNorm();
    if (len_sq < 1e-6f) return -1; // 无效观测（相机在Voxel内部）

    Eigen::Vector3f normalized = view_dir.normalized();

    int best_idx = 0;
    float max_dot = -2.0f;

    // 遍历20个方向求点积最大值
    for (int i = 0; i < 20; ++i) {
      float dot = normalized.dot(face_normals[i]);
      if (dot > max_dot) {
        max_dot = dot;
        best_idx = i;
      }
    }
    return best_idx;
  }

  // 获取第 i 个方向的面法向量
  static const Eigen::Vector3f& get_face_normal(int i) {
    return face_normals[i];
  }
};

// ==========================================
// VoxelHash - 空间哈希函数
// ==========================================
struct VoxelHash {
  size_t operator()(const Eigen::Vector3i &k) const {
    // 使用三个大的质数来打散分布，这是处理空间哈希的标准做法 (Spatial Hashing)
    // 这里的数值选自 Teschner et al. 的论文，专门针对 3D 网格优化
    const size_t p1 = 73856093;
    const size_t p2 = 19349663;
    const size_t p3 = 83492791;

    // 使用异或 (^) 结合乘法，速度极快
    return ((k.x() * p1) ^ (k.y() * p2) ^ (k.z() * p3));
  }
};

// ==========================================
// RegionHash - Region空间哈希函数
// ==========================================
struct RegionHash {
  size_t operator()(const Eigen::Vector3i &k) const {
    const size_t p1 = 73856093;
    const size_t p2 = 19349663;
    const size_t p3 = 83492791;
    return ((k.x() * p1) ^ (k.y() * p2) ^ (k.z() * p3));
  }
};

// ==========================================
// ProjectedClusterData - 2D投影数据结构（用于分水岭算法）
// ==========================================
struct ProjectedClusterData {
  cv::Mat binary_image;        // 2D二值图 (255=有free voxel, 0=无)
  cv::Mat distance_transform;  // 2D ESDF (距离变换结果)
  Eigen::Vector2i aabb_min;    // XY平面AABB最小点
  Eigen::Vector2i aabb_max;    // XY平面AABB最大点
  std::map<std::pair<int,int>, std::vector<Eigen::Vector3i>> xy_to_voxels; // (x,y) -> 所有Z上的voxels
};

// ==========================================
// FreeCluster - 自由空间聚类信息
// ==========================================
struct FreeCluster {
  Eigen::Vector3i center;  // Representative voxel at geometric centroid
  int voxel_count;         // Number of voxels in this cluster
  bool has_poorly_observed_neighbor;  // 是否邻近 poorly-observed 的 occupied 区域
  std::unordered_set<Eigen::Vector3i, VoxelHash, std::equal_to<Eigen::Vector3i>> poorly_observed_neighbors;  // 所有邻近的 poorly-observed 体素索引
  std::unordered_set<Eigen::Vector3i, VoxelHash, std::equal_to<Eigen::Vector3i>> free_voxels;  // 该 cluster 的所有 free voxels（连通分量）

  // Watershed 分区信息 (用于可视化)
  std::unordered_map<Eigen::Vector3i, int, VoxelHash, std::equal_to<Eigen::Vector3i>> voxel_to_region_id;  // voxel -> region_id (用于彩色点云显示)

  // 2D边界线段（用于可视化，在XY平面Z=0）
  // 每对相邻的Vector2f构成一条线段
  std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> boundary_segments_2d;
};
// ==========================================
// VoxelCell - 存储体素的几何信息
// ==========================================
struct VoxelCell {
  Eigen::Vector3f voxel_center = Eigen::Vector3f::Zero();
  Eigen::Vector3f normal = Eigen::Vector3f::Zero();
  float geometric_complexity = 0.0f;
  float texture_complexity = 0.0f;
  float observation_score = 0.0f;  // 观测质量分数（基于观测方向与法向量的关系）
  float max_possible_score = 0.0f;  // 理论最大可能观测分数（基于available_direction_mask，由GPU raycast计算）
  bool well_observed = false;  // 是否充分观测：observation_score > 1 + texture_complexity + geometric_complexity
  int geo_cnt = 0;
  int tex_cnt = 0;
  int normal_bin_idx = -1;  // 法向量对应的球面分箱索引（-1表示未初始化）

  // 几何增量统计数据
  Eigen::Vector3f sum_pos_local = Eigen::Vector3f::Zero();
  Eigen::Matrix3f sum_pp_local = Eigen::Matrix3f::Zero();

  // 纹理增量统计数据
  Eigen::Vector3f sum_color = Eigen::Vector3f::Zero();
  Eigen::Vector3f sum_color_sq = Eigen::Vector3f::Zero();

  // 10x10x10 子空间占用掩码（1000位）- 几何
  std::bitset<1000> geometry_occupancy_mask;
  // 10x10x10 子空间占用掩码（1000位）- 纹理
  std::bitset<1000> texture_occupancy_mask;

  // 20个观测方向掩码（正二十面体的20个面）
  std::bitset<20> observation_direction_mask;

  // 20个可用观测方向掩码（raycast后计算，表示该voxel可以从哪些方向观察）
  // 初始化为全1（0xFFFFF），表示所有方向都可见，后续只能单向更新：可见→不可见
  std::bitset<20> available_direction_mask{0};

  // 计算点在cell内的子空间索引（0-999）
  int getSubVoxelIndex(const Eigen::Vector3f &point_global,
                       float voxel_size) const {
    // 计算点相对cell中心的偏移
    Eigen::Vector3f offset = point_global - voxel_center;

    // 归一化到[-0.5, 0.5]，再转换到[0, 1]
    Eigen::Vector3f normalized = (offset / voxel_size).array() + 0.5f;

    // 映射到[0, 9]的整数索引，并clamp到有效范围
    int idx_x = std::max(0, std::min(9, static_cast<int>(normalized.x() * 10.0f)));
    int idx_y = std::max(0, std::min(9, static_cast<int>(normalized.y() * 10.0f)));
    int idx_z = std::max(0, std::min(9, static_cast<int>(normalized.z() * 10.0f)));

    // 计算线性索引：x*100 + y*10 + z
    return idx_x * 100 + idx_y * 10 + idx_z;
  }

  // 批量添加几何点并更新几何信息
  void addGeometryPoints(const std::vector<Eigen::Vector3f> &points, float voxel_size,
                         const Eigen::Vector3f &camera_pos,
                         float base_score, float tex_weight, float geo_weight) {
    for (const auto &p_global : points) {
      // 计算子空间索引
      int sub_idx = getSubVoxelIndex(p_global, voxel_size);

      // 检查该子空间是否已被占用
      if (geometry_occupancy_mask[sub_idx]) {
        // 该子空间已有点云，跳过此点
        continue;
      }

      // 标记该子空间为已占用
      geometry_occupancy_mask[sub_idx] = true;

      // 正常添加点并更新统计量
      Eigen::Vector3f p_local = p_global - voxel_center;
      sum_pos_local += p_local;
      sum_pp_local += p_local * p_local.transpose();
      geo_cnt++;
    }
    // 计算观测方向：从voxel中心指向相机
    Eigen::Vector3f view_dir = camera_pos - voxel_center;
    updateGeometry(view_dir, base_score, tex_weight, geo_weight);
  }

  // 更新法向量和几何复杂度
  void updateGeometry(const Eigen::Vector3f &view_dir,
                      float base_score, float tex_weight, float geo_weight) {
    if (geo_cnt < 3) {
      normal = Eigen::Vector3f::Zero();
      geometric_complexity = 0.0f;
      normal_bin_idx = -1;
      return;
    }

    // 计算局部均值
    Eigen::Vector3f mu = sum_pos_local / static_cast<float>(geo_cnt);

    // 计算协方差矩阵
    Eigen::Matrix3f cov =
        (sum_pp_local / static_cast<float>(geo_cnt)) - (mu * mu.transpose());

    // 特征值分解
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
    Eigen::Vector3f lambdas = solver.eigenvalues();

    // 法向量：最小特征值对应的特征向量
    normal = solver.eigenvectors().col(0);

    // 根据 view_dir 调整法向量方向（使其朝向观测方向）
    float dot_product = normal.dot(view_dir);
    if (dot_product < 0) {
      normal = -normal;
    }

    // 几何复杂度：Surface Variation（归一化到 [0, 1]）
    // 原始公式：lambda_min / sum_lambdas，取值范围 [0, 1/3]
    // 理论最大值 1/3 出现在三个特征值相等时（各向同性）
    // 乘以 3 归一化到 [0, 1]
    float lambda_min = std::max(0.0f, lambdas[0]);
    float sum_lambdas = std::max(1e-6f, lambdas.sum());
    geometric_complexity = std::min(1.0f, 3.0f * lambda_min / sum_lambdas);

    // 计算新的 normal_bin_idx
    int new_bin_idx = SphericalBinning::get_bin_index(normal);

    // 如果 bin_idx 改变了，重新计算 observation_score
    if (new_bin_idx != normal_bin_idx && new_bin_idx >= 0) {
      normal_bin_idx = new_bin_idx;
    //   // 如果已有观测数据，重新计算分数（会自动调用 updateWellObserved）
    //   if (observation_direction_mask.count() > 0) {
    //     updateObservationScore(base_score, tex_weight, geo_weight);
    //   } else {
    //     updateWellObserved(base_score, tex_weight, geo_weight);
    //   }
    // } else {
    //   updateWellObserved(base_score, tex_weight, geo_weight);
    }
  }

  // 批量添加纹理点并更新纹理复杂度
  void addTexturePoints(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> &points_with_colors,
                        float voxel_size,
                        const Eigen::Vector3f &camera_pos,
                        float base_score, float tex_weight, float geo_weight) {
    // 计算观测方向：从voxel中心指向相机
    Eigen::Vector3f view_dir = camera_pos - voxel_center;
    int obs_bin_idx = SphericalBinning::get_bin_index(view_dir);

    // 调试：记录设置前的mask状态
    size_t mask_before = observation_direction_mask.count();

    // 检查是否为新观测方向
    bool is_new_observation = false;
    if (obs_bin_idx >= 0) {
      is_new_observation = !observation_direction_mask[obs_bin_idx];
      if (is_new_observation) {
        observation_direction_mask[obs_bin_idx] = true;
      }
    }

    // 调试：检查是否成功设置了 observation_direction_mask
    if (mask_before == 0 && observation_direction_mask.count() == 0) {
      ROS_WARN_THROTTLE(1.0, "addTexturePoints FAILED to set mask: obs_bin_idx=%d, view_dir=[%.3f,%.3f,%.3f], len=%.6f, voxel_center=[%.2f,%.2f,%.2f], this=%p",
               obs_bin_idx, view_dir.x(), view_dir.y(), view_dir.z(), view_dir.norm(),
               voxel_center.x(), voxel_center.y(), voxel_center.z(), (void*)this);
    }
    // 调试：成功设置mask时也打印，方便对比this指针
    // if (mask_before == 0 && observation_direction_mask.count() > 0) {
    //   ROS_INFO_THROTTLE(1.0, "addTexturePoints SET mask: voxel=[%.2f,%.2f,%.2f], obs_bin_idx=%d, mask=%lu, this=%p",
    //            voxel_center.x(), voxel_center.y(), voxel_center.z(),
    //            obs_bin_idx, observation_direction_mask.to_ulong(), (void*)this);
    // }

    // 处理纹理点（无论是否为新观测，都要更新纹理复杂度）
    for (const auto &[p_global, color] : points_with_colors) {
      // 计算子空间索引
      int sub_idx = getSubVoxelIndex(p_global, voxel_size);

      // 检查该子空间是否已被占用
      if (texture_occupancy_mask[sub_idx]) {
        // 该子空间已有点云，跳过此点
        continue;
      }

      // 标记该子空间为已占用
      texture_occupancy_mask[sub_idx] = true;

      // 更新颜色统计量（RGB值归一化到[0,1]）
      sum_color += color;
      sum_color_sq += color.cwiseProduct(color);  // 逐元素平方
      tex_cnt++;
    }
    updateTexture(base_score, tex_weight, geo_weight);

    // 仅当有新观测方向时更新分数
    // if (is_new_observation) {
    //   updateObservationScore(base_score, tex_weight, geo_weight);
    // }
  }

  // 更新纹理复杂度（RGB方差）
  void updateTexture(float base_score, float tex_weight, float geo_weight) {
    // 方差计算至少需要2个样本，否则无意义
    if (tex_cnt < 2) {
      texture_complexity = 0.0f;
      return;
    }

    float N = static_cast<float>(tex_cnt);

    // 1. 计算均值 E[X]
    Eigen::Vector3f mean_color = sum_color / N;

    // 2. 计算平方的均值 E[X²]
    Eigen::Vector3f mean_sq = sum_color_sq / N;

    // 3. 计算方差 Var = E[X²] - (E[X])²
    // cwiseProduct 计算 mean_color 的平方
    Eigen::Vector3f variance = mean_sq - mean_color.cwiseProduct(mean_color);

    // 4. 计算总纹理复杂度（归一化到 [0, 1]）
    // 简单求和：R方差 + G方差 + B方差
    // 理论最大值：3 * 0.25 = 0.75（每个通道在[0,1]范围内的最大方差为0.25）
    // 除以0.75归一化到[0,1]，等价于乘以 4/3
    float raw_complexity = std::max(0.0f, variance.sum());
    texture_complexity = std::min(1.0f, raw_complexity * (4.0f / 3.0f));
    updateWellObserved(base_score, tex_weight, geo_weight);
  }

  // 更新观测质量分数（使用已保存的 normal_bin_idx）
  void updateObservationScore(float base_score, float tex_weight, float geo_weight) {
    // 使用评分表计算分数：遍历所有观测方向，累加评分
    observation_score = 0.0f;
    for (int i = 0; i < 20; ++i) {
      if (observation_direction_mask[i]) {
        observation_score += SphericalBinning::scoring_table[normal_bin_idx][i];
      }
    }
    updateWellObserved(base_score, tex_weight, geo_weight);
  }

  // 更新 well_observed 标志
  void updateWellObserved(float base_score = 0.5f, float tex_weight = 0.5f, float geo_weight = 0.5f) {
    // 点数充足的情况：使用正常逻辑
    if (geo_cnt >= 3 && tex_cnt >= 2) {
      // 满足以下任一条件即为 well_observed：
      // 1. 观测分数超过阈值（base_score + tex_weight*纹理复杂度 + geo_weight*几何复杂度）
      // 2. 当前观测分数达到或超过理论最大可能分数（所有可见方向都已观测）
      well_observed = (observation_score > base_score + tex_weight * texture_complexity + geo_weight * geometric_complexity) ||
                      (observation_score >= max_possible_score && max_possible_score > 0.0f);
      // well_observed = true;
      return;
    }

    // 点数不足的情况：检查是否已经从某个方向观测过
    if (observation_direction_mask.count() >= 3) {
      well_observed = true;
      return;
    }
    ROS_ASSERT(well_observed == false);
  }

  // 计算理论最大可能观测分数（基于 available_direction_mask）
  float calculateMaxPossibleScore() const {
    if (normal_bin_idx < 0) {
      return 0.0f;
    }
    float score = 0.0f;
    for (int i = 0; i < 20; ++i) {
      if (available_direction_mask[i]) {
        score += SphericalBinning::scoring_table[normal_bin_idx][i];
      }
    }
    return score;
  }
};


class FreeRegion {
public:
  enum class VoxelState : uint8_t { FREE = 0, OCCUPIED = 1, FRONTIER = 2 };

  static constexpr float REGION_SIZE = 6.0f; // 6m x 6m x 6m
  // 当标记为 true 时，表示该 Region 中的目标已无法通过新视点改进观测质量，后续全局规划忽略该 Region
  bool isAllWellObserved() const { return all_well_observed_; }
  void setAllWellObserved(bool value) { all_well_observed_ = value; }
  vector<int> observation_odom_idx;

  // 存储已知体素的状态，缺失视为 unknown
  std::unordered_map<Eigen::Vector3i, VoxelState, VoxelHash,
                     std::equal_to<Eigen::Vector3i>>
      voxel_states;

  // 添加自由体素（frontier 也视作 free） - O(1)
  void addFreeVoxel(const Eigen::Vector3i& voxel_idx,
                    VoxelState state = VoxelState::FREE) {
    auto it = voxel_states.find(voxel_idx);
    if (it != voxel_states.end() && it->second == VoxelState::OCCUPIED) {
      return; // 已知被占据，保持占据状态
    }
    voxel_states[voxel_idx] = state;
  }

  // 添加占据体素 - O(1)
  void addOccupiedVoxel(const Eigen::Vector3i& voxel_idx) {
    voxel_states[voxel_idx] = VoxelState::OCCUPIED;
  }

  bool getVoxelState(const Eigen::Vector3i& voxel_idx,
                     VoxelState& state) const {
    auto it = voxel_states.find(voxel_idx);
    if (it == voxel_states.end()) {
      return false;
    }
    state = it->second;
    return true;
  }

  size_t freeVoxelCount() const {
    size_t cnt = 0;
    for (const auto& [_, state] : voxel_states) {
      if (isFreeState(state)) {
        ++cnt;
      }
    }
    return cnt;
  }

  size_t occupiedVoxelCount() const {
    size_t cnt = 0;
    for (const auto& [_, state] : voxel_states) {
      if (state == VoxelState::OCCUPIED) {
        ++cnt;
      }
    }
    return cnt;
  }

  template <typename Func>
  void forEachFreeVoxel(Func&& f) const {
    for (const auto& [idx, state] : voxel_states) {
      if (isFreeState(state)) {
        f(idx, state);
      }
    }
  }

  template <typename Func>
  void forEachOccupiedVoxel(Func&& f) const {
    for (const auto& [idx, state] : voxel_states) {
      if (state == VoxelState::OCCUPIED) {
        f(idx);
      }
    }
  }

  // 分水岭调试可视化参数
  struct WatershedDebugParams {
    Eigen::Vector3f odom_position;           // 当前odom位置
    Eigen::Vector3f region_origin;            // Region坐标系原点
    float voxel_size;                         // Voxel尺寸
    ros::Publisher* labels_pub;               // labels图像发布器
    ros::Publisher* distance_pub;             // distance图像发布器
    ros::Publisher* binary_pub;               // binary图像发布器

    WatershedDebugParams() : labels_pub(nullptr), distance_pub(nullptr), binary_pub(nullptr) {}
    bool isValid() const { return labels_pub && distance_pub && binary_pub; }
  };

  // 计算连通分量并生成聚类信息
  // 排除距离 occupied voxels 切比雪夫距离 ≤ 1 的 free voxels（障碍物膨胀 0.2m）
  // 同时检查每个 cluster 是否邻近 poorly-observed 的 occupied 区域
  // debug_params: 可选的调试可视化参数，用于发布分水岭中间结果
  void computeClusterCenters(const std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                                                       std::equal_to<Eigen::Vector3i>>& spatial_hash,
                              const WatershedDebugParams* debug_params = nullptr);

  // 获取聚类信息
  const std::vector<FreeCluster>& getClusters() const { return clusters_; }

  static bool isFreeState(VoxelState state) {
    return state == VoxelState::FREE || state == VoxelState::FRONTIER;
  }

  // 已完成观测的 cluster centers 管理
  void addCompletedClusterCenter(const Eigen::Vector3i& center);
  const std::vector<Eigen::Vector3i>& getCompletedClusterCenters() const;

private:
  bool all_well_observed_ = false;
  std::vector<FreeCluster> clusters_;
  std::vector<Eigen::Vector3i> completed_cluster_centers_;

  // 分水岭二次划分参数
  static constexpr int kMinClusterSizeForSubdivision = 10;
  static constexpr float kWatershedMarkerThreshold = 0.3f;
  static constexpr int kMinSubClusterSize = 50;
  static constexpr float kPcaSplitStdThreshold = 6.0f;   // PCA最大轴的std阈值(以voxel为单位)
  static constexpr int kMaxPcaSplitDepth = 3;

  // 分水岭二次划分函数（实现在watershed_subdivision.cpp）
  ProjectedClusterData projectClusterToXY(const FreeCluster& cluster);
  cv::Mat applyWatershed(const cv::Mat& binary_image, const cv::Mat& distance_transform);
  std::vector<FreeCluster> subdivideCluster(const FreeCluster& original_cluster, const cv::Mat& labels, const ProjectedClusterData& proj_data);

  static bool areNeighbors(const Eigen::Vector3i& a, const Eigen::Vector3i& b) {
    return (std::abs(a.x() - b.x()) + std::abs(a.y() - b.y()) + std::abs(a.z() - b.z()) == 1);
  }
};


class ObservationQualityManager {
  struct OdomPose {
    Eigen::Vector3f position;
    Eigen::Quaternionf orientation;
  };
  // 该区域内出现过的里程计姿态（位置 + 朝向）
private:
  LIOInterface::Ptr lidar_map_interface_;
  TopoGraph::Ptr graph_;
  Eigen::Vector3f map_min_bd_, map_max_bd_;
  Eigen::Vector3f region_origin_;  // Region coordinate system origin (equals map_min_bd_)
  ros::Publisher raycast_vis_pub_;
  ros::Publisher cluster_boundary_vis_pub_;  // 分水岭cluster边界可视化
  ros::Publisher raycast_viewpoints_pub_;    // raycast使用的viewpoints点云可视化
  ros::Publisher raycast_viewpoints_axis_pub_;  // raycast使用的viewpoints axis可视化
  ros::Publisher gpu_voxels_vis_pub_;  // GPU体素可视化

  // 分水岭调试图像发布器
  ros::Publisher watershed_labels_pub_;
  ros::Publisher watershed_distance_pub_;
  ros::Publisher watershed_binary_pub_;

  // GlobalPlanner for TSP path planning
  std::shared_ptr<GlobalPlanner> global_planner_;

  // 空间哈希表
  float voxel_size_;
  float max_ray_length_; // 最大光线长度，超过此距离的点不参与更新
  float horizontal_fov_rad_ = static_cast<float>(M_PI); // 默认无限FOV（禁用朝向裁剪）
  float vertical_fov_rad_ = static_cast<float>(M_PI);

  // well_observed 阈值系数
  float well_observed_base_score_;       // 基础分数 (默认0.5)
  float well_observed_texture_weight_;   // 纹理复杂度权重 (默认0.5)
  float well_observed_geo_weight_;       // 几何复杂度权重 (默认0.5)

  std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                     std::equal_to<Eigen::Vector3i>>
      spatial_hash_;

  // Region管理 (自由空间)
  std::unordered_map<Eigen::Vector3i, FreeRegion, RegionHash,
                     std::equal_to<Eigen::Vector3i>>
      region_map_;

  // 累积的待更新 regions（延迟聚类更新，减少冗余计算）
  std::unordered_set<Eigen::Vector3i, RegionHash, std::equal_to<Eigen::Vector3i>>
      pending_modified_regions_;

  bool isInBox(const PointType &pt) const;
  bool isInBox(const Eigen::Vector3f &pt) const;

  // Region管理相关方法
  // 添加自由体素到对应的区域
  void addFreeVoxelToRegion(const Eigen::Vector3i& voxel_idx,
                            FreeRegion::VoxelState state = FreeRegion::VoxelState::FREE);

  // 添加占据体素到对应的区域
  void addOccupiedVoxelToRegion(const Eigen::Vector3i& voxel_idx);

  bool tryGetVoxelState(const Eigen::Vector3i& voxel_idx,
                        FreeRegion::VoxelState& state) const;

  bool hasUnknownNeighbor(const Eigen::Vector3i& voxel_idx) const;


  // 光线投射：从起点到终点，记录自由空间体素
  void raycastFreeSpace(const Eigen::Vector3f& start, const Eigen::Vector3f& end,
                        std::unordered_set<Eigen::Vector3i, VoxelHash,
                                           std::equal_to<Eigen::Vector3i>>& unknown_to_free,
                        std::unordered_set<Eigen::Vector3i, VoxelHash,
                                           std::equal_to<Eigen::Vector3i>>& frontier_to_free);

  void processNewlyFreedVoxels(const std::unordered_set<Eigen::Vector3i, VoxelHash,
                                                         std::equal_to<Eigen::Vector3i>>& unknown_to_free,
                               const std::unordered_set<Eigen::Vector3i, VoxelHash,
                                                         std::equal_to<Eigen::Vector3i>>& frontier_to_free);

  // 更新指定 regions 的聚类中心
  void updateClustersForRegions(const std::unordered_set<Eigen::Vector3i, RegionHash,
                                                          std::equal_to<Eigen::Vector3i>>& modified_regions);

  // 可视化所有cluster的边界（分水岭划分后的边界voxels）
  void publishClusterBoundaryVisualization();

public:
  std::vector<OdomPose> odom_poses;
  std::unordered_set<Eigen::Vector3i, RegionHash, std::equal_to<Eigen::Vector3i>> modified_regions_set;
  ros::NodeHandle nh_;
  typedef std::shared_ptr<ObservationQualityManager> Ptr;

  void init(ros::NodeHandle &nh, LIOInterface::Ptr &lio_interface,
            TopoGraph::Ptr graph);

  // 批量添加点云并更新几何信息
  void addPointCloudOnlyGeometry(const pcl::PointCloud<PointType>::Ptr &cloud,
                                  const Eigen::Vector3f &odom_position);

  // 批量添加彩色点云并更新纹理复杂度
  void addPointCloudWithTexture(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                                const Eigen::Vector3f &odom_position);

  // Utility conversions exposed for external modules (e.g., FSM callbacks)
  void pos2idx(const PointType &pt, Eigen::Vector3i &idx);
  void pos2idx(const Eigen::Vector3f &pt, Eigen::Vector3i &idx);
  void idx2pos(const Eigen::Vector3i &idx, PointType &pt) const;

  // 将体素索引转换为区域索引
  Eigen::Vector3i voxelToRegion(const Eigen::Vector3i& voxel_idx) const;

  // Getters for visualization
  const std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                           std::equal_to<Eigen::Vector3i>> &
  getSpatialHash() const { return spatial_hash_; }

  // Non-const getter for modification (e.g., GPU raycast updates)
  std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                     std::equal_to<Eigen::Vector3i>> &
  getSpatialHash() { return spatial_hash_; }

  float getVoxelSize() const { return voxel_size_; }

  // Getter for region map (free space)
  std::unordered_map<Eigen::Vector3i, FreeRegion, RegionHash,
                           std::equal_to<Eigen::Vector3i>>&
  getRegionMap() { return region_map_; }

  // Getter for GlobalPlanner
  std::shared_ptr<GlobalPlanner> getGlobalPlanner() { return global_planner_; }

  // Getter for LIOInterface
  LIOInterface::Ptr getLIOInterface() { return lidar_map_interface_; }

  // Getter for TopoGraph
  TopoGraph::Ptr getTopoGraph() { return graph_; }

  // Getters for well_observed threshold parameters
  float getWellObservedBaseScore() const { return well_observed_base_score_; }
  float getWellObservedTextureWeight() const { return well_observed_texture_weight_; }
  float getWellObservedGeoWeight() const { return well_observed_geo_weight_; }

  // Getter for region origin
  Eigen::Vector3f getRegionOrigin() const { return region_origin_; }

  // GPU raycast: check poorly-observed voxels in a region against current odom
  // and mark visible observation directions as observed.
  // additional_viewpoints/orientations: 额外的viewpoints用于raycast
  // target_region_for_additional: 只对该region添加额外viewpoints（nullptr表示不添加）
  void raycastPoorlyObservedToOdoms(
      const std::vector<Eigen::Vector3f>& additional_viewpoints = {},
      const std::vector<Eigen::Quaternionf>& additional_orientations = {},
      const Eigen::Vector3i* target_region_for_additional = nullptr);

  // 批量更新累积的 modified regions 的聚类
  // 在 planGlobalTSPPath 之前调用，将多帧点云的修改聚合在一起更新
  void updatePendingClusters();
};
