/**
 * @file watershed_subdivision.cpp
 * @brief 基于分水岭算法的Cluster二次划分
 *
 * 实现流程:
 * 1. 将3D cluster沿Z轴投影到2D XY平面
 * 2. 计算2D ESDF (Euclidean Distance Transform)
 * 3. 基于ESDF局部极大值确定分水岭seeds (使用连通域标记解决平顶山问题)
 * 4. 应用OpenCV分水岭算法划分区域
 * 5. 将2D划分结果映射回3D cluster
 */

#include <observation_quality_manager/observation_quality_manager.h>
#include <functional>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/ColorRGBA.h>

/**
 * @brief 将3D cluster投影到XY平面
 * @param cluster 原始3D cluster
 * @return ProjectedClusterData 包含2D投影图和映射关系
 */
ProjectedClusterData FreeRegion::projectClusterToXY(const FreeCluster& cluster) {
  ProjectedClusterData result;

  if (cluster.free_voxels.empty()) {
    return result;
  }

  // Step 1: 计算XY平面的AABB
  auto it = cluster.free_voxels.begin();
  int min_x = it->x(), max_x = it->x();
  int min_y = it->y(), max_y = it->y();

  for (const auto& voxel : cluster.free_voxels) {
    min_x = std::min(min_x, voxel.x());
    max_x = std::max(max_x, voxel.x());
    min_y = std::min(min_y, voxel.y());
    max_y = std::max(max_y, voxel.y());
  }

  result.aabb_min = Eigen::Vector2i(min_x, min_y);
  result.aabb_max = Eigen::Vector2i(max_x, max_y);

  int width = max_x - min_x + 1;
  int height = max_y - min_y + 1;

  // Step 2: 创建2D二值图像并填充
  // 注意: OpenCV的Mat是行优先(y, x)，即(rows=height, cols=width)
  result.binary_image = cv::Mat::zeros(height, width, CV_8UC1);

  // Step 3: 遍历所有voxels，投影到XY平面
  for (const auto& voxel : cluster.free_voxels) {
    int local_x = voxel.x() - min_x;
    int local_y = voxel.y() - min_y;

    // 标记为有free voxel
    result.binary_image.at<uchar>(local_y, local_x) = 255;

    // 记录(x,y)到所有Z层voxels的映射
    auto key = std::make_pair(local_x, local_y);
    result.xy_to_voxels[key].push_back(voxel);
  }

  return result;
}

/**
 * @brief 爬山法分水岭分割 (Hill Climbing Watershed)
 *        对每个像素，沿ESDF梯度向上爬到局部最大值，用峰值坐标作为区域ID
 * @param binary_image 2D二值图像 (255=free, 0=occupied)
 * @param distance_transform 距离变换结果 (ESDF)
 * @return 分割后的标签图像 (每个区域有不同的正整数ID，边界/墙壁为-1)
 */
// 简单的并查集查找函数，用于最后的合并
int findRoot(int id, std::map<int, int>& parent) {
    if (parent.find(id) == parent.end()) return id; // 如果不在map中，说明是独立的或者还没处理，暂且当做根
    if (parent[id] == id) return id;
    return parent[id] = findRoot(parent[id], parent); // 路径压缩
}

cv::Mat FreeRegion::applyWatershed(const cv::Mat& binary_image,
                                    const cv::Mat& distance_transform) {
  if (binary_image.empty()) return cv::Mat();

  int width = distance_transform.cols;
  int height = distance_transform.rows;

  // Step 1: 初始化 markers, -1 表示未处理/墙壁
  cv::Mat markers = cv::Mat::ones(binary_image.size(), CV_32S) * -1;

  // 8邻域方向
  const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
  const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};

  // Step 2: 爬山法 (带路径压缩/记忆化优化)
  // 这种优化可以将复杂度从 O(N*Path) 降低到接近 O(N)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // 如果是障碍物或者已经计算过(被之前的路径覆盖)，直接跳过
      if (binary_image.at<uchar>(y, x) == 0 || markers.at<int>(y, x) != -1) {
        continue;
      }

      // 记录本次爬山路径
      std::vector<cv::Point> path;
      int curr_x = x;
      int curr_y = y;
      int final_id = -1;

      int max_steps = 2000; // 防止死循环
      
      while (max_steps-- > 0) {
        path.push_back(cv::Point(curr_x, curr_y));
        
        // 如果我们撞到了一个已经标记过的点，直接加入那个组织
        if (markers.at<int>(curr_y, curr_x) != -1) {
            final_id = markers.at<int>(curr_y, curr_x);
            break;
        }

        float current_dist = distance_transform.at<float>(curr_y, curr_x);
        int current_idx = curr_y * width + curr_x;

        int best_x = curr_x;
        int best_y = curr_y;
        float best_dist = current_dist;
        int best_idx = current_idx;
        bool move_flag = false;

        for (int i = 0; i < 8; ++i) {
          int nx = curr_x + dx[i];
          int ny = curr_y + dy[i];

          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            if (binary_image.at<uchar>(ny, nx) > 0) { // 只看 Free 区域
              float n_dist = distance_transform.at<float>(ny, nx);
              int n_idx = ny * width + nx;

              // --- 确定性 Tie-Breaking ---
              // 1. 确实更高
              bool is_higher = (n_dist > best_dist);
              // 2. 高度一样(误差允许范围内)，但ID更大 (强制平顶流动)
              bool is_equal_dominant = (std::abs(n_dist - best_dist) < 1e-5f && n_idx > best_idx);

              if (is_higher || is_equal_dominant) {
                best_dist = n_dist;
                best_x = nx;
                best_y = ny;
                best_idx = n_idx;
                move_flag = true;
              }
            }
          }
        }

        if (!move_flag) {
          // 到达局部峰值
          final_id = curr_y * width + curr_x;
          break;
        }
        
        curr_x = best_x;
        curr_y = best_y;
      }

      // 回溯：把路径上所有点都设为这个 ID
      for (const auto& p : path) {
        markers.at<int>(p.y, p.x) = final_id;
      }
    }
  }

  // Step 3: 区域合并 (解决噪声过分割)
  // 逻辑：如果两个区域连接处的 ESDF 值很高（接近峰值），说明没有真正的门，应该合并。

  std::map<int, float> region_peaks; // 每个 ID 的最大 ESDF 值
  std::map<std::pair<int, int>, float> boundaries; // 相邻 ID 之间的最大边界值
  std::map<int, int> parent_map; // 并查集数组

  // 3.1 收集统计信息 (峰值高度 和 边界高度)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int id = markers.at<int>(y, x);
      if (id == -1) continue;

      float d = distance_transform.at<float>(y, x);

      // 更新该区域最高点
      if (d > region_peaks[id]) region_peaks[id] = d;

      // 初始化并查集
      if (parent_map.find(id) == parent_map.end()) parent_map[id] = id;

      // 检查右方和下方的邻居 (建立邻接图)
      int nbs[2][2] = {{1, 0}, {0, 1}};
      for (auto& nb : nbs) {
        int nx = x + nb[0];
        int ny = y + nb[1];
        if (nx < width && ny < height) {
          int nid = markers.at<int>(ny, nx);
          if (nid != -1 && nid != id) {
            // 找到边界
            int min_id = std::min(id, nid);
            int max_id = std::max(id, nid);
            std::pair<int, int> key = {min_id, max_id};

            // 边界强度 = 两个像素中较小的那个 ESDF (瓶颈宽度的一半)
            float border_strength = std::min(d, distance_transform.at<float>(ny, nx));

            if (boundaries.find(key) == boundaries.end() || border_strength > boundaries[key]) {
              boundaries[key] = border_strength;
            }
          }
        }
      }
    }
  }

  // 3.2 执行合并决策
  // 阈值说明：如果 (门宽 / 较小的房间半径) > 0.9，认为这是同一个房间
  const float MERGE_RATIO = 0.90f;

  for (auto const& [key, border_val] : boundaries) {
    int id1 = key.first;
    int id2 = key.second;

    float peak1 = region_peaks[id1];
    float peak2 = region_peaks[id2];
    float min_peak = std::min(peak1, peak2);

    // 如果边界高度非常接近房间的峰值高度，说明中间很平坦，没有显著的"门"
    if (border_val > min_peak * MERGE_RATIO) {
      int root1 = findRoot(id1, parent_map);
      int root2 = findRoot(id2, parent_map);
      if (root1 != root2) {
        // 合并：通常让 ID 小的或峰值高的做父亲，这里简化处理
        parent_map[root1] = root2;
      }
    }
  }

  // 3.3 将合并结果应用回 Markers
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int id = markers.at<int>(y, x);
      if (id != -1) {
        markers.at<int>(y, x) = findRoot(id, parent_map);
      }
    }
  }

  return markers;
}
/**
 * @brief 将2D划分结果映射回3D并创建子clusters
 * @param original_cluster 原始cluster
 * @param labels 分水岭输出的标签图像
 * @param proj_data 投影数据
 * @return 子cluster列表
 */
std::vector<FreeCluster> FreeRegion::subdivideCluster(const FreeCluster& original_cluster,
                                                       const cv::Mat& labels,
                                                       const ProjectedClusterData& proj_data) {
  std::vector<FreeCluster> sub_clusters;

  if (labels.empty()) {
    sub_clusters.push_back(original_cluster);
    return sub_clusters;
  }

  // Step 1: 找到所有唯一的label ID
  std::set<int> unique_labels;
  for (int y = 0; y < labels.rows; ++y) {
    for (int x = 0; x < labels.cols; ++x) {
      int label = labels.at<int>(y, x);
      if (label > 0) {  // 正整数为有效区域，-1为边界/墙壁，0为背景
        unique_labels.insert(label);
      }
    }
  }

  if (unique_labels.empty()) {
    sub_clusters.push_back(original_cluster);
    return sub_clusters;
  }

  // Step 2: 为每个label创建voxel集合
  using VoxelSet = std::unordered_set<Eigen::Vector3i, VoxelHash, std::equal_to<Eigen::Vector3i>>;
  std::map<int, VoxelSet> label_to_voxels;

  for (int y = 0; y < labels.rows; ++y) {
    for (int x = 0; x < labels.cols; ++x) {
      int label = labels.at<int>(y, x);
      if (label <= 0) continue;

      // 获取该(x,y)对应的所有3D voxels
      auto key = std::make_pair(x, y);
      auto it = proj_data.xy_to_voxels.find(key);
      if (it != proj_data.xy_to_voxels.end()) {
        for (const auto& voxel : it->second) {
          label_to_voxels[label].insert(voxel);
        }
      }
    }
  }

  // Step 3: 基于PCA递归二分每个label
  int next_region_id = 0;

  auto makeCluster = [&](const std::vector<Eigen::Vector3i>& voxels, int region_id) {
    FreeCluster sub_cluster;
    sub_cluster.free_voxels.insert(voxels.begin(), voxels.end());
    sub_cluster.voxel_count = static_cast<int>(voxels.size());

    for (const auto& voxel : voxels) {
      sub_cluster.voxel_to_region_id[voxel] = region_id;
    }

    // 计算中心点（选择距离几何中心最近的voxel）
    Eigen::Vector3f centroid_sum = Eigen::Vector3f::Zero();
    for (const auto& v : voxels) {
      centroid_sum += v.cast<float>();
    }
    Eigen::Vector3f centroid = centroid_sum / static_cast<float>(voxels.size());

    Eigen::Vector3i center = voxels.front();
    float min_dist_sq = (center.cast<float>() - centroid).squaredNorm();
    for (const auto& v : voxels) {
      float dist_sq = (v.cast<float>() - centroid).squaredNorm();
      if (dist_sq < min_dist_sq) {
        min_dist_sq = dist_sq;
        center = v;
      }
    }
    sub_cluster.center = center;

    sub_clusters.push_back(std::move(sub_cluster));
  };

  std::function<void(const std::vector<Eigen::Vector3i>&, int)> splitClusterPCA =
      [&](const std::vector<Eigen::Vector3i>& voxels, int depth) {
        // if (static_cast<int>(voxels.size()) < kMinSubClusterSize) {
        //   return;  // 太小，不生成子cluster
        // }

        // 计算均值和协方差（仅使用XY平面）
        Eigen::Vector2d mean = Eigen::Vector2d::Zero();
        for (const auto& v : voxels) {
          mean += v.head<2>().cast<double>();
        }
        mean /= static_cast<double>(voxels.size());

        Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
        for (const auto& v : voxels) {
          Eigen::Vector2d diff = v.head<2>().cast<double>() - mean;
          cov += diff * diff.transpose();
        }
        cov /= static_cast<double>(voxels.size());

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
        if (es.info() != Eigen::Success) {
          makeCluster(voxels, next_region_id++);
          return;
        }

        Eigen::Vector2d eigenvalues = es.eigenvalues();
        Eigen::Matrix2d eigenvectors = es.eigenvectors();
        int max_idx = (eigenvalues(1) > eigenvalues(0)) ? 1 : 0;
        Eigen::Vector2d first_pc = eigenvectors.col(max_idx);
        double max_std = std::sqrt(std::max(0.0, eigenvalues(max_idx)));

        bool need_split = (max_std > kPcaSplitStdThreshold) &&
                          (depth < kMaxPcaSplitDepth);
        if (!need_split) {
          makeCluster(voxels, next_region_id++);
          return;
        }

        // 沿第一主成分二分
        std::vector<Eigen::Vector3i> group_pos, group_neg;
        group_pos.reserve(voxels.size());
        group_neg.reserve(voxels.size());

        for (const auto& v : voxels) {
          Eigen::Vector2d diff = v.head<2>().cast<double>() - mean;
          if (diff.dot(first_pc) >= 0.0) {
            group_pos.push_back(v);
          } else {
            group_neg.push_back(v);
          }
        }

        // 如果分割后有一组过小，则不再拆分
        // if (static_cast<int>(group_pos.size()) < kMinSubClusterSize ||
        //     static_cast<int>(group_neg.size()) < kMinSubClusterSize) {
        //   makeCluster(voxels, next_region_id++);
        //   return;
        // }

        splitClusterPCA(group_pos, depth + 1);
        splitClusterPCA(group_neg, depth + 1);
      };

  for (const auto& [label, voxels] : label_to_voxels) {
    // if (static_cast<int>(voxels.size()) < kMinSubClusterSize) {
    //   continue;  // 跳过太小的子cluster
    // }

    std::vector<Eigen::Vector3i> voxel_vec(voxels.begin(), voxels.end());
    splitClusterPCA(voxel_vec, 0);
  }

  // Step 4: 重新分配poorly_observed_neighbors
  // 对原cluster的每个poorly_observed voxel，检查其26邻域属于哪个子cluster
  for (const auto& poor_obs : original_cluster.poorly_observed_neighbors) {
    std::set<int> adjacent_sub_clusters;  // 记录所有邻接的sub_cluster索引

    // 检查26邻域
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          if (dx == 0 && dy == 0 && dz == 0) continue;

          Eigen::Vector3i neighbor = poor_obs + Eigen::Vector3i(dx, dy, dz);

          // 检查这个邻居属于哪个子cluster
          for (size_t i = 0; i < sub_clusters.size(); ++i) {
            if (sub_clusters[i].free_voxels.count(neighbor)) {
              adjacent_sub_clusters.insert(static_cast<int>(i));
            }
          }
        }
      }
    }

    // 分配给所有邻接的子cluster
    for (int idx : adjacent_sub_clusters) {
      sub_clusters[idx].poorly_observed_neighbors.insert(poor_obs);
    }
  }

  // Step 5: 更新has_poorly_observed_neighbor标志
  for (auto& sub : sub_clusters) {
    sub.has_poorly_observed_neighbor = !sub.poorly_observed_neighbors.empty();
  }

  // 如果没有生成有效的子cluster，返回原cluster
  if (sub_clusters.empty()) {
    sub_clusters.push_back(original_cluster);
  }

  return sub_clusters;
}

/**
 * @brief 可视化所有cluster的2D边界线段
 *        在Z=0平面上使用LINE_LIST显示
 */
void ObservationQualityManager::publishClusterBoundaryVisualization() {
  if (cluster_boundary_vis_pub_.getNumSubscribers() == 0) {
    return;
  }

  // 使用 HSV 生成颜色，为每个 region_id 分配不同颜色
  auto hsvToRgb = [](float hue) -> std::tuple<uint8_t, uint8_t, uint8_t> {
    // hue: 0-360
    float h = hue / 60.0f;
    int i = static_cast<int>(h);
    float f = h - i;
    float p = 0.0f;
    float q = 255.0f * (1.0f - f);
    float t = 255.0f * f;

    uint8_t r, g, b;
    switch (i % 6) {
      case 0: r = 255; g = t;   b = p;   break;
      case 1: r = q;   g = 255; b = p;   break;
      case 2: r = p;   g = 255; b = t;   break;
      case 3: r = p;   g = q;   b = 255; break;
      case 4: r = t;   g = p;   b = 255; break;
      case 5: r = 255; g = p;   b = q;   break;
      default: r = 255; g = 255; b = 255; break;
    }
    return std::make_tuple(r, g, b);
  };

  // 创建 PointCloud2 消息
  sensor_msgs::PointCloud2 cloud_msg;
  cloud_msg.header.frame_id = "world";
  cloud_msg.header.stamp = ros::Time::now();

  // 收集所有点
  std::vector<float> x_values, y_values, z_values;
  std::vector<uint32_t> rgb_values;  // 使用 uint32 打包的 RGB

  // 首先统计最大 region_id
  int max_region_id = -1;
  size_t total_voxels_with_region = 0;
  size_t total_clusters = 0;
  size_t clusters_with_region = 0;

  for (const auto& [region_idx, region] : region_map_) {
    for (const auto& cluster : region.getClusters()) {
      total_clusters++;
      if (!cluster.voxel_to_region_id.empty()) {
        clusters_with_region++;
      }
      for (const auto& [voxel, rid] : cluster.voxel_to_region_id) {
        max_region_id = std::max(max_region_id, rid);
        total_voxels_with_region++;
      }
    }
  }

  ROS_INFO("Watershed vis: %zu/%zu clusters have region data, max_region_id=%d, total voxels=%zu",
           clusters_with_region, total_clusters, max_region_id, total_voxels_with_region);

  for (const auto& [region_idx, region] : region_map_) {
    for (const auto& cluster : region.getClusters()) {
      if (cluster.voxel_to_region_id.empty()) {
        continue;
      }

      // 使用 set 去重，确保每个 (x, y) 只添加一个点
      std::set<std::pair<int, int>> xy_added;

      for (const auto& [voxel, region_id] : cluster.voxel_to_region_id) {
        // 2D 投影：每个 (x, y) 只显示一个点
        auto xy_key = std::make_pair(voxel.x(), voxel.y());
        if (xy_added.count(xy_key)) continue;
        xy_added.insert(xy_key);

        // 将 voxel index 转换为世界坐标 (只取 XY，Z 设为 0)
        Eigen::Vector3f world_pos = region_origin_ + Eigen::Vector3f(voxel.x(), voxel.y(), 0).cast<float>() * voxel_size_;

        // 计算颜色 (HSV 环形分布)
        float hue = (max_region_id > 0) ? (360.0f * region_id / (max_region_id + 1)) : 0.0f;
        auto [r, g, b] = hsvToRgb(hue);

        // 打��� RGB 为 uint32 (0x00RRGGBB 格式)
        uint32_t rgb_packed = (static_cast<uint32_t>(r) << 16) |
                              (static_cast<uint32_t>(g) << 8) |
                              static_cast<uint32_t>(b);

        x_values.push_back(world_pos.x());
        y_values.push_back(world_pos.y());
        z_values.push_back(0.0f);  // Z=0 平面
        rgb_values.push_back(rgb_packed);
      }
    }
  }

  if (x_values.empty()) {
    ROS_WARN("Watershed vis: No points to publish!");
    return;
  }

  ROS_INFO("Watershed vis: Publishing %zu points", x_values.size());

  // 设置点云字段
  cloud_msg.height = 1;
  cloud_msg.width = x_values.size();
  cloud_msg.is_bigendian = false;
  cloud_msg.is_dense = true;

  sensor_msgs::PointField field;
  field.name = "x";
  field.offset = 0;
  field.datatype = sensor_msgs::PointField::FLOAT32;
  field.count = 1;
  cloud_msg.fields.push_back(field);

  field.name = "y";
  field.offset = 4;
  cloud_msg.fields.push_back(field);

  field.name = "z";
  field.offset = 8;
  cloud_msg.fields.push_back(field);

  // 使用标准的 rgb 字段 (uint32)
  field.name = "rgb";
  field.offset = 12;
  field.datatype = sensor_msgs::PointField::UINT32;
  field.count = 1;
  cloud_msg.fields.push_back(field);

  cloud_msg.point_step = 16;  // 3*float + 1*uint32 = 16 bytes
  cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;

  cloud_msg.data.resize(cloud_msg.row_step * cloud_msg.height);

  // 填充数据
  for (size_t i = 0; i < x_values.size(); ++i) {
    size_t offset = i * cloud_msg.point_step;
    memcpy(&cloud_msg.data[offset + 0], &x_values[i], 4);
    memcpy(&cloud_msg.data[offset + 4], &y_values[i], 4);
    memcpy(&cloud_msg.data[offset + 8], &z_values[i], 4);
    memcpy(&cloud_msg.data[offset + 12], &rgb_values[i], 4);  // rgb uint32
  }

  cluster_boundary_vis_pub_.publish(cloud_msg);
}
