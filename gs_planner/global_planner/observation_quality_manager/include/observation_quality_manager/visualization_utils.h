#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Eigen>
#include <unordered_map>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <vector>
#include <tuple>

// Forward declarations
struct VoxelHash;
struct VoxelCell;
struct RegionHash;
class FreeRegion;
struct ClusterInfo;

namespace observation_quality {

class VisualizationUtils {
public:
  // Jet colormap: value -> RGB (0-255)
  static Eigen::Vector3i jetColorMap(float value, float min_val = 0.0f,
                                      float max_val = 0.333333f);

  // Generate colored point cloud from spatial hash
  static pcl::PointCloud<pcl::PointXYZRGB>::Ptr generateGeometryComplexityCloud(
      const std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                                std::equal_to<Eigen::Vector3i>> &spatial_hash,
      float voxel_size);

  // Generate colored point cloud for texture complexity
  static pcl::PointCloud<pcl::PointXYZRGB>::Ptr generateTextureComplexityCloud(
      const std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                                std::equal_to<Eigen::Vector3i>> &spatial_hash,
      float voxel_size);

  // Generate colored point cloud for observation score
  static pcl::PointCloud<pcl::PointXYZRGB>::Ptr generateObservationScoreCloud(
      const std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                                std::equal_to<Eigen::Vector3i>> &spatial_hash,
      float voxel_size);

  // Generate colored point cloud for well observed status
  static pcl::PointCloud<pcl::PointXYZRGB>::Ptr generateWellObservedCloud(
      const std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                                std::equal_to<Eigen::Vector3i>> &spatial_hash,
      float voxel_size);

  // ==========================================
  // 新增：可见性连线可视化
  // ==========================================
  // Forward declaration for VisibilityLine
  struct VisibilityLine {
      Eigen::Vector3f start;
      Eigen::Vector3f end;
      bool visible;
      float distance;
      Eigen::Vector3i voxel_idx;
  };

  // Generate LINE_LIST marker for visibility check
  // 绿色线条 = 可见路径
  // 红色线条 = 被遮挡路径
  static visualization_msgs::Marker generateVisibilityLinesMarker(
      const std::vector<VisibilityLine>& visibility_lines,
      const std::string& frame_id = "world");

  // Generate sphere markers for free cluster centers
  // Colors: red (has poorly-observed neighbors) vs green (all neighbors well-observed)
  static visualization_msgs::Marker generateFreeClusterCentersMarker(
      const std::unordered_map<Eigen::Vector3i, FreeRegion, RegionHash,
                                std::equal_to<Eigen::Vector3i>>& region_map,
      const std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                                std::equal_to<Eigen::Vector3i>>& spatial_hash,
      float voxel_size,
      const Eigen::Vector3f& region_origin,
      const std::string& frame_id = "world");

  // Generate point cloud for free space regions
  // All free voxels are visualized with cyan color for easy debugging
  static pcl::PointCloud<pcl::PointXYZRGB>::Ptr generateFreeSpaceCloud(
      const std::unordered_map<Eigen::Vector3i, FreeRegion, RegionHash,
                                std::equal_to<Eigen::Vector3i>>& region_map,
      float voxel_size,
      const Eigen::Vector3f& region_origin);

  // Generate point cloud for frontier voxels (free/unknown boundary)
  static pcl::PointCloud<pcl::PointXYZRGB>::Ptr generateFrontierCloud(
      const std::unordered_map<Eigen::Vector3i, FreeRegion, RegionHash,
                                std::equal_to<Eigen::Vector3i>>& region_map,
      float voxel_size,
      const Eigen::Vector3f& region_origin);

  // Generate LINE_LIST marker for unobserved but observable directions
  // Shows directions that are available (not occluded) but haven't been observed yet
  // for the poorly_observed_voxels in a cluster
  static visualization_msgs::Marker generateUnobservedDirectionsMarker(
      const ClusterInfo& cluster,
      const std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                                std::equal_to<Eigen::Vector3i>>& spatial_hash,
      float voxel_size,
      const std::string& frame_id = "world");

  // Generate TEXT_VIEW_FACING markers for voxel scores (observation, max_possible, target)
  // All scores are combined into a single MarkerArray
  static void generateVoxelScoreTextMarkers(
      const std::unordered_map<Eigen::Vector3i, VoxelCell, VoxelHash,
                                std::equal_to<Eigen::Vector3i>> &spatial_hash,
      float voxel_size,
      float base_score,
      float tex_weight,
      float geo_weight,
      visualization_msgs::MarkerArray &score_markers,
      const std::string& frame_id = "world");
};

} // namespace observation_quality
