#pragma once

#include <observation_quality_manager/observation_quality_manager.h>
#include <observation_quality_manager/visualization_utils.h>
#include <observation_quality_manager/visibility_checker.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

namespace observation_quality {

class OQMVisualization {
public:
  typedef std::shared_ptr<OQMVisualization> Ptr;

  OQMVisualization(ros::NodeHandle &nh, ObservationQualityManager::Ptr oqm);

  // 发布所有可视化内容
  void publishAll(const Eigen::Vector3f &latest_odom_position);

  // 可视化：将当前更新的 region 以方块形式发布到 RViz
  void publishUpdatedRegions(
      const std::unordered_set<Eigen::Vector3i, RegionHash,
                               std::equal_to<Eigen::Vector3i>> &regions);

private:
  ros::NodeHandle nh_;
  ObservationQualityManager::Ptr oqm_;

  // Publishers
  ros::Publisher vis_pub_;
  ros::Publisher texture_vis_pub_;
  ros::Publisher observation_score_vis_pub_;
  ros::Publisher well_observed_vis_pub_;
  ros::Publisher geometry_complexity_data_pub_;
  ros::Publisher texture_complexity_data_pub_;
  ros::Publisher observation_score_data_pub_;
  ros::Publisher visibility_lines_pub_;
  ros::Publisher free_cluster_centers_pub_;
  ros::Publisher free_space_vis_pub_;
  ros::Publisher cluster_count_pub_;
  ros::Publisher updated_regions_pub_;
  ros::Publisher voxel_score_text_pub_;

  // 内部方法
  void publishComplexityClouds();
  void publishRawData();
  void publishVisibilityLines(const Eigen::Vector3f &odom_position);
  void publishFreeSpaceVisualization();
};

} // namespace observation_quality
