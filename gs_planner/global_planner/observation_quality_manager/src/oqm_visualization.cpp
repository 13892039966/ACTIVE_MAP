#include <observation_quality_manager/oqm_visualization.h>
#include <observation_quality_manager/global_planner.h>
#include <pcl_conversions/pcl_conversions.h>

namespace observation_quality {

OQMVisualization::OQMVisualization(ros::NodeHandle &nh,
                                   ObservationQualityManager::Ptr oqm)
    : nh_(nh), oqm_(oqm) {
  // Setup visualization publishers (5 Hz)
  vis_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      "/geometry_complexity_visualization", 1);
  texture_vis_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      "/texture_complexity_visualization", 1);
  observation_score_vis_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      "/observation_score_visualization", 1);
  well_observed_vis_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      "/well_observed_visualization", 1);

  // Setup data publishers for histogram plotting
  geometry_complexity_data_pub_ = nh_.advertise<std_msgs::Float32MultiArray>(
      "/geometry_complexity_data", 1);
  texture_complexity_data_pub_ = nh_.advertise<std_msgs::Float32MultiArray>(
      "/texture_complexity_data", 1);
  observation_score_data_pub_ = nh_.advertise<std_msgs::Float32MultiArray>(
      "/observation_score_data", 1);

  // Visibility and free space publishers
  visibility_lines_pub_ =
      nh_.advertise<visualization_msgs::Marker>("/visibility_lines", 1);
  free_cluster_centers_pub_ =
      nh_.advertise<visualization_msgs::Marker>("/free_cluster_centers", 1);
  free_space_vis_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      "/free_space_visualization", 1);
  cluster_count_pub_ = nh_.advertise<std_msgs::Int32>("/cluster_count", 1);
  updated_regions_pub_ =
      nh_.advertise<visualization_msgs::Marker>("/updated_regions", 1);
  voxel_score_text_pub_ =
      nh_.advertise<visualization_msgs::MarkerArray>("/voxel_score_text", 1);
}

void OQMVisualization::publishAll(const Eigen::Vector3f &latest_odom_position) {
  publishComplexityClouds();
  publishRawData();
  // publishVisibilityLines(latest_odom_position);
  publishFreeSpaceVisualization();
}

void OQMVisualization::publishUpdatedRegions(
    const std::unordered_set<Eigen::Vector3i, RegionHash,
                             std::equal_to<Eigen::Vector3i>> &regions) {
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = "updated_regions";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::CUBE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.scale.x = FreeRegion::REGION_SIZE;
  marker.scale.y = FreeRegion::REGION_SIZE;
  marker.scale.z = FreeRegion::REGION_SIZE;

  std_msgs::ColorRGBA color;
  color.r = 1.0f;   // orange-ish
  color.g = 0.6f;
  color.b = 0.0f;
  color.a = 0.25f;

  marker.points.reserve(regions.size());
  marker.colors.reserve(regions.size());

  // Get region origin from OQM for correct visualization
  Eigen::Vector3f region_origin = oqm_->getRegionOrigin();

  for (const auto &region_idx : regions) {
    geometry_msgs::Point p;
    Eigen::Vector3f offset = ((region_idx.cast<float>().array() + 0.5f) * FreeRegion::REGION_SIZE).matrix();
    Eigen::Vector3f region_center = region_origin + offset;
    p.x = region_center.x();
    p.y = region_center.y();
    p.z = region_center.z();
    marker.points.push_back(p);
    marker.colors.push_back(color);
  }

  updated_regions_pub_.publish(marker);
}

void OQMVisualization::publishComplexityClouds() {
  // Generate and publish geometry complexity cloud
  auto vis_cloud = VisualizationUtils::generateGeometryComplexityCloud(
      oqm_->getSpatialHash(), oqm_->getVoxelSize());

  if (!vis_cloud->points.empty()) {
    sensor_msgs::PointCloud2 vis_msg;
    pcl::toROSMsg(*vis_cloud, vis_msg);
    vis_msg.header.frame_id = "world";
    vis_msg.header.stamp = ros::Time::now();
    vis_pub_.publish(vis_msg);
  }

  // Generate and publish texture complexity cloud
  auto texture_cloud = VisualizationUtils::generateTextureComplexityCloud(
      oqm_->getSpatialHash(), oqm_->getVoxelSize());

  if (!texture_cloud->points.empty()) {
    sensor_msgs::PointCloud2 texture_msg;
    pcl::toROSMsg(*texture_cloud, texture_msg);
    texture_msg.header.frame_id = "world";
    texture_msg.header.stamp = ros::Time::now();
    texture_vis_pub_.publish(texture_msg);
  }

  // Generate and publish observation score cloud
  auto observation_score_cloud =
      VisualizationUtils::generateObservationScoreCloud(
          oqm_->getSpatialHash(), oqm_->getVoxelSize());

  if (!observation_score_cloud->points.empty()) {
    sensor_msgs::PointCloud2 observation_score_msg;
    pcl::toROSMsg(*observation_score_cloud, observation_score_msg);
    observation_score_msg.header.frame_id = "world";
    observation_score_msg.header.stamp = ros::Time::now();
    observation_score_vis_pub_.publish(observation_score_msg);
  }

  // Generate and publish well observed cloud
  auto well_observed_cloud = VisualizationUtils::generateWellObservedCloud(
      oqm_->getSpatialHash(), oqm_->getVoxelSize());

  if (!well_observed_cloud->points.empty()) {
    sensor_msgs::PointCloud2 well_observed_msg;
    pcl::toROSMsg(*well_observed_cloud, well_observed_msg);
    well_observed_msg.header.frame_id = "world";
    well_observed_msg.header.stamp = ros::Time::now();
    well_observed_vis_pub_.publish(well_observed_msg);
  }

  ROS_INFO_THROTTLE(
      2.0,
      "Published visualization clouds - geometry: %zu, texture: %zu, "
      "observation_score: %zu, well_observed: %zu",
      vis_cloud->points.size(), texture_cloud->points.size(),
      observation_score_cloud->points.size(),
      well_observed_cloud->points.size());

  // Generate and publish voxel score text markers
  visualization_msgs::MarkerArray score_markers;
  VisualizationUtils::generateVoxelScoreTextMarkers(
      oqm_->getSpatialHash(), oqm_->getVoxelSize(),
      oqm_->getWellObservedBaseScore(),
      oqm_->getWellObservedTextureWeight(),
      oqm_->getWellObservedGeoWeight(),
      score_markers);
  voxel_score_text_pub_.publish(score_markers);
}

void OQMVisualization::publishRawData() {
  // Collect and publish raw data for histogram plotting
  std::vector<float> geometry_data;
  std::vector<float> texture_data;
  std::vector<float> observation_score_data;

  const auto &spatial_hash = oqm_->getSpatialHash();
  geometry_data.reserve(spatial_hash.size());
  texture_data.reserve(spatial_hash.size());
  observation_score_data.reserve(spatial_hash.size());

  for (const auto &[idx, cell] : spatial_hash) {
    // Collect geometry complexity (only if geo_cnt >= 3)
    if (cell.geo_cnt >= 3) {
      geometry_data.push_back(cell.geometric_complexity);
    }

    // Collect texture complexity (only if tex_cnt >= 2)
    if (cell.tex_cnt >= 2) {
      texture_data.push_back(cell.texture_complexity);
    }

    // Collect observation score (only if score > 0)
    if (cell.observation_score > 0.0f) {
      observation_score_data.push_back(cell.observation_score);
    }
  }

  // Publish geometry complexity data
  if (!geometry_data.empty()) {
    std_msgs::Float32MultiArray geometry_msg;
    geometry_msg.data = geometry_data;
    geometry_complexity_data_pub_.publish(geometry_msg);
  }

  // Publish texture complexity data
  if (!texture_data.empty()) {
    std_msgs::Float32MultiArray texture_msg;
    texture_msg.data = texture_data;
    texture_complexity_data_pub_.publish(texture_msg);
  }

  // Publish observation score data
  if (!observation_score_data.empty()) {
    std_msgs::Float32MultiArray observation_score_msg;
    observation_score_msg.data = observation_score_data;
    observation_score_data_pub_.publish(observation_score_msg);
  }
}

void OQMVisualization::publishFreeSpaceVisualization() {
  // 自由空间聚类中心可视化
  auto cluster_marker = VisualizationUtils::generateFreeClusterCentersMarker(
      oqm_->getRegionMap(), oqm_->getSpatialHash(), oqm_->getVoxelSize(),
      oqm_->getRegionOrigin(), "world");
  free_cluster_centers_pub_.publish(cluster_marker);

  // 发布聚类数量
  std_msgs::Int32 cluster_count_msg;
  cluster_count_msg.data = static_cast<int32_t>(cluster_marker.points.size());
  cluster_count_pub_.publish(cluster_count_msg);

  ROS_INFO_THROTTLE(2.0, "Published %zu cluster centers (size > 30)",
                    cluster_marker.points.size());

  // 自由空间点云可视化
  auto free_space_cloud = VisualizationUtils::generateFreeSpaceCloud(
      oqm_->getRegionMap(), oqm_->getVoxelSize(), oqm_->getRegionOrigin());

  if (!free_space_cloud->points.empty()) {
    sensor_msgs::PointCloud2 free_space_msg;
    pcl::toROSMsg(*free_space_cloud, free_space_msg);
    free_space_msg.header.frame_id = "world";
    free_space_msg.header.stamp = ros::Time::now();
    free_space_vis_pub_.publish(free_space_msg);

    ROS_INFO_THROTTLE(2.0, "Published free space cloud with %zu voxels",
                      free_space_cloud->points.size());
  }
}

} // namespace observation_quality
