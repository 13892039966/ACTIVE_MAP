#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <array>
#include <cstdint>
#include <vector>

namespace active_mapping {

class HeatmapClusterNode {
public:
  HeatmapClusterNode() : nh_(), pnh_("~") {
    pnh_.param<std::string>("input_topic", input_topic_, "/active_mapping/global_map");
    pnh_.param<std::string>("frame_id", frame_id_, "world");
    pnh_.param<double>("voxel_leaf_size", voxel_leaf_size_, 0.15);
    pnh_.param<double>("heat_threshold", heat_threshold_, 0.55);
    pnh_.param<double>("cluster_tolerance", cluster_tolerance_, 0.35);
    pnh_.param<int>("cluster_min_size", cluster_min_size_, 20);
    pnh_.param<int>("cluster_max_size", cluster_max_size_, 200000);

    voxel_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/active_mapping/voxelized_map", 1);
    heat_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/active_mapping/high_heat_map", 1);
    cluster_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/active_mapping/high_heat_clusters", 1);

    sub_ = nh_.subscribe(input_topic_, 1, &HeatmapClusterNode::cloudCallback, this);

    ROS_INFO(
        "Heatmap cluster node started. input=%s voxel=%.3f heat_thresh=%.3f tol=%.3f min_cluster=%d",
        input_topic_.c_str(), voxel_leaf_size_, heat_threshold_, cluster_tolerance_, cluster_min_size_);
  }

private:
  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*msg, *input_cloud);
    if (input_cloud->empty()) {
      return;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setInputCloud(input_cloud);
    voxel_filter.setLeafSize(
        static_cast<float>(voxel_leaf_size_),
        static_cast<float>(voxel_leaf_size_),
        static_cast<float>(voxel_leaf_size_));
    voxel_filter.filter(*voxel_cloud);

    publishCloud(*voxel_cloud, msg->header, voxel_pub_);

    pcl::PointCloud<pcl::PointXYZI>::Ptr heat_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    heat_cloud->reserve(voxel_cloud->size());
    for (const auto& pt : voxel_cloud->points) {
      if (pt.intensity >= heat_threshold_) {
        heat_cloud->points.push_back(pt);
      }
    }
    heat_cloud->width = heat_cloud->points.size();
    heat_cloud->height = 1;
    heat_cloud->is_dense = true;

    publishCloud(*heat_cloud, msg->header, heat_pub_);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    if (!heat_cloud->empty()) {
      pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
      tree->setInputCloud(heat_cloud);

      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
      ec.setClusterTolerance(cluster_tolerance_);
      ec.setMinClusterSize(cluster_min_size_);
      ec.setMaxClusterSize(cluster_max_size_);
      ec.setSearchMethod(tree);
      ec.setInputCloud(heat_cloud);
      ec.extract(cluster_indices);

      cluster_cloud->points.reserve(heat_cloud->size());
      std::size_t cluster_id = 0;
      for (const auto& indices : cluster_indices) {
        const auto color = colorForCluster(cluster_id++);
        for (const int idx : indices.indices) {
          const auto& src = heat_cloud->points[static_cast<std::size_t>(idx)];
          pcl::PointXYZRGB dst;
          dst.x = src.x;
          dst.y = src.y;
          dst.z = src.z;
          dst.r = color[0];
          dst.g = color[1];
          dst.b = color[2];
          cluster_cloud->points.push_back(dst);
        }
      }
    }

    cluster_cloud->width = cluster_cloud->points.size();
    cluster_cloud->height = 1;
    cluster_cloud->is_dense = true;

    sensor_msgs::PointCloud2 cluster_msg;
    pcl::toROSMsg(*cluster_cloud, cluster_msg);
    cluster_msg.header = msg->header;
    if (!frame_id_.empty()) {
      cluster_msg.header.frame_id = frame_id_;
    }
    cluster_pub_.publish(cluster_msg);
  }

  void publishCloud(const pcl::PointCloud<pcl::PointXYZI>& cloud,
                    const std_msgs::Header& input_header,
                    ros::Publisher& pub) {
    sensor_msgs::PointCloud2 out;
    pcl::toROSMsg(cloud, out);
    out.header = input_header;
    if (!frame_id_.empty()) {
      out.header.frame_id = frame_id_;
    }
    pub.publish(out);
  }

  std::array<std::uint8_t, 3> colorForCluster(std::size_t cluster_id) const {
    static const std::array<std::array<std::uint8_t, 3>, 8> palette = {{
        {{239, 83, 80}}, {{66, 165, 245}}, {{102, 187, 106}}, {{255, 202, 40}},
        {{171, 71, 188}}, {{255, 112, 67}}, {{38, 198, 218}}, {{255, 238, 88}},
    }};
    return palette[cluster_id % palette.size()];
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber sub_;
  ros::Publisher voxel_pub_;
  ros::Publisher heat_pub_;
  ros::Publisher cluster_pub_;

  std::string input_topic_;
  std::string frame_id_;
  double voxel_leaf_size_;
  double heat_threshold_;
  double cluster_tolerance_;
  int cluster_min_size_;
  int cluster_max_size_;
};

}  // namespace active_mapping

int main(int argc, char** argv) {
  ros::init(argc, argv, "heatmap_cluster_node");
  active_mapping::HeatmapClusterNode node;
  ros::spin();
  return 0;
}
