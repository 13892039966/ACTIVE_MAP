#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <string>

namespace active_mapping {

class HeatmapClusterNode {
public:
  HeatmapClusterNode() : nh_(), pnh_("~") {
    pnh_.param<std::string>("input_topic", input_topic_, "/active_mapping/global_map");
    pnh_.param<std::string>("output_topic", output_topic_, "/active_mapping/high_intensity_map");
    pnh_.param<std::string>("frame_id", frame_id_, "");
    pnh_.param<double>("intensity_low_threshold", intensity_low_threshold_, 0.05);
    pnh_.param<double>("intensity_high_threshold", intensity_high_threshold_, 0.15);
    pnh_.param<double>("min_publish_intensity", min_publish_intensity_, 0.01);

    filtered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 1);
    sub_ = nh_.subscribe(input_topic_, 1, &HeatmapClusterNode::cloudCallback, this);

    ROS_INFO("Intensity filter node started. input=%s output=%s low=%.3f high=%.3f min_pub=%.3f",
             input_topic_.c_str(), output_topic_.c_str(), intensity_low_threshold_,
             intensity_high_threshold_, min_publish_intensity_);
  }

private:
  float smoothGate(float intensity) const {
    const float low = static_cast<float>(intensity_low_threshold_);
    const float high = static_cast<float>(
        std::max(intensity_high_threshold_, intensity_low_threshold_ + 1e-6));

    if (intensity <= low) {
      return 0.0f;
    }
    if (intensity >= high) {
      return 1.0f;
    }

    const float x = (intensity - low) / (high - low);
    return x * x * (3.0f - 2.0f * x);
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*msg, *input_cloud);
    if (input_cloud->empty()) {
      return;
    }

    pcl::PointCloud<pcl::PointXYZI> filtered_cloud;
    filtered_cloud.points.reserve(input_cloud->size());
    for (auto pt : input_cloud->points) {
      pt.intensity *= smoothGate(pt.intensity);
      if (pt.intensity >= static_cast<float>(min_publish_intensity_)) {
        filtered_cloud.points.push_back(pt);
      }
    }
    filtered_cloud.width = filtered_cloud.points.size();
    filtered_cloud.height = 1;
    filtered_cloud.is_dense = true;

    sensor_msgs::PointCloud2 out;
    pcl::toROSMsg(filtered_cloud, out);
    out.header = msg->header;
    if (!frame_id_.empty()) {
      out.header.frame_id = frame_id_;
    }
    filtered_pub_.publish(out);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber sub_;
  ros::Publisher filtered_pub_;

  std::string input_topic_;
  std::string output_topic_;
  std::string frame_id_;
  double intensity_low_threshold_;
  double intensity_high_threshold_;
  double min_publish_intensity_;
};

}  // namespace active_mapping

int main(int argc, char** argv) {
  ros::init(argc, argv, "heatmap_cluster_node");
  active_mapping::HeatmapClusterNode node;
  ros::spin();
  return 0;
}
