#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/filters/filter.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>

class PointCloudSanitizer {
public:
  explicit PointCloudSanitizer(ros::NodeHandle& nh)
      : tf_buffer_(ros::Duration(30.0)), tf_listener_(tf_buffer_) {
    std::string input_topic;
    std::string output_topic;
    int queue_size = 5;

    nh.param<std::string>("input_topic", input_topic, "/airsim/point_cloud_raw");
    nh.param<std::string>("output_topic", output_topic, "/airsim/point_cloud");
    nh.param<std::string>("target_frame", target_frame_, std::string("world"));
    nh.param("queue_size", queue_size, queue_size);

    pub_ = nh.advertise<sensor_msgs::PointCloud2>(output_topic, queue_size);
    sub_ = nh.subscribe(input_topic, queue_size,
                        &PointCloudSanitizer::cloudCallback, this,
                        ros::TransportHints().tcpNoDelay());

    ROS_INFO("PointCloudSanitizer: %s -> %s (target_frame=%s)",
             input_topic.c_str(), output_topic.c_str(), target_frame_.c_str());
  }

private:
  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    sensor_msgs::PointCloud2 cloud_world_msg;
    try {
      geometry_msgs::TransformStamped transform_stamped =
          tf_buffer_.lookupTransform(target_frame_, msg->header.frame_id,
                                     msg->header.stamp, ros::Duration(0.2));
      tf2::doTransform(*msg, cloud_world_msg, transform_stamped);
    } catch (const tf2::TransformException& ex) {
      ROS_WARN_THROTTLE(2.0,
                        "PointCloudSanitizer: TF lookup failed, dropping cloud. "
                        "target=%s source=%s reason=%s",
                        target_frame_.c_str(), msg->header.frame_id.c_str(),
                        ex.what());
      return;
    }

    pcl::PointCloud<pcl::PointXYZRGB> input_cloud;
    pcl::fromROSMsg(cloud_world_msg, input_cloud);

    pcl::PointCloud<pcl::PointXYZRGB> filtered_cloud;
    std::vector<int> valid_indices;
    pcl::removeNaNFromPointCloud(input_cloud, filtered_cloud, valid_indices);

    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(filtered_cloud, output_msg);
    output_msg.header = cloud_world_msg.header;
    output_msg.is_dense = true;

    if (filtered_cloud.size() != input_cloud.size()) {
      ROS_INFO_THROTTLE(2.0,
                        "PointCloudSanitizer: removed %zu invalid points "
                        "(kept %zu/%zu)",
                        input_cloud.size() - filtered_cloud.size(),
                        filtered_cloud.size(), input_cloud.size());
    }

    pub_.publish(output_msg);
  }

  ros::Subscriber sub_;
  ros::Publisher pub_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string target_frame_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "point_cloud_sanitizer");
  ros::NodeHandle nh("~");

  PointCloudSanitizer sanitizer(nh);
  ros::spin();
  return 0;
}
