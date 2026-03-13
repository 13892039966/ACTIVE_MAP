#include <cmath>
#include <limits>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

class DepthToPointCloudNode
{
public:
    DepthToPointCloudNode()
        : nh_(), pnh_("~"), has_camera_info_(false)
    {
        pnh_.param("depth_topic", depth_topic_, std::string("/airsim_node/drone_1/front_center/DepthPlanner"));
        pnh_.param("camera_info_topic", camera_info_topic_, std::string("/airsim_node/drone_1/front_center/DepthPlanner/camera_info"));
        pnh_.param("output_topic", output_topic_, std::string("/airsim_depth_points"));
        pnh_.param("output_frame", output_frame_, std::string(""));
        pnh_.param("min_depth", min_depth_, 0.1);
        pnh_.param("max_depth", max_depth_, 100.0);
        pnh_.param("stride", stride_, 1);

        if (stride_ < 1)
        {
            ROS_WARN("stride must be >= 1. Resetting to 1.");
            stride_ = 1;
        }

        cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 1);
        camera_info_sub_ = nh_.subscribe(camera_info_topic_, 1, &DepthToPointCloudNode::cameraInfoCallback, this);
        depth_sub_ = nh_.subscribe(depth_topic_, 1, &DepthToPointCloudNode::depthCallback, this);

        ROS_INFO_STREAM("depth_to_pointcloud_node subscribed depth_topic=" << depth_topic_
                        << ", camera_info_topic=" << camera_info_topic_
                        << ", publishing " << output_topic_);
    }

private:
    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg)
    {
        camera_info_ = *msg;
        has_camera_info_ = true;
    }

    void depthCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        if (!has_camera_info_)
        {
            ROS_WARN_THROTTLE(5.0, "Waiting for camera_info before converting depth image to point cloud");
            return;
        }

        if (msg->encoding != sensor_msgs::image_encodings::TYPE_32FC1 &&
            msg->encoding != "32FC1")
        {
            ROS_WARN_THROTTLE(5.0, "Unsupported depth encoding: %s", msg->encoding.c_str());
            return;
        }

        cv_bridge::CvImageConstPtr depth_ptr;
        try
        {
            depth_ptr = cv_bridge::toCvShare(msg);
        }
        catch (const cv_bridge::Exception& e)
        {
            ROS_ERROR_THROTTLE(5.0, "cv_bridge exception: %s", e.what());
            return;
        }

        const double fx = camera_info_.K[0];
        const double fy = camera_info_.K[4];
        const double cx = camera_info_.K[2];
        const double cy = camera_info_.K[5];

        if (fx <= 0.0 || fy <= 0.0)
        {
            ROS_WARN_THROTTLE(5.0, "Invalid camera intrinsics fx=%f fy=%f", fx, fy);
            return;
        }

        pcl::PointCloud<pcl::PointXYZ> cloud;
        cloud.points.reserve((msg->width / stride_) * (msg->height / stride_));

        for (int v = 0; v < depth_ptr->image.rows; v += stride_)
        {
            for (int u = 0; u < depth_ptr->image.cols; u += stride_)
            {
                const float depth = depth_ptr->image.at<float>(v, u);
                if (!std::isfinite(depth) || depth < min_depth_ || depth > max_depth_)
                {
                    continue;
                }

                pcl::PointXYZ pt;
                pt.x = static_cast<float>((u - cx) * depth / fx);
                pt.y = static_cast<float>((v - cy) * depth / fy);
                pt.z = depth;
                cloud.points.push_back(pt);
            }
        }

        cloud.width = cloud.points.size();
        cloud.height = 1;
        cloud.is_dense = false;

        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(cloud, cloud_msg);
        cloud_msg.header = msg->header;
        if (!output_frame_.empty())
        {
            cloud_msg.header.frame_id = output_frame_;
        }

        cloud_pub_.publish(cloud_msg);
    }

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber depth_sub_;
    ros::Subscriber camera_info_sub_;
    ros::Publisher cloud_pub_;

    sensor_msgs::CameraInfo camera_info_;
    bool has_camera_info_;

    std::string depth_topic_;
    std::string camera_info_topic_;
    std::string output_topic_;
    std::string output_frame_;
    double min_depth_;
    double max_depth_;
    int stride_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "depth_to_pointcloud_node");
    DepthToPointCloudNode node;
    ros::spin();
    return 0;
}
