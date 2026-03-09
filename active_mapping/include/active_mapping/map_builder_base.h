#pragma once

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <deque>

#define PCL_NO_PRECOMPILE 
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Eigenvalues>

// =========================================================
// 🌟 核心修复：彻底抛弃 PCL_ADD_POINT4D 和 EIGEN 对齐宏
// 改为纯朴素 C++ 结构体，彻底免疫 ikd-Tree 底层的 new 内存不对齐问题！
// ==========================================
struct PointExploration {
    float x, y, z;
    union {
        struct {
            uint8_t b;
            uint8_t g;
            uint8_t r;
            uint8_t a;
        };
        float rgb;
        uint32_t rgba;
    };
    float normal_x, normal_y, normal_z;
    float curvature;      
    float color_entropy;  
    float intensity;      
    float radius;         
    float confidence;     
};

// PCL 依然可以完美识别它
POINT_CLOUD_REGISTER_POINT_STRUCT(PointExploration,
                                  (float, x, x)(float, y, y)(float, z, z)
                                  (uint32_t, rgba, rgba)
                                  (float, normal_x, normal_x)(float, normal_y, normal_y)(float, normal_z, normal_z)
                                  (float, curvature, curvature)(float, color_entropy, color_entropy)(float, intensity, intensity)
                                  (float, radius, radius)(float, confidence, confidence)
)

#include "active_mapping/ikd-Tree/ikd_Tree.h"
#include "active_mapping/ikd-Tree/ikd_Tree_impl.h"

#include <memory>
#include <string>
#include <cstddef>

namespace active_mapping {

class MapBuilderBase {
public:
    using PointT = PointExploration;
    using CloudT = pcl::PointCloud<PointT>;
    virtual ~MapBuilderBase() = default;
    void init(ros::NodeHandle& nh);
    void run();

protected:
    virtual void processAndInsertCloud(CloudT::Ptr& cloud) = 0;
    virtual void publishMap() = 0;
    void updatePointCountStats(std::size_t raw_points,
                               std::size_t near_points,
                               std::size_t downsampled_points,
                               std::size_t output_points);

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);

    ros::NodeHandle nh_;
    ros::Subscriber sub_cloud_;
    ros::Publisher pub_global_map_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<ikdtreeNS::KD_TREE<PointT>> ikd_tree_;

    std::deque<sensor_msgs::PointCloud2ConstPtr> cloud_queue_;

    std::string global_frame_id_;
    double downsample_size_;
    double publish_freq_;
    double max_depth_;
    bool enable_backend_voxel_filter_;
    bool ikd_tree_downsample_;
    bool publish_only_on_update_;
    std::size_t max_queue_size_;
    ros::Time last_publish_time_;
    bool map_dirty_;

    ros::Time last_stats_report_time_;
    std::size_t cloud_age_count_;
    std::size_t processed_count_;
    std::size_t tf_lookup_fail_count_;
    std::size_t queue_drop_count_;
    double cloud_age_sum_sec_;
    double cloud_age_max_sec_;

    std::size_t point_stats_count_;
    std::size_t raw_points_sum_;
    std::size_t near_points_sum_;
    std::size_t downsampled_points_sum_;
    std::size_t output_points_sum_;
    std::size_t raw_points_max_;
    std::size_t near_points_max_;
    std::size_t downsampled_points_max_;
    std::size_t output_points_max_;
};

} // namespace active_mapping
