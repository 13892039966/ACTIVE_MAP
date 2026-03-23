#pragma once

#include <deque>
#include <memory>
#include <string>
#include <cstddef>
#include <cstdint>
#include <unordered_set>

#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include "active_mapping/BackendBenchmark.h"

#define PCL_NO_PRECOMPILE 
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Eigenvalues>

// Plain point struct avoids alignment issues with the current ikd-tree usage.
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
        std::uint32_t rgba;
    };
    float normal_x, normal_y, normal_z;
    float curvature;      
    // Surface texture score approximation after edge suppression.
    float color_entropy;
    // Published texture score mirror for RViz/intensity-based consumers.
    float intensity;
    // Mean valid neighbor radius used by the texture analysis.
    float radius;
    // Color edge response around the query point.
    float confidence;
};

POINT_CLOUD_REGISTER_POINT_STRUCT(PointExploration,
                                  (float, x, x)(float, y, y)(float, z, z)
                                  (std::uint32_t, rgba, rgba)
                                  (float, normal_x, normal_x)(float, normal_y, normal_y)(float, normal_z, normal_z)
                                  (float, curvature, curvature)(float, color_entropy, color_entropy)(float, intensity, intensity)
                                  (float, radius, radius)(float, confidence, confidence)
)

#include "active_mapping/ikd-Tree/ikd_Tree.h"
#include "active_mapping/ikd-Tree/ikd_Tree_impl.h"

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
    virtual bool publishMap() = 0;
    virtual void publishBenchmark(const ros::Time& stamp,
                                  double window_duration_sec,
                                  std::size_t processed_count,
                                  std::size_t queue_drop_count,
                                  double cloud_age_mean,
                                  double cloud_age_max) = 0;
    void updatePointCountStats(std::size_t raw_points,
                               std::size_t near_points,
                               std::size_t downsampled_points,
                               std::size_t output_points);

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);

    ros::NodeHandle nh_;
    ros::Subscriber sub_cloud_;
    ros::Publisher pub_global_map_;
    ros::Publisher pub_benchmark_;

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
    bool publish_benchmark_;
};

class CloudMapBuilder : public MapBuilderBase {
public:
    CloudMapBuilder() = default;
    ~CloudMapBuilder() override = default;

    void initParams(ros::NodeHandle& nh);

protected:
    void processAndInsertCloud(CloudT::Ptr& cloud) override;
    bool publishMap() override;
    void publishBenchmark(const ros::Time& stamp,
                          double window_duration_sec,
                          std::size_t processed_count,
                          std::size_t queue_drop_count,
                          double cloud_age_mean,
                          double cloud_age_max) override;

private:
    struct VoxelKey {
        int x;
        int y;
        int z;

        bool operator==(const VoxelKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct VoxelKeyHash {
        std::size_t operator()(const VoxelKey& key) const {
            std::size_t seed = 0;
            seed ^= std::hash<int>{}(key.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= std::hash<int>{}(key.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= std::hash<int>{}(key.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };

    int knn_k_;
    double knn_max_radius_;
    bool compute_normals_;
    int analysis_frame_window_;
    double low_texture_threshold_;
    double low_texture_downsample_size_;
    bool debug_texture_stats_;
    double benchmark_support_ms_ = 0.0;
    double benchmark_search_ms_ = 0.0;
    double benchmark_score_ms_ = 0.0;
    double benchmark_total_ms_ = 0.0;
    double benchmark_raw_neighbors_mean_ = 0.0;
    double benchmark_zeroed_ratio_ = 0.0;
    double benchmark_edge_mean_ = 0.0;
    double benchmark_texture_mean_ = 0.0;
    CloudT cached_cloud_;
    bool cache_valid_ = false;
    std::deque<CloudT> recent_analysis_frames_;
};

} // namespace active_mapping
