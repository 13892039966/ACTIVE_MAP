#include "active_mapping/map_builder_base.h"
#include <pcl/filters/voxel_grid.h>
#include <algorithm>

namespace active_mapping {

void MapBuilderBase::updatePointCountStats(std::size_t raw_points,
                                           std::size_t near_points,
                                           std::size_t downsampled_points,
                                           std::size_t output_points) {
    ++point_stats_count_;
    raw_points_sum_ += raw_points;
    near_points_sum_ += near_points;
    downsampled_points_sum_ += downsampled_points;
    output_points_sum_ += output_points;

    raw_points_max_ = std::max(raw_points_max_, raw_points);
    near_points_max_ = std::max(near_points_max_, near_points);
    downsampled_points_max_ = std::max(downsampled_points_max_, downsampled_points);
    output_points_max_ = std::max(output_points_max_, output_points);
}

void MapBuilderBase::init(ros::NodeHandle& nh) {
    nh_ = nh;
    nh_.param<std::string>("global_frame_id", global_frame_id_, "world");
    nh_.param<double>("downsample_size", downsample_size_, 0.05);
    nh_.param<double>("publish_freq", publish_freq_, 2.0);
    nh_.param<double>("max_depth", max_depth_, 10.0);
    nh_.param<bool>("enable_backend_voxel_filter", enable_backend_voxel_filter_, true);
    nh_.param<bool>("ikd_tree_downsample", ikd_tree_downsample_, true);
    nh_.param<bool>("publish_only_on_update", publish_only_on_update_, true);
    int max_queue_size_param = 3;
    nh_.param<int>("max_queue_size", max_queue_size_param, 3);
    max_queue_size_ = std::max(1, max_queue_size_param);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(ros::Duration(30.0));
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    ikd_tree_ = std::make_unique<ikdtreeNS::KD_TREE<PointT>>();
    ikd_tree_->InitializeKDTree(0.3, 0.6, downsample_size_);

    sub_cloud_ = nh_.subscribe("/cloud_alin_color", 10, &MapBuilderBase::cloudCallback, this);
    pub_global_map_ = nh_.advertise<sensor_msgs::PointCloud2>("/active_mapping/global_map", 1);

    last_publish_time_ = ros::Time::now();
    last_stats_report_time_ = last_publish_time_;
    cloud_age_count_ = 0;
    processed_count_ = 0;
    tf_lookup_fail_count_ = 0;
    queue_drop_count_ = 0;
    cloud_age_sum_sec_ = 0.0;
    cloud_age_max_sec_ = 0.0;
    point_stats_count_ = 0;
    raw_points_sum_ = 0;
    near_points_sum_ = 0;
    downsampled_points_sum_ = 0;
    output_points_sum_ = 0;
    raw_points_max_ = 0;
    near_points_max_ = 0;
    downsampled_points_max_ = 0;
    output_points_max_ = 0;
    map_dirty_ = false;
}

void MapBuilderBase::cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    cloud_queue_.push_back(msg);
    // Keep queue short to avoid stale cloud timestamps falling outside TF cache.
    while (cloud_queue_.size() > max_queue_size_) {
        cloud_queue_.pop_front();
        ++queue_drop_count_;
    }
}

void MapBuilderBase::run() {
    ros::Rate rate(100);
    while (ros::ok()) {
        ros::spinOnce();

        if (!cloud_queue_.empty()) {
            // Always process the newest cloud and discard stale backlog.
            auto process_msg = cloud_queue_.back();
            cloud_queue_.clear();
            const double cloud_age = (ros::Time::now() - process_msg->header.stamp).toSec();
            cloud_age_sum_sec_ += cloud_age;
            cloud_age_max_sec_ = std::max(cloud_age_max_sec_, cloud_age);
            ++cloud_age_count_;

            geometry_msgs::TransformStamped transform_stamped;
            try {
                transform_stamped = tf_buffer_->lookupTransform(
                    global_frame_id_,
                    process_msg->header.frame_id,
                    process_msg->header.stamp,
                    ros::Duration(0.2));
            } catch (tf2::TransformException& ex) {
                ++tf_lookup_fail_count_;
                ROS_WARN_THROTTLE(
                    2.0,
                    "TF lookup failed, dropping frame. target=%s source=%s stamp=%.6f cloud_age=%.3fs reason=%s",
                    global_frame_id_.c_str(),
                    process_msg->header.frame_id.c_str(),
                    process_msg->header.stamp.toSec(),
                    cloud_age,
                    ex.what());
                rate.sleep();
                continue;
            }

            sensor_msgs::PointCloud2 cloud_world_msg;
            tf2::doTransform(*process_msg, cloud_world_msg, transform_stamped);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_world(new pcl::PointCloud<pcl::PointXYZRGB>());
            pcl::fromROSMsg(cloud_world_msg, *cloud_world);
            const std::size_t raw_points = cloud_world->size();

            double max_sq_dist = max_depth_ * max_depth_;
            double cx = transform_stamped.transform.translation.x;
            double cy = transform_stamped.transform.translation.y;
            double cz = transform_stamped.transform.translation.z;

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_near(new pcl::PointCloud<pcl::PointXYZRGB>());
            cloud_near->reserve(cloud_world->size());

            for (const auto& pt_in : cloud_world->points) {
                if (!std::isfinite(pt_in.x) || !std::isfinite(pt_in.y) || !std::isfinite(pt_in.z)) {
                    continue;
                }
                double sq_dist = (pt_in.x - cx) * (pt_in.x - cx) +
                                 (pt_in.y - cy) * (pt_in.y - cy) +
                                 (pt_in.z - cz) * (pt_in.z - cz);
                if (sq_dist <= max_sq_dist) {
                    cloud_near->points.push_back(pt_in);
                }
            }
            const std::size_t near_points = cloud_near->size();

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZRGB>());
            if (enable_backend_voxel_filter_ && downsample_size_ > 0.0) {
                pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
                voxel_filter.setInputCloud(cloud_near);
                voxel_filter.setLeafSize(
                    static_cast<float>(downsample_size_),
                    static_cast<float>(downsample_size_),
                    static_cast<float>(downsample_size_));
                voxel_filter.filter(*cloud_downsampled);
            } else {
                *cloud_downsampled = *cloud_near;
            }
            const std::size_t downsampled_points = cloud_downsampled->size();

            CloudT::Ptr cloud_filtered(new CloudT());
            cloud_filtered->reserve(cloud_downsampled->size());
            for (const auto& pt_in : cloud_downsampled->points) {
                PointT pt_out;
                pt_out.x = pt_in.x;
                pt_out.y = pt_in.y;
                pt_out.z = pt_in.z;
                pt_out.r = pt_in.r;
                pt_out.g = pt_in.g;
                pt_out.b = pt_in.b;
                pt_out.normal_x = 0.0f;
                pt_out.normal_y = 0.0f;
                pt_out.normal_z = 0.0f;
                pt_out.curvature = 0.0f;
                pt_out.color_entropy = 0.0f;
                pt_out.intensity = 0.0f;
                pt_out.radius = 0.0f;
                pt_out.confidence = 0.0f;
                cloud_filtered->points.push_back(pt_out);
            }
            const std::size_t output_points = cloud_filtered->size();
            updatePointCountStats(raw_points, near_points, downsampled_points, output_points);

            if (!cloud_filtered->empty()) {
                const std::size_t tree_size_before = ikd_tree_->size();
                processAndInsertCloud(cloud_filtered);
                map_dirty_ = map_dirty_ || (ikd_tree_->size() > tree_size_before);
                ++processed_count_;
            }
        }

        ros::Time now = ros::Time::now();
        if ((now - last_publish_time_).toSec() >= (1.0 / publish_freq_)) {
            const bool should_publish = !publish_only_on_update_ || map_dirty_;
            if (ikd_tree_->size() > 0 && should_publish) {
                publishMap();
                map_dirty_ = false;
            }
            last_publish_time_ = now;
        }

        if ((now - last_stats_report_time_).toSec() >= 2.0) {
            const double mean_cloud_age = cloud_age_count_ > 0
                ? (cloud_age_sum_sec_ / static_cast<double>(cloud_age_count_))
                : 0.0;
            const double mean_raw_points = point_stats_count_ > 0
                ? (static_cast<double>(raw_points_sum_) / static_cast<double>(point_stats_count_))
                : 0.0;
            const double mean_near_points = point_stats_count_ > 0
                ? (static_cast<double>(near_points_sum_) / static_cast<double>(point_stats_count_))
                : 0.0;
            const double mean_downsampled_points = point_stats_count_ > 0
                ? (static_cast<double>(downsampled_points_sum_) / static_cast<double>(point_stats_count_))
                : 0.0;
            const double mean_output_points = point_stats_count_ > 0
                ? (static_cast<double>(output_points_sum_) / static_cast<double>(point_stats_count_))
                : 0.0;
            ROS_INFO(
                "Cloud stats (last %.2fs): age_mean=%.3fs age_max=%.3fs samples=%zu processed=%zu tf_fail=%zu queue_drop=%zu tree_size=%d | "
                "points mean raw/near/down/out=%.0f/%.0f/%.0f/%.0f max raw/near/down/out=%zu/%zu/%zu/%zu",
                (now - last_stats_report_time_).toSec(),
                mean_cloud_age,
                cloud_age_max_sec_,
                cloud_age_count_,
                processed_count_,
                tf_lookup_fail_count_,
                queue_drop_count_,
                ikd_tree_->size(),
                mean_raw_points,
                mean_near_points,
                mean_downsampled_points,
                mean_output_points,
                raw_points_max_,
                near_points_max_,
                downsampled_points_max_,
                output_points_max_);

            last_stats_report_time_ = now;
            cloud_age_count_ = 0;
            processed_count_ = 0;
            tf_lookup_fail_count_ = 0;
            queue_drop_count_ = 0;
            cloud_age_sum_sec_ = 0.0;
            cloud_age_max_sec_ = 0.0;
            point_stats_count_ = 0;
            raw_points_sum_ = 0;
            near_points_sum_ = 0;
            downsampled_points_sum_ = 0;
            output_points_sum_ = 0;
            raw_points_max_ = 0;
            near_points_max_ = 0;
            downsampled_points_max_ = 0;
            output_points_max_ = 0;
        }

        rate.sleep();
    }
}

} // namespace active_mapping
