#include "active_mapping/map_builder_base.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <unordered_set>

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
    nh_.param<bool>("enable_backend_voxel_filter", enable_backend_voxel_filter_, false);
    nh_.param<bool>("ikd_tree_downsample", ikd_tree_downsample_, false);
    nh_.param<bool>("publish_only_on_update", publish_only_on_update_, true);
    nh_.param<bool>("publish_benchmark", publish_benchmark_, true);
    int max_queue_size_param = 3;
    nh_.param<int>("max_queue_size", max_queue_size_param, 3);
    max_queue_size_ = std::max(1, max_queue_size_param);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(ros::Duration(30.0));
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    ikd_tree_ = std::make_unique<ikdtreeNS::KD_TREE<PointT>>();
    ikd_tree_->InitializeKDTree(0.3, 0.6, downsample_size_);

    sub_cloud_ = nh_.subscribe("/cloud", 10, &MapBuilderBase::cloudCallback, this);
    pub_global_map_ = nh_.advertise<sensor_msgs::PointCloud2>("/global_map", 1);
    if (publish_benchmark_) {
        pub_benchmark_ = nh_.advertise<active_mapping::BackendBenchmark>("/active_mapping/backend_benchmark", 1);
    }

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
                pt_out.a = 255;
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
                if (publishMap()) {
                    map_dirty_ = false;
                }
            }
            last_publish_time_ = now;
        }

        if ((now - last_stats_report_time_).toSec() >= 2.0) {
            const double stats_window_sec = (now - last_stats_report_time_).toSec();
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
                stats_window_sec,
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

            if (publish_benchmark_ && pub_benchmark_) {
                publishBenchmark(now,
                                 stats_window_sec,
                                 processed_count_,
                                 queue_drop_count_,
                                 mean_cloud_age,
                                 cloud_age_max_sec_);
            }

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

void CloudMapBuilder::initParams(ros::NodeHandle& nh) {
    nh.param<int>("knn_k", knn_k_, 15);
    nh.param<double>("knn_max_radius", knn_max_radius_, 0.12);
    nh.param<bool>("compute_normals", compute_normals_, false);
    nh.param<int>("analysis_frame_window", analysis_frame_window_, 1);
    nh.param<double>("low_texture_threshold", low_texture_threshold_, 0.03);
    nh.param<double>("low_texture_downsample_size", low_texture_downsample_size_, 0.15);
    nh.param<bool>("debug_texture_stats", debug_texture_stats_, false);

    ROS_INFO("Color texture analysis started. Output topic: /active_mapping/global_map");
    ROS_INFO("Texture params: knn_k=%d knn_max_radius=%.3f analysis_frame_window=%d low_texture_threshold=%.3f low_texture_downsample_size=%.3f",
             knn_k_,
             knn_max_radius_,
             analysis_frame_window_,
             low_texture_threshold_,
             low_texture_downsample_size_);
}

void CloudMapBuilder::publishBenchmark(const ros::Time& stamp,
                                       double window_duration_sec,
                                       std::size_t processed_count,
                                       std::size_t queue_drop_count,
                                       double cloud_age_mean,
                                       double cloud_age_max) {
    if (!pub_benchmark_) {
        return;
    }

    active_mapping::BackendBenchmark msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = global_frame_id_;
    msg.window_duration_sec = window_duration_sec;
    msg.processed_count = static_cast<std::uint32_t>(processed_count);
    msg.queue_drop_count = static_cast<std::uint32_t>(queue_drop_count);
    msg.cloud_age_mean = cloud_age_mean;
    msg.cloud_age_max = cloud_age_max;
    msg.support_ms = benchmark_support_ms_;
    msg.search_ms = benchmark_search_ms_;
    msg.score_ms = benchmark_score_ms_;
    msg.total_ms = benchmark_total_ms_;
    msg.raw_neighbors_mean = benchmark_raw_neighbors_mean_;
    msg.zeroed_ratio = benchmark_zeroed_ratio_;
    msg.edge_mean = benchmark_edge_mean_;
    msg.texture_mean = benchmark_texture_mean_;
    pub_benchmark_.publish(msg);
}

void CloudMapBuilder::processAndInsertCloud(CloudT::Ptr& cloud) {
    using Clock = std::chrono::steady_clock;
    const auto t_start = Clock::now();
    CloudT::Ptr analysis_support;
    if (analysis_frame_window_ <= 1) {
        analysis_support = cloud;
    } else {
        recent_analysis_frames_.push_back(*cloud);
        while (recent_analysis_frames_.size() > static_cast<std::size_t>(analysis_frame_window_)) {
            recent_analysis_frames_.pop_front();
        }

        analysis_support.reset(new CloudT());
        std::size_t support_size = 0;
        for (const auto& frame : recent_analysis_frames_) {
            support_size += frame.points.size();
        }
        analysis_support->points.reserve(support_size);
        for (const auto& frame : recent_analysis_frames_) {
            analysis_support->points.insert(analysis_support->points.end(), frame.points.begin(), frame.points.end());
        }
        analysis_support->width = analysis_support->points.size();
        analysis_support->height = 1;
        analysis_support->is_dense = false;
    }
    const auto t_support_done = Clock::now();

    if (debug_texture_stats_) {
        ROS_INFO_THROTTLE(2.0,
                          "Texture analysis support: frame_window=%d buffered_frames=%zu support_points=%zu query_points=%zu",
                          analysis_frame_window_,
                          recent_analysis_frames_.size(),
                          analysis_support->points.size(),
                          cloud->points.size());
    }

    pcl::search::KdTree<PointT> analysis_tree;
    if (!analysis_support->empty()) {
        analysis_tree.setInputCloud(analysis_support);
    }
    std::vector<int> neighbor_indices;
    std::vector<float> neighbor_distances;
    neighbor_indices.reserve(static_cast<std::size_t>(std::max(knn_k_, 1)));
    neighbor_distances.reserve(static_cast<std::size_t>(std::max(knn_k_, 1)));
    const float max_radius = std::max(static_cast<float>(knn_max_radius_), 1e-3f);
    const float max_radius_sq = max_radius * max_radius;
    const float self_match_epsilon_sq = 1e-10f;

    std::size_t total_neighbors = 0;
    std::size_t zeroed_points = 0;
    float edge_score_sum = 0.0f;
    float texture_score_sum = 0.0f;

    std::chrono::steady_clock::duration search_duration = Clock::duration::zero();
    std::chrono::steady_clock::duration score_duration = Clock::duration::zero();

    for (auto& pt : cloud->points) {
        neighbor_indices.clear();
        neighbor_distances.clear();
        int found_neighbors = 0;
        if (!analysis_support->empty()) {
            const auto t_search_begin = Clock::now();
            found_neighbors = analysis_tree.nearestKSearch(pt, knn_k_, neighbor_indices, neighbor_distances);
            search_duration += Clock::now() - t_search_begin;
        }
        std::vector<int> valid_neighbor_indices;
        std::vector<float> valid_neighbor_distances;
        valid_neighbor_indices.reserve(static_cast<std::size_t>(std::max(found_neighbors, 0)));
        valid_neighbor_distances.reserve(static_cast<std::size_t>(std::max(found_neighbors, 0)));
        for (int i = 0; i < found_neighbors; ++i) {
            const std::size_t slot = static_cast<std::size_t>(i);
            const float sq_dist = neighbor_distances[slot];
            if (sq_dist <= self_match_epsilon_sq) {
                continue;
            }
            if (sq_dist > max_radius_sq) {
                continue;
            }
            valid_neighbor_indices.push_back(neighbor_indices[slot]);
            valid_neighbor_distances.push_back(sq_dist);
        }

        total_neighbors += valid_neighbor_indices.size();
        if (valid_neighbor_indices.size() < 3U) {
            pt.intensity = 0.0f;
            pt.curvature = 0.0f;
            pt.color_entropy = 0.0f;
            pt.radius = 0.0f;
            pt.confidence = 0.0f;
            ++zeroed_points;
            continue;
        }

        const auto t_score_begin = Clock::now();
        float mean_r = 0.0f;
        float mean_g = 0.0f;
        float mean_b = 0.0f;
        for (int idx : valid_neighbor_indices) {
            const auto& n = analysis_support->points[static_cast<std::size_t>(idx)];
            mean_r += n.r;
            mean_g += n.g;
            mean_b += n.b;
        }

        const float inv_n = 1.0f / static_cast<float>(valid_neighbor_indices.size());
        mean_r *= inv_n;
        mean_g *= inv_n;
        mean_b *= inv_n;

        float color_energy = 0.0f;
        float mean_distance = 0.0f;
        for (std::size_t i = 0; i < valid_neighbor_indices.size(); ++i) {
            const auto& n = analysis_support->points[
                static_cast<std::size_t>(valid_neighbor_indices[i])];
            const float dr = static_cast<float>(n.r) - mean_r;
            const float dg = static_cast<float>(n.g) - mean_g;
            const float db = static_cast<float>(n.b) - mean_b;
            color_energy += dr * dr + dg * dg + db * db;
            mean_distance += std::sqrt(valid_neighbor_distances[i]);
        }
        color_energy *= inv_n;
        mean_distance *= inv_n;

        const float qdr = static_cast<float>(pt.r) - mean_r;
        const float qdg = static_cast<float>(pt.g) - mean_g;
        const float qdb = static_cast<float>(pt.b) - mean_b;
        const float edge_score = std::min((qdr * qdr + qdg * qdg + qdb * qdb) /
                                              (3.0f * 255.0f * 255.0f),
                                          1.0f);
        const float texture_score = std::min(color_energy / (3.0f * 255.0f * 255.0f), 1.0f);

        pt.color_entropy = texture_score;
        pt.intensity = texture_score;
        pt.confidence = edge_score;
        pt.radius = mean_distance;

        edge_score_sum += edge_score;
        texture_score_sum += texture_score;
        score_duration += Clock::now() - t_score_begin;
    }

    const auto t_end = Clock::now();
    const auto ms = [](const Clock::duration& d) {
        return std::chrono::duration_cast<std::chrono::microseconds>(d).count() / 1000.0;
    };

    if (!cloud->points.empty()) {
        const double point_count = static_cast<double>(cloud->points.size());
        const double mean_neighbors = point_count > 0.0
            ? static_cast<double>(total_neighbors) / point_count
            : 0.0;
        const double zero_ratio = point_count > 0.0
            ? static_cast<double>(zeroed_points) / point_count
            : 0.0;

        if (debug_texture_stats_) {
            ROS_INFO_THROTTLE(2.0,
                              "Texture stats: mean_neighbors=%.2f zeroed_ratio=%.3f edge_mean=%.4f texture_mean=%.4f",
                              mean_neighbors,
                              zero_ratio,
                              edge_score_sum / point_count,
                              texture_score_sum / point_count);
            ROS_INFO_THROTTLE(2.0,
                              "Texture timing: support_ms=%.3f search_ms=%.3f score_ms=%.3f total_ms=%.3f",
                              ms(t_support_done - t_start),
                              ms(search_duration),
                              ms(score_duration),
                              ms(t_end - t_start));
        }

        benchmark_support_ms_ = ms(t_support_done - t_start);
        benchmark_search_ms_ = ms(search_duration);
        benchmark_score_ms_ = ms(score_duration);
        benchmark_total_ms_ = ms(t_end - t_start);
        benchmark_raw_neighbors_mean_ = mean_neighbors;
        benchmark_zeroed_ratio_ = zero_ratio;
        benchmark_edge_mean_ = edge_score_sum / point_count;
        benchmark_texture_mean_ = texture_score_sum / point_count;
    }

    ikdtreeNS::KD_TREE<PointT>::PointVector points_to_add;
    points_to_add.reserve(cloud->points.size());
    std::unordered_set<VoxelKey, VoxelKeyHash> low_texture_voxels;
    const float voxel_size = std::max(static_cast<float>(low_texture_downsample_size_), 1e-3f);

    for (auto pt : cloud->points) {
        pt.r = 255;
        pt.g = 255;
        pt.b = 255;
        pt.a = 255;

        if (pt.color_entropy < static_cast<float>(low_texture_threshold_)) {
            const VoxelKey key{
                static_cast<int>(std::floor(pt.x / voxel_size)),
                static_cast<int>(std::floor(pt.y / voxel_size)),
                static_cast<int>(std::floor(pt.z / voxel_size))};
            if (!low_texture_voxels.insert(key).second) {
                continue;
            }
        }
        points_to_add.push_back(pt);
    }

    if (!points_to_add.empty()) {
        ikd_tree_->Add_Points(points_to_add, ikd_tree_downsample_);
    }
}

bool CloudMapBuilder::publishMap() {
    if (pub_global_map_.getNumSubscribers() == 0) {
        return false;
    }

    if (!cache_valid_ || map_dirty_) {
        ikdtreeNS::KD_TREE<PointT>::PointVector global_points;
        global_points.reserve(ikd_tree_->size());
        ikd_tree_->flatten(ikd_tree_->Root_Node, global_points, ikdtreeNS::NOT_RECORD);

        for (auto& pt : global_points) {
            pt.r = 255;
            pt.g = 255;
            pt.b = 255;
            pt.a = 255;
        }

        cached_cloud_.points.swap(global_points);
        cached_cloud_.width = cached_cloud_.points.size();
        cached_cloud_.height = 1;
        cached_cloud_.is_dense = true;
        cache_valid_ = true;
    }
    if (cached_cloud_.empty()) {
        return false;
    }

    sensor_msgs::PointCloud2 msg_combined;
    pcl::toROSMsg(cached_cloud_, msg_combined);
    msg_combined.header.stamp = ros::Time::now();
    msg_combined.header.frame_id = global_frame_id_;
    pub_global_map_.publish(msg_combined);

    return true;
}

} // namespace active_mapping
