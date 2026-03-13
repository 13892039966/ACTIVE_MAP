#include "active_mapping/cloud_map_builder.h"
#include <algorithm>

namespace active_mapping {

void CloudMapBuilder::initParams(ros::NodeHandle& nh) {
    nh.param<int>("knn_k", knn_k_, 15);
    nh.param<double>("w_geom", w_geom_, 0.4);         
    nh.param<double>("w_color", w_color_, 0.3);       
    nh.param<double>("w_sparse", w_sparse_, 0.3);
    nh.param<bool>("compute_normals", compute_normals_, false);
    nh.param<bool>("publish_geom_map", publish_geom_map_, true);
    nh.param<bool>("publish_color_map", publish_color_map_, true);
    
    if (publish_geom_map_) {
        pub_geom_map_ = nh.advertise<sensor_msgs::PointCloud2>("/active_mapping/geom_entropy_map", 1);
    }
    if (publish_color_map_) {
        pub_color_map_ = nh.advertise<sensor_msgs::PointCloud2>("/active_mapping/color_entropy_map", 1);
    }

    ROS_INFO("✅ Paradigm A Started. 3 Maps will be published: Combined, Geometric, Color.");
}

void CloudMapBuilder::processAndInsertCloud(CloudT::Ptr& cloud) {
    if (ikd_tree_->size() < knn_k_) {
        for (auto& pt : cloud->points) {
            pt.intensity = 1.0f; 
            pt.curvature = 0.0f;
            pt.color_entropy = 0.0f; // 🌟 赋值
        }
        ikd_tree_->Add_Points(cloud->points, ikd_tree_downsample_);
        return;
    }

    ikdtreeNS::KD_TREE<PointT>::PointVector neighbors;
    std::vector<float> distances;
    neighbors.reserve(static_cast<std::size_t>(knn_k_));
    distances.reserve(static_cast<std::size_t>(knn_k_));

    for (auto& pt : cloud->points) {
        ikd_tree_->Nearest_Search(pt, knn_k_, neighbors, distances);

        const std::size_t neighbor_count = neighbors.size();
        const float sparsity = (neighbor_count < 3U)
            ? 1.0f
            : 1.0f - (static_cast<float>(neighbor_count) / static_cast<float>(knn_k_));

        if (neighbor_count < 3U) {
            pt.intensity = w_sparse_ * sparsity; 
            pt.curvature = 0.0f;
            pt.color_entropy = 0.0f; // 🌟 赋值
            continue;
        }

        float mean_x = 0.0f;
        float mean_y = 0.0f;
        float mean_z = 0.0f;
        float sum_gray = 0.0f;
        float sum_gray_sq = 0.0f;
        for (const auto& n : neighbors) {
            mean_x += n.x;
            mean_y += n.y;
            mean_z += n.z;
            const float gray = 0.299f * n.r + 0.587f * n.g + 0.114f * n.b;
            sum_gray += gray;
            sum_gray_sq += gray * gray;
        }
        const float inv_n = 1.0f / static_cast<float>(neighbor_count);
        mean_x *= inv_n;
        mean_y *= inv_n;
        mean_z *= inv_n;
        const float mean_gray = sum_gray * inv_n;

        float c00 = 0.0f;
        float c01 = 0.0f;
        float c02 = 0.0f;
        float c11 = 0.0f;
        float c12 = 0.0f;
        float c22 = 0.0f;
        for (const auto& n : neighbors) {
            const float dx = n.x - mean_x;
            const float dy = n.y - mean_y;
            const float dz = n.z - mean_z;
            c00 += dx * dx;
            c01 += dx * dy;
            c02 += dx * dz;
            c11 += dy * dy;
            c12 += dy * dz;
            c22 += dz * dz;
        }
        const float inv_n_minus_1 = 1.0f / static_cast<float>(neighbor_count - 1U);
        Eigen::Matrix3f cov;
        cov(0, 0) = c00 * inv_n_minus_1;
        cov(0, 1) = c01 * inv_n_minus_1;
        cov(0, 2) = c02 * inv_n_minus_1;
        cov(1, 0) = cov(0, 1);
        cov(1, 1) = c11 * inv_n_minus_1;
        cov(1, 2) = c12 * inv_n_minus_1;
        cov(2, 0) = cov(0, 2);
        cov(2, 1) = cov(1, 2);
        cov(2, 2) = c22 * inv_n_minus_1;

        float geom_entropy = 0.0f;
        int pca_options = compute_normals_ ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> pca(cov, pca_options);
        
        Eigen::Vector3f evals = pca.eigenvalues(); 
        float sum_evals = evals(0) + evals(1) + evals(2);
        
        if (sum_evals > 1e-6) geom_entropy = evals(0) / sum_evals; 
        pt.curvature = geom_entropy; 

        if (compute_normals_) {
            Eigen::Vector3f normal = pca.eigenvectors().col(0);
            pt.normal_x = normal.x(); pt.normal_y = normal.y(); pt.normal_z = normal.z();
        }

        // 计算色彩熵
        float color_variance = (sum_gray_sq * inv_n) - (mean_gray * mean_gray);
        color_variance = std::max(color_variance, 0.0f);
        float color_entropy = std::min(color_variance / 3600.0f, 1.0f); 
        pt.color_entropy = color_entropy; // 🌟 保存至字段

        // 加权总分
        pt.intensity = w_geom_ * geom_entropy + w_color_ * color_entropy + w_sparse_ * sparsity;
    }

    ikd_tree_->Add_Points(cloud->points, ikd_tree_downsample_);
}

void CloudMapBuilder::publishMap() {
    const bool need_geom = publish_geom_map_ && pub_geom_map_.getNumSubscribers() > 0;
    const bool need_color = publish_color_map_ && pub_color_map_.getNumSubscribers() > 0;

    // 性能优化：如果没有人订阅任何话题，直接跳过展平动作
    if (pub_global_map_.getNumSubscribers() == 0 &&
        !need_geom &&
        !need_color) {
        return;
    }

    // 性能优化：地图无变化时复用缓存，避免重复 flatten 整棵树。
    if (!cache_valid_ || map_dirty_) {
        ikdtreeNS::KD_TREE<PointT>::PointVector global_points;
        global_points.reserve(ikd_tree_->size());
        ikd_tree_->flatten(ikd_tree_->Root_Node, global_points, ikdtreeNS::NOT_RECORD);

        cached_cloud_.points.swap(global_points);
        cached_cloud_.width = cached_cloud_.points.size();
        cached_cloud_.height = 1;
        cached_cloud_.is_dense = true;
        cache_valid_ = true;
    }
    if (cached_cloud_.empty()) {
        return;
    }

    // ==========================================
    // 地图 1：综合加权热力图 (默认)
    // ==========================================
    if (pub_global_map_.getNumSubscribers() > 0) {
        sensor_msgs::PointCloud2 msg_combined;
        pcl::toROSMsg(cached_cloud_, msg_combined);
        msg_combined.header.stamp = ros::Time::now();
        msg_combined.header.frame_id = global_frame_id_;
        pub_global_map_.publish(msg_combined);
    }

    // ==========================================
    // 地图 2：纯几何熵热力图 (动态替换 intensity)
    // ==========================================
    if (need_geom) {
        CloudT geom_cloud = cached_cloud_; // 拷贝一份
        for (auto& pt : geom_cloud.points) {
            pt.intensity = pt.curvature; // 🌟 狸猫换太子
        }
        sensor_msgs::PointCloud2 msg_geom;
        pcl::toROSMsg(geom_cloud, msg_geom);
        msg_geom.header.stamp = ros::Time::now();
        msg_geom.header.frame_id = global_frame_id_;
        pub_geom_map_.publish(msg_geom);
    }

    // ==========================================
    // 地图 3：纯色彩熵热力图 (动态替换 intensity)
    // ==========================================
    if (need_color) {
        CloudT color_cloud = cached_cloud_; // 拷贝一份
        for (auto& pt : color_cloud.points) {
            pt.intensity = pt.color_entropy; // 🌟 狸猫换太子
        }
        sensor_msgs::PointCloud2 msg_color;
        pcl::toROSMsg(color_cloud, msg_color);
        msg_color.header.stamp = ros::Time::now();
        msg_color.header.frame_id = global_frame_id_;
        pub_color_map_.publish(msg_color);
    }
}

} // namespace active_mapping
