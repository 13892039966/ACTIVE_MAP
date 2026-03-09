#include "active_mapping/surfel_map_builder.h"

namespace active_mapping {

void SurfelMapBuilder::initParams(ros::NodeHandle& nh) {
    nh.param<float>("surfel_radius", surfel_radius_, 0.15f);
    surfel_radius_sq_ = surfel_radius_ * surfel_radius_;
    
    nh.param<float>("depth_tolerance", depth_tolerance_, 0.05f);
    nh.param<int>("init_pca_k", init_pca_k_, 10);
    
    nh.param<double>("w_geom", w_geom_, 0.6);
    nh.param<double>("w_hole", w_hole_, 0.4);

    ROS_INFO("✅ Paradigm B (Surfel Mapping) Started. Radius: %.2f, Depth Tol: %.2f", 
             surfel_radius_, depth_tolerance_);
}

void SurfelMapBuilder::processAndInsertCloud(CloudT::Ptr& cloud) {
    if (ikd_tree_->size() < init_pca_k_) {
        // 初始地图为空，强制计算法向并全量插入作为初始面元
        for (auto& pt : cloud->points) {
            pt.radius = surfel_radius_;
            pt.intensity = 1.0f; // 初始全为未知前线
            pt.confidence = 1.0f;
            // 简化：第一帧没有足够的邻居，暂设法向为朝向相机 (假设 Z 向外)
            pt.normal_x = 0.0f; pt.normal_y = 0.0f; pt.normal_z = 1.0f; 
        }
        ikd_tree_->Add_Points(cloud->points, ikd_tree_downsample_);
        return;
    }

    CloudT::Ptr new_surfels(new CloudT);
    ikdtreeNS::KD_TREE<PointT>::PointVector neighbors;
    std::vector<float> distances;

    for (auto& pt : cloud->points) {
        // 只搜最近的 1 个面元进行匹配验证
        ikd_tree_->Nearest_Search(pt, 1, neighbors, distances);
        
        if (neighbors.empty()) continue;

        const auto& nearest_surfel = neighbors[0];
        float dist_sq = distances[0];

        // 判定 1：覆盖率检查 (Coverage & Holes)
        // 如果当前点距离最近的面元中心超过半径，说明落在“空洞”里
        if (dist_sq > surfel_radius_sq_) {
            pt.radius = surfel_radius_;
            pt.intensity = w_hole_ * 1.0f; // 空洞，高探索价值
            pt.confidence = 1.0f;
            new_surfels->points.push_back(pt);
            continue;
        }

        // 判定 2：法向杂乱度与切平面检查 (Geometric Clutter)
        // 点到面元切平面的垂直距离：d = |(P - S) · N|
        Eigen::Vector3f diff(pt.x - nearest_surfel.x, pt.y - nearest_surfel.y, pt.z - nearest_surfel.z);
        Eigen::Vector3f normal(nearest_surfel.normal_x, nearest_surfel.normal_y, nearest_surfel.normal_z);
        
        float point_to_plane_dist = std::abs(diff.dot(normal));

        if (point_to_plane_dist > depth_tolerance_) {
            // 虽然落在圆盘内，但深度不连续（比如桌子边缘和地面）。
            // 产生结构分裂，生成新面元！
            pt.radius = surfel_radius_;
            pt.intensity = w_geom_ * 1.0f; // 结构复杂区，高探索价值
            pt.confidence = 1.0f;
            new_surfels->points.push_back(pt);
        } else {
            // 🌟 完美落在已有面元上，冗余数据直接丢弃，极大节约内存和算力！
            // (未来可在此处增加 IKD-Tree 现有节点的颜色融合逻辑)
            continue; 
        }
    }

    if (new_surfels->empty()) return;

    // --- 仅对**新生面元**进行 PCA 法向提取 ---
    // 这是核心算力优化的关键：将 O(N) 的 PCA 降维成了 O(N_new)
    for (auto& new_pt : new_surfels->points) {
        ikd_tree_->Nearest_Search(new_pt, init_pca_k_, neighbors, distances);
        if (neighbors.size() < 3) {
            new_pt.normal_x = 0.0f; new_pt.normal_y = 0.0f; new_pt.normal_z = 1.0f;
            continue;
        }

        Eigen::Matrix<float, 3, Eigen::Dynamic> coords(3, neighbors.size());
        for (size_t i = 0; i < neighbors.size(); ++i) {
            coords.col(i) << neighbors[i].x, neighbors[i].y, neighbors[i].z;
        }
        
        Eigen::Vector3f centroid = coords.rowwise().mean();
        coords.colwise() -= centroid;
        Eigen::Matrix3f cov = (coords * coords.transpose()) / float(neighbors.size() - 1);
        
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> pca(cov, Eigen::ComputeEigenvectors);
        Eigen::Vector3f normal = pca.eigenvectors().col(0);
        
        new_pt.normal_x = normal.x();
        new_pt.normal_y = normal.y();
        new_pt.normal_z = normal.z();
    }

    // 批量插入新生面元
    ikd_tree_->Add_Points(new_surfels->points, ikd_tree_downsample_);
}

void SurfelMapBuilder::publishMap() {
    ikdtreeNS::KD_TREE<PointT>::PointVector global_points;
    ikd_tree_->flatten(ikd_tree_->Root_Node, global_points, ikdtreeNS::NOT_RECORD);

    CloudT global_cloud;
    global_cloud.points = global_points;
    global_cloud.width = global_points.size();
    global_cloud.height = 1;
    global_cloud.is_dense = true;

    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(global_cloud, output_msg);
    output_msg.header.stamp = ros::Time::now();
    output_msg.header.frame_id = global_frame_id_;

    pub_global_map_.publish(output_msg);
    
    // 💡 提示：在 RViz 中，可将 PointCloud2 的渲染风格改为 "Spheres" 
    // 并适当调节 Size 来近似观察面元的覆盖情况。
}

} // namespace active_mapping
