#include <observation_quality_manager/global_planner.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/ColorRGBA.h>

void GlobalPlanner::publishVisualization() {
  visualization_msgs::MarkerArray marker_array;

  // 先清除所有旧的 marker 避免累帧
  visualization_msgs::Marker delete_all;
  delete_all.action = visualization_msgs::Marker::DELETEALL;
  marker_array.markers.push_back(delete_all);

  // Marker 1: cluster 中心点
  visualization_msgs::Marker points_marker;
  points_marker.header.frame_id = "world";
  points_marker.header.stamp = ros::Time::now();
  points_marker.ns = "global_tsp_clusters";
  points_marker.id = 1;
  points_marker.type = visualization_msgs::Marker::SPHERE_LIST;
  points_marker.action = visualization_msgs::Marker::ADD;
  points_marker.scale.x = 0.3;
  points_marker.scale.y = 0.3;
  points_marker.scale.z = 0.3;
  points_marker.color.r = 0.0;
  points_marker.color.g = 1.0;
  points_marker.color.b = 0.0;
  points_marker.color.a = 1.0;

  for (const auto& cluster : selected_clusters_) {
    geometry_msgs::Point p;
    p.x = cluster.position.x();
    p.y = cluster.position.y();
    p.z = cluster.position.z();
    points_marker.points.push_back(p);
  }

  marker_array.markers.push_back(points_marker);

  // Marker 2: 精选视点（蓝色小球）
  if (!selected_views_.empty()) {
    visualization_msgs::Marker viewpoints_marker;
    viewpoints_marker.header.frame_id = "world";
    viewpoints_marker.header.stamp = ros::Time::now();
    viewpoints_marker.ns = "global_tsp_viewpoints";
    viewpoints_marker.id = 2;
    viewpoints_marker.type = visualization_msgs::Marker::SPHERE_LIST;
    viewpoints_marker.action = visualization_msgs::Marker::ADD;
    viewpoints_marker.scale.x = 0.2;
    viewpoints_marker.scale.y = 0.2;
    viewpoints_marker.scale.z = 0.2;
    viewpoints_marker.color.r = 0.0;
    viewpoints_marker.color.g = 0.5;
    viewpoints_marker.color.b = 1.0;  // 蓝色
    viewpoints_marker.color.a = 1.0;

    for (const auto& view : selected_views_) {
      geometry_msgs::Point p;
      p.x = view.position.x();
      p.y = view.position.y();
      p.z = view.position.z();
      viewpoints_marker.points.push_back(p);
    }

    marker_array.markers.push_back(viewpoints_marker);

    // Marker 3: 视点朝向箭头
    const float arrow_length = 0.5f;  // 箭头长度
    int arrow_id = 100;
    for (const auto& view : selected_views_) {
      visualization_msgs::Marker arrow_marker;
      arrow_marker.header.frame_id = "world";
      arrow_marker.header.stamp = ros::Time::now();
      arrow_marker.ns = "global_tsp_viewpoint_arrows";
      arrow_marker.id = arrow_id++;
      arrow_marker.type = visualization_msgs::Marker::ARROW;
      arrow_marker.action = visualization_msgs::Marker::ADD;

      // 箭头尺寸: shaft_diameter, head_diameter, head_length (if points used)
      arrow_marker.scale.x = 0.05;  // shaft diameter
      arrow_marker.scale.y = 0.1;   // head diameter
      arrow_marker.scale.z = 0.1;   // head length

      // 颜色：青色
      arrow_marker.color.r = 0.0;
      arrow_marker.color.g = 1.0;
      arrow_marker.color.b = 1.0;
      arrow_marker.color.a = 1.0;

      // 计算方向向量 (从 yaw, pitch)
      // ROS Coordinate: Z is Up.
      // Positive Pitch = Down (Nose Down).
      // So when Pitch > 0, Z component should be negative.
      // z = -sin(pitch)
      float cos_pitch = std::cos(view.pitch);
      float sin_pitch = std::sin(view.pitch);
      float cos_yaw = std::cos(view.yaw);
      float sin_yaw = std::sin(view.yaw);

      Eigen::Vector3f direction(cos_pitch * cos_yaw,
                                 cos_pitch * sin_yaw,
                                 -sin_pitch);

      // 箭头起点和终点
      geometry_msgs::Point start_point, end_point;
      start_point.x = view.position.x();
      start_point.y = view.position.y();
      start_point.z = view.position.z();
      end_point.x = view.position.x() + direction.x() * arrow_length;
      end_point.y = view.position.y() + direction.y() * arrow_length;
      end_point.z = view.position.z() + direction.z() * arrow_length;

      arrow_marker.points.push_back(start_point);
      arrow_marker.points.push_back(end_point);

      marker_array.markers.push_back(arrow_marker);
    }
  }

  // 发布到独立的话题
  viewpoint_vis_pub_.publish(marker_array);
}

void GlobalPlanner::publishViewpointVisibilityLines(
    const std::vector<SelectedView>& selected_views,
    const std::vector<Eigen::Vector3i>& target_voxels,
    const VisibilityCSR& csr_result,
    bool clear_history) {
  if (viewpoint_visibility_lines_pub_.getNumSubscribers() == 0) {
    return;
  }

  // 静态计数器：用于为每次调用分配唯一的 marker id
  static int lines_marker_id = 50;
  static int text_id_offset = 0;

  visualization_msgs::MarkerArray marker_array;

  // 根据 clear_history 参数决定是否清空历史
  if (clear_history) {
    visualization_msgs::Marker delete_marker;
    delete_marker.header.frame_id = "world";
    delete_marker.header.stamp = ros::Time::now();
    delete_marker.ns = "viewpoint_visible_targets";
    delete_marker.id = 50;
    delete_marker.action = visualization_msgs::Marker::DELETEALL;
    marker_array.markers.push_back(delete_marker);

    // 清空文本标记 (DELETEALL 删除该 namespace 下所有标记)
    visualization_msgs::Marker delete_text;
    delete_text.header.frame_id = "world";
    delete_text.header.stamp = ros::Time::now();
    delete_text.ns = "viewpoint_visible_count";
    delete_text.action = visualization_msgs::Marker::DELETEALL;
    marker_array.markers.push_back(delete_text);

    // 清空历史时重置计数器
    lines_marker_id = 50;
    text_id_offset = 0;
  }

  visualization_msgs::Marker lines_marker;
  lines_marker.header.frame_id = "world";
  lines_marker.header.stamp = ros::Time::now();
  lines_marker.ns = "viewpoint_visible_targets";
  lines_marker.id = lines_marker_id++;  // 使用计数器，每次递增
  lines_marker.type = visualization_msgs::Marker::LINE_LIST;
  lines_marker.action = visualization_msgs::Marker::ADD;
  lines_marker.scale.x = 0.02;  // 线宽
  lines_marker.color.r = 1.0;
  lines_marker.color.g = 0.85;
  lines_marker.color.b = 0.0;
  lines_marker.color.a = 0.5;

  std::vector<Eigen::Vector3f> target_positions;
  target_positions.reserve(target_voxels.size());
  for (const auto& idx : target_voxels) {
    target_positions.emplace_back(voxelIdxToPosition(idx));
  }

  bool has_lines = false;

  // 用于存储每个 viewpoint 的可见数量和位置
  std::vector<std::pair<Eigen::Vector3f, int>> viewpoint_counts;

  for (size_t view_idx = 0; view_idx < selected_views.size(); ++view_idx) {
    // 使用原始 viewpoint_idx 来访问 CSR (原始 GPU CSR 包含所有采样视点的可见性)
    int csr_idx = selected_views[view_idx].viewpoint_idx;

    if (csr_idx < 0 || csr_idx >= static_cast<int>(csr_result.viewpoint_offsets.size()) - 1) {
      continue;
    }

    int start = csr_result.viewpoint_offsets[csr_idx];
    int end = csr_result.viewpoint_offsets[csr_idx + 1];
    if (start == end) {
      continue;
    }

    geometry_msgs::Point vp_point;
    vp_point.x = selected_views[view_idx].position.x();
    vp_point.y = selected_views[view_idx].position.y();
    vp_point.z = selected_views[view_idx].position.z();

    int visible_count = 0;
    for (int offset = start; offset < end; ++offset) {
      int target_idx = csr_result.viewpoint_to_targets[offset];
      if (target_idx < 0 || target_idx >= static_cast<int>(target_positions.size())) {
        continue;
      }

      geometry_msgs::Point target_point;
      const Eigen::Vector3f& target = target_positions[target_idx];
      target_point.x = target.x();
      target_point.y = target.y();
      target_point.z = target.z();

      lines_marker.points.push_back(vp_point);
      lines_marker.points.push_back(target_point);
      has_lines = true;
      visible_count++;
    }

    // 只记录有可见目标的 viewpoint
    if (visible_count > 0) {
      viewpoint_counts.push_back({selected_views[view_idx].position, visible_count});
    }
  }

  // 只有当有可见线段时才添加 lines_marker（LINE_LIST 不能为空）
  if (has_lines) {
    marker_array.markers.push_back(lines_marker);
  }

  // 添加文本标记，显示每个 viewpoint 可见的 target 数量
  // 使用 text_id_offset 避免不同调用之间的 id 冲突
  for (size_t i = 0; i < viewpoint_counts.size(); ++i) {
    visualization_msgs::Marker text_marker;
    text_marker.header.frame_id = "world";
    text_marker.header.stamp = ros::Time::now();
    text_marker.ns = "viewpoint_visible_count";
    text_marker.id = text_id_offset + i;  // 使用偏移量避免覆盖
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;

    // 文本位置在 viewpoint 上方 0.3m
    text_marker.pose.position.x = viewpoint_counts[i].first.x();
    text_marker.pose.position.y = viewpoint_counts[i].first.y();
    text_marker.pose.position.z = viewpoint_counts[i].first.z() + 0.3f;
    text_marker.pose.orientation.w = 1.0;

    // 显示可见数量
    text_marker.text = std::to_string(viewpoint_counts[i].second);

    // 文本样式
    text_marker.scale.z = 0.15;  // 字体高度
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.color.a = 1.0;

    marker_array.markers.push_back(text_marker);
  }

  // 更新文本偏移量，为下次调用做准备
  text_id_offset += viewpoint_counts.size();

  viewpoint_visibility_lines_pub_.publish(marker_array);
}

void GlobalPlanner::visualizeCompleteViewpointPath() {
  if (complete_viewpoint_path_.empty()) {
    return;
  }

  visualization_msgs::MarkerArray marker_array;

  // Create sphere markers for each path point
  visualization_msgs::Marker sphere_marker;
  sphere_marker.header.frame_id = "world";
  sphere_marker.header.stamp = ros::Time::now();
  sphere_marker.ns = "complete_viewpoint_path_points";
  sphere_marker.id = 0;
  sphere_marker.type = visualization_msgs::Marker::SPHERE_LIST;
  sphere_marker.action = visualization_msgs::Marker::ADD;

  // Set sphere size (large for good visibility)
  sphere_marker.scale.x = 0.2;
  sphere_marker.scale.y = 0.2;
  sphere_marker.scale.z = 0.2;

  // Set color (orange to distinguish from other markers)
  sphere_marker.color.r = 1.0;
  sphere_marker.color.g = 0.5;
  sphere_marker.color.b = 0.0;
  sphere_marker.color.a = 0.8;

  // Add all path points
  for (const auto& pos : complete_viewpoint_path_) {
    geometry_msgs::Point p;
    p.x = pos.x();
    p.y = pos.y();
    p.z = pos.z();
    sphere_marker.points.push_back(p);
  }

  marker_array.markers.push_back(sphere_marker);
  path_vis_pub_.publish(marker_array);
}

void GlobalPlanner::visualizeGlobalPath(const std::vector<int>& cluster_indices,
                                         const Eigen::Vector3f& start_pos) {
  // 如果提供了 cluster_indices，则可视化cluster顺序
  if (!cluster_indices.empty()) {
    visualization_msgs::MarkerArray marker_array;

    // 创建红色折线连接cluster中心
    visualization_msgs::Marker line_marker;
    line_marker.header.frame_id = "world";
    line_marker.header.stamp = ros::Time::now();
    line_marker.ns = "cluster_order";
    line_marker.id = 0;
    line_marker.type = visualization_msgs::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::Marker::ADD;
    line_marker.scale.x = 0.15;  // 线宽
    line_marker.color.r = 1.0;
    line_marker.color.g = 0.0;
    line_marker.color.b = 0.0;
    line_marker.color.a = 1.0;

    // 添加起始位置
    geometry_msgs::Point start_point;
    start_point.x = start_pos.x();
    start_point.y = start_pos.y();
    start_point.z = start_pos.z();
    line_marker.points.push_back(start_point);

    // 按顺序添加每个cluster的中心
    for (int cluster_idx : cluster_indices) {
      const ClusterInfo& cluster = selected_clusters_[cluster_idx];
      geometry_msgs::Point p;
      p.x = cluster.position.x();
      p.y = cluster.position.y();
      p.z = cluster.position.z();
      line_marker.points.push_back(p);
    }

    marker_array.markers.push_back(line_marker);
    path_vis_pub_.publish(marker_array);
    return;
  }

  // 否则，可视化全局TSP路径（原有逻辑）
  if (global_path_.empty()) {
    return;
  }

  visualization_msgs::MarkerArray marker_array;

  // 可视化全局TSP路径（红色折线）
  visualization_msgs::Marker line_marker;
  line_marker.header.frame_id = "world";
  line_marker.header.stamp = ros::Time::now();
  line_marker.ns = "global_tsp_path";
  line_marker.id = 0;
  line_marker.type = visualization_msgs::Marker::LINE_STRIP;
  line_marker.action = visualization_msgs::Marker::ADD;
  line_marker.scale.x = 0.1;  // 线宽
  line_marker.color.r = 1.0;
  line_marker.color.g = 0.0;
  line_marker.color.b = 0.0;
  line_marker.color.a = 1.0;

  for (const auto& pos : global_path_) {
    geometry_msgs::Point p;
    p.x = pos.x();
    p.y = pos.y();
    p.z = pos.z();
    line_marker.points.push_back(p);
  }

  marker_array.markers.push_back(line_marker);
  path_vis_pub_.publish(marker_array);
}

void GlobalPlanner::visualizeClusterReachability(
    const std::vector<ClusterInfo>& clusters,
    const std::vector<bool>& reachable) {

  visualization_msgs::MarkerArray marker_array;

  // 先清除所有旧的 marker
  visualization_msgs::Marker delete_all;
  delete_all.action = visualization_msgs::Marker::DELETEALL;
  delete_all.ns = "cluster_reachability";
  marker_array.markers.push_back(delete_all);

  // 为每个 cluster 创建一个球体 marker
  for (size_t i = 0; i < clusters.size(); ++i) {
    visualization_msgs::Marker sphere;
    sphere.header.frame_id = "world";
    sphere.header.stamp = ros::Time::now();
    sphere.ns = "cluster_reachability";
    sphere.id = static_cast<int>(i);
    sphere.type = visualization_msgs::Marker::SPHERE;
    sphere.action = visualization_msgs::Marker::ADD;
    sphere.pose.position.x = clusters[i].position.x();
    sphere.pose.position.y = clusters[i].position.y();
    sphere.pose.position.z = clusters[i].position.z();
    sphere.pose.orientation.w = 1.0;
    sphere.scale.x = 0.5;  // 球体直径
    sphere.scale.y = 0.5;
    sphere.scale.z = 0.5;

    if (reachable[i]) {
      // 绿色 = 可达
      sphere.color.r = 0.0;
      sphere.color.g = 1.0;
      sphere.color.b = 0.0;
    } else {
      // 红色 = 不可达
      sphere.color.r = 1.0;
      sphere.color.g = 0.0;
      sphere.color.b = 0.0;
    }
    sphere.color.a = 0.8;

    marker_array.markers.push_back(sphere);

    // 添加文本标签显示 cluster 索引
    visualization_msgs::Marker text;
    text.header.frame_id = "world";
    text.header.stamp = ros::Time::now();
    text.ns = "cluster_reachability_text";
    text.id = static_cast<int>(i);
    text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::Marker::ADD;
    text.pose.position.x = clusters[i].position.x();
    text.pose.position.y = clusters[i].position.y();
    text.pose.position.z = clusters[i].position.z() + 0.4;  // 稍微上移
    text.scale.z = 0.3;  // 文字大小
    text.color.r = 1.0;
    text.color.g = 1.0;
    text.color.b = 1.0;
    text.color.a = 1.0;
    text.text = std::to_string(i) + (reachable[i] ? "" : " X");

    marker_array.markers.push_back(text);
  }

  cluster_reachability_pub_.publish(marker_array);
  ROS_INFO("Published cluster reachability visualization: %zu clusters", clusters.size());
}

void GlobalPlanner::visualizeAABB(const Eigen::Vector3i& aabb_min,
                                   const Eigen::Vector3i& aabb_max,
                                   const std::string& ns,
                                   float r, float g, float b) {
  if (target_aabb_pub_.getNumSubscribers() == 0) {
    return;
  }

  // 将体素索引转换为世界坐标 (AABB的外边界，需要加1体素大小)
  Eigen::Vector3f min_pos = voxelIdxToPosition(aabb_min);
  Eigen::Vector3f max_pos = voxelIdxToPosition(aabb_max + Eigen::Vector3i(1, 1, 1));

  // 计算中心点和尺寸
  Eigen::Vector3f center = (min_pos + max_pos) * 0.5f;
  Eigen::Vector3f size = max_pos - min_pos;

  visualization_msgs::Marker box_marker;
  box_marker.header.frame_id = "world";
  box_marker.header.stamp = ros::Time::now();
  box_marker.ns = ns;
  box_marker.id = 0;
  box_marker.type = visualization_msgs::Marker::CUBE;
  box_marker.action = visualization_msgs::Marker::ADD;

  box_marker.pose.position.x = center.x();
  box_marker.pose.position.y = center.y();
  box_marker.pose.position.z = center.z();
  box_marker.pose.orientation.w = 1.0;

  box_marker.scale.x = size.x();
  box_marker.scale.y = size.y();
  box_marker.scale.z = size.z();

  box_marker.color.r = r;
  box_marker.color.g = g;
  box_marker.color.b = b;
  box_marker.color.a = 0.25f;  // 半透明

  target_aabb_pub_.publish(box_marker);
  ROS_DEBUG("Published AABB visualization: min [%.2f, %.2f, %.2f], max [%.2f, %.2f, %.2f], size [%.2f, %.2f, %.2f]",
            min_pos.x(), min_pos.y(), min_pos.z(),
            max_pos.x(), max_pos.y(), max_pos.z(),
            size.x(), size.y(), size.z());
}
