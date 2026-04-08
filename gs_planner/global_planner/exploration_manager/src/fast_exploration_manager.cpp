/***
 * @Author: ning-zelin && zl.ning@qq.com
 * @Date: 2024-02-25 15:00:51
 * @LastEditTime: 2024-03-12 22:15:11
 * @Description:
 * @
 * @Copyright (c) 2024 by ning-zelin, All Rights Reserved.
 */

#include <boost/lexical_cast.hpp>
#include <epic_planner/expl_data.h>
#include <epic_planner/fast_exploration_manager.h>
#include <fstream>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <plan_manage/planner_manager.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <visualization_msgs/Marker.h>
using namespace std;
using namespace Eigen;

namespace fast_planner {
// SECTION interfaces for setup and query

FastExplorationManager::FastExplorationManager() {}

FastExplorationManager::~FastExplorationManager() {}

void FastExplorationManager::initialize(
    ros::NodeHandle &nh,
    ::ObservationQualityManager::Ptr oqm,
    FastPlannerManager::Ptr planner_manager) {

  oqm_ = oqm;
  planner_manager_ = planner_manager;

  ed_.reset(new ExplorationData);
  ep_.reset(new ExplorationParam);
  ed_->next_goal_idx_ = -1;

  ep_->a_avg_ = tan(planner_manager_->gcopter_config_->maxTiltAngle) *
                planner_manager_->gcopter_config_->gravAcc;
  ep_->v_max_ = planner_manager_->gcopter_config_->maxVelMag;
  ep_->yaw_v_max_ = planner_manager_->gcopter_config_->yaw_max_vel;
  nh.param("exploration/tsp_dir", ep_->tsp_dir_, string("null"));
  nh.getParam("viewpoint_param/global_viewpoint_num",
              ep_->global_viewpoint_num_);
  nh.getParam("view_graph", ep_->view_graph_);
  nh.getParam("viewpoint_param/local_viewpoint_num", ep_->local_viewpoint_num_);
  nh.getParam("global_planning/w_vdir", ep_->w_vdir_);
  nh.getParam("global_planning/w_yawdir", ep_->w_yawdir_);

  goal_yaw = 0.0;
  goal_pitch = 0.0;

  ROS_INFO("FastExplorationManager initialized with OQM");
}

void FastExplorationManager::goalCallback(
    const geometry_msgs::PoseStampedConstPtr &msg) {
  // 提取四元数
  double roll, pitch;
  tf::Quaternion quat;
  tf::quaternionMsgToTF(msg->pose.orientation, quat);

  // 将四元数转换为Euler角
  tf::Matrix3x3(quat).getRPY(roll, pitch, goal_yaw);
  goal_pitch = pitch;
}

int FastExplorationManager::planGlobalPath(const Eigen::Vector3d &pos,
                                           const Eigen::Vector3d &vel,
                                           float odom_yaw,
                                           float odom_pitch) {
  ros::Time start = ros::Time::now();
  // 获取当前位置
  Eigen::Vector3f odom_pos = pos.cast<float>();
  // ros::Time start_time = planner_manager_->local_data_.start_time_;
  auto global_planner = oqm_->getGlobalPlanner();

  // 获取已经过和未经过的viewpoints（在raycast之前）
  std::vector<PathSegmentWithView> passed_viewpoints, unpassed_viewpoints;
  oqm_->getGlobalPlanner()->getPassedAndUnpassedViewpoints(
      passed_viewpoints, unpassed_viewpoints);
  ROS_INFO("FastExplorationManager: Viewpoints status - %lu passed, %lu unpassed",
           passed_viewpoints.size(), unpassed_viewpoints.size());

  // 批量更新累积的 modified regions 的聚类（多帧点云聚合，减少冗余计算）
  ROS_INFO("FastExplorationManager: Updating %d pending clusters before global planning", (int)oqm_->modified_regions_set.size());

  // 在raycast之前，准备passed_viewpoints数据

  // 可视化passed_viewpoints
  static ros::Publisher passed_viewpoints_pub;
  static bool pub_initialized = false;
  if (!pub_initialized) {
    passed_viewpoints_pub = oqm_->nh_.advertise<visualization_msgs::MarkerArray>("/exploration/passed_viewpoints", 10);
    pub_initialized = true;
  }

  if (global_planner->has_previous_planning_state_ && !passed_viewpoints.empty()) {
    std::vector<Eigen::Vector3f> passed_positions;
    std::vector<Eigen::Quaternionf> passed_orientations;
    passed_positions.reserve(passed_viewpoints.size());
    passed_orientations.reserve(passed_viewpoints.size());

    for (const auto& segment : passed_viewpoints) {
      passed_positions.push_back(segment.target_view.position);
      // 从yaw和pitch构建四元数
      // Pitch already follows "positive = down" convention in viewpoint selection; keep the same sign here.
      Eigen::Quaternionf q = Eigen::AngleAxisf(segment.target_view.yaw, Eigen::Vector3f::UnitZ()) *
                             Eigen::AngleAxisf(segment.target_view.pitch, Eigen::Vector3f::UnitY());
      passed_orientations.push_back(q);
    }

    // 可视化passed_viewpoints
    if (passed_viewpoints_pub.getNumSubscribers() > 0) {
      visualization_msgs::MarkerArray marker_array;

      for (size_t i = 0; i < passed_positions.size(); ++i) {
        const Eigen::Vector3f& pos = passed_positions[i];

        // 创建球体marker显示位置
        visualization_msgs::Marker sphere_marker;
        sphere_marker.header.frame_id = "world";
        sphere_marker.header.stamp = ros::Time::now();
        sphere_marker.ns = "passed_viewpoints";
        sphere_marker.id = i;
        sphere_marker.type = visualization_msgs::Marker::SPHERE;
        sphere_marker.action = visualization_msgs::Marker::ADD;

        // 设置位置
        sphere_marker.pose.position.x = pos.x();
        sphere_marker.pose.position.y = pos.y();
        sphere_marker.pose.position.z = pos.z();
        sphere_marker.pose.orientation.w = 1.0;

        // 设置大小 (半径0.2m)
        sphere_marker.scale.x = 0.4;
        sphere_marker.scale.y = 0.4;
        sphere_marker.scale.z = 0.4;

        // 设置颜色 (绿色表示已通过)
        sphere_marker.color.r = 0.0;
        sphere_marker.color.g = 1.0;
        sphere_marker.color.b = 0.0;
        sphere_marker.color.a = 0.8;

        marker_array.markers.push_back(sphere_marker);

        // 添加文本标签显示索引
        visualization_msgs::Marker text_marker;
        text_marker.header.frame_id = "world";
        text_marker.header.stamp = ros::Time::now();
        text_marker.ns = "passed_viewpoints_text";
        text_marker.id = i;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        text_marker.pose.position.x = pos.x();
        text_marker.pose.position.y = pos.y();
        text_marker.pose.position.z = pos.z() + 0.5;
        text_marker.pose.orientation.w = 1.0;
        text_marker.text = "passed_" + std::to_string(i);
        text_marker.scale.z = 0.2;
        text_marker.color.r = 0.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 0.0;
        text_marker.color.a = 1.0;

        marker_array.markers.push_back(text_marker);
      }

      passed_viewpoints_pub.publish(marker_array);
      ROS_INFO("Published %zu passed viewpoints visualization", passed_positions.size());
    }

    oqm_->raycastPoorlyObservedToOdoms(
        passed_positions,
        passed_orientations,
        &global_planner->previous_target_region_idx_);
  } else {
    oqm_->raycastPoorlyObservedToOdoms();
  }
  // 现在标签状态都不更新了，留在别的地方更新
  oqm_->updatePendingClusters();

  // 调用 OQM 的 GlobalPlanner 进行规划
  bool success = oqm_->getGlobalPlanner()->planGlobalTSPPath(
      odom_pos, odom_yaw, odom_pitch, oqm_->getRegionMap(), unpassed_viewpoints);

  if (!success) {
    planner_manager_->graph_visualizer_->vizTour({}, VizColor::RED, "global");
    return NO_FRONTIER;
  }

  // 获取规划结果
  const auto &path = oqm_->getGlobalPlanner()->getPath();
  const auto &selected_views = oqm_->getGlobalPlanner()->getSelectedViews();

  if (path.empty() || selected_views.empty()) {
    planner_manager_->graph_visualizer_->vizTour({}, VizColor::RED, "global");
    return NO_FRONTIER;
  }

  // 转换 SelectedView 到 ViewPose
  ed_->selected_views_.clear();
  ed_->selected_views_.reserve(selected_views.size());
  for (const auto &sv : selected_views) {
    ed_->selected_views_.emplace_back(
        ViewPose(sv.position, sv.yaw, sv.pitch, sv.score));
  }

  // 更新全局路径
  ed_->global_tour_ = path;

  // 获取到第一个viewpoint的路径（TSP矩阵计算时已记录）
  ed_->path_next_goal_ = oqm_->getGlobalPlanner()->getPathToFirstViewpoint();

  // 防御性检查：确保路径不为空
  if (ed_->path_next_goal_.empty()) {
    ROS_ERROR("FastExplorationManager: path_next_goal_ is empty after successful planning, this should not happen");
    planner_manager_->graph_visualizer_->vizTour({}, VizColor::RED, "global");
    return NO_FRONTIER;
  }

  // 获取完整的A*路径（经过所有viewpoints，TSP顺序）
  std::vector<Eigen::Vector3f> complete_astar_path = oqm_->getGlobalPlanner()->getCompleteViewpointPath();

  // 更新目标节点
  updateGoalNode();

  // 设置目标 yaw (用于局部规划)
  if (!ed_->selected_views_.empty()) {
    planner_manager_->local_data_.end_yaw_ = ed_->selected_views_[0].yaw;
  }

  // 可视化
  planner_manager_->graph_visualizer_->vizTour(ed_->global_tour_, VizColor::RED,
                                               "global");

  // 可视化完整的A*路径（经过所有viewpoints的详细路径）
  planner_manager_->graph_visualizer_->vizTour(complete_astar_path, VizColor::ORANGE,
                                               "complete_astar");

  ros::Time end = ros::Time::now();
  ROS_INFO("FastExplorationManager: planGlobalPath completed in %.3f ms",
           (end - start).toSec() * 1000.0);

  return SUCCEED;
}

void FastExplorationManager::updateGoalNode() {
  if (ed_->selected_views_.empty()) {
    return;
  }

  // 选择第一个视点作为下一个目标
  ed_->next_goal_idx_ = 0;
  ed_->next_goal_ = ed_->selected_views_[0];
  ed_->next_goal_.yaw = ed_->next_goal_.yaw;

  // 更新目标 yaw 和 pitch
  goal_yaw = ed_->next_goal_.yaw;
  goal_pitch = ed_->next_goal_.pitch;

  ROS_DEBUG("FastExplorationManager: Next goal set to [%.2f, %.2f, %.2f], "
            "yaw=%.2f, pitch=%.2f",
            ed_->next_goal_.position.x(), ed_->next_goal_.position.y(),
            ed_->next_goal_.position.z(), goal_yaw, goal_pitch);
}

} // namespace fast_planner
