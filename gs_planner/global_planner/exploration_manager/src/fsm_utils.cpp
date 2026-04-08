#include <epic_planner/expl_data.h>
#include <epic_planner/fast_exploration_fsm.h>
#include <plan_manage/planner_manager.h>
#include <Eigen/Geometry>
#include <cmath>
#include <observation_quality_manager/observation_quality_manager.h>
#include <observation_quality_manager/global_planner.h>  // For PathSegmentWithView
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>  // For downsampling
#include <tf/transform_datatypes.h>
#include <unordered_set>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <sstream>
#include <ctime>  // For timestamp generation

void FastExplorationFSM::pubState() {

  std_msgs::Empty heartbeat_msg;
  heartbeat_pub_.publish(heartbeat_msg);
  std_msgs::Bool msg;
  msg.data = fd_->static_state_;
  static_pub_.publish(msg);
  Marker state_marker;
  state_marker.type = Marker::TEXT_VIEW_FACING;
  state_marker.pose.position.x = fd_->odom_pos_.x();
  state_marker.pose.position.y = fd_->odom_pos_.y();
  state_marker.pose.position.z = fd_->odom_pos_.z();
  state_marker.pose.orientation.w = 1.0;
  state_marker.scale.x = state_marker.scale.y = state_marker.scale.z = 0.5;
  state_marker.action = Marker::ADD;
  state_marker.color.r = 1.0;
  state_marker.color.a = 1.0;
  state_marker.text = fd_->state_str_[int(state_)];
  state_marker.header.frame_id = "world";
  state_marker.header.stamp = ros::Time::now();

  state_pub_.publish(state_marker);
}

int FastExplorationFSM::callExplorationPlanner() {
  // if (planner_manager_->lidar_map_interface_->getDisToOcc(fd_->odom_pos_) < planner_manager_->gcopter_config_->dilateRadiusHard)
  //   return START_FAIL;
  if (planner_manager_->topo_graph_->odom_node_->neighbors_.empty())
    return START_FAIL;
  if (expl_manager_->ed_->global_tour_.size() < 2)
    return NO_FRONTIER;

  // debug: 检查下一个目标是否在障碍物中
  Eigen::Vector3f next_goal_pos = expl_manager_->ed_->next_goal_.position;
  // if (planner_manager_->lidar_map_interface_->getDisToOcc(next_goal_pos) <
  //     planner_manager_->topo_graph_->bubble_min_radius_) { // TODO:
  //   cout << "410:  next goal in occ, update it" << endl;
  //   updateTopoAndGlobalPath();
  //   return FAIL;
  // }
  // 直接使用 planGlobalPath 中记录的 A* 路径（TSP矩阵计算时已记录）
  // expl_manager_->planGlobalPath(fd_->odom_pos_.cast<double>(), fd_->odom_vel_.cast<double>(), fd_->odom_yaw_, fd_->odom_pitch_);
  vector<Eigen::Vector3f> path_next_goal = expl_manager_->ed_->path_next_goal_;

  if (path_next_goal.empty()) {
    ROS_ERROR("ExplorationPlanner: path_next_goal_ is empty, planGlobalPath may have failed");
    return FAIL;
  }

  auto info = &planner_manager_->local_data_;

  if (!fd_->static_state_) {
    double plan_finish_time_exp = (ros::Time::now() - info->start_time_).toSec() + fp_->replan_time_;
    if (plan_finish_time_exp > info->duration_) {
      plan_finish_time_exp = info->duration_;
    }
    Eigen::Vector3d start_exp = info->minco_traj_.getPos(plan_finish_time_exp);
    path_next_goal.insert(path_next_goal.begin(), start_exp.cast<float>());
  }
  vector<Eigen::Vector3f> path_next_goal_tmp;
  path_next_goal_tmp.push_back(path_next_goal[0]);

  for (size_t i = 1; i < path_next_goal.size();) {
    Eigen::Vector3f end_pt = path_next_goal_tmp.back();
    if ((path_next_goal[i] - end_pt).norm() > 1.0) {
      Eigen::Vector3f dir = (path_next_goal[i] - end_pt).normalized();
      path_next_goal_tmp.push_back(end_pt + 1.0 * dir);
    } else if ((path_next_goal[i] - end_pt).norm() < 0.01) {
      i++;
    } else {
      path_next_goal_tmp.push_back(path_next_goal[i]);
      i++;
    }
  }
  expl_manager_->ed_->path_next_goal_.swap(path_next_goal_tmp);

  // ==========================================
  // Multi-viewpoint trajectory planning
  // ==========================================

  // Get path segments from GlobalPlanner
  auto global_planner = expl_manager_->oqm_->getGlobalPlanner();
  const auto& path_segments = global_planner->getPathSegmentsWithViews();

  int plan_result;

  if (path_segments.empty()) {
    ROS_ERROR("callExplorationPlanner: path_segments is empty!");
    return FAIL;
  }

  // ==========================================
  // 重规划第一段路径：补偿全局规划到轨迹优化的延迟
  // ==========================================
  Eigen::Vector3f current_start_pos;
  if (fd_->static_state_) {
    // 静态状态：使用当前 odom 位置
    current_start_pos = fd_->odom_pos_;
  } else {
    // 运动状态：使用轨迹预测的位置
    double plan_finish_time = (ros::Time::now() - info->start_time_).toSec() ;
    if (plan_finish_time > info->duration_) {
      plan_finish_time = info->duration_;
    }
    current_start_pos = info->minco_traj_.getPos(plan_finish_time).cast<float>();
  }

  // 调用 GlobalPlanner 重规划第一段路径
  global_planner->replanFirstSegment(current_start_pos);

  ROS_INFO("Planning trajectory through %lu viewpoint(s)", path_segments.size());
  plan_result = planner_manager_->planLongExploreTraj(path_segments, expl_manager_->oqm_, fd_->static_state_);

  // ==========================================
  // Trajectory publishing (unchanged)
  // ==========================================

  if (plan_result == fast_planner::PLAN_SUCCEED) {
    // 更新previous_path_segments_with_views_中每个viewpoint的到达时间
    const auto& arrival_offsets = planner_manager_->getViewpointArrivalTimes();
    ros::Time start_time = planner_manager_->local_data_.start_time_;
    global_planner->updatePreviousPathSegmentArrivalTimes(start_time, arrival_offsets);

    traj_utils::PolyTraj poly_traj_msg;
    planner_manager_->polyTraj2ROSMsg(poly_traj_msg, info->start_time_);
    fd_->newest_traj_ = poly_traj_msg;
    traj_utils::PolyTraj poly_yaw_traj_msg;
    planner_manager_->polyYawTraj2ROSMsg(poly_yaw_traj_msg, info->start_time_);
    fd_->newest_yaw_traj_ = poly_yaw_traj_msg;
    traj_utils::PolyTraj poly_pitch_traj_msg;
    planner_manager_->polyPitchTraj2ROSMsg(poly_pitch_traj_msg, info->start_time_);
    fd_->newest_pitch_traj_ = poly_pitch_traj_msg;
    return SUCCEED;
  } else {
    return FAIL;
  }
}

void FastExplorationFSM::triggerCallback(const nav_msgs::PathConstPtr &msg) {
  if (msg->poses[0].pose.position.z < -0.1)
    return;

  if (state_ != WAIT_TRIGGER)
    return;
  fd_->trigger_ = true;
  cout << "Triggered!" << endl;
  total_time_ = ros::Time::now().toSec();
  transitState(PLAN_TRAJ, "triggerCallback");
}

void FastExplorationFSM::odometryCallback(
    const nav_msgs::OdometryConstPtr &msg) {
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    latest_odom_msg_ = msg;
  }
  fd_->odom_pos_ = Eigen::Vector3f(msg->pose.pose.position.x,
                                   msg->pose.pose.position.y,
                                   msg->pose.pose.position.z);
  fd_->odom_vel_ = Eigen::Vector3f(msg->twist.twist.linear.x,
                                   msg->twist.twist.linear.y,
                                   msg->twist.twist.linear.z);

  tf::Quaternion quat;
  tf::quaternionMsgToTF(msg->pose.pose.orientation, quat);
  double roll, pitch, yaw;
  tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
  fd_->odom_yaw_ = static_cast<float>(yaw);
  fd_->odom_pitch_ = static_cast<float>(pitch);

  fd_->have_odom_ = true;
}

void FastExplorationFSM::SingleCloudPointCloudCallback(
    const sensor_msgs::PointCloud2ConstPtr &colored_cloud) {
  nav_msgs::OdometryConstPtr odom_msg;
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    odom_msg = latest_odom_msg_;
  }
  if (!odom_msg) {
    return;
  }
  CloudOdomCallback(colored_cloud, colored_cloud, odom_msg);
}

void FastExplorationFSM::CloudOdomCallback(const sensor_msgs::PointCloud2ConstPtr &colored_cloud, const sensor_msgs::PointCloud2ConstPtr &nocolor_cloud, const nav_msgs::Odometry::ConstPtr &odom_) {
  // !step 0 更新视角
  auto& oqm = expl_manager_->oqm_;
  bool update_view_pose = false;
  Eigen::Vector3f current_position(odom_->pose.pose.position.x,
                                   odom_->pose.pose.position.y,
                                   odom_->pose.pose.position.z);
  Eigen::Quaternionf current_orientation(odom_->pose.pose.orientation.w,
                                         odom_->pose.pose.orientation.x,
                                         odom_->pose.pose.orientation.y,
                                         odom_->pose.pose.orientation.z);
  current_orientation.normalize();

  if (oqm->odom_poses.empty()) {
    oqm->odom_poses.push_back({current_position, current_orientation});
    update_view_pose = true;
  } else {
    const auto& last_pose = oqm->odom_poses.back();
    float distance = (current_position - last_pose.position).norm();

    Eigen::Quaternionf last_orientation = last_pose.orientation;
    last_orientation.normalize();
    float dot = std::abs(current_orientation.dot(last_orientation));
    if (dot > 1.0f) {
      dot = 1.0f;
    }
    float angle_diff = 2.0f * std::acos(dot);

    constexpr float kPositionThreshold = 0.2f;  // 20 cm
    constexpr float kAngleThreshold = 10.0f * 3.14159265358979323846f / 180.0f;  // 10 deg

    if (distance > kPositionThreshold || angle_diff > kAngleThreshold) {
      oqm->odom_poses.push_back({current_position, current_orientation});
      update_view_pose = true;
    }
  }

  // 检查 odom 是否接近第一个 viewpoint，如果是则用精确的 viewpoint 位姿替换
  if (update_view_pose && !oqm->odom_poses.empty()) {
    const auto& selected_views = oqm->getGlobalPlanner()->getSelectedViews();
    if (!selected_views.empty()) {
      const auto& first_view = selected_views[0];

      // 阈值定义
      constexpr float POSITION_THRESHOLD = 0.3f;  // 0.3m
      constexpr float ANGLE_THRESHOLD = 10.0f * M_PI / 180.0f;  // 10度

      // 获取刚添加的最后一个 odom pose
      const auto& last_odom_pose = oqm->odom_poses.back();
      float pos_dist = (last_odom_pose.position - first_view.position).norm();

      // 从 odom 四元数中提取 yaw 和 pitch
      tf::Quaternion quat(last_odom_pose.orientation.x(),
                          last_odom_pose.orientation.y(),
                          last_odom_pose.orientation.z(),
                          last_odom_pose.orientation.w());
      double roll, pitch, yaw;
      tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);

      float yaw_diff = std::abs(static_cast<float>(yaw) - first_view.yaw);
      // 处理 yaw 的环绕 (e.g., -π 和 π 实际相差很小)
      if (yaw_diff > M_PI) yaw_diff = 2.0f * M_PI - yaw_diff;

      float pitch_diff = std::abs(static_cast<float>(pitch) - first_view.pitch);

      if (pos_dist < POSITION_THRESHOLD &&
          yaw_diff < ANGLE_THRESHOLD &&
          pitch_diff < ANGLE_THRESHOLD) {

        ROS_INFO("CloudOdomCallback: Odom close to first viewpoint (dist=%.3f, yaw_diff=%.1f deg, pitch_diff=%.1f deg), replacing odom pose with exact viewpoint pose",
                 pos_dist, yaw_diff * 180.0f / M_PI, pitch_diff * 180.0f / M_PI);

        // 将 viewpoint 的 yaw/pitch 转换为四元数
        tf::Quaternion viewpoint_quat;
        // Flip pitch to match "positive pitch = down" convention used by viewpoint selection.
        viewpoint_quat.setRPY(0.0, -first_view.pitch, first_view.yaw);
        Eigen::Quaternionf viewpoint_orientation(viewpoint_quat.w(),
                                                  viewpoint_quat.x(),
                                                  viewpoint_quat.y(),
                                                  viewpoint_quat.z());

        // 替换最后一个 odom pose 为精确的 viewpoint 位姿
        oqm->odom_poses.back() = {first_view.position, viewpoint_orientation};
      }
    }
  }

  // Visualize the last odom pose
  static ros::Publisher odom_pose_vis_pub;
  static bool pub_initialized = false;
  if (!pub_initialized) {
    ros::NodeHandle nh;
    odom_pose_vis_pub = nh.advertise<nav_msgs::Odometry>("/odom_pose_visualization", 10);
    pub_initialized = true;
  }

  if (!oqm->odom_poses.empty() && update_view_pose) {
    const auto& last_pose = oqm->odom_poses.back();

    nav_msgs::Odometry odom_msg;
    odom_msg.header.frame_id = "world";
    odom_msg.header.stamp = ros::Time::now();

    odom_msg.pose.pose.position.x = last_pose.position.x();
    odom_msg.pose.pose.position.y = last_pose.position.y();
    odom_msg.pose.pose.position.z = last_pose.position.z();

    odom_msg.pose.pose.orientation.w = last_pose.orientation.w();
    odom_msg.pose.pose.orientation.x = last_pose.orientation.x();
    odom_msg.pose.pose.orientation.y = last_pose.orientation.y();
    odom_msg.pose.pose.orientation.z = last_pose.orientation.z();

    odom_pose_vis_pub.publish(odom_msg);
  }

  int odom_idx = oqm->odom_poses.size() - 1;
  // step 1 更新点云地图，用于避障（TODO: 去除点云地图）
  ros::Time t1 = ros::Time::now();
  planner_manager_->lidar_map_interface_->updateCloudMapOdometry(colored_cloud, nocolor_cloud, odom_);
  double collision_time;
  bool safe;

  if(planner_manager_->local_data_.traj_id_ < 1){
    safe = true;
  }else{
    safe = planner_manager_->checkTrajCollision(collision_time);
  }
  if (!safe) {
    transitState(PLAN_TRAJ, "safetyCallback: not safe, time:" + to_string(collision_time), true);
    if (collision_time < fp_->replan_time_ + 0.2)
      stopTraj();
  }
  ros::Time t2 = ros::Time::now();
  ros::Time t3 = ros::Time::now();

  if (planner_manager_->lidar_map_interface_->ld_->lidar_cloud_.points.empty())
    return;
  auto& ld = planner_manager_->lidar_map_interface_->ld_;
  fd_->odom_pos_ = ld->lidar_pose_;
  fd_->odom_vel_ = ld->lidar_vel_;

  // 从 odom orientation 中提取 yaw 和 pitch
  tf::Quaternion quat;
  tf::quaternionMsgToTF(odom_->pose.pose.orientation, quat);
  double roll, pitch, yaw;
  tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
  fd_->odom_yaw_ = static_cast<float>(yaw);
  fd_->odom_pitch_ = static_cast<float>(pitch);  // 保持标准定义: 正值=向上, 负值=向下

  planner_manager_->local_data_.curr_pos_ = fd_->odom_pos_.cast<double>();
  planner_manager_->local_data_.curr_vel_ = fd_->odom_vel_.cast<double>();
  planner_manager_->local_data_.curr_yaw_ = fd_->odom_yaw_;
  planner_manager_->local_data_.curr_pitch_ = fd_->odom_pitch_;
  planner_manager_->topo_graph_->odom_node_->center_ = fd_->odom_pos_;
  fd_->have_odom_ = true;
  fd_->have_cloud_odom_ = true;

  // !step 2 更新 OQM 
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<PointType>::Ptr nocolor_pcl(new pcl::PointCloud<PointType>);
  pcl::fromROSMsg(*colored_cloud, *colored_pcl);
  pcl::fromROSMsg(*nocolor_cloud, *nocolor_pcl);

  // 合并点云用于几何更新
  pcl::PointCloud<PointType>::Ptr colored_geometry(new pcl::PointCloud<PointType>);
  pcl::copyPointCloud(*colored_pcl, *colored_geometry);
  pcl::PointCloud<PointType>::Ptr merged_geometry(new pcl::PointCloud<PointType>);
  *merged_geometry = *nocolor_pcl + *colored_geometry;

  // 更新 OQM
  // 使用无色点云进行raycast. 更新frontier
  expl_manager_->oqm_->addPointCloudOnlyGeometry(merged_geometry, fd_->odom_pos_);
  expl_manager_->oqm_->addPointCloudWithTexture(colored_pcl, fd_->odom_pos_);

  // 触发 GPU 可见性检查：本帧几何覆盖到的 region + 彩色命中的 region 都要做可见性检查
  Eigen::Quaternionf odom_q(odom_->pose.pose.orientation.w,
                            odom_->pose.pose.orientation.x,
                            odom_->pose.pose.orientation.y,
                            odom_->pose.pose.orientation.z);
  std::unordered_set<Eigen::Vector3i, RegionHash, std::equal_to<Eigen::Vector3i>> regions_to_update;

  // 根据 merged_geometry 覆盖的体素选择 region 
  double max_ray_length = planner_manager_->lidar_map_interface_->lp_->max_ray_length_;
  for (const auto& pt : merged_geometry->points) {
    Eigen::Vector3f pt_pos = pt.getVector3fMap();
    if (!std::isfinite(pt_pos.x()) || !std::isfinite(pt_pos.y()) ||
        !std::isfinite(pt_pos.z())) {
      continue;
    }
    if (!planner_manager_->lidar_map_interface_->IsInBox(pt_pos))
      continue;
    if ((pt_pos - fd_->odom_pos_).norm() >= max_ray_length)
      continue;
    Eigen::Vector3i voxel_idx;
    expl_manager_->oqm_->pos2idx(pt, voxel_idx);
    if ((voxel_idx.array().abs() > 1000000).any()) {
      continue;
    }
    regions_to_update.insert(expl_manager_->oqm_->voxelToRegion(voxel_idx));
  }
  // ROS_INFO("CloudOdomCallback: updating %lu regions", regions_to_update.size());
  oqm_visualization_->publishUpdatedRegions(regions_to_update);
  auto& region_map = oqm->getRegionMap();
  if(update_view_pose) {
    for (const auto& r : regions_to_update) {
      region_map[r].observation_odom_idx.push_back(odom_idx);
      oqm->modified_regions_set.insert(r);
    }
  }

  ros::Time t4 = ros::Time::now();

  ROS_INFO_STREAM_THROTTLE(1.0, "cloud odom callback cost: " << "ikd-tree insert:" << (t2 - t1).toSec() * 1000 << "ms  "
                                                             << "OQM update: " << (t4 - t3).toSec() * 1000 << "ms  "
                                                             << "total: " << (t4 - t1).toSec() * 1000 << "ms" << endl);
}

void FastExplorationFSM::SingleCloudOdomCallback(
    const sensor_msgs::PointCloud2ConstPtr &colored_cloud,
    const nav_msgs::Odometry::ConstPtr &odom_) {
  CloudOdomCallback(colored_cloud, colored_cloud, odom_);
}

void FastExplorationFSM::transitState(EXPL_STATE new_state, string pos_call, bool red) {
  int pre_s = int(state_);
  state_ = new_state;
  if (!red) {
    cout << "\033[32m[" + pos_call + "]\033[0m: from " + fd_->state_str_[pre_s] + " to " + fd_->state_str_[int(new_state)] << endl;
  } else {
    cout << "\033[31m[" + pos_call + "]\033[0m: from " + fd_->state_str_[pre_s] + " to " + fd_->state_str_[int(new_state)] << endl;
  }
}

void FastExplorationFSM::stopTraj() {
  replan_pub_.publish(std_msgs::Empty());
  ros::Time time_now = ros::Time::now();
  ros::Time start_time = planner_manager_->local_data_.start_time_;
  double curr_dur = planner_manager_->local_data_.duration_;
  planner_manager_->local_data_.duration_ = min(curr_dur, (time_now - start_time).toSec() + fp_->replan_time_);
  if (planner_manager_->local_data_.duration_ <= (time_now - start_time).toSec())
    fd_->static_state_ = true;
}

void FastExplorationFSM::dataRecordCallback(
    const sensor_msgs::ImageConstPtr &rgb_image,
    const sensor_msgs::PointCloud2ConstPtr &colored_cloud,
    const sensor_msgs::PointCloud2ConstPtr &nocolor_cloud,
    const nav_msgs::OdometryConstPtr &odom) {

  if (!enable_data_recording_) return;

  std::lock_guard<std::mutex> lock(record_mutex_);

  // Get current position and orientation
  Eigen::Vector3f current_position(odom->pose.pose.position.x,
                                   odom->pose.pose.position.y,
                                   odom->pose.pose.position.z);
  Eigen::Quaternionf current_orientation(odom->pose.pose.orientation.w,
                                         odom->pose.pose.orientation.x,
                                         odom->pose.pose.orientation.y,
                                         odom->pose.pose.orientation.z);
  current_orientation.normalize();

  // Check if we should record this frame (based on position/angle change)
  bool should_record = first_record_frame_;
  if (!first_record_frame_) {
    float distance = (current_position - last_record_position_).norm();

    Eigen::Quaternionf last_orientation = last_record_orientation_;
    last_orientation.normalize();
    float dot = std::abs(current_orientation.dot(last_orientation));
    if (dot > 1.0f) dot = 1.0f;
    float angle_diff_rad = 2.0f * std::acos(dot);
    float angle_diff_deg = angle_diff_rad * 180.0f / M_PI;

    if (distance > record_position_threshold_ || angle_diff_deg > record_angle_threshold_) {
      should_record = true;
    }
  }

  if (!should_record) return;

  // Update last recorded pose
  last_record_position_ = current_position;
  last_record_orientation_ = current_orientation;
  first_record_frame_ = false;

  // Generate frame ID string with zero padding
  std::ostringstream frame_id_ss;
  frame_id_ss << std::setw(6) << std::setfill('0') << record_frame_id_;
  std::string frame_id_str = frame_id_ss.str();

  // 1. Save RGB image
  try {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(rgb_image, "bgr8");
    std::string image_path = data_record_path_ + "/images/" + frame_id_str + ".png";
    cv::imwrite(image_path, cv_ptr->image);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // 2. Save colored point cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(*colored_cloud, *colored_pcl);
  std::string colored_pcd_path = data_record_path_ + "/pointcloud_colored/" + frame_id_str + ".pcd";
  pcl::io::savePCDFileBinary(colored_pcd_path, *colored_pcl);

  // 3. Save uncolored point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr nocolor_pcl(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*nocolor_cloud, *nocolor_pcl);
  std::string nocolor_pcd_path = data_record_path_ + "/pointcloud_nocolor/" + frame_id_str + ".pcd";
  pcl::io::savePCDFileBinary(nocolor_pcd_path, *nocolor_pcl);

  // 4. Append pose to poses.txt file
  // Format: frame_id tx ty tz qx qy qz qw timestamp
  std::string poses_path = data_record_path_ + "/poses.txt";
  std::ofstream poses_file(poses_path, std::ios::app);
  if (poses_file.is_open()) {
    poses_file << std::fixed << std::setprecision(9);
    poses_file << frame_id_str << " "
               << current_position.x() << " "
               << current_position.y() << " "
               << current_position.z() << " "
               << current_orientation.x() << " "
               << current_orientation.y() << " "
               << current_orientation.z() << " "
               << current_orientation.w() << " "
               << odom->header.stamp.toSec() << std::endl;
    poses_file.close();
  }

  // 5. Save camera intrinsics (only once)
  std::string intrinsics_path = data_record_path_ + "/camera_intrinsics.txt";
  if (record_frame_id_ == 0) {
    std::ofstream intrinsics_file(intrinsics_path);
    if (intrinsics_file.is_open()) {
      intrinsics_file << "# Camera intrinsics: fx fy cx cy width height" << std::endl;
      intrinsics_file << std::fixed << std::setprecision(6);
      intrinsics_file << record_cam_fx_ << " " << record_cam_fy_ << " "
                      << record_cam_cx_ << " " << record_cam_cy_ << " "
                      << record_cam_width_ << " " << record_cam_height_ << std::endl;
      intrinsics_file.close();
    }
  }

  ROS_INFO_THROTTLE(1.0, "Recorded frame %d at position (%.2f, %.2f, %.2f)",
                    record_frame_id_, current_position.x(), current_position.y(), current_position.z());

  record_frame_id_++;
}

void FastExplorationFSM::singleCloudDataRecordCallback(
    const sensor_msgs::ImageConstPtr &rgb_image,
    const sensor_msgs::PointCloud2ConstPtr &colored_cloud,
    const nav_msgs::OdometryConstPtr &odom) {
  dataRecordCallback(rgb_image, colored_cloud, colored_cloud, odom);
}

void FastExplorationFSM::exportFreeVoxelsToPCD() {
  if (!enable_free_voxel_export_ || free_voxel_exported_) {
    return;
  }

  ros::Time start_time = ros::Time::now();
  ROS_INFO("Starting FREE voxel export to PCD...");

  // Get OQM and region map
  auto& oqm = expl_manager_->oqm_;
  auto& region_map = oqm->getRegionMap();
  float voxel_size = oqm->getVoxelSize();
  Eigen::Vector3f region_origin = oqm->getRegionOrigin();

  // Step 1: Collect all FREE voxels from all regions
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->points.reserve(100000);  // Pre-allocate

  size_t total_free_voxels = 0;

  // Iterate through all regions
  for (const auto& region_pair : region_map) {
    const auto& region = region_pair.second;

    // Use forEachFreeVoxel callback
    region.forEachFreeVoxel([&](const Eigen::Vector3i& voxel_idx,
                                const FreeRegion::VoxelState& state) {
      // Convert voxel index to world position
      Eigen::Vector3f voxel_center = region_origin +
          ((voxel_idx.cast<float>().array() + 0.5f) * voxel_size).matrix();

      // Add point to cloud
      cloud->points.emplace_back(voxel_center.x(), voxel_center.y(), voxel_center.z());
      total_free_voxels++;
    });
  }

  if (total_free_voxels == 0) {
    ROS_WARN("No FREE voxels found to export!");
    return;
  }

  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = true;

  ROS_INFO("Collected %zu FREE voxels from %zu regions",
           total_free_voxels, region_map.size());

  // Step 2: Downsample using PCL VoxelGrid
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setInputCloud(cloud);
  voxel_filter.setLeafSize(free_voxel_downsample_resolution_,
                          free_voxel_downsample_resolution_,
                          free_voxel_downsample_resolution_);
  voxel_filter.filter(*downsampled);

  ROS_INFO("Downsampled to %zu points (%.2fm resolution)",
           downsampled->points.size(), free_voxel_downsample_resolution_);

  // Step 3: Generate filename with timestamp
  std::ostringstream filename_ss;
  filename_ss << free_voxel_export_path_ << "/free_voxels_";

  time_t now = time(0);
  struct tm* tstruct = localtime(&now);
  char buf[80];
  strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", tstruct);
  filename_ss << buf << ".pcd";

  std::string pcd_path = filename_ss.str();

  // Step 4: Save to PCD file
  if (pcl::io::savePCDFileBinary(pcd_path, *downsampled) == -1) {
    ROS_ERROR("Failed to save PCD file to: %s", pcd_path.c_str());
    return;
  }

  // Step 5: Log summary
  ros::Time end_time = ros::Time::now();
  double export_time_ms = (end_time - start_time).toSec() * 1000.0;

  ROS_INFO("Successfully exported FREE voxels to: %s", pcd_path.c_str());
  ROS_INFO("Export summary:");
  ROS_INFO("  - Original voxels: %zu", total_free_voxels);
  ROS_INFO("  - Downsampled points: %zu", downsampled->points.size());
  ROS_INFO("  - Export time: %.2f ms", export_time_ms);

  // Mark as exported
  free_voxel_exported_ = true;
}
