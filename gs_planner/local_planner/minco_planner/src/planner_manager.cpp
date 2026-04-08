/***
 * @Author: ning-zelin && zl.ning@qq.com
 * @Date: 2023-12-28 14:48:50
 * @LastEditTime: 2023-12-30 15:02:15
 * @Description:
 * @
 * @Copyright (c) 2023 by ning-zelin, All Rights Reserved.
 */
// #include <fstream>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <algorithm>
#include <plan_manage/planner_manager.h>
#include <std_msgs/Int32.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/tf.h>
#include <thread>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

namespace fast_planner {

// Thresholds for skipping optimization (when start/end are close)
static constexpr double POS_CLOSE_THRESHOLD = 0.01;     // 1cm
static constexpr double YAW_CLOSE_THRESHOLD = 0.03;    // ~18 degrees
static constexpr double PITCH_CLOSE_THRESHOLD = 0.03;  // ~18 degrees
static constexpr double MAX_VEL_SIMPLE = 0.5;          // m/s for proportional duration
static constexpr double MAX_ANGVEL_SIMPLE = 1.0;       // rad/s for proportional duration

// SECTION interfaces for setup and query

FastPlannerManager::FastPlannerManager() {
  esdf_valid_ = false;
  oqm_ = nullptr;
}

FastPlannerManager::~FastPlannerManager() {
  lidar_map_interface_.reset();
  gcopter_viz_.reset();
  std::cout << "des manager" << std::endl;
}

void FastPlannerManager::printTimeCost(double time_threhold, double time_cost,
                                       string printInfo) {
  if (time_cost > time_threhold) {
    std::cout << "\033[31m " << printInfo << time_cost << " ms" << "\033[0m"
              << std::endl;
  } else {
    std::cout << "\033[32m " << printInfo << time_cost << " ms" << "\033[0m"
              << std::endl;
  }
}

void FastPlannerManager::initPlanModules(
    ros::NodeHandle &nh, ParallelBubbleAstar::Ptr &parallel_path_finder,
    TopoGraph::Ptr &graph) {

  local_data_.traj_id_ = 0;

  lidar_map_interface_ = graph->lidar_map_interface_;
  nh.getParam("max_traj_len", max_traj_len_);
  nh.getParam("lidar_perception/max_ray_length", max_ray_length);
  nh.getParam("lidar_perception/fov_up", fov_up);
  nh.getParam("lidar_perception/fov_down", fov_down);
  nh.getParam("lidar_perception/lidar_pitch", lidar_pitch);

  gcopter_viz_.reset(new Visualizer);
  gcopter_viz_->init(nh);
  gcopter_config_.reset(new GcopterConfig);
  gcopter_config_->init(nh);

  graph_visualizer_.reset(new GraphVisualizer);
  graph_visualizer_->init(nh);
  bubble_path_finder_.reset(new BubbleAstar);
  bubble_path_finder_->init(nh, lidar_map_interface_);
  topo_graph_ = graph;

  parallel_path_finder_ = parallel_path_finder;
  fast_searcher_.reset(new FastSearcher);
  fast_searcher_->init(topo_graph_, bubble_path_finder_);

  pos_sub = nh.subscribe("/quad_0/lidar_slam/odom", 10,
                         &FastPlannerManager::posCallback, this);
  goal_sub = nh.subscribe("/move_base_simple/goal", 10,
                          &FastPlannerManager::goalCallback, this);
  yaw_state_pub = nh.advertise<std_msgs::Int32>("/quad_0/yaw_state", 10);

  // Add static publisher for yaw waypoint visualization
  yaw_waypoints_pub = nh.advertise<visualization_msgs::MarkerArray>("/planning/yaw_waypoints", 1);
  waypoints_opt_pub = nh.advertise<visualization_msgs::MarkerArray>("/planning/waypoints_opt_status", 1);
  waypoints_grad_pub = nh.advertise<visualization_msgs::MarkerArray>("/planning/waypoints_gradients", 1);
  esdf_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/planning/esdf_slice", 1);
  path_segments_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/planning/path_segments", 1);
}

// test_gs
void FastPlannerManager::posCallback(const nav_msgs::OdometryConstPtr &msg) {

  // 提取四元数
  double roll, pitch;
  tf::Quaternion quat;
  tf::quaternionMsgToTF(msg->pose.pose.orientation, quat);

  // 将四元数转换为Euler角
  tf::Matrix3x3(quat).getRPY(roll, pitch, local_data_.curr_yaw_);
  local_data_.curr_pitch_ = pitch;
}

void FastPlannerManager::goalCallback(
    const geometry_msgs::PoseStampedConstPtr &msg) {
  // 提取四元数
  double roll, pitch;
  tf::Quaternion quat;
  tf::quaternionMsgToTF(msg->pose.orientation, quat);

  // 将四元数转换为Euler角
  tf::Matrix3x3(quat).getRPY(roll, pitch, local_data_.end_yaw_);
}

bool FastPlannerManager::checkTrajVelocity() {
  auto traj = local_data_.minco_traj_;
  double duration = local_data_.duration_;
  double curr_time = (ros::Time::now() - local_data_.start_time_).toSec();
  while (curr_time < duration) {
    Vector3d curr_vel = traj.getVel(curr_time);
    if (curr_vel.norm() > gcopter_config_->maxVelMag + 1.0) {
      return false;
    }
    curr_time += 0.3;
  }
  return true;
}

bool FastPlannerManager::checkTrajCollision(double &collision_time) {
  return true;
  PointType target;
  PointVector nearest_point;
  vector<float> PointDist;

  auto traj = local_data_.minco_traj_;
  double duration = local_data_.duration_;
  double curr_time = (ros::Time::now() - local_data_.start_time_).toSec();
  Vector3d last_sphere_cen_;
  if (curr_time > duration) {
    collision_time = duration;
    return true;
  }

  last_sphere_cen_ = traj.getPos(curr_time);
  double last_radius_ = lidar_map_interface_->getDisToOcc(last_sphere_cen_) -
                        0.15;
  while (curr_time < duration) {
    Vector3d curr_pos = traj.getPos(curr_time);
    if ((curr_pos - last_sphere_cen_).norm() < last_radius_) {
      curr_time += 0.05;
      continue;
    }
    // 超出了上一个球的范围, 更新一个球
    last_radius_ = lidar_map_interface_->getDisToOcc(curr_pos) -
                   0.15;
    last_sphere_cen_ = curr_pos;
    if (last_radius_ < 0) {
      collision_time = curr_time;
      return false;
    }
  }
  return true;
}

int FastPlannerManager::planExploreTraj(const vector<Eigen::Vector3f> &path,
                                         const ViewPoint5D &target_viewpoint,
                                         bool is_static) {
  ros::Time start = ros::Time::now();
  auto local_data_backup = local_data_;

  // Visualize input path before trajectory planning
  gcopter_viz_->visualizeRoute(path);

  // === 计算 pos/yaw/pitch 差值，判断哪些轨迹需要规划 ===
  Eigen::Vector3d start_pos = path.empty() ? local_data_.curr_pos_ : path.front().cast<double>();
  Eigen::Vector3d end_pos = path.size() < 2 ? start_pos : path.back().cast<double>();
  double pos_diff = (end_pos - start_pos).norm();

  double yaw_diff = target_viewpoint.yaw - local_data_.curr_yaw_;
  angleLimite(yaw_diff);

  double pitch_diff = target_viewpoint.pitch - local_data_.curr_pitch_;

  // 判断哪些轨迹需要规划
  bool pos_needed = (path.size() >= 2) && (pos_diff >= POS_CLOSE_THRESHOLD);
  bool yaw_needed = fabs(yaw_diff) >= YAW_CLOSE_THRESHOLD;
  bool pitch_needed = fabs(pitch_diff) >= PITCH_CLOSE_THRESHOLD;

  ROS_INFO_STREAM("Current yaw: " << local_data_.curr_yaw_ << " pitch: " << local_data_.curr_pitch_ 
      << " Target yaw: " << target_viewpoint.yaw << " pitch: " << target_viewpoint.pitch);

  ROS_INFO_STREAM("Traj needed: pos=" << pos_needed << " (diff=" << pos_diff << ")"
                  << " yaw=" << yaw_needed << " (diff=" << yaw_diff << ")"
                  << " pitch=" << pitch_needed << " (diff=" << pitch_diff << ")");

  // 如果三条轨迹都不需要规划，直接返回

  // === 独立规划三条轨迹 ===
  Trajectory<7> pos_traj;
  Trajectory<5> yaw_traj;
  Trajectory<5> pitch_traj;
  double pos_duration = 0.0;
  double yaw_duration = 0.0;
  double pitch_duration = 0.0;
  double computed_end_yaw = target_viewpoint.yaw;

  // === Position 轨迹 ===
  if (pos_needed) {
    if (!PositionTrajPlan(path, target_viewpoint, is_static,
                          pos_duration, computed_end_yaw, pos_traj)) {
      ROS_ERROR("planExploreTraj: position trajectory planning failed");
      local_data_ = local_data_backup;
      return PLAN_FAIL;
    }

    // 碰撞和速度检查
    local_data_.minco_traj_ = pos_traj;
    local_data_.duration_ = pos_duration;

    double collision_time = 10.0;
    if (!checkTrajCollision(collision_time) && collision_time < 1.0) {
      ROS_ERROR("planExploreTraj: trajectory collision check failed");
      local_data_ = local_data_backup;
      return PLAN_FAIL;
    }
    // if (!checkTrajVelocity()) {
    //   ROS_ERROR("planExploreTraj: trajectory velocity check failed");
    //   local_data_ = local_data_backup;
    //   return PLAN_FAIL;
    // }
  } else {
    // Use interpolation for small distance
    double dist = (end_pos - local_data_.curr_pos_).norm();
    pos_duration = std::max(0.1, dist / MAX_VEL_SIMPLE);
    GenerateSimplePositionTraj(local_data_.curr_pos_, end_pos, pos_duration, pos_traj);
    ROS_INFO("Position: generating simple interpolation trajectory");
  }

  // === Yaw 轨迹 ===
  if (yaw_needed) {
    yaw_duration = fabs(yaw_diff) / MAX_ANGVEL_SIMPLE;
    if (yaw_duration < 0.01) yaw_duration = 0.01;  // 最小时长

    double end_yaw_for_traj = pos_needed ? computed_end_yaw : target_viewpoint.yaw;
    ROS_INFO_STREAM("current yaw: " << local_data_.curr_yaw_ << " goal yaw: " << end_yaw_for_traj);
    if (!YawTrajPlan(local_data_.curr_yaw_, end_yaw_for_traj,
                     is_static, yaw_duration, yaw_traj)) {
      ROS_ERROR("planExploreTraj: yaw trajectory planning failed");
      local_data_ = local_data_backup;
      return PLAN_FAIL;
    }
  } else {
    // Use interpolation for small angle
    yaw_duration = std::max(0.1, fabs(yaw_diff) / MAX_ANGVEL_SIMPLE);
    // Get current yaw velocity and acceleration from the existing trajectory
    double current_yaw_vel = 0.0;
    double current_yaw_acc = 0.0;
    if (local_data_.minco_yaw_traj_.getPieceNum() > 0) {
      double traj_end_time = local_data_.minco_yaw_traj_.getTotalDuration();
      current_yaw_vel = local_data_.minco_yaw_traj_.getVel(traj_end_time).x();
      current_yaw_acc = local_data_.minco_yaw_traj_.getAcc(traj_end_time).x();
    }
    GenerateSimpleAngleTraj(local_data_.curr_yaw_, current_yaw_vel, current_yaw_acc,
                          local_data_.curr_yaw_ + yaw_diff, 0.0, 0.0, yaw_duration, yaw_traj);
    ROS_INFO("Yaw: generating simple interpolation trajectory");
  }

  // === Pitch 轨迹 ===
  if (pitch_needed) {
    pitch_duration = fabs(pitch_diff) / MAX_ANGVEL_SIMPLE;
    if (pitch_duration < 0.01) pitch_duration = 0.01;  // 最小时长

    ROS_INFO_STREAM("current pitch: " << local_data_.curr_pitch_ << " goal pitch: " << target_viewpoint.pitch);
    if (!PitchTrajPlan(local_data_.curr_pitch_, target_viewpoint.pitch,
                       is_static, pitch_duration, pitch_traj)) {
      ROS_ERROR("planExploreTraj: pitch trajectory planning failed");
      local_data_ = local_data_backup;
      return PLAN_FAIL;
    }
  } else {
    // Use interpolation for small angle
    pitch_duration = std::max(0.1, fabs(pitch_diff) / MAX_ANGVEL_SIMPLE);
    // Get current pitch velocity and acceleration from the existing trajectory
    double current_pitch_vel = 0.0;
    double current_pitch_acc = 0.0;
    if (local_data_.minco_pitch_traj_.getPieceNum() > 0) {
      double traj_end_time = local_data_.minco_pitch_traj_.getTotalDuration();
      current_pitch_vel = local_data_.minco_pitch_traj_.getVel(traj_end_time).x();
      current_pitch_acc = local_data_.minco_pitch_traj_.getAcc(traj_end_time).x();
    }
    GenerateSimpleAngleTraj(local_data_.curr_pitch_, current_pitch_vel, current_pitch_acc,
                          local_data_.curr_pitch_ + pitch_diff, 0.0, 0.0, pitch_duration, pitch_traj);
    ROS_INFO("Pitch: generating simple interpolation trajectory");
  }

  ROS_INFO_STREAM("Duration: pos=" << pos_duration << " yaw=" << yaw_duration << " pitch=" << pitch_duration);

  // === 更新状态 - 三条轨迹都保存 ===
  local_data_.minco_traj_ = pos_traj;
  local_data_.minco_yaw_traj_ = yaw_traj;
  local_data_.minco_pitch_traj_ = pitch_traj;

  // 保存独立 duration 和标志
  local_data_.pos_duration_ = pos_duration;
  local_data_.yaw_duration_ = yaw_duration;
  local_data_.pitch_duration_ = pitch_duration;
  local_data_.pos_needed_ = pos_needed;
  local_data_.yaw_needed_ = yaw_needed;
  local_data_.pitch_needed_ = pitch_needed;

  // 保持 duration_ 为最大值 (用于兼容其他代码)
  local_data_.duration_ = std::max({pos_duration, yaw_duration, pitch_duration});
  local_data_.end_yaw_ = pos_needed ? computed_end_yaw : target_viewpoint.yaw;
  local_data_.end_pitch_ = target_viewpoint.pitch;
  local_data_.traj_id_ += 1;
  local_data_.start_time_ = ros::Time::now();
  local_data_.start_pos_ = start_pos;

  // === Visualize ===
  visualizeTrajWithCamera();

  return PLAN_SUCCEED;
}

int FastPlannerManager::planLongExploreTraj(
    const vector<PathSegmentWithView> &path_segments,
    ObservationQualityManager::Ptr oqm,
    bool is_static)
{
  ros::Time start_time = ros::Time::now();
  auto local_data_backup = local_data_;

  // Store OQM pointer for ESDF construction
  oqm_ = oqm;

  ROS_INFO("planLongExploreTraj: Received %lu path segments", path_segments.size());

  // Visualize input path segments
  visualizePathSegments(path_segments);

  // ==============================
  // STEP 1: Input Validation & Fallback
  // ==============================
  if (path_segments.empty()) {
    ROS_ERROR("planLongExploreTraj: empty path_segments");
    return PLAN_FAIL;
  }


  // ==============================
  // STEP 1.5: Truncate path_segments to max 10 meters
  // ==============================

  // Use truncated segments for planning
  // const std::vector<PathSegmentWithView>& segments_to_plan = truncated_segments;
  const std::vector<PathSegmentWithView>& segments_to_plan = path_segments;

  // ==============================
  // STEP 2: Extract Waypoints
  // ==============================
  ref_path_points_.clear();
  ref_path_segments_.clear();
  {
    const double min_spacing = oqm_ ? oqm_->getVoxelSize() : 0.2;
    ref_path_segments_.reserve(segments_to_plan.size());
    Eigen::Vector3d last_pt = Eigen::Vector3d::Zero();
    bool has_last = false;
    for (const auto& seg : segments_to_plan) {
      std::vector<Eigen::Vector3d> seg_points;
      seg_points.reserve(seg.path.size() + 1);
      for (const auto& p : seg.path) {
        const Eigen::Vector3d pd = p.cast<double>();
        if (!has_last || (pd - last_pt).norm() >= min_spacing) {
          seg_points.push_back(pd);
          ref_path_points_.push_back(pd);
          last_pt = pd;
          has_last = true;
        }
      }
      const Eigen::Vector3d view_p = seg.target_view.position.cast<double>();
      if (!has_last || (view_p - last_pt).norm() >= min_spacing) {
        seg_points.push_back(view_p);
        ref_path_points_.push_back(view_p);
        last_pt = view_p;
        has_last = true;
      }
      ref_path_segments_.push_back(std::move(seg_points));
    }
  }
  Eigen::Matrix3Xd all_wps;
  Eigen::Matrix<double, 3, 4> iniState, finState;
  std::vector<bool> opt_indi;
  std::vector<double> target_yaws, target_pitches;
  std::vector<int> viewpoint_piece_indices;

  prepareLongTrajWaypoints(segments_to_plan, is_static,
                           all_wps, opt_inistate_, opt_finstate_,
                           opt_indi, target_yaws, target_pitches,
                           viewpoint_piece_indices);

  // Visualize waypoints: blue = optimizable, green = fixed
  visualizeWaypointsOptStatus(all_wps, opt_indi);

  // Build local ESDF for smooth collision gradients
  buildLocalESDF(segments_to_plan);

  // Visualize ESDF slice
  publishESDF();

  // Store for later use
  target_yaws_ = target_yaws;
  target_pitches_ = target_pitches;
  opt_indi_ = opt_indi;
  viewpoint_piece_indices_ = viewpoint_piece_indices;
  opt_wps_ = all_wps;

  // 保存起始角度供联合优化使用
  opt_start_yaw_ = local_data_.curr_yaw_;
  opt_start_pitch_ = local_data_.curr_pitch_;

  ROS_INFO("planLongExploreTraj: Total waypoints=%ld, viewpoints=%lu",
           all_wps.cols(), target_yaws.size());

  // ==============================
  // STEP 3: Setup Selection Matrices
  // ==============================
  setupSelectionMatrices(all_wps, opt_indi);

  piece_nums_ = all_wps.cols() + 1;
  ROS_INFO("planLongExploreTraj: piece_nums=%d, waypt_count=%d", piece_nums_, waypt_count_);

  // ==============================
  // STEP 4: Initialize Time Allocation
  // ==============================
  initializeLongTimeAllocation();

  // ==============================
  // STEP 5: Setup MINCO
  // ==============================
  minco_long_.setConditions(opt_inistate_, opt_finstate_, piece_nums_);

  // ==============================
  // STEP 6: Setup L-BFGS Optimization
  // ==============================
  // Decision variables: [tau (piece_nums), waypoints (3 * waypt_count)]
  int opt_dim = piece_nums_ + 3 * waypt_count_;
  Eigen::VectorXd x(opt_dim);

  // Initialize tau from opt_times using backwardT
  Eigen::VectorXd init_tau;
  backwardT_local(opt_times_, init_tau);
  x.segment(0, piece_nums_) = init_tau;

  // Initialize waypoints from way_wps (column-major)
  if (waypt_count_ > 0) {
    Eigen::Map<Eigen::VectorXd> waypt_vec(way_wps_.data(), 3 * waypt_count_);
    x.segment(piece_nums_, 3 * waypt_count_) = waypt_vec;
  }

  // Set velocity/acceleration bounds
  max_v_squared_ = gcopter_config_->maxVelMag * gcopter_config_->maxVelMag;
  max_a_squared_ = 3.0 * 3.0;  // Default max acceleration 10 m/s^2

  // Configure L-BFGS
  lbfgs::lbfgs_parameter_t lbfgs_params;
  lbfgs_params.mem_size = 64;
  lbfgs_params.past = 3;
  lbfgs_params.g_epsilon = 0;
  lbfgs_params.min_step = 1.0e-32;
  lbfgs_params.delta = 1.0e-2;
  lbfgs_params.max_linesearch = 256;
  lbfgs_params.max_iterations = 1000;

  // ==============================
  // STEP 7: Run Optimization
  // ==============================
  double inner_cost;
  int result = lbfgs::lbfgs_optimize(
      x, inner_cost, &FastPlannerManager::innerCallbackLong,
      nullptr, nullptr, this, lbfgs_params);

  if (result == lbfgs::LBFGS_CONVERGENCE ||
      result == lbfgs::LBFGS_CANCELED ||
      result == lbfgs::LBFGS_STOP ||
      result == lbfgs::LBFGSERR_MAXIMUMITERATION) {
    ROS_INFO("\033[32m[planLongExploreTraj] Optimization success! cost=%.2f, result=%d\033[0m",
             inner_cost, result);
  } else {
    ROS_WARN("[planLongExploreTraj] Optimization warning: result=%d, %s",
             result, lbfgs::lbfgs_strerror(result));
    // Continue anyway - the trajectory may still be usable
  }

  // ==============================
  // STEP 8: Extract Optimized Trajectory
  // ==============================
  Trajectory<7> pos_traj;
  minco_long_.getTrajectory(pos_traj);
  double pos_duration = pos_traj.getTotalDuration();

  // Get optimized times for viewpoint arrival computation
  Eigen::Map<const Eigen::VectorXd> final_tau(x.data(), piece_nums_);
  Eigen::VectorXd final_T;
  forwardT_local(final_tau, final_T);


  ROS_INFO("planLongExploreTraj: Total trajectory duration=%.2fs", pos_duration);

  // ==============================
  // STEP 9: Compute Viewpoint Arrival Times
  // ==============================
  computeViewpointArrivalTimes(final_T);

  // ==============================
  // STEP 10: Generate Yaw/Pitch Trajectories
  // ==============================
  Trajectory<5> yaw_traj, pitch_traj;

  if (!generateYawPitchTrajByTime(local_data_.curr_yaw_, local_data_.curr_pitch_,
                                   is_static, yaw_traj, pitch_traj)) {
    ROS_ERROR("planLongExploreTraj: yaw/pitch trajectory generation failed");
    local_data_ = local_data_backup;
    return PLAN_FAIL;
  }

  // ==============================
  // STEP 11: Collision and Velocity Checks
  // ==============================
  local_data_.minco_traj_ = pos_traj;
  local_data_.duration_ = pos_duration;

  double collision_time = 10.0;
  if (!checkTrajCollision(collision_time) && collision_time < 1.0) {
    ROS_ERROR("planLongExploreTraj: trajectory collision check failed at t=%.2fs", collision_time);
    local_data_ = local_data_backup;
    return PLAN_FAIL;
  }

  // if (!checkTrajVelocity()) {
  //   ROS_ERROR("planLongExploreTraj: trajectory velocity check failed");
  //   local_data_ = local_data_backup;
  //   return PLAN_FAIL;
  // }

  // ==============================
  // STEP 12: Update local_data_
  // ==============================
  local_data_.minco_traj_ = pos_traj;
  local_data_.minco_yaw_traj_ = yaw_traj;
  local_data_.minco_pitch_traj_ = pitch_traj;

  local_data_.pos_duration_ = pos_duration;
  local_data_.yaw_duration_ = yaw_traj.getTotalDuration();
  local_data_.pitch_duration_ = pitch_traj.getTotalDuration();

  local_data_.duration_ = std::max({
      local_data_.pos_duration_,
      local_data_.yaw_duration_,
      local_data_.pitch_duration_});

  local_data_.pos_needed_ = true;
  local_data_.yaw_needed_ = true;
  local_data_.pitch_needed_ = true;

  local_data_.end_yaw_ = target_yaws_.back();
  local_data_.end_pitch_ = target_pitches_.back();
  local_data_.traj_id_ += 1;
  local_data_.start_time_ = ros::Time::now();
  local_data_.start_pos_ = opt_inistate_.col(0);

  // ==============================
  // STEP 13: Visualization
  // ==============================
  visualizeTrajWithCamera();

  double total_time = (ros::Time::now() - start_time).toSec() * 1000.0;
  ROS_INFO("\033[32m[planLongExploreTraj] SUCCESS: duration=%.2fs, %d pieces, %lu viewpoints, time=%.1fms\033[0m",
           local_data_.duration_, piece_nums_, target_yaws_.size(), total_time);

  return PLAN_SUCCEED;
}

// ==========================================
// Helper functions for long trajectory planning
// ==========================================

void FastPlannerManager::prepareLongTrajWaypoints(
    const std::vector<PathSegmentWithView>& path_segments,
    bool is_static,
    Eigen::Matrix3Xd& wps,
    Eigen::Matrix<double, 3, 4>& iniState,
    Eigen::Matrix<double, 3, 4>& finState,
    std::vector<bool>& opt_indi,
    std::vector<double>& target_yaws,
    std::vector<double>& target_pitches,
    std::vector<int>& viewpoint_piece_indices)
{
  // Count total waypoints (excluding final viewpoint which becomes finState)
  // For each segment:
  //   - path[0] is skipped (it's the start point or previous viewpoint)
  //   - path[1] to path[size-2] are optimizable waypoints
  //   - path[size-1] is skipped (it's close to the viewpoint)
  //   - viewpoint is added as frozen (except for last segment which becomes finState)
  int total_wps = 0;
  for (size_t seg_idx = 0; seg_idx < path_segments.size(); seg_idx++) {
    const auto& segment = path_segments[seg_idx];
    // Path points: skip first (start/prev viewpoint) and last (near viewpoint)
    if (segment.path.size() > 2) {
      total_wps += segment.path.size() - 2;
    }
    // Viewpoint position (except for the last segment, which becomes finState)
    if (seg_idx < path_segments.size() - 1) {
      total_wps += 1;
    }
  }

  wps.resize(3, total_wps);
  opt_indi.clear();
  opt_indi.reserve(total_wps);
  target_yaws.clear();
  target_pitches.clear();
  viewpoint_piece_indices.clear();

  int wp_idx = 0;
  int piece_idx = 0;

  for (size_t seg_idx = 0; seg_idx < path_segments.size(); seg_idx++) {
    const auto& segment = path_segments[seg_idx];

    // Add path waypoints (optimizable)
    // Skip path[0] (start point or previous viewpoint) and path[size-1] (near viewpoint)
    for (size_t i = 1; i + 1 < segment.path.size(); i++) {
      wps.col(wp_idx) = segment.path[i].cast<double>();
      opt_indi.push_back(true);  // Path point is optimizable
      wp_idx++;
      piece_idx++;
    }

    // Add viewpoint position (frozen) - except for the last segment
    if (seg_idx < path_segments.size() - 1) {
      wps.col(wp_idx) = segment.target_view.position.cast<double>();
      opt_indi.push_back(false);  // Viewpoint is frozen
      wp_idx++;
      piece_idx++;
    }

    // Record viewpoint info with yaw continuity handling
    double raw_yaw = static_cast<double>(segment.target_view.yaw);
    if (!target_yaws.empty()) {
      // Make yaw continuous relative to previous yaw
      double prev_yaw = target_yaws.back();
      double yaw_diff = raw_yaw - prev_yaw;
      // Normalize to [-pi, pi]
      while (yaw_diff > M_PI) yaw_diff -= 2 * M_PI;
      while (yaw_diff < -M_PI) yaw_diff += 2 * M_PI;
      raw_yaw = prev_yaw + yaw_diff;
    }
    target_yaws.push_back(raw_yaw);
    target_pitches.push_back(static_cast<double>(segment.target_view.pitch));
    viewpoint_piece_indices.push_back(piece_idx);  // This viewpoint is reached at end of this piece
  }

  // Set initial state [P, V, A, J]
  Eigen::Vector3d start_pos;
  Eigen::Vector3d start_vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d start_acc = Eigen::Vector3d::Zero();
  Eigen::Vector3d start_jerk = Eigen::Vector3d::Zero();

  if (!path_segments[0].path.empty()) {
    start_pos = path_segments[0].path[0].cast<double>();
  } else {
    start_pos = local_data_.curr_pos_;
  }

  if (!is_static && local_data_.minco_traj_.getPieceNum() > 0) {
    double time_now = (ros::Time::now() - local_data_.start_time_).toSec();
    time_now = std::min(time_now, local_data_.minco_traj_.getTotalDuration());
    start_pos = local_data_.minco_traj_.getPos(time_now);
    start_vel = local_data_.minco_traj_.getVel(time_now);
    start_acc = local_data_.minco_traj_.getAcc(time_now);
  }

  iniState.col(0) = start_pos;
  iniState.col(1) = start_vel;
  iniState.col(2) = start_acc;
  iniState.col(3) = start_jerk;

  // Set final state (last viewpoint position, zero derivatives)
  const auto& last_view = path_segments.back().target_view;
  finState.col(0) = last_view.position.cast<double>();
  finState.col(1) = Eigen::Vector3d::Zero();
  finState.col(2) = Eigen::Vector3d::Zero();
  finState.col(3) = Eigen::Vector3d::Zero();
}

void FastPlannerManager::setupSelectionMatrices(
    const Eigen::Matrix3Xd& all_wps,
    const std::vector<bool>& opt_indi)
{
  int N = all_wps.cols();
  if (N == 0) {
    waypt_count_ = 0;
    return;
  }

  // Create sparse identity matrix
  Eigen::SparseMatrix<double> onehot(N, N);
  onehot.setIdentity();

  // Count optimizable waypoints
  waypt_count_ = std::count(opt_indi.begin(), opt_indi.end(), true);
  int viewpt_count = N - waypt_count_;

  // Build selection matrices
  select_waypt_.resize(N, waypt_count_);
  select_viewpt_.resize(N, viewpt_count);
  way_wps_.resize(3, waypt_count_);
  view_wps_.resize(3, viewpt_count);

  int way_idx = 0, view_idx = 0;
  for (int i = 0; i < N; i++) {
    if (opt_indi[i]) {
      select_waypt_.col(way_idx) = onehot.col(i);
      way_wps_.col(way_idx) = all_wps.col(i);
      way_idx++;
    } else {
      select_viewpt_.col(view_idx) = onehot.col(i);
      view_wps_.col(view_idx) = all_wps.col(i);
      view_idx++;
    }
  }
}

void FastPlannerManager::initializeLongTimeAllocation()
{
  opt_times_.resize(piece_nums_);
  double max_vel = gcopter_config_->maxVelMag;

  // First piece: from iniState to first waypoint
  if (opt_wps_.cols() > 0) {
    opt_times_[0] = ((opt_inistate_.col(0) - opt_wps_.col(0)).norm() + 0.1) * 2.0 / max_vel;
  } else {
    opt_times_[0] = ((opt_inistate_.col(0) - opt_finstate_.col(0)).norm() + 0.1) * 2.0 / max_vel;
    return;
  }

  // Middle pieces
  for (int i = 1; i < opt_wps_.cols(); i++) {
    opt_times_[i] = ((opt_wps_.col(i) - opt_wps_.col(i-1)).norm() + 0.1) * 2.0 / max_vel;
  }

  // Last piece: to finState
  opt_times_[piece_nums_-1] = ((opt_finstate_.col(0) - opt_wps_.col(opt_wps_.cols()-1)).norm() + 0.1) * 2.0 / max_vel;

  // Ensure minimum time per piece
  for (int i = 0; i < piece_nums_; i++) {
    if (opt_times_[i] < 0.1) opt_times_[i] = 0.1;
  }
}

bool FastPlannerManager::smoothedL1(const double& x, const double& mu, double& f, double& df)
{
  if (x < 0.0) {
    return false;  // No violation
  } else if (x > mu) {
    f = x - 0.5 * mu;  // Linear penalty
    df = 1.0;
    return true;
  } else {
    // Smooth quadratic penalty for small violations
    const double xdmu = x / mu;
    f = (mu - 0.5 * x) * xdmu * xdmu * xdmu;
    df = xdmu * xdmu * (3.0 * (mu - 0.5 * x) / mu - 0.5 * xdmu);
    return true;
  }
}

// ==========================================
// ESDF Construction and Query
// ==========================================

void FastPlannerManager::buildLocalESDF(
  const std::vector<PathSegmentWithView>& path_segments)
{
  if (path_segments.empty() || !oqm_) {
    esdf_valid_ = false;
    return;
  }

  const float voxel_size = oqm_->getVoxelSize();  // 0.2m
  constexpr float EXPANSION = 2.5f;
  constexpr float MAX_DIST = EXPANSION;  // cap ESDF to box margin to keep cost small
  const float max_dist_sq = (MAX_DIST * MAX_DIST) / (voxel_size * voxel_size);

  // Step 1: Compute bounding box with expansion
  Eigen::Vector3f bd_min(1e6f, 1e6f, 1e6f);
  Eigen::Vector3f bd_max(-1e6f, -1e6f, -1e6f);

  for (const auto& seg : path_segments) {
    for (const auto& pt : seg.path) {
      bd_min = bd_min.cwiseMin(pt);
      bd_max = bd_max.cwiseMax(pt);
    }
    bd_min = bd_min.cwiseMin(seg.target_view.position);
    bd_max = bd_max.cwiseMax(seg.target_view.position);
  }

  bd_min.array() -= EXPANSION;
  bd_max.array() += EXPANSION;

  // Step 2: Convert to voxel indices using OQM's pos2idx
  oqm_->pos2idx(bd_min, esdf_bound_min_);
  oqm_->pos2idx(bd_max, esdf_bound_max_);

  esdf_size_ = esdf_bound_max_ - esdf_bound_min_ + Eigen::Vector3i::Ones();
  int total_size = esdf_size_.x() * esdf_size_.y() * esdf_size_.z();

  // Allocate buffers
  if ((int)esdf_distance_buffer_.size() < total_size) {
    esdf_distance_buffer_.resize(total_size);
    esdf_neg_distance_buffer_.resize(total_size);
    esdf_tmp_buffer1_.resize(total_size);
    esdf_tmp_buffer2_.resize(total_size);
  }

  auto& region_map = oqm_->getRegionMap();

  // Helper lambda to query voxel state
  // Returns: 0 = FREE/FRONTIER, 1 = OCCUPIED, 2 = UNKNOWN
  auto getVoxelType = [&](const Eigen::Vector3i& global_idx) -> int {
    Eigen::Vector3i region_idx = oqm_->voxelToRegion(global_idx);
    auto region_it = region_map.find(region_idx);
    if (region_it != region_map.end()) {
      auto& voxel_states = region_it->second.voxel_states;
      auto voxel_it = voxel_states.find(global_idx);
      if (voxel_it != voxel_states.end()) {
        // In hash table: FREE, FRONTIER, or OCCUPIED
        if (voxel_it->second == FreeRegion::VoxelState::OCCUPIED) {
          return 1;  // OCCUPIED
        } else {
          return 0;  // FREE or FRONTIER
        }
      }
    }
    return 2;  // Not in hash table → UNKNOWN
  };

  // ========================================
  // PART 1: Positive distance field
  // Distance from FREE to nearest OCCUPIED/UNKNOWN
  // ========================================
  for (int x = 0; x < esdf_size_.x(); ++x) {
    for (int y = 0; y < esdf_size_.y(); ++y) {
      for (int z = 0; z < esdf_size_.z(); ++z) {
        int addr = esdfAddress(x, y, z);
        Eigen::Vector3i global_idx = esdf_bound_min_ + Eigen::Vector3i(x, y, z);

        // Boundary voxels → treat as obstacle (distance = 0)
        if (x == 0 || x == esdf_size_.x()-1 ||
            y == 0 || y == esdf_size_.y()-1 ||
            z == 0 || z == esdf_size_.z()-1) {
          esdf_tmp_buffer1_[addr] = 0.0f;
          continue;
        }

        int voxel_type = getVoxelType(global_idx);
        // For positive distance: OCCUPIED or UNKNOWN → 0, FREE → max
        esdf_tmp_buffer1_[addr] = (voxel_type == 0) ? max_dist_sq : 0.0f;
      }
    }
  }

  // 3D separable distance transform for positive distance
  // Z-axis pass
  for (int x = 0; x < esdf_size_.x(); ++x) {
    for (int y = 0; y < esdf_size_.y(); ++y) {
      fillESDF1D(
          [&](int z) { return esdf_tmp_buffer1_[esdfAddress(x, y, z)]; },
          [&](int z, float val) { esdf_tmp_buffer2_[esdfAddress(x, y, z)] = val; },
          0, esdf_size_.z() - 1);
    }
  }

  // Y-axis pass
  for (int x = 0; x < esdf_size_.x(); ++x) {
    for (int z = 0; z < esdf_size_.z(); ++z) {
      fillESDF1D(
          [&](int y) { return esdf_tmp_buffer2_[esdfAddress(x, y, z)]; },
          [&](int y, float val) { esdf_tmp_buffer1_[esdfAddress(x, y, z)] = val; },
          0, esdf_size_.y() - 1);
    }
  }

  // X-axis pass (store in distance buffer)
  for (int y = 0; y < esdf_size_.y(); ++y) {
    for (int z = 0; z < esdf_size_.z(); ++z) {
          fillESDF1D(
              [&](int x) { return esdf_tmp_buffer1_[esdfAddress(x, y, z)]; },
              [&](int x, float val) {
                float dist = voxel_size * std::sqrt(val);
                esdf_distance_buffer_[esdfAddress(x, y, z)] = std::min(dist, MAX_DIST);
              },
              0, esdf_size_.x() - 1);
    }
  }

  // ========================================
  // PART 2: Negative distance field
  // Distance from OCCUPIED/UNKNOWN to nearest FREE
  // ========================================
  for (int x = 0; x < esdf_size_.x(); ++x) {
    for (int y = 0; y < esdf_size_.y(); ++y) {
      for (int z = 0; z < esdf_size_.z(); ++z) {
        int addr = esdfAddress(x, y, z);
        Eigen::Vector3i global_idx = esdf_bound_min_ + Eigen::Vector3i(x, y, z);

        // Boundary voxels → max (so they get distance computed from interior FREE)
        if (x == 0 || x == esdf_size_.x()-1 ||
            y == 0 || y == esdf_size_.y()-1 ||
            z == 0 || z == esdf_size_.z()-1) {
          esdf_tmp_buffer1_[addr] = max_dist_sq;
          continue;
        }

        int voxel_type = getVoxelType(global_idx);
        // For negative distance: FREE → 0, OCCUPIED/UNKNOWN → max
        esdf_tmp_buffer1_[addr] = (voxel_type == 0) ? 0.0f : max_dist_sq;
      }
    }
  }

  // 3D separable distance transform for negative distance
  // Z-axis pass
  for (int x = 0; x < esdf_size_.x(); ++x) {
    for (int y = 0; y < esdf_size_.y(); ++y) {
      fillESDF1D(
          [&](int z) { return esdf_tmp_buffer1_[esdfAddress(x, y, z)]; },
          [&](int z, float val) { esdf_tmp_buffer2_[esdfAddress(x, y, z)] = val; },
          0, esdf_size_.z() - 1);
    }
  }

  // Y-axis pass
  for (int x = 0; x < esdf_size_.x(); ++x) {
    for (int z = 0; z < esdf_size_.z(); ++z) {
      fillESDF1D(
          [&](int y) { return esdf_tmp_buffer2_[esdfAddress(x, y, z)]; },
          [&](int y, float val) { esdf_tmp_buffer1_[esdfAddress(x, y, z)] = val; },
          0, esdf_size_.y() - 1);
    }
  }

  // X-axis pass (store in neg distance buffer)
  for (int y = 0; y < esdf_size_.y(); ++y) {
    for (int z = 0; z < esdf_size_.z(); ++z) {
          fillESDF1D(
              [&](int x) { return esdf_tmp_buffer1_[esdfAddress(x, y, z)]; },
              [&](int x, float val) {
                float dist = voxel_size * std::sqrt(val);
                esdf_neg_distance_buffer_[esdfAddress(x, y, z)] = std::min(dist, MAX_DIST);
              },
              0, esdf_size_.x() - 1);
    }
  }

  // ========================================
  // PART 3: Merge positive and negative distance
  // Final: positive for FREE, negative for OCCUPIED/UNKNOWN
  // ========================================
  for (int x = 0; x < esdf_size_.x(); ++x) {
    for (int y = 0; y < esdf_size_.y(); ++y) {
      for (int z = 0; z < esdf_size_.z(); ++z) {
        int addr = esdfAddress(x, y, z);
        float neg_dist = esdf_neg_distance_buffer_[addr];
        // If inside OCCUPIED/UNKNOWN (neg_dist > 0), make distance negative
        if (neg_dist > 0.0f) {
          esdf_distance_buffer_[addr] = -std::min(neg_dist, MAX_DIST);
        }
      }
    }
  }

  esdf_valid_ = true;
  ROS_INFO("Built local ESDF with negative distance: size=(%d,%d,%d), bounds=[%.1f,%.1f,%.1f]-[%.1f,%.1f,%.1f]",
           esdf_size_.x(), esdf_size_.y(), esdf_size_.z(),
           bd_min.x(), bd_min.y(), bd_min.z(), bd_max.x(), bd_max.y(), bd_max.z());
}

void FastPlannerManager::fillESDF1D(
    const std::function<float(int)>& get_val,
    const std::function<void(int, float)>& set_val,
    int start, int end)
{
  int length = end - start + 1;
  if (length <= 0) return;

  std::vector<float> f(length);
  for (int i = 0; i < length; ++i) f[i] = get_val(start + i);

  std::vector<int> v(length);
  std::vector<float> z(length + 1);

  int k = 0;
  v[0] = 0;
  z[0] = -std::numeric_limits<float>::max();
  z[1] = std::numeric_limits<float>::max();

  for (int q = 1; q < length; ++q) {
    float s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0f * q - 2.0f * v[k]);
    while (s <= z[k]) {
      --k;
      s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0f * q - 2.0f * v[k]);
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = std::numeric_limits<float>::max();
  }

  k = 0;
  for (int q = 0; q < length; ++q) {
    while (z[k + 1] < q) ++k;
    set_val(start + q, (q - v[k]) * (q - v[k]) + f[v[k]]);
  }
}

double FastPlannerManager::getDistanceESDF(const Eigen::Vector3i& local_idx) const {
  if (local_idx.x() < 0 || local_idx.x() >= esdf_size_.x() ||
      local_idx.y() < 0 || local_idx.y() >= esdf_size_.y() ||
      local_idx.z() < 0 || local_idx.z() >= esdf_size_.z()) {
    return 0.0;  // Out of bounds → occupied
  }
  return esdf_distance_buffer_[esdfAddress(local_idx.x(), local_idx.y(), local_idx.z())];
}

double FastPlannerManager::getDistWithGradESDF(
    const Eigen::Vector3d& pos,
    Eigen::Vector3d& grad)
{
  if (!oqm_ || !esdf_valid_) {
    grad.setZero();
    return 0.0;
  }

  const double resolution = oqm_->getVoxelSize();
  const double resolution_inv = 1.0 / resolution;

  // Step 1: Shift position (same as reference code)
  Eigen::Vector3d pos_m = pos - 0.5 * resolution * Eigen::Vector3d::Ones();

  // Step 2: Convert to global voxel index
  Eigen::Vector3i idx;
  oqm_->pos2idx(pos_m.cast<float>(), idx);

  // Step 3: Convert to local index for ESDF buffer access
  Eigen::Vector3i local_idx = idx - esdf_bound_min_;

  // Step 4: Bounds check
  const bool out_of_bounds =
      local_idx.x() < 0 || local_idx.x() + 1 >= esdf_size_.x() ||
      local_idx.y() < 0 || local_idx.y() + 1 >= esdf_size_.y() ||
      local_idx.z() < 0 || local_idx.z() + 1 >= esdf_size_.z();
  if (out_of_bounds) {
    // Push query back toward ESDF box with unit gradient and distance-to-box magnitude
    const Eigen::Vector3d region_origin = oqm_->getRegionOrigin().cast<double>();
    const Eigen::Vector3d box_min = region_origin + esdf_bound_min_.cast<double>() * resolution;
    const Eigen::Vector3d box_max =
        region_origin + (esdf_bound_max_.cast<double>() + Eigen::Vector3d::Ones()) * resolution;

    const Eigen::Vector3d projected = pos.cwiseMax(box_min).cwiseMin(box_max);
    const Eigen::Vector3d to_box = projected - pos;
    const double outside_dist = to_box.norm();

    if (outside_dist > 1e-6) {
      grad = to_box / outside_dist;
      return -outside_dist;
    } else {
      grad.setZero();
      return 0.0;
    }
  }

  // Step 5: Get idx position (voxel center in world coordinates)
  // Use OQM's idx2pos to correctly apply region_origin offset
  PointType pt;
  oqm_->idx2pos(idx, pt);
  Eigen::Vector3d idx_pos = pt.getVector3fMap().cast<double>();

  // Step 6: Compute diff using ORIGINAL pos (not pos_m!)
  Eigen::Vector3d diff = (pos - idx_pos) * resolution_inv;

  // Step 7: Sample 8 corner values using LOCAL index
  double values[2][2][2];
  for (int x = 0; x < 2; x++)
    for (int y = 0; y < 2; y++)
      for (int z = 0; z < 2; z++) {
        Eigen::Vector3i current_local_idx = local_idx + Eigen::Vector3i(x, y, z);
        values[x][y][z] = getDistanceESDF(current_local_idx);
      }

  // Step 8: Trilinear interpolation (same as reference)
  double v00 = (1 - diff[0]) * values[0][0][0] + diff[0] * values[1][0][0];
  double v01 = (1 - diff[0]) * values[0][0][1] + diff[0] * values[1][0][1];
  double v10 = (1 - diff[0]) * values[0][1][0] + diff[0] * values[1][1][0];
  double v11 = (1 - diff[0]) * values[0][1][1] + diff[0] * values[1][1][1];
  double v0 = (1 - diff[1]) * v00 + diff[1] * v10;
  double v1 = (1 - diff[1]) * v01 + diff[1] * v11;
  double dist = (1 - diff[2]) * v0 + diff[2] * v1;

  // Step 9: Gradient computation (same as reference)
  grad[2] = (v1 - v0) * resolution_inv;
  grad[1] = ((1 - diff[2]) * (v10 - v00) + diff[2] * (v11 - v01)) * resolution_inv;
  grad[0] = (1 - diff[2]) * (1 - diff[1]) * (values[1][0][0] - values[0][0][0]);
  grad[0] += (1 - diff[2]) * diff[1] * (values[1][1][0] - values[0][1][0]);
  grad[0] += diff[2] * (1 - diff[1]) * (values[1][0][1] - values[0][0][1]);
  grad[0] += diff[2] * diff[1] * (values[1][1][1] - values[0][1][1]);
  grad[0] *= resolution_inv;

  return dist;
}

void FastPlannerManager::queryDistanceWithGrad(
    const Eigen::Vector3d& pos,
    double& dist,
    Eigen::Vector3d& grad)
{
  // Use ESDF if available
    dist = getDistWithGradESDF(pos, grad);
    return;
}

void FastPlannerManager::computeConstraintCostGrad(
    double& cost,
    Eigen::MatrixX3d& gdC,
    Eigen::VectorXd& gdT)
{
  const int K = 64;  // Integration points per piece (reduced for speed)
  const double rho_v = 1000.0;
  const double rho_a = 50.0;
  const double rho_coll = gcopter_config_->rho_collision;
  const double safe_dist = gcopter_config_->safe_distance;
  const double smoothFactor = 0.1;

  cost = 0.0;
  gdC.setZero(8 * piece_nums_, 3);
  gdT.setZero(piece_nums_);

  const auto& coeffs = minco_long_.getCoeffs();

  for (int i = 0; i < piece_nums_; i++) {
    double T = opt_times_[i];
    double step = T / K;

    for (int j = 0; j <= K; j++) {
      // Trapezoidal rule weight: 0.5 at endpoints, 1.0 at interior points
      double node = (j == 0 || j == K) ? 0.5 : 1.0;
      double s = j * step;
      double s2 = s * s;
      double s3 = s2 * s;
      double s4 = s3 * s;
      double s5 = s4 * s;
      double s6 = s5 * s;
      double s7 = s6 * s;

      // Polynomial basis vectors
      Eigen::Matrix<double, 8, 1> beta0, beta1, beta2;
      beta0 << 1.0, s, s2, s3, s4, s5, s6, s7;
      beta1 << 0.0, 1.0, 2.0*s, 3.0*s2, 4.0*s3, 5.0*s4, 6.0*s5, 7.0*s6;
      beta2 << 0.0, 0.0, 2.0, 6.0*s, 12.0*s2, 20.0*s3, 30.0*s4, 42.0*s5;

      // Get coefficients for this piece
      Eigen::Matrix<double, 8, 3> c = coeffs.block<8, 3>(8 * i, 0);

      // Evaluate trajectory at this point
      Eigen::Vector3d pos = c.transpose() * beta0;
      Eigen::Vector3d vel = c.transpose() * beta1;
      Eigen::Vector3d acc = c.transpose() * beta2;

      // Velocity constraint
      double v_snorm = vel.squaredNorm();
      double vViola = v_snorm - max_v_squared_;
      double violaPena, violaPenaD;
      if (smoothedL1(vViola, smoothFactor, violaPena, violaPenaD)) {
        Eigen::Vector3d gradVel = rho_v * violaPenaD * 2.0 * vel;
        cost += rho_v * violaPena * node * step;
        gdC.block<8, 3>(8 * i, 0) += node * step * beta1 * gradVel.transpose();
      }

      // Acceleration constraint
      // double a_snorm = acc.squaredNorm();
      // double aViola = a_snorm - max_a_squared_;
      // if (smoothedL1(aViola, smoothFactor, violaPena, violaPenaD)) {
      //   Eigen::Vector3d gradAcc = rho_a * violaPenaD * 2.0 * acc;
      //   cost += rho_a * violaPena * node * step;
      //   gdC.block<8, 3>(8 * i, 0) += node * step * beta2 * gradAcc.transpose();
      // }

      // Collision constraint
      double dist_obs;
      Eigen::Vector3d grad_obs;
      queryDistanceWithGrad(pos, dist_obs, grad_obs);

      double pen = safe_dist - dist_obs;
      double violaCollPena, violaCollPenaD;
      if (smoothedL1(pen, smoothFactor, violaCollPena, violaCollPenaD)) {
        // Gradient: push away from obstacle (negative sign because pen = safe_dist - dist_obs)
        Eigen::Vector3d gradColl = -rho_coll * violaCollPenaD * grad_obs;
        cost += rho_coll * violaCollPena * node * step;
        gdC.block<8, 3>(8 * i, 0) += node * step * beta0 * gradColl.transpose();
      }

      // Path adherence cost: stay close to front-end A* path
      const double rho_path = gcopter_config_->rho_path;
      if (rho_path > 0.0 && !ref_path_points_.empty()) {
        int seg_idx = -1;
        if (!viewpoint_piece_indices_.empty()) {
          for (int k = 0; k < static_cast<int>(viewpoint_piece_indices_.size()); k++) {
            if (i < viewpoint_piece_indices_[k]) {
              seg_idx = k;
              break;
            }
          }
        }
        const std::vector<Eigen::Vector3d>& ref_pts =
            (seg_idx >= 0 && seg_idx < static_cast<int>(ref_path_segments_.size()) &&
             !ref_path_segments_[seg_idx].empty())
                ? ref_path_segments_[seg_idx]
                : ref_path_points_;

        double best_sq = std::numeric_limits<double>::infinity();
        Eigen::Vector3d best_pt = ref_pts.front();
        for (const auto& p : ref_pts) {
          const double sq = (pos - p).squaredNorm();
          if (sq < best_sq) {
            best_sq = sq;
            best_pt = p;
          }
        }
        const Eigen::Vector3d diff = pos - best_pt;
        cost += 0.5 * rho_path * best_sq * node * step;
        const Eigen::Vector3d gradPath = rho_path * diff;
        gdC.block<8, 3>(8 * i, 0) += node * step * beta0 * gradPath.transpose();
      }

      // Gradient w.r.t. time (from constraint violations)
      // The penalty is integrated, so dCost/dT ~ penalty * dstep/dT
      // For simplicity, we add a small time gradient contribution
    }
  }
}

void FastPlannerManager::computeYawPitchConstraint(
    const Eigen::VectorXd& T,
    double& cost,
    Eigen::VectorXd& gradT)
{
  cost = 0.0;
  gradT.setZero(T.size());

  if (viewpoint_piece_indices_.empty() || target_yaws_.empty()) return;

  int num_viewpoints = static_cast<int>(target_yaws_.size());
  if (num_viewpoints < 1) return;

  // Step 1: 计算 arrival_times 和 yaw_times
  std::vector<double> arrival_times(num_viewpoints);
  Eigen::VectorXd yaw_times(num_viewpoints);
  double prev_arrival = 0.0;

  for (int k = 0; k < num_viewpoints; k++) {
    double arrival = 0.0;
    int piece_end = viewpoint_piece_indices_[k];
    for (int i = 0; i < piece_end && i < T.size(); i++) {
      arrival += T(i);
    }
    arrival_times[k] = arrival;
    yaw_times(k) = std::max(0.01, arrival - prev_arrival);
    prev_arrival = arrival;
  }

  // 对最后一个视点，补上终端段（finState 不在 waypoint 列表里），避免少算一段时间
  double total_T = T.sum();
  if (arrival_times.back() < total_T) {
    double prev = (num_viewpoints > 1) ? arrival_times[num_viewpoints - 2] : 0.0;
    arrival_times.back() = total_T;
    yaw_times(num_viewpoints - 1) = std::max(0.01, arrival_times.back() - prev);
  }

  // Step 2: 构建航点 (3D格式，只用x分量)
  int num_inner_wps = num_viewpoints - 1;
  Eigen::Matrix3Xd wpsYaw(3, num_inner_wps);
  Eigen::Matrix3Xd wpsPitch(3, num_inner_wps);

  // 为避免 [-pi, pi] 跳变导致虚假的大角度，引入连续 yaw 序列
  std::vector<double> continuous_yaws(num_viewpoints);
  double accumulated_yaw = opt_start_yaw_;
  for (int i = 0; i < num_viewpoints; i++) {
    double yaw_diff = target_yaws_[i] - accumulated_yaw;
    while (yaw_diff > M_PI) yaw_diff -= 2 * M_PI;
    while (yaw_diff < -M_PI) yaw_diff += 2 * M_PI;
    accumulated_yaw += yaw_diff;
    continuous_yaws[i] = accumulated_yaw;
  }

  for (int i = 0; i < num_inner_wps; i++) {
    wpsYaw(0, i) = continuous_yaws[i];
    wpsYaw(1, i) = 0.0;
    wpsYaw(2, i) = 0.0;
    wpsPitch(0, i) = target_pitches_[i];
    wpsPitch(1, i) = 0.0;
    wpsPitch(2, i) = 0.0;
  }

  // Step 3: 设置边界条件
  Eigen::Matrix3d iniYaw, finYaw, iniPitch, finPitch;
  iniYaw << Eigen::Vector3d(opt_start_yaw_, 0, 0),
            Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero();
  finYaw << Eigen::Vector3d(continuous_yaws.back(), 0, 0),
            Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero();
  iniPitch << Eigen::Vector3d(opt_start_pitch_, 0, 0),
              Eigen::Vector3d::Zero(),
              Eigen::Vector3d::Zero();
  finPitch << Eigen::Vector3d(target_pitches_.back(), 0, 0),
              Eigen::Vector3d::Zero(),
              Eigen::Vector3d::Zero();

  // Step 4: 生成 MINCO 轨迹
  minco::MINCO_S3NU yaw_minco, pitch_minco;
  yaw_minco.setConditions(iniYaw, finYaw, num_viewpoints);
  yaw_minco.setParameters(wpsYaw, yaw_times);

  pitch_minco.setConditions(iniPitch, finPitch, num_viewpoints);
  pitch_minco.setParameters(wpsPitch, yaw_times);

  // Step 5: 采样角速度并计算惩罚和系数梯度
  double max_yaw_vel = gcopter_config_->yaw_max_vel;
  double max_yaw_vel_sq = max_yaw_vel * max_yaw_vel;
  double rho = gcopter_config_->yaw_rho_vis;
  double smoothFactor = 0.1;

  // 系数梯度
  Eigen::MatrixX3d gdC_yaw(6 * num_viewpoints, 3);
  Eigen::MatrixX3d gdC_pitch(6 * num_viewpoints, 3);
  Eigen::VectorXd gdT_yaw(num_viewpoints);
  Eigen::VectorXd gdT_pitch(num_viewpoints);
  gdC_yaw.setZero();
  gdC_pitch.setZero();
  gdT_yaw.setZero();
  gdT_pitch.setZero();

  const auto& yaw_coeffs = yaw_minco.getCoeffs();
  const auto& pitch_coeffs = pitch_minco.getCoeffs();

  int K = 16;  // 每段采样点数
  for (int k = 0; k < num_viewpoints; k++) {
    double duration = yaw_times(k);
    double step = duration / K;

    for (int j = 0; j <= K; j++) {
      double node = (j == 0 || j == K) ? 0.5 : 1.0;
      double s = j * step;
      double s2 = s * s, s3 = s2 * s, s4 = s3 * s;

      // 多项式基向量 (MINCO_S3NU 用 6 阶多项式, 5th order)
      Eigen::Matrix<double, 6, 1> beta1;  // 速度基
      beta1 << 0.0, 1.0, 2.0*s, 3.0*s2, 4.0*s3, 5.0*s4;

      // 获取系数并计算速度
      Eigen::Matrix<double, 6, 3> c_yaw = yaw_coeffs.block<6, 3>(6 * k, 0);
      Eigen::Matrix<double, 6, 3> c_pitch = pitch_coeffs.block<6, 3>(6 * k, 0);

      Eigen::Vector3d yaw_vel_vec = c_yaw.transpose() * beta1;
      Eigen::Vector3d pitch_vel_vec = c_pitch.transpose() * beta1;
      double yaw_vel = yaw_vel_vec.x();
      double pitch_vel = pitch_vel_vec.x();

      // Yaw 角速度约束
      double yaw_viola = yaw_vel * yaw_vel - max_yaw_vel_sq;
      double pena, penaD;
      if (smoothedL1(yaw_viola, smoothFactor, pena, penaD)) {
        cost += rho * pena * node * step;
        Eigen::Vector3d gradV_yaw = rho * penaD * 2.0 * yaw_vel_vec;
        gdC_yaw.block<6, 3>(6 * k, 0) += node * step * beta1 * gradV_yaw.transpose();
      }

      // Pitch 角速度约束
      double pitch_viola = pitch_vel * pitch_vel - max_yaw_vel_sq;
      if (smoothedL1(pitch_viola, smoothFactor, pena, penaD)) {
        cost += rho * pena * node * step;
        Eigen::Vector3d gradV_pitch = rho * penaD * 2.0 * pitch_vel_vec;
        gdC_pitch.block<6, 3>(6 * k, 0) += node * step * beta1 * gradV_pitch.transpose();
      }
    }
  }

  // Step 6: 通过 MINCO propogateGrad 传播梯度
  Eigen::Matrix3Xd gradP_yaw, gradP_pitch;
  Eigen::VectorXd gradT_yaw_minco, gradT_pitch_minco;

  yaw_minco.propogateGrad(gdC_yaw, gdT_yaw, gradP_yaw, gradT_yaw_minco);
  pitch_minco.propogateGrad(gdC_pitch, gdT_pitch, gradP_pitch, gradT_pitch_minco);

  // 合并 yaw 和 pitch 的时间梯度
  Eigen::VectorXd grad_yaw_times = gradT_yaw_minco + gradT_pitch_minco;

  // Step 7: 将 grad_yaw_times 传播回 pos 轨迹的 grad_T
  // yaw_times[k] = arrival_times[k] - arrival_times[k-1]
  // arrival_times[k] = Σ_{i<piece_idx[k]} T[i]
  // ∂yaw_times[k]/∂T[i] = 1 当 i ∈ [piece_idx[k-1], piece_idx[k])
  for (int i = 0; i < T.size(); i++) {
    // 找到 piece i 属于哪个 viewpoint 段
    int vp_idx = -1;
    for (int k = 0; k < num_viewpoints; k++) {
      if (i < viewpoint_piece_indices_[k]) {
        vp_idx = k;
        break;
      }
    }
    if (vp_idx >= 0 && vp_idx < num_viewpoints) {
      gradT(i) += grad_yaw_times(vp_idx);
    }
  }
}

double FastPlannerManager::innerCallbackLong(
    void* ptrObj,
    const Eigen::VectorXd& x,
    Eigen::VectorXd& grad)
{
  FastPlannerManager& obj = *(FastPlannerManager*)ptrObj;

  // 1. Map optimization variables
  Eigen::Map<const Eigen::VectorXd> tau(x.data(), obj.piece_nums_);
  Eigen::Map<Eigen::VectorXd> grad_tau(grad.data(), obj.piece_nums_);

  // 2. Forward: tau -> T
  Eigen::VectorXd T;
  forwardT_local(tau, T);
  obj.opt_times_ = T;

  // 3. Fuse waypoints (key FC-Planner pattern)
  Eigen::Matrix3Xd wps_opt;
  if (obj.waypt_count_ > 0) {
    Eigen::Map<const Eigen::Matrix3Xd> wps(x.data() + obj.piece_nums_, 3, obj.waypt_count_);
    obj.fused_wps_ = wps * obj.select_waypt_.transpose() +
                     obj.view_wps_ * obj.select_viewpt_.transpose();
  } else {
    obj.fused_wps_ = obj.view_wps_ * obj.select_viewpt_.transpose();
  }

  // 3.5 Hard ESDF bounds for all waypoints (project to box)
  if (obj.esdf_valid_ && obj.oqm_ && obj.fused_wps_.cols() > 0) {
    const double resolution = obj.oqm_->getVoxelSize();
    const Eigen::Vector3d region_origin = obj.oqm_->getRegionOrigin().cast<double>();
    const Eigen::Vector3d box_min =
        region_origin + obj.esdf_bound_min_.cast<double>() * resolution + 1e-2 * Eigen::Vector3d::Ones();
    const Eigen::Vector3d box_max =
        region_origin + (obj.esdf_bound_max_.cast<double>() + Eigen::Vector3d::Ones()) * resolution  - 1e-2 * Eigen::Vector3d::Ones();

    for (int k = 0; k < obj.fused_wps_.cols(); k++) {
      const Eigen::Vector3d wp = obj.fused_wps_.col(k);
      const Eigen::Vector3d projected = wp.cwiseMax(box_min).cwiseMin(box_max);
      if ((projected - wp).squaredNorm() > 1e-12) {
        obj.fused_wps_.col(k) = projected;
      }
    }
  }

  // 4. Set MINCO parameters
  obj.minco_long_.setParameters(obj.fused_wps_, T);

  // 5. Compute snap energy cost
  double snap_cost;
  obj.minco_long_.getEnergy(snap_cost);

  Eigen::MatrixX3d gdC_snap;
  obj.minco_long_.getEnergyPartialGradByCoeffs(gdC_snap);

  Eigen::VectorXd gdT_snap;
  obj.minco_long_.getEnergyPartialGradByTimes(gdT_snap);

  // 6. Compute constraint costs
  double constrain_cost;
  Eigen::MatrixX3d gdC_constrain;
  Eigen::VectorXd gdT_constrain;
  obj.computeConstraintCostGrad(constrain_cost, gdC_constrain, gdT_constrain);

  // 6.5 Compute yaw/pitch angular velocity constraint 
  double yaw_pitch_cost = 0.0;
  Eigen::VectorXd gdT_yaw_pitch(T.size());
  gdT_yaw_pitch.setZero();
  obj.computeYawPitchConstraint(T, yaw_pitch_cost, gdT_yaw_pitch);
  constrain_cost += yaw_pitch_cost;
  gdT_constrain += gdT_yaw_pitch;

  // 7. Time penalty
  double time_cost = obj.gcopter_config_->weightT * T.sum();

  // 8. Total gradients
  Eigen::MatrixX3d gdC = gdC_snap + gdC_constrain;
  Eigen::VectorXd gdT = gdT_snap + gdT_constrain;
  gdT.array() += obj.gcopter_config_->weightT;

  // 9. Propagate gradients through MINCO
  Eigen::Matrix3Xd gradP;
  Eigen::VectorXd gradT;
  obj.minco_long_.propogateGrad(gdC, gdT, gradP, gradT);

  // 10. Map gradients back
  grad_tau.setZero();
  backwardGradT_local(tau, gradT, grad_tau);

  if (obj.waypt_count_ > 0) {
    Eigen::Map<Eigen::Matrix3Xd> grad_wps(grad.data() + obj.piece_nums_, 3, obj.waypt_count_);
    grad_wps = gradP * obj.select_waypt_;  // Only for optimizable waypoints
  }

  return snap_cost + constrain_cost + time_cost;
}

void FastPlannerManager::computeViewpointArrivalTimes(const Eigen::VectorXd& T)
{
  viewpoint_arrival_times_.clear();
  double cumulative = 0.0;

  // Accumulate times for each viewpoint based on their piece indices
  for (size_t vp_idx = 0; vp_idx < viewpoint_piece_indices_.size(); vp_idx++) {
    int piece_end = viewpoint_piece_indices_[vp_idx];
    while (cumulative == 0.0 || (int)viewpoint_arrival_times_.size() < (int)vp_idx) {
      // Sum up pieces until we reach this viewpoint
      break;
    }
    // Sum all pieces up to and including piece_end-1
    double arrival_time = 0.0;
    for (int i = 0; i < piece_end && i < T.size(); i++) {
      arrival_time += T[i];
    }
    viewpoint_arrival_times_.push_back(arrival_time);
  }

  // If the logic above failed, use a simpler approach: evenly distribute viewpoints
  if (viewpoint_arrival_times_.empty() && target_yaws_.size() > 0) {
    double total_time = T.sum();
    for (size_t i = 0; i < target_yaws_.size(); i++) {
      viewpoint_arrival_times_.push_back(total_time * (i + 1) / target_yaws_.size());
    }
  }

  // Ensure last arrival time equals total duration
  if (!viewpoint_arrival_times_.empty()) {
    viewpoint_arrival_times_.back() = T.sum();
  }
}

bool FastPlannerManager::generateYawPitchTrajByTime(
    double start_yaw, double start_pitch,
    bool is_static,
    Trajectory<5>& yaw_traj, Trajectory<5>& pitch_traj)
{
  // Use dense interpolation for smoother trajectories
  // Each original segment is divided into SUB_SEGMENTS sub-segments with linear interpolation

  const int INTERP_POINTS_PER_SEGMENT = 10;  // 10 interpolation points per segment
  const int SUB_SEGMENTS = INTERP_POINTS_PER_SEGMENT + 1;  // = 11 sub-segments
  const double MAX_PITCH = gcopter_config_->max_pitch;
  const double MIN_PITCH = gcopter_config_->min_pitch;

  auto clampPitch = [MIN_PITCH, MAX_PITCH](double pitch) {
    return std::max(MIN_PITCH, std::min(MAX_PITCH, pitch));
  };

  if (viewpoint_arrival_times_.empty() || target_yaws_.empty() || target_pitches_.empty()) {
    ROS_ERROR("generateYawPitchTrajByTime: Invalid data (arrival_times=%lu, yaws=%lu, pitches=%lu)",
              viewpoint_arrival_times_.size(), target_yaws_.size(), target_pitches_.size());
    return false;
  }

  int num_viewpoints = target_yaws_.size();

  // Compute original piece times from viewpoint arrival times
  Eigen::VectorXd orig_piece_times(num_viewpoints);
  double prev_time = 0.0;
  for (int i = 0; i < num_viewpoints; i++) {
    orig_piece_times(i) = viewpoint_arrival_times_[i] - prev_time;
    if (orig_piece_times(i) < 0.01) orig_piece_times(i) = 0.01;  // Minimum time
    prev_time = viewpoint_arrival_times_[i];
  }

  // Get current state
  double yaw_sp = start_yaw, yaw_sv = 0.0, yaw_sa = 0.0;
  double pitch_sp = start_pitch, pitch_sv = 0.0, pitch_sa = 0.0;

  if (!is_static && local_data_.minco_yaw_traj_.getPieceNum() > 0) {
    double time_now = (ros::Time::now() - local_data_.start_time_).toSec();
    time_now = std::min(time_now, local_data_.minco_yaw_traj_.getTotalDuration());
    yaw_sp = local_data_.minco_yaw_traj_.getPos(time_now).x();
    yaw_sv = local_data_.minco_yaw_traj_.getVel(time_now).x();
    yaw_sa = local_data_.minco_yaw_traj_.getAcc(time_now).x();
  }
  if (!is_static && local_data_.minco_pitch_traj_.getPieceNum() > 0) {
    double time_now = (ros::Time::now() - local_data_.start_time_).toSec();
    time_now = std::min(time_now, local_data_.minco_pitch_traj_.getTotalDuration());
    pitch_sp = local_data_.minco_pitch_traj_.getPos(time_now).x();
    pitch_sv = local_data_.minco_pitch_traj_.getVel(time_now).x();
    pitch_sa = local_data_.minco_pitch_traj_.getAcc(time_now).x();
  }

  // Clamp initial pitch
  pitch_sp = clampPitch(pitch_sp);

  // First, make all target_yaws continuous relative to start_yaw
  std::vector<double> continuous_yaws(num_viewpoints);
  double accumulated_yaw = yaw_sp;
  for (int i = 0; i < num_viewpoints; i++) {
    double target_yaw = target_yaws_[i];
    double yaw_diff = target_yaw - accumulated_yaw;
    // Normalize to [-pi, pi] to find shortest path
    while (yaw_diff > M_PI) yaw_diff -= 2 * M_PI;
    while (yaw_diff < -M_PI) yaw_diff += 2 * M_PI;
    accumulated_yaw += yaw_diff;
    continuous_yaws[i] = accumulated_yaw;
  }

  // Dense interpolation: each original segment is divided into SUB_SEGMENTS sub-segments
  int total_pieces = num_viewpoints * SUB_SEGMENTS;
  int total_waypoints = total_pieces - 1;

  Eigen::VectorXd piece_times(total_pieces);
  Eigen::Matrix3Xd wpsYaw(3, total_waypoints);
  Eigen::Matrix3Xd wpsPitch(3, total_waypoints);

  double prev_yaw = yaw_sp;
  double prev_pitch = pitch_sp;

  for (int seg = 0; seg < num_viewpoints; seg++) {
    double target_yaw = continuous_yaws[seg];
    double target_pitch = clampPitch(target_pitches_[seg]);
    double sub_time = orig_piece_times(seg) / SUB_SEGMENTS;

    for (int k = 0; k < SUB_SEGMENTS; k++) {
      int piece_idx = seg * SUB_SEGMENTS + k;
      piece_times(piece_idx) = sub_time;

      // Linear interpolation ratio (uniform angular velocity)
      double ratio = (k + 1.0) / SUB_SEGMENTS;
      double interp_yaw = prev_yaw + ratio * (target_yaw - prev_yaw);
      double interp_pitch = prev_pitch + ratio * (target_pitch - prev_pitch);

      // Last point becomes finState, not added to waypoints
      if (piece_idx < total_waypoints) {
        wpsYaw(0, piece_idx) = interp_yaw;
        wpsYaw(1, piece_idx) = 0.0;
        wpsYaw(2, piece_idx) = 0.0;

        wpsPitch(0, piece_idx) = interp_pitch;
        wpsPitch(1, piece_idx) = 0.0;
        wpsPitch(2, piece_idx) = 0.0;
      }
    }
    prev_yaw = target_yaw;
    prev_pitch = target_pitch;
  }

  // Final state (last viewpoint with clamped pitch)
  double final_yaw = continuous_yaws.back();
  double final_pitch = clampPitch(target_pitches_.back());

  ROS_INFO("Yaw trajectory: start=%.2f -> end=%.2f (change=%.1f deg), pieces=%d (dense interp)",
           yaw_sp, final_yaw, (final_yaw - yaw_sp) * 180.0 / M_PI, total_pieces);

  // Set initial and final states
  Eigen::Matrix3d iniStateYaw, finStateYaw;
  iniStateYaw << Eigen::Vector3d(yaw_sp, 0.0, 0.0),
                 Eigen::Vector3d(yaw_sv, 0.0, 0.0),
                 Eigen::Vector3d(yaw_sa, 0.0, 0.0);
  finStateYaw << Eigen::Vector3d(final_yaw, 0.0, 0.0),
                 Eigen::Vector3d::Zero(),
                 Eigen::Vector3d::Zero();

  Eigen::Matrix3d iniStatePitch, finStatePitch;
  iniStatePitch << Eigen::Vector3d(pitch_sp, 0.0, 0.0),
                   Eigen::Vector3d(pitch_sv, 0.0, 0.0),
                   Eigen::Vector3d(pitch_sa, 0.0, 0.0);
  finStatePitch << Eigen::Vector3d(final_pitch, 0.0, 0.0),
                   Eigen::Vector3d::Zero(),
                   Eigen::Vector3d::Zero();

  // Use MINCO to generate trajectories with dense waypoints
  minco::MINCO_S3NU yaw_minco, pitch_minco;

  yaw_minco.setConditions(iniStateYaw, finStateYaw, total_pieces);
  yaw_minco.setParameters(wpsYaw, piece_times);
  yaw_traj.clear();
  yaw_minco.getTrajectory(yaw_traj);

  pitch_minco.setConditions(iniStatePitch, finStatePitch, total_pieces);
  pitch_minco.setParameters(wpsPitch, piece_times);
  pitch_traj.clear();
  pitch_minco.getTrajectory(pitch_traj);

  ROS_INFO("generateYawPitchTrajByTime: pieces=%d, duration=%.2fs (yaw=%.2fs, pitch=%.2fs)",
           total_pieces, piece_times.sum(), yaw_traj.getTotalDuration(), pitch_traj.getTotalDuration());

  return true;
}

void FastPlannerManager::angleLimite(double &angle) {
  while (angle > M_PI) {
    angle -= (M_PI * 2);
  }
  while (angle < -M_PI) {
    angle += (M_PI * 2);
  }
}

bool FastPlannerManager::YawTrajOpt(double &start_yaw, double &end_yaw,
                                    bool is_static, bool use_shorten_path) {
  Eigen::Matrix3d iniStateYaw, finStateYaw;
  Eigen::MatrixXd wpsYaw;
  Eigen::VectorXd opt_times_Yaw;
  double end_yaw_temp;
  double init_yaw;
  double time_now = (ros::Time::now() - local_data_.start_time_).toSec();
  double yaw_sp, yaw_sv(0.0), yaw_sa(0.0), yaw_ep(end_yaw), yaw_ev(0.0),
      yaw_ea(0.0);

  if (is_static) {
    yaw_sp = start_yaw;
  } else {
    time_now = time_now > local_data_.minco_yaw_traj_.getTotalDuration()
                   ? local_data_.minco_yaw_traj_.getTotalDuration()
                   : time_now;
    yaw_sp = local_data_.minco_yaw_traj_.getPos(time_now).x();
    yaw_sv = local_data_.minco_yaw_traj_.getVel(time_now).x();
    yaw_sa = local_data_.minco_yaw_traj_.getAcc(time_now).x();
  }
  angleLimite(yaw_sp);

  // Use uniform interpolation similar to pitch planning (instead of look-forward)
  static double yaw_dur = 0.3;
  vector<double> wp;
  wp.push_back(yaw_sp);

  // Calculate yaw difference with angle wrapping
  double yaw_diff = yaw_ep - yaw_sp;
  angleLimite(yaw_diff);
  double time_dur = local_data_.duration_;

  // Generate yaw waypoints with smooth linear interpolation
  for (double t = yaw_dur; t < time_dur + yaw_dur; t += yaw_dur) {
    if (t > time_dur) {
      wp.push_back(yaw_sp + yaw_diff);  // Final point at end_yaw
      break;
    }
    // Linear interpolation for yaw waypoints
    double alpha = t / time_dur;
    double interpolated_yaw = yaw_sp + alpha * yaw_diff;
    wp.push_back(interpolated_yaw);
  }

  // Update end_yaw to the final interpolated value
  local_data_.end_yaw_ = wp.back();
  yaw_ep = local_data_.end_yaw_;
  iniStateYaw << Eigen::Vector3d(yaw_sp, 0.0, 0.0),
      Eigen::Vector3d(yaw_sv, 0.0, 0.0), Eigen::Vector3d(yaw_sa, 0.0, 0.0);
  finStateYaw << Eigen::Vector3d(yaw_ep, 0.0, 0.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero();

  gcopter::GCOPTER_PolytopeSFC gcopter_yaw;
  if (!gcopter_yaw.setup_yaw(gcopter_config_->yaw_rho_vis,
                             gcopter_config_->integralIntervs)) {
    cout << "setup_yaw failed!" << endl;
    return false;
  }
  int pieceNUM = wp.size() - 2;
  if (pieceNUM <= 1) {
    opt_times_Yaw.resize(2);
    opt_times_Yaw[0] = local_data_.duration_ / 2.0;
    opt_times_Yaw[1] = local_data_.duration_ / 2.0;
    wpsYaw.resize(3, 1);
    wpsYaw(0, 0) = (wp[0] + wp[1]) / 2.0;
    wpsYaw(1, 0) = 0.0;
    wpsYaw(2, 0) = 0.0;
    pieceNUM = 2;
  } else {
    // 干掉最后一个值
    opt_times_Yaw.resize(wp.size() - 2);
    for (int i = 0; i < wp.size() - 3; i++) {
      opt_times_Yaw[i] = yaw_dur;
    }
    opt_times_Yaw[wp.size() - 3] =
        local_data_.duration_ - (wp.size() - 3) * yaw_dur;
    wpsYaw.resize(3, wp.size() - 3);
    for (int i = 1; i < wp.size() - 2; i++) {
      wpsYaw(0, i - 1) = wp[i];
      wpsYaw(1, i - 1) = 0.0;
      wpsYaw(2, i - 1) = 0.0;
    }
  }
  // double dur_yaw = 0.0;
  // for (int i = 0; i < opt_times_Yaw.size(); i++) {
  //   dur_yaw += opt_times_Yaw[i];
  // }
  // cout << "dur_p= " << local_data_.duration_ << " dur_yaw= " << dur_yaw <<
  // endl; cout << "start yaw = " << iniStateYaw.col(0).transpose() << endl;
  // cout << "end yaw = " << finStateYaw.col(0).transpose() << endl;
  // cout << "wpsYaw = " << endl;
  // for (int i = 0; i < wpsYaw.cols(); i++) {
  //   cout << wpsYaw.col(i).transpose()(0) << " ";
  // }
  // cout << endl;

  local_data_.minco_yaw_traj_.clear();
  if (std::isinf(gcopter_yaw.optimize_yaw(iniStateYaw, finStateYaw, pieceNUM,
                                          wpsYaw, opt_times_Yaw,
                                          local_data_.minco_yaw_traj_))) {
    std::cout << "optimize yaw failed!" << std::endl;
    return false;
  }
  return true;
}

bool FastPlannerManager::flyToSafeRegion(bool is_static) {
  Eigen::Vector3f min_bd, max_bd;
  for (int i = 0; i < 3; i++) {
    min_bd[i] = topo_graph_->odom_node_->center_[i] - 2.0;
    max_bd[i] = topo_graph_->odom_node_->center_[i] + 2.0;
  }
  PointVector Searched_Points;
  lidar_map_interface_->boxSearch(min_bd, max_bd, Searched_Points);
  std::vector<Eigen::Vector3d> surf_points;
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(
      new pcl::PointCloud<pcl::PointXYZ>);
  cloud_origin->points = Searched_Points;
  sor.setInputCloud(cloud_origin);
  sor.setLeafSize(0.2, 0.2, 0.2);
  sor.filter(*cloud_tmp);

  surf_points.reserve(cloud_tmp->points.size());
  for (const pcl::PointXYZ &point : cloud_tmp->points) {
    surf_points.emplace_back(point.x, point.y, point.z);
  }
  Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
  bd(0, 0) = 1.0;
  bd(1, 0) = -1.0;
  bd(2, 1) = 1.0;
  bd(3, 1) = -1.0;
  bd(4, 2) = 1.0;
  bd(5, 2) = -1.0;
  bd(0, 3) =
      -(std::min(topo_graph_->odom_node_->center_(0) + 2.0f,
                 lidar_map_interface_->lp_->global_map_max_boundary_[0]));
  bd(1, 3) = std::max(topo_graph_->odom_node_->center_(0) - 2.0f,
                      lidar_map_interface_->lp_->global_box_min_boundary_[0]);
  bd(2, 3) =
      -(std::min(topo_graph_->odom_node_->center_(1) + 2.0f,
                 lidar_map_interface_->lp_->global_map_max_boundary_[1]));
  bd(3, 3) = std::max(topo_graph_->odom_node_->center_(1) - 2.0f,
                      lidar_map_interface_->lp_->global_box_min_boundary_[1]);
  bd(4, 3) =
      -(std::min(topo_graph_->odom_node_->center_(2) + 1.0f,
                 lidar_map_interface_->lp_->global_map_max_boundary_[2]));
  bd(5, 3) = std::max(topo_graph_->odom_node_->center_(2) - 1.0f,
                      lidar_map_interface_->lp_->global_box_min_boundary_[2]);
  Eigen::Map<const Eigen::Matrix<double, 3, -1, Eigen::ColMajor>> pc(
      surf_points[0].data(), 3, surf_points.size());
  Eigen::MatrixX4d hp;
  firi::firi(bd, pc, topo_graph_->odom_node_->center_.cast<double>(),
             topo_graph_->odom_node_->center_.cast<double>(),
             hp); // 计算出包含a和b的凸包
  std::vector<Eigen::MatrixX4d> hPolys;
  hPolys.push_back(hp);
  hPolys.push_back(hp);
  Eigen::Vector3d inner;
  geo_utils::findInterior(hp, inner);
  Eigen::Vector4d bh;
  double time_now = (ros::Time::now() - local_data_.start_time_).toSec();
  Eigen::Matrix<double, 3, 4> iniState;
  Eigen::Vector3d dir =
      (inner - topo_graph_->odom_node_->center_.cast<double>()).normalized();
  if (is_static) {
    // TSP、更新地图等会阻塞里程计回调函数，导致这里的数据不准，所以只有static才用
    iniState << local_data_.curr_pos_, dir * 0.2, Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero();
    // iniState << topo_graph_->odom_node_->center_, local_data_.curr_vel_,
    // Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
  } else {
    time_now =
        time_now > local_data_.duration_ ? local_data_.duration_ : time_now;
    Eigen::Vector3d current_pose = local_data_.minco_traj_.getPos(time_now);
    Eigen::Vector3d curr_vel = local_data_.minco_traj_.getVel(time_now);
    Eigen::Vector3d curr_acc = local_data_.minco_traj_.getAcc(time_now);
    Eigen::Vector3d curr_jerk = local_data_.minco_traj_.getJer(time_now);
    // iniState << current_pose, curr_vel, curr_acc, curr_jerk;
    iniState << current_pose, dir * 0.2, Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero();
  }
  Eigen::Matrix<double, 3, 4> finState;
  ros::Time hpoly_gen_end = ros::Time::now();
  // iniState << topo_graph_->odom_node_->center_, local_data_.curr_vel_,
  // Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
  finState << inner, dir, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
  bh << iniState.topLeftCorner<3, 1>(), 1.0;
  int start_idx = -1;
  for (int i = hPolys.size() - 1; i >= 0; i--) {
    Eigen::MatrixX4d hp = hPolys[i];
    if ((((hp * bh).array() > -1.0e-6).cast<int>().sum() <= 0)) {
      start_idx = i;
      break;
    }
  }
  if (start_idx == -1) {
    ROS_ERROR("current position not in corridor");
    return false;
  }
  if (start_idx != 0) {
    hPolys.erase(hPolys.begin(), hPolys.begin() + start_idx);
  }
  sfc_gen::shortCut(hPolys);

  // ros::Duration(1.0).sleep();

  if (hPolys.size() < 2) {
    cout << "hPolys size < 2" << endl;
    return false;
  }
  gcopter_viz_->visualizePolytope(hPolys);
  gcopter::GCOPTER_PolytopeSFC gcopter;
  Eigen::VectorXd magnitudeBounds(5);
  Eigen::VectorXd penaltyWeights(5);
  Eigen::VectorXd physicalParams(6);
  magnitudeBounds(0) = gcopter_config_->maxVelMag;
  magnitudeBounds(1) = gcopter_config_->maxBdrMag;
  magnitudeBounds(2) = gcopter_config_->maxTiltAngle;
  magnitudeBounds(3) = gcopter_config_->minThrust;
  magnitudeBounds(4) = gcopter_config_->maxThrust;
  penaltyWeights(0) = (gcopter_config_->chiVec)[0] * 2.0;
  penaltyWeights(1) = (gcopter_config_->chiVec)[1] / 2.0;
  penaltyWeights(2) = (gcopter_config_->chiVec)[2] / 2.0;
  penaltyWeights(3) = (gcopter_config_->chiVec)[3] / 2.0;
  penaltyWeights(4) = (gcopter_config_->chiVec)[4] / 2.0;
  physicalParams(0) = gcopter_config_->vehicleMass;
  physicalParams(1) = gcopter_config_->gravAcc;
  physicalParams(2) = gcopter_config_->horizDrag;
  physicalParams(3) = gcopter_config_->vertDrag;
  physicalParams(4) = gcopter_config_->parasDrag;
  physicalParams(5) = gcopter_config_->speedEps;
  const int quadratureRes = gcopter_config_->integralIntervs;

  if (!gcopter.setup(
          gcopter_config_->WeightSafeT, gcopter_config_->dilateRadiusSoft,
          iniState, finState, hPolys, INFINITY, gcopter_config_->smoothingEps,
          quadratureRes, magnitudeBounds, penaltyWeights, physicalParams)) {
    std::cout << "\n\n\n\n\n\n\n\nsetup failed!" << std::endl;

    return false;
  }
  auto local_data_backup = local_data_;
  local_data_.minco_traj_.clear();
  if (std::isinf(gcopter.optimize(local_data_.minco_traj_,
                                  gcopter_config_->relCostTol, 0.0))) {
    std::cout << "optimize failed!" << std::endl;
    local_data_ = local_data_backup;
    return false;
  }
  if (local_data_.minco_traj_.getPieceNum() > 0) {
    // ROS_INFO_STREAM(
    // "local_data_.minco_traj_.getPieceNum(): " <<
    // local_data_.minco_traj_.getPieceNum());
    gcopter_viz_->visualize(local_data_.minco_traj_,
                            gcopter_config_->maxVelMag);
  } else {
    local_data_ = local_data_backup;
    std::cout << "traj empty!" << std::endl;
    return false;
  }
  ros::Time optimize_end_stamp = ros::Time::now();
  local_data_.traj_id_ += 1;
  local_data_.start_time_ = hpoly_gen_end;
  local_data_.start_pos_ = topo_graph_->odom_node_->center_.cast<double>();
  local_data_.duration_ = local_data_.minco_traj_.getTotalDuration();
  return true;
}

void FastPlannerManager::polyTraj2ROSMsg(traj_utils::PolyTraj &poly_msg,
                                         const ros::Time &start_time) {
  Eigen::VectorXd durs = local_data_.minco_traj_.getDurations();
  int piece_num = local_data_.minco_traj_.getPieceNum();
  poly_msg.drone_id = 0;
  poly_msg.traj_id = local_data_.traj_id_;
  poly_msg.start_time = start_time;
  poly_msg.order = 7;
  poly_msg.duration.resize(piece_num);
  poly_msg.coef_x.resize(8 * piece_num);
  poly_msg.coef_y.resize(8 * piece_num);
  poly_msg.coef_z.resize(8 * piece_num);
  for (int i = 0; i < piece_num; ++i) {
    poly_msg.duration[i] = durs(i);
    Eigen::Matrix<double, 3, 8> cMat =
        local_data_.minco_traj_.pieces[i].getCoeffMat();
    int i6 = i * 8;
    for (int j = 0; j < 8; j++) {
      poly_msg.coef_x[i6 + j] = cMat(0, j);
      poly_msg.coef_y[i6 + j] = cMat(1, j);
      poly_msg.coef_z[i6 + j] = cMat(2, j);
    }
  }
}

void FastPlannerManager::polyYawTraj2ROSMsg(traj_utils::PolyTraj &poly_msg,
                                            const ros::Time &start_time) {
  Eigen::VectorXd durs = local_data_.minco_yaw_traj_.getDurations();
  int piece_num = local_data_.minco_yaw_traj_.getPieceNum();
  poly_msg.drone_id = 0;
  poly_msg.traj_id = local_data_.traj_id_;
  poly_msg.start_time = start_time;
  poly_msg.order = 5;
  poly_msg.duration.resize(piece_num);
  poly_msg.coef_x.resize(6 * piece_num);
  poly_msg.coef_y.resize(6 * piece_num);
  poly_msg.coef_z.resize(6 * piece_num);
  for (int i = 0; i < piece_num; ++i) {
    poly_msg.duration[i] = durs(i);
    Eigen::Matrix<double, 3, 6> cMat =
        local_data_.minco_yaw_traj_.pieces[i].getCoeffMat();
    int i6 = i * 6;
    for (int j = 0; j < 6; j++) {
      poly_msg.coef_x[i6 + j] = cMat(0, j);
      poly_msg.coef_y[i6 + j] = cMat(1, j);
      poly_msg.coef_z[i6 + j] = cMat(2, j);
    }
  }
}

void FastPlannerManager::calculateTimelb(
    const vector<Eigen::Vector3d> &path2next_goal, const double &current_yaw,
    const double &goal_yaw, double &time_lb) {
  double start2fwd = 0.0, fwd2end = 0.0;
  if (path2next_goal.size() == 2) {
    Eigen::Vector3d diff = path2next_goal.back() - path2next_goal.front();
    if (pow(diff.x(), 2) + pow(diff.y(), 2) >= pow(diff.z(), 2) &&
        (diff.squaredNorm() - pow(diff.z(), 2) > 0.01)) {
      double fwd_yaw = atan2(diff.y(), diff.x());
      start2fwd = fwd_yaw - current_yaw;
      angleLimite(start2fwd);
      fwd2end = goal_yaw - fwd_yaw;
      angleLimite(fwd2end);

    } else {
      double diff = goal_yaw - current_yaw;
      angleLimite(diff);
      time_lb = fabs(diff) / gcopter_config_->yaw_max_vel;
      return;
    }
  } else {
    Eigen::Vector3d diff = path2next_goal[1] - path2next_goal[0];
    if (pow(diff.x(), 2) + pow(diff.y(), 2) >= pow(diff.z(), 2)) {
      double fwd_yaw = atan2(diff.y(), diff.x());
      start2fwd = fwd_yaw - current_yaw;
      angleLimite(start2fwd);
    } else {
      start2fwd = 0.0;
    }
    diff = path2next_goal[path2next_goal.size() - 1] -
           path2next_goal[path2next_goal.size() - 2];
    if (pow(diff.x(), 2) + pow(diff.y(), 2) >= pow(diff.z(), 2)) {
      double fwd_yaw = atan2(diff.y(), diff.x());
      fwd2end = fwd_yaw - goal_yaw;
      angleLimite(fwd2end);
    } else {
      fwd2end = 0.0;
    }
  }

  time_lb = fabs(start2fwd) / gcopter_config_->yaw_max_vel +
            fabs(fwd2end) / gcopter_config_->yaw_max_vel;
  return;
}

bool FastPlannerManager::PitchTrajOpt(double &start_pitch, double &end_pitch, bool is_static) {
  Eigen::Matrix3d iniStatePitch, finStatePitch;
  Eigen::MatrixXd wpsPitch;
  Eigen::VectorXd opt_times_Pitch;
  double time_now = (ros::Time::now() - local_data_.start_time_).toSec();
  double pitch_sp, pitch_sv(0.0), pitch_sa(0.0), pitch_ep(end_pitch), pitch_ev(0.0), pitch_ea(0.0);

  if (is_static) {
    pitch_sp = start_pitch;
  } else {
    time_now = time_now > local_data_.minco_pitch_traj_.getTotalDuration() ? local_data_.minco_pitch_traj_.getTotalDuration() : time_now;
    pitch_sp = local_data_.minco_pitch_traj_.getPos(time_now).x();
    pitch_sv = local_data_.minco_pitch_traj_.getVel(time_now).x();
    pitch_sa = local_data_.minco_pitch_traj_.getAcc(time_now).x();
  }

  static double pitch_dur = 0.15;
  vector<double> way_pts_pitch;
  way_pts_pitch.push_back(pitch_sp);
  double pitch_diff = std::abs(pitch_ep - pitch_sp);
  double time_dur = max(local_data_.duration_, pitch_diff / 2.0) ;

  // Generate pitch waypoints with smooth interpolation
  for (double t = pitch_dur; t < time_dur + pitch_dur; t += pitch_dur) {
    if (t > time_dur) {
      way_pts_pitch.push_back(pitch_ep);
      break;
    }
    // Linear interpolation for pitch waypoints
    double alpha = t / time_dur;
    double interpolated_pitch = pitch_sp + alpha * (pitch_ep - pitch_sp);
    interpolated_pitch = std::max(-M_PI/3, std::min(M_PI/3, interpolated_pitch));
    way_pts_pitch.push_back(interpolated_pitch);
  }

  iniStatePitch << Eigen::Vector3d(pitch_sp, 0.0, 0.0), Eigen::Vector3d(pitch_sv, 0.0, 0.0), Eigen::Vector3d(pitch_sa, 0.0, 0.0);
  finStatePitch << Eigen::Vector3d(pitch_ep, 0.0, 0.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

  gcopter::GCOPTER_PolytopeSFC gcopter_pitch;
  if (!gcopter_pitch.setup_yaw(gcopter_config_->yaw_rho_vis, gcopter_config_->integralIntervs)) {
    cout << "setup_pitch failed!" << endl;
    return false;
  }

  int pieceNUM = way_pts_pitch.size() - 2;
  if (pieceNUM <= 1) {
    opt_times_Pitch.resize(2);
    opt_times_Pitch[0] = time_dur / 2.0;
    opt_times_Pitch[1] = time_dur / 2.0;
    wpsPitch.resize(3, 1);
    wpsPitch(0, 0) = (way_pts_pitch[0] + way_pts_pitch[1]) / 2.0;
    wpsPitch(1, 0) = 0.0;
    wpsPitch(2, 0) = 0.0;
    pieceNUM = 2;
  } else {
    opt_times_Pitch.resize(way_pts_pitch.size() - 2);
    for (int i = 0; i < way_pts_pitch.size() - 3; i++) {
      opt_times_Pitch[i] = pitch_dur;
    }
    opt_times_Pitch[way_pts_pitch.size() - 3] = time_dur - (way_pts_pitch.size() - 3) * pitch_dur;
    wpsPitch.resize(3, way_pts_pitch.size() - 3);
    for (int i = 1; i < way_pts_pitch.size() - 2; i++) {
      wpsPitch(0, i - 1) = way_pts_pitch[i];
      wpsPitch(1, i - 1) = 0.0;
      wpsPitch(2, i - 1) = 0.0;
    }
  }

  local_data_.minco_pitch_traj_.clear();
  if (std::isinf(gcopter_pitch.optimize_yaw(iniStatePitch, finStatePitch, pieceNUM, wpsPitch, opt_times_Pitch, local_data_.minco_pitch_traj_))) {
    std::cout << "optimize pitch failed!" << std::endl;
    return false;
  }
  return true;
}

// ============================================================================
// Simple trajectory generation helpers (no optimization, just interpolation)
// ============================================================================

bool FastPlannerManager::GenerateSimpleAngleTraj(double start_angle, double start_vel, double start_acc,
                                                  double end_angle, double end_vel, double end_acc,
                                                  double duration,
                                                  Trajectory<5> &traj) {
  // Use MINCO_S3NU with 2 pieces for smooth quintic interpolation
  Eigen::Matrix3d iniState, finState;
  iniState << Eigen::Vector3d(start_angle, 0.0, 0.0),
              Eigen::Vector3d(start_vel, 0.0, 0.0),
              Eigen::Vector3d(start_acc, 0.0, 0.0);
  finState << Eigen::Vector3d(end_angle, 0.0, 0.0),
              Eigen::Vector3d(end_vel, 0.0, 0.0),
              Eigen::Vector3d(end_acc, 0.0, 0.0);

  minco::MINCO_S3NU minco;
  minco.setConditions(iniState, finState, 2);

  Eigen::Matrix3Xd waypoints(3, 1);
  waypoints(0, 0) = (start_angle + end_angle) / 2.0;
  waypoints(1, 0) = 0.0;
  waypoints(2, 0) = 0.0;

  Eigen::VectorXd times(2);
  times << duration / 2.0, duration / 2.0;

  minco.setParameters(waypoints, times);
  minco.getTrajectory(traj);
  return true;
}

bool FastPlannerManager::GenerateSimplePositionTraj(const Eigen::Vector3d &start,
                                                     const Eigen::Vector3d &end,
                                                     double duration,
                                                     Trajectory<7> &traj) {
  // Use MINCO_S4NU with 2 pieces for smooth septic interpolation
  // Note: This function assumes initial velocity/acceleration/jerk are zero (static case)
  // For non-static case, the caller should use the full optimization path
  Eigen::Matrix<double, 3, 4> iniState, finState;
  iniState << start, Eigen::Vector3d::Zero(),
              Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
  finState << end, Eigen::Vector3d::Zero(),
              Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

  minco::MINCO_S4NU minco;
  minco.setConditions(iniState, finState, 2);

  Eigen::Matrix3Xd waypoints(3, 1);
  waypoints.col(0) = (start + end) / 2.0;

  Eigen::VectorXd times(2);
  times << duration / 2.0, duration / 2.0;

  minco.setParameters(waypoints, times);
  minco.getTrajectory(traj);
  return true;
}

// ============================================================================
// New trajectory planning functions (separated position/yaw/pitch)
// ============================================================================

bool FastPlannerManager::PositionTrajPlan(const vector<Eigen::Vector3f> &path,
                                           const ViewPoint5D &target_viewpoint,
                                           bool is_static,
                                           double &duration,
                                           double &end_yaw,
                                           Trajectory<7> &traj) {
  // === 边界检查：路径只有 1 个点时，不应该调用此函数 ===
  if (path.size() < 2) {
    ROS_WARN("PositionTrajPlan: path size %zu < 2, should not be called", path.size());
    duration = 0.0;
    end_yaw = target_viewpoint.yaw;
    return false;
  }

  // === Path shortening logic (moved from planExploreTraj) ===
  vector<Eigen::Vector3d> path_shorten;
  bool use_shorten_path = false;
  int end_idx = path.size() - 1;

  int i = 0;
  int j = 0;
  for (j = path.size() - 1; j > 0; j--) {
    if ((path[j] - path[0]).norm() <= max_traj_len_ / 2.0)
      break;
  }
  double len = 0.0;
  for (i = 1; i < (int)path.size();) {
    len += (path[i] - path[i - 1]).norm();
    if (len > max_traj_len_ || i == (int)path.size() - 1) {
      break;
    }
    i++;
  }
  end_idx = max(i, j);
  // 确保 end_idx 不越界
  if (end_idx >= (int)path.size()) {
    end_idx = path.size() - 1;
  }
  if (end_idx < (int)path.size() - 1) {
    use_shorten_path = true;
  } else {
    use_shorten_path = false;
  }
  for (int k = 0; k <= end_idx; k++) {
    path_shorten.emplace_back(path[k].cast<double>());
  }

  // === Compute end_yaw based on path direction ===
  // 注意：使用路径方向计算的 yaw 需要取反以匹配执行端坐标系
  if (use_shorten_path) {
    Eigen::Vector3f fwd_dir = path[end_idx] - path[end_idx - 1];
    if (fwd_dir.x() * fwd_dir.x() + fwd_dir.y() * fwd_dir.y() >
        fwd_dir.z() * fwd_dir.z()) {
      end_yaw = -atan2(fwd_dir.y(), fwd_dir.x());
    } else {
      end_yaw = target_viewpoint.yaw;
    }
  } else {
    end_yaw = target_viewpoint.yaw;
  }

  Eigen::Vector3d start_pos = path_shorten.front();
  Eigen::Vector3d end_pos = path_shorten.back();
  double dist = (end_pos - start_pos).norm();
  if(end_pos.x() == 0.0 && end_pos.y() == 0.0){
    ROS_ERROR("BUG: end pos is zero");
  }

  // === If close AND static, use simple interpolation ===
  // (Non-static case still needs optimization to handle current velocity/acceleration)
  if (dist < POS_CLOSE_THRESHOLD && is_static) {
    duration = std::max(0.5, dist / MAX_VEL_SIMPLE);  // Proportional duration, min 0.5s
    ROS_ERROR("PositionTrajPlan: using simple interpolation");
    return GenerateSimplePositionTraj(start_pos, end_pos, duration, traj);
  }else{
    ROS_INFO("not using simple interpolation, distance: %f", dist);
    ROS_INFO("start pos: [%f, %f, %f]", start_pos.x(), start_pos.y(), start_pos.z());
    ROS_INFO("end pos: [%f, %f, %f]", end_pos.x(), end_pos.y(), end_pos.z());
  }

  // === Otherwise, run full GCOPTER optimization ===
  // Compute bounding box
  Eigen::Vector3f min_bd, max_bd;
  for (int k = 0; k < 3; k++) {
    min_bd[k] = path_shorten[0][k];
    max_bd[k] = path_shorten[0][k];
  }
  for (const Eigen::Vector3d &waypoint : path_shorten) {
    for (int k = 0; k < 3; k++) {
      if (waypoint[k] < min_bd[k]) {
        min_bd[k] = waypoint[k];
      }
      if (waypoint[k] > max_bd[k]) {
        max_bd[k] = waypoint[k];
      }
    }
  }
  for (int k = 0; k < 2; k++) {
    min_bd[k] = (min_bd[k] - 3.0);
    max_bd[k] = (max_bd[k] + 3.0);
  }
  min_bd[2] -= 1.0;
  max_bd[2] += 1.0;

  // Box search and downsampling
  PointVector Searched_Points;
  lidar_map_interface_->boxSearch(min_bd, max_bd, Searched_Points);

  std::vector<Eigen::Vector3d> surf_points;
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZ>);
  cloud_origin->points = Searched_Points;
  sor.setInputCloud(cloud_origin);
  sor.setLeafSize(0.2, 0.2, 0.2);
  sor.filter(*cloud_tmp);

  surf_points.reserve(cloud_tmp->points.size());
  for (const pcl::PointXYZ &point : cloud_tmp->points) {
    surf_points.emplace_back(point.x, point.y, point.z);
  }

  // Generate flight corridor
  std::vector<Eigen::MatrixX4d> hPolys;
  sfc_gen::convexCover(gcopter_viz_, path_shorten, surf_points,
                       min_bd.cast<double>(), max_bd.cast<double>(), 7.0,
                       gcopter_config_->corridor_size, hPolys, 1e-6,
                       gcopter_config_->dilateRadiusSoft);

  // Set initial and final states based on is_static
  Eigen::Matrix<double, 3, 4> iniState;
  Eigen::Matrix<double, 3, 4> finState;
  double time_now = (ros::Time::now() - local_data_.start_time_).toSec();

  if (is_static) {
    // Static: zero velocity, acceleration, jerk
    iniState << local_data_.curr_pos_, Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
  } else {
    // Non-static: get current state from trajectory
    time_now = time_now > local_data_.duration_ ? local_data_.duration_ : time_now;
    Eigen::Vector3d current_pose = local_data_.minco_traj_.getPos(time_now);
    Eigen::Vector3d curr_vel = local_data_.minco_traj_.getVel(time_now);
    Eigen::Vector3d curr_acc = local_data_.minco_traj_.getAcc(time_now);
    Eigen::Vector3d curr_jerk = local_data_.minco_traj_.getJer(time_now);
    iniState << current_pose, curr_vel, curr_acc, curr_jerk;
  }

  // Check if initial state is in corridor
  Eigen::Vector4d bh;
  bh << iniState.topLeftCorner<3, 1>(), 1.0;
  int start_idx = -1;
  for (int k = hPolys.size() - 1; k >= 0; k--) {
    Eigen::MatrixX4d hp = hPolys[k];
    if ((((hp * bh).array() > -1.0e-6).cast<int>().sum() <= 0)) {
      start_idx = k;
      break;
    }
  }
  if (start_idx == -1) {
    ROS_ERROR("PositionTrajPlan: current position not in corridor");
    return false;
  }
  if (start_idx != 0) {
    hPolys.erase(hPolys.begin(), hPolys.begin() + start_idx);
  }
  sfc_gen::shortCut(hPolys);

  if (hPolys.size() < 2) {
    ROS_ERROR("PositionTrajPlan: hPolys size < 2");
    return false;
  }

  // Check overlap of polytopes
  int front = 0;
  int back = 1;
  while (back < hPolys.size() - 1) {
    bool overlap = geo_utils::overlap(hPolys[front], hPolys[back], 1e-2);
    if (overlap) {
      front += 1;
      back += 1;
    } else {
      break;
    }
  }

  if (front != hPolys.size() - 2) {
    ROS_WARN("PositionTrajPlan: front != hPolys.size() - 2");
    Eigen::Vector3d inner;
    geo_utils::findInterior(hPolys[front], inner);
    finState << inner, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero();
    hPolys.resize(front + 1);
    gcopter_viz_->visualizePolytope(hPolys, true);
  } else {
    finState << path_shorten.back(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
    gcopter_viz_->visualizePolytope(hPolys);
  }
  gcopter_viz_->visualizeRoute(path);

  // Setup GCOPTER optimizer
  gcopter::GCOPTER_PolytopeSFC gcopter;
  Eigen::VectorXd magnitudeBounds(5);
  Eigen::VectorXd penaltyWeights(5);
  Eigen::VectorXd physicalParams(6);
  magnitudeBounds(0) = gcopter_config_->maxVelMag;
  magnitudeBounds(1) = gcopter_config_->maxBdrMag;
  magnitudeBounds(2) = gcopter_config_->maxTiltAngle;
  magnitudeBounds(3) = gcopter_config_->minThrust;
  magnitudeBounds(4) = gcopter_config_->maxThrust;
  penaltyWeights(0) = (gcopter_config_->chiVec)[0];
  penaltyWeights(1) = (gcopter_config_->chiVec)[1];
  penaltyWeights(2) = (gcopter_config_->chiVec)[2];
  penaltyWeights(3) = (gcopter_config_->chiVec)[3];
  penaltyWeights(4) = (gcopter_config_->chiVec)[4];
  physicalParams(0) = gcopter_config_->vehicleMass;
  physicalParams(1) = gcopter_config_->gravAcc;
  physicalParams(2) = gcopter_config_->horizDrag;
  physicalParams(3) = gcopter_config_->vertDrag;
  physicalParams(4) = gcopter_config_->parasDrag;
  physicalParams(5) = gcopter_config_->speedEps;
  const int quadratureRes = gcopter_config_->integralIntervs;

  if (!gcopter.setup(
          gcopter_config_->weightT, gcopter_config_->dilateRadiusSoft, iniState,
          finState, hPolys, INFINITY, gcopter_config_->smoothingEps,
          quadratureRes, magnitudeBounds, penaltyWeights, physicalParams)) {
    ROS_ERROR("PositionTrajPlan: GCOPTER setup failed");
    return false;
  }

  // Calculate time lower bound
  double time_lb;
  calculateTimelb(path_shorten, local_data_.curr_yaw_, end_yaw, time_lb);

  // Optimize trajectory
  traj.clear();
  if (std::isinf(gcopter.optimize(traj, gcopter_config_->relCostTol, time_lb))) {
    ROS_ERROR("PositionTrajPlan: optimization failed");
    return false;
  }

  duration = traj.getTotalDuration();

  if (traj.getPieceNum() <= 0) {
    ROS_ERROR("PositionTrajPlan: trajectory is empty");
    return false;
  }

  return true;
}

bool FastPlannerManager::YawTrajPlan(double start_yaw, double end_yaw,
                                      bool is_static,
                                      double duration,
                                      Trajectory<5> &traj) {
  (void)is_static;  // Odometry delay makes static/active distinction less reliable here.
  // Prefer the predicted state from the last trajectory to avoid delayed odom.
  double yaw_sp = start_yaw;
  double yaw_sv = 0.0;
  double yaw_sa = 0.0;
  bool has_prev_traj = local_data_.minco_yaw_traj_.getPieceNum() > 0 &&
                       local_data_.start_time_.toSec() > 0.0;
  if (has_prev_traj) {
    double query_time = (ros::Time::now() - local_data_.start_time_).toSec();
    if (!std::isfinite(query_time)) {
      query_time = 0.0;
    }
    query_time = std::max(0.0, query_time);
    double traj_total = local_data_.minco_yaw_traj_.getTotalDuration();
    if (traj_total > 0.0) {
      query_time = std::min(query_time, traj_total);
      yaw_sp = local_data_.minco_yaw_traj_.getPos(query_time).x();
      yaw_sv = local_data_.minco_yaw_traj_.getVel(query_time).x();
      yaw_sa = local_data_.minco_yaw_traj_.getAcc(query_time).x();
    } else {
      has_prev_traj = false;
    }
  }
  angleLimite(yaw_sp);
  double yaw_diff = end_yaw - yaw_sp;
  angleLimite(yaw_diff);

  // === If close AND static, use simple interpolation ===
  double min_duration = std::max(0.05, duration);
  if (fabs(yaw_diff) < YAW_CLOSE_THRESHOLD || !std::isfinite(yaw_diff)) {
    // Keep at start_yaw (no movement needed)
    ROS_ERROR("YawTrajPlan: using simple interpolation");
    // Use current predicted state if available.
    if (!has_prev_traj) {
      yaw_sp = start_yaw;
    }
    return GenerateSimpleAngleTraj(yaw_sp, yaw_sv, yaw_sa,
                                   yaw_sp, 0.0, 0.0, min_duration, traj);
  }

  // === Otherwise, run MINCO optimization ===
  // If no previous trajectory is available, fall back to the provided start yaw.
  if (!has_prev_traj) {
    yaw_sp = start_yaw;
    yaw_sv = 0.0;
    yaw_sa = 0.0;
  }

  // Compute proportional duration if angle diff requires more time
  double yaw_time =
      std::max(min_duration, fabs(yaw_diff) / MAX_ANGVEL_SIMPLE);

  // Generate yaw waypoints with uniform interpolation
  static double yaw_dur = 0.3;
  vector<double> wp;
  wp.push_back(yaw_sp);

  double actual_yaw_diff = end_yaw - yaw_sp;
  angleLimite(actual_yaw_diff);

  for (double t = yaw_dur; t < yaw_time + yaw_dur; t += yaw_dur) {
    if (t > yaw_time) {
      wp.push_back(yaw_sp + actual_yaw_diff);  // Final point
      break;
    }
    double alpha = t / yaw_time;
    double interpolated_yaw = yaw_sp + alpha * actual_yaw_diff;
    wp.push_back(interpolated_yaw);
  }

  double yaw_ep = wp.back();

  Eigen::Matrix3d iniStateYaw, finStateYaw;
  iniStateYaw << Eigen::Vector3d(yaw_sp, 0.0, 0.0),
      Eigen::Vector3d(yaw_sv, 0.0, 0.0), Eigen::Vector3d(yaw_sa, 0.0, 0.0);
  finStateYaw << Eigen::Vector3d(yaw_ep, 0.0, 0.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero();

  gcopter::GCOPTER_PolytopeSFC gcopter_yaw;
  if (!gcopter_yaw.setup_yaw(gcopter_config_->yaw_rho_vis,
                             gcopter_config_->integralIntervs)) {
    ROS_ERROR("YawTrajPlan: setup_yaw failed");
    return false;
  }

  int pieceNUM = wp.size() - 2;
  Eigen::MatrixXd wpsYaw;
  Eigen::VectorXd opt_times_Yaw;

  if (pieceNUM <= 1) {
    opt_times_Yaw.resize(2);
    opt_times_Yaw[0] = yaw_time / 2.0;
    opt_times_Yaw[1] = yaw_time / 2.0;
    wpsYaw.resize(3, 1);
    wpsYaw(0, 0) = (wp[0] + wp[1]) / 2.0;
    wpsYaw(1, 0) = 0.0;
    wpsYaw(2, 0) = 0.0;
    pieceNUM = 2;
  } else {
    opt_times_Yaw.resize(wp.size() - 2);
    for (int k = 0; k < wp.size() - 3; k++) {
      opt_times_Yaw[k] = yaw_dur;
    }
    opt_times_Yaw[wp.size() - 3] = yaw_time - (wp.size() - 3) * yaw_dur;
    wpsYaw.resize(3, wp.size() - 3);
    for (int k = 1; k < wp.size() - 2; k++) {
      wpsYaw(0, k - 1) = wp[k];
      wpsYaw(1, k - 1) = 0.0;
      wpsYaw(2, k - 1) = 0.0;
    }
  }

  traj.clear();
  if (std::isinf(gcopter_yaw.optimize_yaw(iniStateYaw, finStateYaw, pieceNUM,
                                          wpsYaw, opt_times_Yaw, traj))) {
    ROS_ERROR("YawTrajPlan: optimization failed");
    return false;
  }
  return true;
}

bool FastPlannerManager::PitchTrajPlan(double start_pitch, double end_pitch,
                                        bool is_static,
                                        double duration,
                                        Trajectory<5> &traj) {
  (void)is_static;
  // Prefer the predicted state from the last trajectory to avoid delayed odom.
  double pitch_sp = start_pitch;
  double pitch_sv = 0.0;
  double pitch_sa = 0.0;
  bool has_prev_traj = local_data_.minco_pitch_traj_.getPieceNum() > 0 &&
                       local_data_.start_time_.toSec() > 0.0;
  if (has_prev_traj) {
    double query_time = (ros::Time::now() - local_data_.start_time_).toSec();
    if (!std::isfinite(query_time)) {
      query_time = 0.0;
    }
    query_time = std::max(0.0, query_time);
    double traj_total = local_data_.minco_pitch_traj_.getTotalDuration();
    if (traj_total > 0.0) {
      query_time = std::min(query_time, traj_total);
      pitch_sp = local_data_.minco_pitch_traj_.getPos(query_time).x();
      pitch_sv = local_data_.minco_pitch_traj_.getVel(query_time).x();
      pitch_sa = local_data_.minco_pitch_traj_.getAcc(query_time).x();
    } else {
      has_prev_traj = false;
    }
  }

  // Clamp the command pitch to a safe range.
  end_pitch = std::max(-M_PI / 3, std::min(M_PI / 3, end_pitch));
  double pitch_diff = fabs(end_pitch - pitch_sp);

  // === If close AND static, use simple interpolation ===
  double min_duration = std::max(0.05, duration);
  if (pitch_diff < PITCH_CLOSE_THRESHOLD || !std::isfinite(pitch_diff)) {
    // Keep at start_pitch (no movement needed)
    ROS_ERROR("PitchTrajPlan: using simple interpolation");
    if (!has_prev_traj) {
      pitch_sp = start_pitch;
    }
    return GenerateSimpleAngleTraj(pitch_sp, pitch_sv, pitch_sa,
                                   pitch_sp, 0.0, 0.0, min_duration, traj);
  }

  // === Otherwise, run MINCO optimization ===
  if (!has_prev_traj) {
    pitch_sp = start_pitch;
    pitch_sv = 0.0;
    pitch_sa = 0.0;
  }

  // Compute proportional duration if pitch diff requires more time
  double pitch_time =
      std::max(min_duration, pitch_diff / MAX_ANGVEL_SIMPLE);

  // Generate pitch waypoints with uniform interpolation
  static double pitch_dur = 0.15;
  vector<double> way_pts_pitch;
  way_pts_pitch.push_back(pitch_sp);

  for (double t = pitch_dur; t < pitch_time + pitch_dur; t += pitch_dur) {
    if (t > pitch_time) {
      way_pts_pitch.push_back(end_pitch);
      break;
    }
    double alpha = t / pitch_time;
    double interpolated_pitch = pitch_sp + alpha * (end_pitch - pitch_sp);
    // Clamp pitch to [-π/3, π/3]
    interpolated_pitch = std::max(-M_PI / 3, std::min(M_PI / 3, interpolated_pitch));
    way_pts_pitch.push_back(interpolated_pitch);
  }

  Eigen::Matrix3d iniStatePitch, finStatePitch;
  iniStatePitch << Eigen::Vector3d(pitch_sp, 0.0, 0.0),
      Eigen::Vector3d(pitch_sv, 0.0, 0.0), Eigen::Vector3d(pitch_sa, 0.0, 0.0);
  finStatePitch << Eigen::Vector3d(end_pitch, 0.0, 0.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero();

  gcopter::GCOPTER_PolytopeSFC gcopter_pitch;
  if (!gcopter_pitch.setup_yaw(gcopter_config_->yaw_rho_vis,
                               gcopter_config_->integralIntervs)) {
    ROS_ERROR("PitchTrajPlan: setup failed");
    return false;
  }

  int pieceNUM = way_pts_pitch.size() - 2;
  Eigen::MatrixXd wpsPitch;
  Eigen::VectorXd opt_times_Pitch;

  if (pieceNUM <= 1) {
    opt_times_Pitch.resize(2);
    opt_times_Pitch[0] = pitch_time / 2.0;
    opt_times_Pitch[1] = pitch_time / 2.0;
    wpsPitch.resize(3, 1);
    wpsPitch(0, 0) = (way_pts_pitch[0] + way_pts_pitch[1]) / 2.0;
    wpsPitch(1, 0) = 0.0;
    wpsPitch(2, 0) = 0.0;
    pieceNUM = 2;
  } else {
    opt_times_Pitch.resize(way_pts_pitch.size() - 2);
    for (int k = 0; k < way_pts_pitch.size() - 3; k++) {
      opt_times_Pitch[k] = pitch_dur;
    }
    opt_times_Pitch[way_pts_pitch.size() - 3] =
        pitch_time - (way_pts_pitch.size() - 3) * pitch_dur;
    wpsPitch.resize(3, way_pts_pitch.size() - 3);
    for (int k = 1; k < way_pts_pitch.size() - 2; k++) {
      wpsPitch(0, k - 1) = way_pts_pitch[k];
      wpsPitch(1, k - 1) = 0.0;
      wpsPitch(2, k - 1) = 0.0;
    }
  }

  traj.clear();
  if (std::isinf(gcopter_pitch.optimize_yaw(iniStatePitch, finStatePitch,
                                            pieceNUM, wpsPitch, opt_times_Pitch,
                                            traj))) {
    ROS_ERROR("PitchTrajPlan: optimization failed");
    return false;
  }
  return true;
}

void FastPlannerManager::polyPitchTraj2ROSMsg(traj_utils::PolyTraj &poly_msg,
                                              const ros::Time &start_time) {
  Eigen::VectorXd durs = local_data_.minco_pitch_traj_.getDurations();
  int piece_num = local_data_.minco_pitch_traj_.getPieceNum();
  poly_msg.drone_id = 0;
  poly_msg.traj_id = local_data_.traj_id_;
  poly_msg.start_time = start_time;
  poly_msg.order = 5;
  poly_msg.duration.resize(piece_num);
  poly_msg.coef_x.resize(6 * piece_num);
  poly_msg.coef_y.resize(6 * piece_num);
  poly_msg.coef_z.resize(6 * piece_num);
  for (int i = 0; i < piece_num; ++i) {
    poly_msg.duration[i] = durs(i);
    Eigen::Matrix<double, 3, 6> cMat =
        local_data_.minco_pitch_traj_.pieces[i].getCoeffMat();
    int i6 = i * 6;
    for (int j = 0; j < 6; j++) {
      poly_msg.coef_x[i6 + j] = cMat(0, j);
      poly_msg.coef_y[i6 + j] = cMat(1, j);
      poly_msg.coef_z[i6 + j] = cMat(2, j);
    }
  }
}

void FastPlannerManager::visualizeTrajWithCamera() {
  if (local_data_.minco_traj_.getPieceNum() <= 0 ||
      local_data_.minco_yaw_traj_.getPieceNum() <= 0 ||
      local_data_.minco_pitch_traj_.getPieceNum() <= 0) {
    return;
  }

  visualization_msgs::MarkerArray marker_array;

  // Sample points along the trajectory
  double sample_dt = 0.2;  // Sample every 0.2 seconds
  int sample_count = 0;

  for (double t = 0.0; t <= local_data_.duration_; t += sample_dt) {
    // Get position from position trajectory
    Eigen::Vector3d pos = local_data_.minco_traj_.getPos(t);

    // Get yaw from yaw trajectory
    double yaw = 0.0;
    if (t <= local_data_.minco_yaw_traj_.getTotalDuration()) {
      yaw = local_data_.minco_yaw_traj_.getPos(t).x();
    } else {
      yaw = local_data_.minco_yaw_traj_.getPos(local_data_.minco_yaw_traj_.getTotalDuration()).x();
    }

    // Get pitch from pitch trajectory
    double pitch = 0.0;
    if (t <= local_data_.minco_pitch_traj_.getTotalDuration()) {
      pitch = local_data_.minco_pitch_traj_.getPos(t).x();
    } else {
      pitch = local_data_.minco_pitch_traj_.getPos(local_data_.minco_pitch_traj_.getTotalDuration()).x();
    }

    // Create camera orientation marker
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.ns = "camera_orientation";
    marker.id = sample_count;
    marker.type = visualization_msgs::Marker::ARROW;

    if (sample_count == 0) {
      marker.action = visualization_msgs::Marker::DELETEALL;
    } else {
      marker.action = visualization_msgs::Marker::ADD;
    }

    // Set position
    marker.pose.position.x = pos.x();
    marker.pose.position.y = pos.y();
    marker.pose.position.z = pos.z();

    // Set orientation based on yaw and pitch
    tf::Quaternion quat;
    quat.setRPY(0, pitch, yaw);  // Roll=0, Pitch=pitch, Yaw=yaw
    marker.pose.orientation.x = quat.x();
    marker.pose.orientation.y = quat.y();
    marker.pose.orientation.z = quat.z();
    marker.pose.orientation.w = quat.w();

    // Set arrow properties
    marker.scale.x = 0.8; // arrow length
    marker.scale.y = 0.08; // arrow width
    marker.scale.z = 0.08; // arrow height

    // Color: blue for camera orientation
    marker.color.r = 0.0;
    marker.color.g = 0.5;
    marker.color.b = 1.0;
    marker.color.a = 0.8;

    marker_array.markers.push_back(marker);
    sample_count++;
  }

  // Publish the marker array
  yaw_waypoints_pub.publish(marker_array);

  // Also call the original trajectory visualization
  gcopter_viz_->visualize(local_data_.minco_traj_, gcopter_config_->maxVelMag);
}

void FastPlannerManager::visualizeWaypointsOptStatus(
    const Eigen::Matrix3Xd& wps, const std::vector<bool>& opt_indi)
{
  visualization_msgs::MarkerArray marker_array;

  // First marker to delete all previous markers
  visualization_msgs::Marker delete_marker;
  delete_marker.header.frame_id = "world";
  delete_marker.header.stamp = ros::Time::now();
  delete_marker.ns = "waypoints_opt_status";
  delete_marker.action = visualization_msgs::Marker::DELETEALL;
  marker_array.markers.push_back(delete_marker);

  // Create markers for each waypoint
  for (int i = 0; i < wps.cols(); i++) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.ns = "waypoints_opt_status";
    marker.id = i + 1;  // Start from 1 since 0 is for delete
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;

    // Set position
    marker.pose.position.x = wps(0, i);
    marker.pose.position.y = wps(1, i);
    marker.pose.position.z = wps(2, i);
    marker.pose.orientation.w = 1.0;

    if (opt_indi[i]) {
      // Optimizable waypoint: BLUE, smaller size
      marker.scale.x = 0.35;
      marker.scale.y = 0.35;
      marker.scale.z = 0.35;
      marker.color.r = 0.0;
      marker.color.g = 0.0;
      marker.color.b = 1.0;
      marker.color.a = 0.9;
    } else {
      // Fixed waypoint: GREEN, 
      marker.scale.x = 0.35;
      marker.scale.y = 0.35;
      marker.scale.z = 0.35;
      marker.color.r = 0.0;
      marker.color.g = 1.0;
      marker.color.b = 0.0;
      marker.color.a = 0.9;
    }

    marker_array.markers.push_back(marker);
  }

  // Add text labels for waypoint indices
  for (int i = 0; i < wps.cols(); i++) {
    visualization_msgs::Marker text_marker;
    text_marker.header.frame_id = "world";
    text_marker.header.stamp = ros::Time::now();
    text_marker.ns = "waypoints_indices";
    text_marker.id = i;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;

    text_marker.pose.position.x = wps(0, i);
    text_marker.pose.position.y = wps(1, i);
    text_marker.pose.position.z = wps(2, i) + 0.3;  // Slightly above the sphere
    text_marker.pose.orientation.w = 1.0;

    text_marker.scale.z = 0.2;  // Text height
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.color.a = 1.0;

    text_marker.text = std::to_string(i);
    marker_array.markers.push_back(text_marker);
  }

  waypoints_opt_pub.publish(marker_array);
  ROS_INFO("Visualized %ld waypoints (blue=optimizable, green=fixed)", wps.cols());
}

void FastPlannerManager::visualizeWaypointsGradients(
    const Eigen::Matrix3Xd& wps,
    const Eigen::Matrix3Xd& grads)
{
  visualization_msgs::MarkerArray marker_array;

  // First, add a DELETE_ALL marker to clear previous visualization
  visualization_msgs::Marker delete_marker;
  delete_marker.header.frame_id = "world";
  delete_marker.header.stamp = ros::Time::now();
  delete_marker.ns = "waypoint_gradients";
  delete_marker.action = visualization_msgs::Marker::DELETEALL;
  marker_array.markers.push_back(delete_marker);

  // Scale factor for gradient arrows (make them visible)
  const double arrow_scale = 1.0;  // Adjust to make gradients visible

  for (int i = 0; i < wps.cols(); i++) {
    Eigen::Vector3d grad = grads.col(i);
    double grad_norm = grad.norm();

    // Skip very small gradients
    if (grad_norm < 1e-6) continue;

    visualization_msgs::Marker arrow;
    arrow.header.frame_id = "world";
    arrow.header.stamp = ros::Time::now();
    arrow.ns = "waypoint_gradients";
    arrow.id = i;
    arrow.type = visualization_msgs::Marker::ARROW;
    arrow.action = visualization_msgs::Marker::ADD;

    // Arrow from waypoint position to waypoint + gradient (scaled)
    geometry_msgs::Point start_pt, end_pt;
    start_pt.x = wps(0, i);
    start_pt.y = wps(1, i);
    start_pt.z = wps(2, i);

    // Negative gradient direction (descent direction)
    Eigen::Vector3d grad_dir = -grad.normalized() * std::min(grad_norm * arrow_scale, 1.0);
    end_pt.x = wps(0, i) + grad_dir.x();
    end_pt.y = wps(1, i) + grad_dir.y();
    end_pt.z = wps(2, i) + grad_dir.z();

    arrow.points.push_back(start_pt);
    arrow.points.push_back(end_pt);

    // Set orientation (required for ARROW type)
    arrow.pose.orientation.x = 0.0;
    arrow.pose.orientation.y = 0.0;
    arrow.pose.orientation.z = 0.0;
    arrow.pose.orientation.w = 1.0;

    // Arrow scale: shaft diameter, head diameter, head length (if 0, auto)
    arrow.scale.x = 0.05;  // Shaft diameter
    arrow.scale.y = 0.1;   // Head diameter
    arrow.scale.z = 0.0;   // Head length (auto)

    // Color: red for gradient arrows
    arrow.color.r = 1.0;
    arrow.color.g = 0.0;
    arrow.color.b = 0.0;
    arrow.color.a = 0.9;

    marker_array.markers.push_back(arrow);
  }


  for (int i = 0; i < wps.cols(); i++) {
    double grad_norm = grads.col(i).norm();

    visualization_msgs::Marker text;
    text.header.frame_id = "world";
    text.header.stamp = ros::Time::now();
    text.ns = "gradient_magnitudes";
    text.id = i;
    text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::Marker::ADD;

    text.pose.position.x = wps(0, i);
    text.pose.position.y = wps(1, i);
    text.pose.position.z = wps(2, i) - 0.3;  // Below the waypoint
    text.pose.orientation.w = 1.0;

    text.scale.z = 0.15;  // Text height
    text.color.r = 1.0;
    text.color.g = 0.5;
    text.color.b = 0.0;
    text.color.a = 1.0;

    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f", grad_norm);
    text.text = buf;
    marker_array.markers.push_back(text);
  }

  waypoints_grad_pub.publish(marker_array);
  ROS_INFO("Visualized %ld waypoint gradients", wps.cols());
}

void FastPlannerManager::publishESDF() {
  if (!esdf_valid_ || !oqm_) {
    return;
  }

  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointXYZI pt;

  const double min_dist = -1.0;  // Red zone (inside obstacles)
  const double max_dist = 2.0;   // Blue zone (far from obstacles)

  const float voxel_size = oqm_->getVoxelSize();

  // Get current drone position for z-slicing
  double slice_z = local_data_.curr_pos_(2);  // Use current drone height

  // Iterate through ESDF grid, create horizontal slice at drone height
  for (int x = 0; x < esdf_size_.x(); ++x) {
    for (int y = 0; y < esdf_size_.y(); ++y) {
      // Find z index closest to drone height
      Eigen::Vector3i global_idx_min = esdf_bound_min_ + Eigen::Vector3i(x, y, 0);
      PointType pt_min;
      oqm_->idx2pos(global_idx_min, pt_min);
      Eigen::Vector3d pos_min = pt_min.getVector3fMap().cast<double>();

      int z_slice = std::max(0, std::min(esdf_size_.z() - 1,
                                          static_cast<int>((slice_z - pos_min.z()) / voxel_size)));

      Eigen::Vector3i local_idx(x, y, z_slice);
      Eigen::Vector3i global_idx = esdf_bound_min_ + local_idx;

      // Get world position (voxel center) - use OQM's idx2pos to apply region_origin offset
      PointType pt_idx;
      oqm_->idx2pos(global_idx, pt_idx);
      Eigen::Vector3d pos = pt_idx.getVector3fMap().cast<double>();

      // Get ESDF distance value
      double dist = getDistanceESDF(local_idx);

      // Clamp distance for visualization
      dist = std::min(dist, max_dist);
      dist = std::max(dist, min_dist);

      pt.x = pos.x();
      pt.y = pos.y();
      pt.z = slice_z;  // Fixed height slice

      // Intensity: 0.0 (red/inside) -> 1.0 (blue/far)
      pt.intensity = (dist - min_dist) / (max_dist - min_dist);

      cloud.push_back(pt);
    }
  }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = "world";

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  esdf_pub_.publish(cloud_msg);

  ROS_INFO("Published ESDF slice with %lu points at height %.2f", cloud.points.size(), slice_z);
}

void FastPlannerManager::visualizePathSegments(
    const std::vector<PathSegmentWithView>& path_segments)
{
  visualization_msgs::MarkerArray marker_array;

  // Delete all previous markers
  visualization_msgs::Marker delete_marker;
  delete_marker.header.frame_id = "world";
  delete_marker.header.stamp = ros::Time::now();
  delete_marker.ns = "path_segments";
  delete_marker.action = visualization_msgs::Marker::DELETEALL;
  marker_array.markers.push_back(delete_marker);

  if (path_segments.empty()) {
    path_segments_pub_.publish(marker_array);
    return;
  }

  // Visualize all paths as a single LINE_LIST
  visualization_msgs::Marker path_marker;
  path_marker.header.frame_id = "world";
  path_marker.header.stamp = ros::Time::now();
  path_marker.ns = "path_segments";
  path_marker.id = 0;
  path_marker.type = visualization_msgs::Marker::LINE_LIST;
  path_marker.action = visualization_msgs::Marker::ADD;

  path_marker.pose.orientation.w = 1.0;
  path_marker.scale.x = 0.05;  // Line width

  path_marker.color.r = 0.0;
  path_marker.color.g = 0.8;
  path_marker.color.b = 1.0;
  path_marker.color.a = 0.8;

  for (const auto& seg : path_segments) {
    if (seg.path.size() >= 2) {
      for (size_t i = 1; i < seg.path.size(); ++i) {
        geometry_msgs::Point p1, p2;
        p1.x = seg.path[i-1].x();
        p1.y = seg.path[i-1].y();
        p1.z = seg.path[i-1].z();
        p2.x = seg.path[i].x();
        p2.y = seg.path[i].y();
        p2.z = seg.path[i].z();
        path_marker.points.push_back(p1);
        path_marker.points.push_back(p2);
      }
    }
  }

  marker_array.markers.push_back(path_marker);

  // Visualize path points as spheres
  int point_id = 0;
  for (const auto& seg : path_segments) {
    for (const auto& pt : seg.path) {
      visualization_msgs::Marker sphere_marker;
      sphere_marker.header.frame_id = "world";
      sphere_marker.header.stamp = ros::Time::now();
      sphere_marker.ns = "path_points";
      sphere_marker.id = point_id++;
      sphere_marker.type = visualization_msgs::Marker::SPHERE;
      sphere_marker.action = visualization_msgs::Marker::ADD;

      sphere_marker.pose.position.x = pt.x();
      sphere_marker.pose.position.y = pt.y();
      sphere_marker.pose.position.z = pt.z();
      sphere_marker.pose.orientation.w = 1.0;

      sphere_marker.scale.x = 0.12;
      sphere_marker.scale.y = 0.12;
      sphere_marker.scale.z = 0.12;

      sphere_marker.color.r = 1.0;
      sphere_marker.color.g = 0.6;
      sphere_marker.color.b = 0.0;
      sphere_marker.color.a = 0.9;

      marker_array.markers.push_back(sphere_marker);
    }
  }

  path_segments_pub_.publish(marker_array);
  ROS_INFO("Visualized %lu path segments", path_segments.size());
}


} // namespace fast_planner
