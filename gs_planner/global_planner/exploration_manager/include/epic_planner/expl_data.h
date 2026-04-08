#pragma once

#include <Eigen/Eigen>
#include <pointcloud_topo/graph.h>
#include <traj_utils/PolyTraj.h>
#include <vector>
using Eigen::Vector3d;
using std::vector;
using namespace std;

namespace fast_planner {

// ViewPose - 统一的视点结构体 (替代 TopoNode::Ptr)
struct ViewPose {
  Eigen::Vector3f position;  // 视点位置 (x, y, z)
  float yaw;                 // 偏航角 [-π, π], 0 = +X方向
  float pitch;               // 俯仰角 [-60°, 60°], 0 = 水平
  float score;               // 视点得分 (可选，用于调试)

  ViewPose() : position(Eigen::Vector3f::Zero()), yaw(0.0f), pitch(0.0f), score(0.0f) {}

  ViewPose(const Eigen::Vector3f& pos, float y, float p, float s = 0.0f)
      : position(pos), yaw(y), pitch(p), score(s) {}

  // 从 SelectedView 构造 (兼容 OQM)
  // 注意: 需要在使用处包含 observation_quality_manager/global_planner.h
};

struct FSMData {
  // FSM data
  bool trigger_, have_odom_, have_cloud_odom_, static_state_, emergency_replan_,
      use_bubble_a_star_, half_resolution;
  vector<string> state_str_;
  int bb_astar_fail_cnt_, fast_search_fial_cnt_;
  double bb_astar_time_out, fast_search_time_out;
  Eigen::Vector3f odom_pos_, odom_vel_; // odometry state
  Eigen::Quaterniond odom_orient_;
  float odom_yaw_;
  float odom_pitch_;

  Eigen::Vector3d start_pt_, start_vel_, start_acc_, start_yaw_; // start state
  vector<Eigen::Vector3d> start_poss;
  traj_utils::PolyTraj newest_traj_;
  traj_utils::PolyTraj newest_yaw_traj_;
  traj_utils::PolyTraj newest_pitch_traj_;
};

struct FSMParam {
  double replan_thresh_;
  double replan_time_after_traj_start_;
  double replan_time_before_traj_end_;
  double replan_time_; // second
  double emergency_replan_control_error;
  double bubble_a_star_resolution;
};

struct ExplorationData {
  // 全局规划路径 (位置序列)
  vector<Eigen::Vector3f> global_tour_;

  // 精选视点列表 (带朝向)
  vector<ViewPose> selected_views_;

  // 下一个目标视点
  ViewPose next_goal_;
  int next_goal_idx_;  // 在 selected_views_ 中的索引

  // 到下一个目标的路径
  vector<Eigen::Vector3f> path_next_goal_;

  // 可视化用 (保留兼容)
  vector<Vector3d> views_vis1_, views_vis2_;
  vector<Vector3d> centers_, scales_;
};

struct ExplorationParam {
  // params
  int local_viewpoint_num_, global_viewpoint_num_;
  int viewpoint_connection_num_;
  double a_avg_, v_max_, yaw_v_max_, viewpoint_gian_lambda_;
  double w_vdir_, w_yawdir_;
  bool view_graph_;
  string tsp_dir_; // resource dir of tsp solver
};

} // namespace fast_planner
