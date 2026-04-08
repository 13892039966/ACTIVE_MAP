#ifndef _PLAN_CONTAINER_H_
#define _PLAN_CONTAINER_H_

#include <Eigen/Eigen>
#include <vector>
#include <ros/ros.h>
#include <gcopter/trajectory.hpp>
#include <poly_traj/polynomial_traj.h>

using std::vector;

namespace fast_planner {
class GlobalTrajData {
private:
public:
  PolynomialTraj global_traj_;

  double global_duration_;
  ros::Time global_start_time_;
  double local_start_time_, local_end_time_;
  double time_change_;
  double last_time_inc_;

  GlobalTrajData(/* args */) {
  }

  ~GlobalTrajData() {
  }


};

struct LocalTrajData {
  /* info of generated traj */

  int traj_id_;
  double duration_;
  ros::Time start_time_;
  Eigen::Vector3d start_pos_;
  Eigen::Vector3d curr_pos_;
  Eigen::Vector3d curr_vel_;
  double curr_yaw_;
  double end_yaw_;
  double curr_pitch_;
  double end_pitch_;
  Eigen::Vector3d view_target_;
  Trajectory<7> minco_traj_;
  Trajectory<5> minco_yaw_traj_;
  Trajectory<5> minco_pitch_traj_;

  // 独立 duration (三条轨迹各自的时长)
  double pos_duration_ = 0.0;
  double yaw_duration_ = 0.0;
  double pitch_duration_ = 0.0;

  // 规划标志 (指示哪些轨迹需要发布)
  bool pos_needed_ = true;
  bool yaw_needed_ = true;
  bool pitch_needed_ = true;
};

// structure of trajectory info
struct LocalTrajState {
  Eigen::Vector3d pos, vel, acc;
  double yaw, yawdot;
  int id;
};

class LocalTrajServer {
private:
  LocalTrajData traj1_, traj2_;

public:
  LocalTrajServer(/* args */) {
    traj1_.traj_id_ = 0;
    traj2_.traj_id_ = 0;
  }
  ~LocalTrajServer() {
  }

  void addTraj(const LocalTrajData& traj) {
    if (traj1_.traj_id_ == 0) {
      // receive the first traj, save in traj1
      traj1_ = traj;
    } else {
      traj2_ = traj;
    }
  }

  // bool evaluate(const ros::Time& time, LocalTrajState& traj_state) {
  //   if (traj1_.traj_id_ == 0) {
  //     // not receive traj yet
  //     return false;
  //   }

  //   if (traj2_.traj_id_ != 0 && time > traj2_.start_time_) {
  //     // have traj2 AND time within range of traj2. should use traj2 now
  //     traj1_ = traj2_;
  //     traj2_.traj_id_ = 0;
  //   }

  //   double t_cur = (time - traj1_.start_time_).toSec();
  //   if (t_cur < 0) {
  //     cout << "[Traj server]: invalid time." << endl;
  //     return false;
  //   } else if (t_cur < traj1_.duration_) {
  //     // time within range of traj 1
  //     traj_state.pos = traj1_.position_traj_.evaluateDeBoorT(t_cur);
  //     traj_state.vel = traj1_.velocity_traj_.evaluateDeBoorT(t_cur);
  //     traj_state.acc = traj1_.acceleration_traj_.evaluateDeBoorT(t_cur);
  //     traj_state.yaw = traj1_.yaw_traj_.evaluateDeBoorT(t_cur)[0];
  //     traj_state.yawdot = traj1_.yawdot_traj_.evaluateDeBoorT(t_cur)[0];
  //     traj_state.id = traj1_.traj_id_;
  //     return true;
  //   } else {
  //     traj_state.pos = traj1_.position_traj_.evaluateDeBoorT(traj1_.duration_);
  //     traj_state.vel.setZero();
  //     traj_state.acc.setZero();
  //     traj_state.yaw = traj1_.yaw_traj_.evaluateDeBoorT(traj1_.duration_)[0];
  //     traj_state.yawdot = 0;
  //     traj_state.id = traj1_.traj_id_;
  //     return true;
  //   }
  // }

};
class MidPlanData {
public:
  MidPlanData(/* args */) {
  }
  ~MidPlanData() {
  }

  vector<Eigen::Vector3d> global_waypoints_;

  // initial trajectory segment
  vector<Eigen::Vector3d> local_start_end_derivative_;

  // kinodynamic path
  vector<Eigen::Vector3d> kino_path_;


  // visibility constraint
  vector<Eigen::Vector3d> block_pts_;
  Eigen::MatrixXd ctrl_pts_;


  // heading planning
  vector<vector<Eigen::Vector3d>> frontiers_;
  vector<double> path_yaw_;
  double dt_yaw_;
  double dt_yaw_path_;

};

}  // namespace fast_planner

#endif