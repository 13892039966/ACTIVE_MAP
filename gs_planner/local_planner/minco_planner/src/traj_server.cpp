#include <gcopter/trajectory.hpp>
#include <misc/visualizer.hpp>
#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float64.h>
#include <traj_utils/PolyTraj.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/tf.h>
#include <fstream>

using namespace Eigen;
using namespace std;
double replan_time_;
ros::Publisher pos_cmd_pub, cmd_vis_pub, traj_pub, tar_pitch_deg_pub;
std::shared_ptr<Trajectory<7>> traj_;
std::shared_ptr<Trajectory<5>> yaw_traj_;
std::shared_ptr<Trajectory<5>> pitch_traj_;
quadrotor_msgs::PositionCommand cmd;
std::vector<Eigen::Matrix<double, 5, 1>> traj_cmd_, traj_real_;
Eigen::Vector3d real_pos_;
double slowly_flip_yaw_target_, slowly_turn_to_center_target_;

bool receive_traj_ = false;
double traj_duration_, t_stop, yaw_traj_duration_, yaw_t_stop, pitch_traj_duration_, pitch_t_stop;
ros::Time start_time_;
ros::Time emergency_stop_time;
int traj_id_;
ros::Time heartbeat_time_(0);
Eigen::Vector3d last_pos_;

double last_yaw_, last_yawdot_;
double last_pitch_, last_pitchdot_;
std::string global_path_file_;

std::pair<double, double> get_yaw(double current_yaw, double t_cur) {
  std::pair<double, double> yaw_yawdot(0, 0);
  if (t_cur > yaw_traj_->getTotalDuration())
    t_cur = yaw_traj_->getTotalDuration();
  Eigen::Vector3d ap = yaw_traj_->getPos(t_cur);
  double next_yaw = ap.x();
  double d_yaw = next_yaw - current_yaw;
  if (d_yaw >= M_PI) {
    d_yaw -= 2 * M_PI;
  }
  if (d_yaw <= -M_PI) {
    d_yaw += 2 * M_PI;
  }
  next_yaw = current_yaw + d_yaw;
  yaw_yawdot.first = next_yaw;
  Eigen::Vector3d av = yaw_traj_->getVel(t_cur);
  yaw_yawdot.second = av(0);
  return yaw_yawdot;
}

std::pair<double, double> get_pitch(double current_pitch, double t_cur) {
  std::pair<double, double> pitch_pitchdot(0, 0);
  if (t_cur > pitch_traj_->getTotalDuration())
    t_cur = pitch_traj_->getTotalDuration();
  Eigen::Vector3d ap = pitch_traj_->getPos(t_cur);
  double next_pitch = ap.x();
  double d_pitch = next_pitch - current_pitch;
  if (d_pitch >= M_PI) {
    d_pitch -= 2 * M_PI;
  }
  if (d_pitch <= -M_PI) {
    d_pitch += 2 * M_PI;
  }
  next_pitch = current_pitch + d_pitch;
  pitch_pitchdot.first = next_pitch;
  Eigen::Vector3d av = pitch_traj_->getVel(t_cur);
  pitch_pitchdot.second = av(0);
  return pitch_pitchdot;
}

void heartbeatCallback(std_msgs::EmptyPtr msg) {
  double fsm_time_cost = (ros::Time::now() - heartbeat_time_).toSec();
  heartbeat_time_ = ros::Time::now();
  static int count;
  static int last_count;
  if (count > 10000)
    count = 0;
  else
    count++;
  if (fsm_time_cost > 0.1) {
    std::cout << count - last_count << "\033[31m [FSM_Spin Time Cost] " << fsm_time_cost
              << "s \033[0m" << std::endl;
    last_count = count;
  }
}

void drawCmd(const Eigen::Vector3d &pos, const Eigen::Vector3d &vec, const int &id,
             const Eigen::Vector4d &color) {
  visualization_msgs::Marker mk_state;
  mk_state.header.frame_id = "world";
  mk_state.header.stamp = ros::Time::now();
  mk_state.id = id;
  mk_state.type = visualization_msgs::Marker::ARROW;
  mk_state.action = visualization_msgs::Marker::ADD;

  mk_state.pose.orientation.w = 1.0;
  mk_state.scale.x = 0.1;
  mk_state.scale.y = 0.2;
  mk_state.scale.z = 0.3;

  geometry_msgs::Point pt;
  pt.x = pos(0);
  pt.y = pos(1);
  pt.z = pos(2);
  mk_state.points.push_back(pt);

  pt.x = pos(0) + vec(0);
  pt.y = pos(1) + vec(1);
  pt.z = pos(2) + vec(2);
  mk_state.points.push_back(pt);

  mk_state.color.r = color(0);
  mk_state.color.g = color(1);
  mk_state.color.b = color(2);
  mk_state.color.a = color(3);

  cmd_vis_pub.publish(mk_state);
}

void polyTrajCallback(traj_utils::PolyTrajPtr msg) {
  if (msg->order != 7) {
    ROS_ERROR("[traj_server] Only support trajectory order equals 7 now!");
    return;
  }
  if (msg->duration.size() * (msg->order + 1) != msg->coef_x.size()) {
    ROS_ERROR("[traj_server] WRONG trajectory parameters, ");
    return;
  }

  int piece_nums = msg->duration.size();
  std::vector<double> dura(piece_nums);
  std::vector<Piece<7>::CoefficientMat> cMats(piece_nums);
  for (int i = 0; i < piece_nums; ++i) {
    int i6 = i * 8;
    cMats[i].row(0) << msg->coef_x[i6 + 0], msg->coef_x[i6 + 1], msg->coef_x[i6 + 2],
    msg->coef_x[i6 + 3], msg->coef_x[i6 + 4], msg->coef_x[i6 + 5], msg->coef_x[i6 + 6],
    msg->coef_x[i6 + 7];
    cMats[i].row(1) << msg->coef_y[i6 + 0], msg->coef_y[i6 + 1], msg->coef_y[i6 + 2],
    msg->coef_y[i6 + 3], msg->coef_y[i6 + 4], msg->coef_y[i6 + 5], msg->coef_y[i6 + 6],
    msg->coef_y[i6 + 7];
    cMats[i].row(2) << msg->coef_z[i6 + 0], msg->coef_z[i6 + 1], msg->coef_z[i6 + 2],
    msg->coef_z[i6 + 3], msg->coef_z[i6 + 4], msg->coef_z[i6 + 5], msg->coef_z[i6 + 6],
    msg->coef_z[i6 + 7];

    dura[i] = msg->duration[i];
  }

  traj_.reset(new Trajectory<7>(dura, cMats));

  start_time_ = msg->start_time;
  traj_duration_ = traj_->getTotalDuration();
  t_stop = traj_duration_;
  traj_id_ = msg->traj_id;

  receive_traj_ = true;
  std::vector<Eigen::Vector3d> a;
}

void polyYawTrajCallback(traj_utils::PolyTrajPtr msg) {
  if (msg->order != 5) {
    ROS_ERROR("[traj_server] Only support trajectory order equals 5 now!");
    return;
  }
  if (msg->duration.size() * (msg->order + 1) != msg->coef_x.size()) {
    ROS_ERROR("[traj_server] WRONG trajectory parameters, ");
    return;
  }
  // std::cout << "receive yaw traj" << yaw_t_stop + start_time_.toSec() - msg->start_time.toSec()
  // << "s to go" << std::endl;

  int piece_nums = msg->duration.size();
  std::vector<double> dura(piece_nums);
  std::vector<Piece<5>::CoefficientMat> cMats(piece_nums);
  for (int i = 0; i < piece_nums; ++i) {
    int i6 = i * 6;
    cMats[i].row(0) << msg->coef_x[i6 + 0], msg->coef_x[i6 + 1], msg->coef_x[i6 + 2],
    msg->coef_x[i6 + 3], msg->coef_x[i6 + 4], msg->coef_x[i6 + 5];
    cMats[i].row(1) << msg->coef_y[i6 + 0], msg->coef_y[i6 + 1], msg->coef_y[i6 + 2],
    msg->coef_y[i6 + 3], msg->coef_y[i6 + 4], msg->coef_y[i6 + 5];
    cMats[i].row(2) << msg->coef_z[i6 + 0], msg->coef_z[i6 + 1], msg->coef_z[i6 + 2],
    msg->coef_z[i6 + 3], msg->coef_z[i6 + 4], msg->coef_z[i6 + 5];

    dura[i] = msg->duration[i];
  }
  yaw_traj_.reset(new Trajectory<5>(dura, cMats));
  yaw_traj_duration_ = yaw_traj_->getTotalDuration();
  yaw_t_stop = yaw_traj_duration_;
}

void polyPitchTrajCallback(traj_utils::PolyTrajPtr msg) {
  if (msg->order != 5) {
    ROS_ERROR("[traj_server] Only support trajectory order equals 5 now!");
    return;
  }
  if (msg->duration.size() * (msg->order + 1) != msg->coef_x.size()) {
    ROS_ERROR("[traj_server] WRONG trajectory parameters, ");
    return;
  }

  int piece_nums = msg->duration.size();
  std::vector<double> dura(piece_nums);
  std::vector<Piece<5>::CoefficientMat> cMats(piece_nums);
  for (int i = 0; i < piece_nums; ++i) {
    int i6 = i * 6;
    cMats[i].row(0) << msg->coef_x[i6 + 0], msg->coef_x[i6 + 1], msg->coef_x[i6 + 2],
    msg->coef_x[i6 + 3], msg->coef_x[i6 + 4], msg->coef_x[i6 + 5];
    cMats[i].row(1) << msg->coef_y[i6 + 0], msg->coef_y[i6 + 1], msg->coef_y[i6 + 2],
    msg->coef_y[i6 + 3], msg->coef_y[i6 + 4], msg->coef_y[i6 + 5];
    cMats[i].row(2) << msg->coef_z[i6 + 0], msg->coef_z[i6 + 1], msg->coef_z[i6 + 2],
    msg->coef_z[i6 + 3], msg->coef_z[i6 + 4], msg->coef_z[i6 + 5];

    dura[i] = msg->duration[i];
  }
  pitch_traj_.reset(new Trajectory<5>(dura, cMats));
  pitch_traj_duration_ = pitch_traj_->getTotalDuration();
  pitch_t_stop = pitch_traj_duration_;
}

void publish_cmd(Vector3d p, Vector3d v, Vector3d a, Vector3d j, double y, double yd, double pit, double pitd) {
  cmd.header.stamp = ros::Time::now();
  cmd.header.frame_id = "world";
  cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
  cmd.trajectory_id = traj_id_;

  cmd.position.x = p(0);
  cmd.position.y = p(1);
  cmd.position.z = p(2);
  cmd.velocity.x = v(0);
  cmd.velocity.y = v(1);
  cmd.velocity.z = v(2);
  cmd.acceleration.x = a(0);
  cmd.acceleration.y = a(1);
  cmd.acceleration.z = a(2);
  cmd.jerk.x = j(0);
  cmd.jerk.y = j(1);
  cmd.jerk.z = j(2);
  cmd.yaw = y;
  cmd.yaw_dot = yd;
  cmd.pitch = pit;
  cmd.pitch_dot = pitd;
  pos_cmd_pub.publish(cmd);
  std_msgs::Float64 tar_pitch_deg_msg;
  tar_pitch_deg_msg.data = pit;
  tar_pitch_deg_pub.publish(tar_pitch_deg_msg);

  last_pos_ = p;
}

void cmdCallback(const ros::TimerEvent &e) {
  /* no publishing before receive traj_ and have heartbeat */
  if (heartbeat_time_.toSec() <= 1e-5) {
    // ROS_ERROR_ONCE("[traj_server] No heartbeat from the planner received");
    return;
  }
  if (!receive_traj_)
    return;

  ros::Time time_now = ros::Time::now();
  static bool printed;
  if ((time_now - heartbeat_time_).toSec() > 0.5) {
    if (!printed) {
      ROS_ERROR("[traj_server] Lost heartbeat from the planner, is it dead?");
      printed = true;
    }
  } else {
    printed = false;
  }

  double t_cur = (time_now - start_time_).toSec();

  Eigen::Vector3d pos(Eigen::Vector3d::Zero()), vel(Eigen::Vector3d::Zero()),
  acc(Eigen::Vector3d::Zero()), jer(Eigen::Vector3d::Zero());
  std::pair<double, double> yaw_yawdot(0, 0);
  std::pair<double, double> pitch_pitchdot(0, 0);

  static ros::Time time_last = ros::Time::now();

  // double traj_dur = traj_duration_;

  // if (traj_duration_ < yaw_traj_duration_)
  //   traj_dur = yaw_traj_duration_;

  // if (t_cur < traj_dur && t_cur >= 0.0) {
  if (t_cur < traj_duration_) {
    pos = traj_->getPos(t_cur);
    vel = traj_->getVel(t_cur);
    acc = traj_->getAcc(t_cur);
    jer = traj_->getJer(t_cur);

  } else {
    pos = last_pos_;
    vel = Eigen::Vector3d::Zero();
    acc = Eigen::Vector3d::Zero();
    jer = Eigen::Vector3d::Zero();
  }

  if (t_cur < yaw_traj_duration_) {
    yaw_yawdot = get_yaw(last_yaw_, t_cur);
  } else {
    yaw_yawdot = std::make_pair(last_yaw_, 0.0);
  }

  if (t_cur < pitch_traj_duration_) {
    pitch_pitchdot = get_pitch(last_pitch_, t_cur);
  } else {
    pitch_pitchdot = std::make_pair(last_pitch_, 0.0);
  }

  time_last = time_now;
  last_yaw_ = yaw_yawdot.first;
  last_pitch_ = pitch_pitchdot.first;
  last_pos_ = pos;

  publish_cmd(pos, vel, acc, jer, yaw_yawdot.first, yaw_yawdot.second, pitch_pitchdot.first, pitch_pitchdot.second);
  if (traj_cmd_.size() == 0) {
    Eigen::Matrix<double, 5, 1> vec;
    vec << pos(0), pos(1), pos(2), last_yaw_, last_pitch_;
    traj_cmd_.emplace_back(vec);
  } else if ((traj_cmd_.back().head<3>() - pos).norm() > 0.05) {
    Eigen::Matrix<double, 5, 1> vec;
    vec << pos(0), pos(1), pos(2), last_yaw_, last_pitch_;
    traj_cmd_.emplace_back(vec);
  }
  // if (traj_cmd_.size() > 10000)
  //   traj_cmd_.erase(traj_cmd_.begin(), traj_cmd_.begin() + 1000);
  drawCmd(pos, vel, 0, Eigen::Vector4d(0, 1, 0, 1));
  // }
}

void odomCallbck(const nav_msgs::Odometry &msg) {
  // if (msg.child_frame_id == "X" || msg.child_frame_id == "O")
  //   return;
  // if (traj_real_.size() == 0) {
  //   traj_real_.push_back(
  //   Eigen::Vector3d(msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z));
  // } else if ((traj_real_.back() - Eigen::Vector3d(msg.pose.pose.position.x,
  //                                                 msg.pose.pose.position.y,
  //                                                 msg.pose.pose.position.z))
  //            .norm() > 0.1)
  //   traj_real_.emplace_back(msg.pose.pose.position.x, msg.pose.pose.position.y,
  //                           msg.pose.pose.position.z);

  // if (traj_real_.size() > 100000)
  //   traj_real_.erase(traj_real_.begin(), traj_real_.begin() + 1000);
}

void displayTrajWithColor(std::vector<Eigen::Matrix<double, 5, 1>> &path, double resolution,
                          Eigen::Vector4d color, int id) {
  visualization_msgs::MarkerArray mk_arr;
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::SPHERE_LIST;
  if (id == 0) {
    mk.ns = "traj_exp";
    mk.action = visualization_msgs::Marker::DELETEALL;
    mk.id = id;
    mk_arr.markers.emplace_back(mk);
  } else {
    mk.ns = "traj_real";
  }

  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;
  mk.pose.orientation.w = 1.0;
  mk.color.r = color(0);
  mk.color.g = color(1);
  mk.color.b = color(2);
  mk.color.a = color(3);
  mk.scale.x = resolution;
  mk.scale.y = resolution;
  mk.scale.z = resolution;
  geometry_msgs::Point pt;
  for (int i = 0; i < int(path.size()); i++) {
    pt.x = path[i](0);
    pt.y = path[i](1);
    pt.z = path[i](2);
    mk.points.push_back(pt);
  }
  mk_arr.markers.push_back(mk);

  // Add yaw+pitch direction arrows
  visualization_msgs::Marker yaw_mk;
  yaw_mk.header.frame_id = "world";
  yaw_mk.header.stamp = ros::Time::now();
  yaw_mk.ns = "yaw_direction";
  yaw_mk.type = visualization_msgs::Marker::ARROW;
  yaw_mk.action = visualization_msgs::Marker::ADD;
  yaw_mk.scale.x = 1.0;  // arrow length
  yaw_mk.scale.y = 0.1;  // arrow width
  yaw_mk.scale.z = 0.1;  // arrow height
  yaw_mk.color.r = 0.0;  // green
  yaw_mk.color.g = 1.0;
  yaw_mk.color.b = 0.0;
  yaw_mk.color.a = 1.0;

  for (int i = 0; i < int(path.size()); i++) {
    yaw_mk.pose.position.x = path[i](0);
    yaw_mk.pose.position.y = path[i](1);
    yaw_mk.pose.position.z = path[i](2);
    double yaw = path[i](3);
    double pitch = path[i](4);  // 5th element is pitch
    tf::Quaternion quat;
    quat.setRPY(0, pitch, yaw);  // combine pitch and yaw
    yaw_mk.pose.orientation.x = quat.x();
    yaw_mk.pose.orientation.y = quat.y();
    yaw_mk.pose.orientation.z = quat.z();
    yaw_mk.pose.orientation.w = quat.w();
    yaw_mk.id = i;
    // mk_arr.markers.push_back(yaw_mk);
  }

  traj_pub.publish(mk_arr);
  ros::Duration(0.001).sleep();
}

void visCallback(const ros::TimerEvent &e) {
  displayTrajWithColor(traj_cmd_, 0.1, Eigen::Vector4d(1,1,1, 1), 0);
  displayTrajWithColor(traj_real_, 0.1, Eigen::Vector4d(0, 0, 1, 1), 1);
}

void replanCallback(const std_msgs::Empty &msg) {
  const double time_out = 0.1;
  ros::Time time_now = ros::Time::now();
  double tmp = (time_now - start_time_).toSec() + replan_time_ + time_out;
  traj_duration_ = std::min(traj_duration_, tmp);
  // yaw_traj_duration_ = std::min(yaw_traj_duration_, tmp);
  // std::cout << "\033[32m [TrajServer] Replan , stop after "
  //           << traj_duration_ - (time_now - start_time_).toSec() << "s \033[0m"
  //           << std::endl;
}

void newCallback(std_msgs::Empty msg) {
  // Clear the executed traj data
  traj_cmd_.clear();
  traj_real_.clear();
}

void writeGlobalPathCallback(const ros::TimerEvent &e) {
  std::ofstream file(global_path_file_, std::ios::out | std::ios::trunc);
  if (!file.is_open()) {
    ROS_ERROR("[traj_server] Failed to open %s for writing", global_path_file_.c_str());
    return;
  }

  for (const auto &point : traj_cmd_) {
    file << point(0) << " " << point(1) << " " << point(2) << "\n";
  }

  file.close();
  // ROS_INFO("[traj_server] Wrote %zu points to %s", traj_cmd_.size(), global_path_file_.c_str());
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "traj_server");
  // ros::NodeHandle node;
  ros::NodeHandle nh("~");
  // ros::Subscriber emergency_sub =
  //     nh.subscribe("/planning/emergency_stop", 10, emergencyStopCb);
  std::string odom_topic;
  nh.getParam("odometry_topic", odom_topic);
  std::string default_global_path_file = "/home/hmq/ACTIVE_MAP/global_path.txt";
  nh.param("global_path_file", global_path_file_, default_global_path_file);
  ROS_INFO("[traj_server] Global path output file: %s", global_path_file_.c_str());

  ros::Subscriber poly_traj_sub = nh.subscribe("/planning/trajectory", 10, polyTrajCallback);
  ros::Subscriber poly_yaw_traj_sub =
  nh.subscribe("/planning/yaw_trajectory", 10, polyYawTrajCallback);
  ros::Subscriber poly_pitch_traj_sub =
  nh.subscribe("/planning/pitch_trajectory", 10, polyPitchTrajCallback);
  ros::Subscriber heartbeat_sub = nh.subscribe("/planning/heartbeat", 10, heartbeatCallback);
  ros::Subscriber odom_sub = nh.subscribe(odom_topic, 50, odomCallbck);
  ros::Subscriber replan_sub = nh.subscribe("/planning/replan", 10, replanCallback);
  ros::Subscriber new_sub = nh.subscribe("planning/new", 10, newCallback);

  nh.param("/fsm/replan_time", replan_time_, 0.1);
  ros::Timer vis_timer = nh.createTimer(ros::Duration(0.25), visCallback);
  ros::Timer write_path_timer = nh.createTimer(ros::Duration(1.0), writeGlobalPathCallback);
  pos_cmd_pub = nh.advertise<quadrotor_msgs::PositionCommand>("/position_cmd", 50);
  tar_pitch_deg_pub = nh.advertise<std_msgs::Float64>("/tar_pitch_deg", 50);
  cmd_vis_pub = nh.advertise<visualization_msgs::Marker>("/planning/position_cmd_vis", 10);
  traj_pub = nh.advertise<visualization_msgs::MarkerArray>("/planning/travel_traj", 10);

  ros::Timer cmd_timer = nh.createTimer(ros::Duration(0.01), cmdCallback);

  last_yaw_ = 0.0;
  last_yawdot_ = 0.0;
  last_pitch_ = 0.0;
  last_pitchdot_ = 0.0;

  ros::Duration(1.0).sleep();

  ROS_INFO("[Traj server]: ready.");

  ros::spin();

  return 0;
}
