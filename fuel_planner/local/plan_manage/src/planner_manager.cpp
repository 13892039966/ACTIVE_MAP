// #include <fstream>
#include <plan_manage/planner_manager.h>
#include <plan_env/sdf_map.h>
#include <plan_env/raycast.h>

#include <thread>
#include <limits>
#include <cmath>
#include <algorithm>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>

namespace fast_planner {
namespace {
struct MincoTrajStats {
  double max_vel = 0.0;
  double max_acc = 0.0;
  double max_jerk = 0.0;
  double min_obs_dist = std::numeric_limits<double>::infinity();
  double min_obs_t = 0.0;
};

struct ExploreAcceptanceLimits {
  double vel = std::numeric_limits<double>::infinity();
  double acc = std::numeric_limits<double>::infinity();
  double jerk = std::numeric_limits<double>::infinity();
  double reject_safe_margin = 0.0;
};

double defaultExploreStep(const SDFMap::Ptr& sdf_map) {
  return sdf_map ? std::max(0.05, std::min(0.2, 0.5 * sdf_map->getResolution())) : 0.1;
}

double wrapAngle(double angle) {
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

double unwrapTowards(double reference, double target) {
  return reference + wrapAngle(target - reference);
}

bool sampleCurrentYawState(const LocalTrajData& local_data, Eigen::Vector3d& yaw_state) {
  if (local_data.minco_yaw_traj_.getPieceNum() <= 0 || local_data.start_time_.toSec() <= 0.0) {
    return false;
  }

  double query_time = (ros::Time::now() - local_data.start_time_).toSec();
  if (!std::isfinite(query_time)) query_time = 0.0;
  query_time = std::max(0.0, query_time);

  const double total_duration = local_data.minco_yaw_traj_.getTotalDuration();
  if (total_duration <= 0.0) return false;

  query_time = std::min(query_time, total_duration);
  yaw_state.x() = local_data.minco_yaw_traj_.getPos(query_time).x();
  yaw_state.y() = local_data.minco_yaw_traj_.getVel(query_time).x();
  yaw_state.z() = local_data.minco_yaw_traj_.getAcc(query_time).x();
  return true;
}

bool lbfgsReturnedUsableSolution(const int result) {
  return result >= 0;
}

MincoTrajStats evaluateMincoTrajStats(const Trajectory<7>& traj,
                                      fast_planner::FastPlannerManager* manager,
                                      const double sample_dt) {
  MincoTrajStats stats;
  if (manager == nullptr || traj.getPieceNum() <= 0) return stats;

  const double duration = traj.getTotalDuration();
  const double dt = std::max(0.01, sample_dt);
  for (double t = 0.0; t <= duration + 1e-6; t += dt) {
    const double tc = std::min(duration, t);
    const Eigen::Vector3d pos = traj.getPos(tc);
    stats.max_vel = std::max(stats.max_vel, traj.getVel(tc).norm());
    stats.max_acc = std::max(stats.max_acc, traj.getAcc(tc).norm());
    stats.max_jerk = std::max(stats.max_jerk, traj.getJer(tc).norm());

    double dist = 0.0;
    Eigen::Vector3d grad = Eigen::Vector3d::Zero();
    if (manager->edt_environment_) {
      manager->edt_environment_->evaluateEDTWithGrad(pos, -1.0, dist, grad);
    }
    if (!std::isfinite(dist)) dist = 0.0;
    if (dist < stats.min_obs_dist) {
      stats.min_obs_dist = dist;
      stats.min_obs_t = tc;
    }
  }

  return stats;
}

ExploreAcceptanceLimits getExploreAcceptanceLimits(const fast_planner::PlanParameters& pp) {
  ExploreAcceptanceLimits limits;
  const double soft_safe_margin = std::max(0.05, 0.5 * pp.minco_safe_distance_);
  limits.reject_safe_margin =
      std::max(soft_safe_margin, std::max(0.12, 0.35 * pp.minco_safe_distance_));
  limits.vel =
      (pp.accept_vel_ > 0.0 ? pp.accept_vel_ : std::max(0.2, pp.max_vel_ * 1.05));
  limits.acc =
      (pp.accept_acc_ > 0.0 ? pp.accept_acc_ : std::max(0.2, pp.max_acc_ * 1.05));
  limits.jerk =
      (pp.max_jerk_ > 0.0 ? std::max(0.2, pp.max_jerk_ * 1.1) : std::numeric_limits<double>::infinity());
  return limits;
}

double computeExploreTimeStretch(const MincoTrajStats& stats,
                                 const fast_planner::PlanParameters& pp) {
  const ExploreAcceptanceLimits limits = getExploreAcceptanceLimits(pp);
  double stretch = 1.0;
  if (limits.vel > 1e-6 && std::isfinite(limits.vel) && stats.max_vel > limits.vel) {
    stretch = std::max(stretch, stats.max_vel / limits.vel);
  }
  if (limits.acc > 1e-6 && std::isfinite(limits.acc) && stats.max_acc > limits.acc) {
    stretch = std::max(stretch, std::sqrt(stats.max_acc / limits.acc));
  }
  if (limits.jerk > 1e-6 && std::isfinite(limits.jerk) && stats.max_jerk > limits.jerk) {
    stretch = std::max(stretch, std::cbrt(stats.max_jerk / limits.jerk));
  }
  if (stretch <= 1.02) return 1.0;
  return std::min(3.0, stretch * 1.05);
}

double evaluateExploreProjectionScore(const SDFMap::Ptr& sdf_map,
                                      const EDTEnvironment::Ptr& edt_environment,
                                      const Eigen::Vector3d& candidate,
                                      const Eigen::Vector3d& reference,
                                      double target_clearance) {
  double obs_dist = 0.0;
  Eigen::Vector3d grad = Eigen::Vector3d::Zero();
  if (edt_environment) {
    edt_environment->evaluateEDTWithGrad(candidate, -1.0, obs_dist, grad);
  }
  if (!std::isfinite(obs_dist)) obs_dist = 0.0;

  const double clearance_term =
      std::min(obs_dist, target_clearance) / std::max(1e-3, target_clearance);
  const double extra_clearance_bonus =
      obs_dist > target_clearance
          ? std::min(1.0, (obs_dist - target_clearance) / std::max(0.1, target_clearance))
          : 0.0;
  const double distance_penalty =
      (candidate - reference).norm() / std::max(0.2, target_clearance);

  (void)sdf_map;
  return 3.0 * clearance_term + 0.5 * extra_clearance_bonus - 0.35 * distance_penalty;
}

bool sampleCurrentMincoState(const LocalTrajData& local_data, Eigen::Vector3d& pos,
                             Eigen::Vector3d& vel, Eigen::Vector3d& acc,
                             Eigen::Vector3d& jerk) {
  if (local_data.minco_traj_.getPieceNum() <= 0 || local_data.start_time_.toSec() <= 0.0) {
    return false;
  }

  double query_time = (ros::Time::now() - local_data.start_time_).toSec();
  if (!std::isfinite(query_time)) query_time = 0.0;
  query_time = std::max(0.0, query_time);

  const double total_duration = local_data.minco_traj_.getTotalDuration();
  if (total_duration <= 0.0) return false;

  query_time = std::min(query_time, total_duration);
  pos = local_data.minco_traj_.getPos(query_time);
  vel = local_data.minco_traj_.getVel(query_time);
  acc = local_data.minco_traj_.getAcc(query_time);
  jerk = local_data.minco_traj_.getJer(query_time);
  return true;
}

double inferYawFromPath(const vector<Eigen::Vector3d>& path) {
  for (size_t i = 1; i < path.size(); ++i) {
    const Eigen::Vector3d diff = path[i] - path[i - 1];
    if (diff.head<2>().norm() > 1e-3) {
      return std::atan2(diff.y(), diff.x());
    }
  }
  return 0.0;
}

Eigen::Vector3d computeConstrainedTerminalVelocity(const vector<Eigen::Vector3d>& path,
                                                   const Eigen::Vector3d& requested_vel,
                                                   const fast_planner::PlanParameters& pp) {
  if (path.size() < 2) return Eigen::Vector3d::Zero();

  Eigen::Vector3d dir = Eigen::Vector3d::Zero();
  double last_seg_len = 0.0;
  for (int i = static_cast<int>(path.size()) - 1; i > 0; --i) {
    const Eigen::Vector3d diff = path[i] - path[i - 1];
    if (diff.norm() > 1e-3) {
      dir = diff.normalized();
      last_seg_len = diff.norm();
      break;
    }
  }

  if (dir.squaredNorm() < 1e-6) return Eigen::Vector3d::Zero();

  double speed = requested_vel.dot(dir);
  if (!std::isfinite(speed) || speed <= 0.05) return Eigen::Vector3d::Zero();

  const double vel_cap = std::max(0.2, pp.max_vel_ * 0.8);
  const double acc_cap =
      std::sqrt(std::max(0.0, 2.0 * std::max(0.1, pp.max_acc_) * std::max(0.2, last_seg_len)));
  speed = std::min(speed, std::min(vel_cap, acc_cap));
  return dir * speed;
}

bool buildCoupledMincoYawTraj(const Eigen::VectorXd& raw_times, const Eigen::Vector3d& start_yaw,
                              const double target_yaw, Trajectory<5>& yaw_traj) {
  const int piece_num = raw_times.size();
  if (piece_num <= 0 || !std::isfinite(target_yaw)) return false;

  Eigen::VectorXd times = raw_times;
  for (int i = 0; i < piece_num; ++i) {
    times(i) = std::max(0.03, times(i));
  }

  const double yaw_start = wrapAngle(start_yaw.x());
  const double yaw_end = unwrapTowards(yaw_start, target_yaw);
  const double yaw_diff = yaw_end - yaw_start;

  Eigen::Matrix3d ini_state, fin_state;
  ini_state << Eigen::Vector3d(yaw_start, 0.0, 0.0), Eigen::Vector3d(start_yaw.y(), 0.0, 0.0),
      Eigen::Vector3d(start_yaw.z(), 0.0, 0.0);
  fin_state << Eigen::Vector3d(yaw_end, 0.0, 0.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

  Eigen::Matrix3Xd inner_points(3, std::max(0, piece_num - 1));
  for (int k = 1; k < piece_num; ++k) {
    const double alpha = static_cast<double>(k) / static_cast<double>(piece_num);
    inner_points.col(k - 1) << yaw_start + alpha * yaw_diff, 0.0, 0.0;
  }

  minco::MINCO_S3NU yaw_minco;
  yaw_minco.setConditions(ini_state, fin_state, piece_num);
  yaw_minco.setParameters(inner_points, times);
  yaw_minco.getTrajectory(yaw_traj);
  return true;
}

void subdividePath(vector<Eigen::Vector3d>& path) {
  if (path.size() < 2) return;

  vector<Eigen::Vector3d> dense_path;
  dense_path.reserve(path.size() * 2);
  dense_path.push_back(path.front());
  for (size_t i = 1; i < path.size(); ++i) {
    dense_path.push_back(0.5 * (path[i - 1] + path[i]));
    dense_path.push_back(path[i]);
  }
  path.swap(dense_path);
}

void subdividePathSegments(std::vector<fast_planner::PathSegmentWithYaw>& segments) {
  for (auto& segment : segments) {
    subdividePath(segment.path);
    if (segment.path.empty() || (segment.path.back() - segment.viewpoint).norm() > 1e-3) {
      segment.path.push_back(segment.viewpoint);
    }
  }
}

PolynomialTraj bsplineToPolynomialTraj(NonUniformBspline bspline) {
  const double duration = bspline.getTimeSum();
  const double dt = bspline.getKnotSpan();
  const int piece_num = std::max(1, static_cast<int>(std::round(duration / dt)));

  Eigen::MatrixXd positions(piece_num + 1, 3);
  Eigen::VectorXd times(piece_num);

  for (int i = 0; i <= piece_num; ++i) {
    const double t = std::min(duration, i * dt);
    positions.row(i) = bspline.evaluateDeBoorT(t).transpose();
    if (i < piece_num) {
      const double next_t = std::min(duration, (i + 1) * dt);
      times(i) = std::max(1e-3, next_t - t);
    }
  }

  NonUniformBspline vel_traj = bspline.getDerivative();
  NonUniformBspline acc_traj = vel_traj.getDerivative();

  PolynomialTraj poly_traj;
  PolynomialTraj::waypointsTraj(positions, vel_traj.evaluateDeBoorT(0.0),
                                vel_traj.evaluateDeBoorT(duration),
                                acc_traj.evaluateDeBoorT(0.0),
                                acc_traj.evaluateDeBoorT(duration), times, poly_traj);
  return poly_traj;
}

PolynomialTraj yawBsplineToPolynomialTraj(NonUniformBspline bspline) {
  const double duration = bspline.getTimeSum();
  const double dt = bspline.getKnotSpan();
  const int piece_num = std::max(1, static_cast<int>(std::round(duration / dt)));

  Eigen::MatrixXd positions(piece_num + 1, 3);
  Eigen::VectorXd times(piece_num);
  double yaw_offset = 0.0;
  double prev_yaw = 0.0;

  for (int i = 0; i <= piece_num; ++i) {
    const double t = std::min(duration, i * dt);
    double yaw = bspline.evaluateDeBoorT(t)(0);
    if (i > 0) {
      while (yaw + yaw_offset - prev_yaw > M_PI)
        yaw_offset -= 2.0 * M_PI;
      while (yaw + yaw_offset - prev_yaw < -M_PI)
        yaw_offset += 2.0 * M_PI;
    }
    yaw += yaw_offset;
    positions.row(i) << yaw, 0.0, 0.0;
    prev_yaw = yaw;
    if (i < piece_num) {
      const double next_t = std::min(duration, (i + 1) * dt);
      times(i) = std::max(1e-3, next_t - t);
    }
  }

  NonUniformBspline yawdot_traj = bspline.getDerivative();
  NonUniformBspline yawdotdot_traj = yawdot_traj.getDerivative();
  Eigen::Vector3d start_vel(yawdot_traj.evaluateDeBoorT(0.0)(0), 0.0, 0.0);
  Eigen::Vector3d end_vel(yawdot_traj.evaluateDeBoorT(duration)(0), 0.0, 0.0);
  Eigen::Vector3d start_acc(yawdotdot_traj.evaluateDeBoorT(0.0)(0), 0.0, 0.0);
  Eigen::Vector3d end_acc(yawdotdot_traj.evaluateDeBoorT(duration)(0), 0.0, 0.0);

  PolynomialTraj poly_traj;
  PolynomialTraj::waypointsTraj(positions, start_vel, end_vel, start_acc, end_acc, times, poly_traj);
  return poly_traj;
}

void polynomialTrajToRosMsg(const PolynomialTraj& poly_traj, const int traj_id,
                            const ros::Time& start_time, traj_utils::PolyTraj& poly_msg) {
  const int piece_num = poly_traj.getPieceNum();
  poly_msg.drone_id = 0;
  poly_msg.traj_id = traj_id;
  poly_msg.start_time = start_time;
  poly_msg.order = 5;
  poly_msg.duration.resize(piece_num);
  poly_msg.coef_x.resize(6 * piece_num);
  poly_msg.coef_y.resize(6 * piece_num);
  poly_msg.coef_z.resize(6 * piece_num);

  for (int i = 0; i < piece_num; ++i) {
    const auto& piece = poly_traj.getPiece(i);
    poly_msg.duration[i] = piece.getTime();
    const auto& cx = piece.getCoeffX();
    const auto& cy = piece.getCoeffY();
    const auto& cz = piece.getCoeffZ();
    const int offset = i * 6;
    for (int j = 0; j < 6; ++j) {
      poly_msg.coef_x[offset + j] = cx(j);
      poly_msg.coef_y[offset + j] = cy(j);
      poly_msg.coef_z[offset + j] = cz(j);
    }
  }
}

void mincoTrajToRosMsg(const Trajectory<7>& traj, const int traj_id, const ros::Time& start_time,
                       traj_utils::PolyTraj& poly_msg) {
  const Eigen::VectorXd durs = traj.getDurations();
  const int piece_num = traj.getPieceNum();
  poly_msg.drone_id = 0;
  poly_msg.traj_id = traj_id;
  poly_msg.start_time = start_time;
  poly_msg.order = 7;
  poly_msg.duration.resize(piece_num);
  poly_msg.coef_x.resize(8 * piece_num);
  poly_msg.coef_y.resize(8 * piece_num);
  poly_msg.coef_z.resize(8 * piece_num);
  for (int i = 0; i < piece_num; ++i) {
    poly_msg.duration[i] = durs(i);
    const auto cmat = traj[i].getCoeffMat();
    const int offset = i * 8;
    for (int j = 0; j < 8; ++j) {
      poly_msg.coef_x[offset + j] = cmat(0, j);
      poly_msg.coef_y[offset + j] = cmat(1, j);
      poly_msg.coef_z[offset + j] = cmat(2, j);
    }
  }
}

void mincoYawTrajToRosMsg(const Trajectory<5>& traj, const int traj_id, const ros::Time& start_time,
                          traj_utils::PolyTraj& poly_msg) {
  const Eigen::VectorXd durs = traj.getDurations();
  const int piece_num = traj.getPieceNum();
  poly_msg.drone_id = 0;
  poly_msg.traj_id = traj_id;
  poly_msg.start_time = start_time;
  poly_msg.order = 5;
  poly_msg.duration.resize(piece_num);
  poly_msg.coef_x.resize(6 * piece_num);
  poly_msg.coef_y.resize(6 * piece_num);
  poly_msg.coef_z.resize(6 * piece_num);
  for (int i = 0; i < piece_num; ++i) {
    const auto cmat = traj[i].getCoeffMat();
    poly_msg.duration[i] = durs(i);
    const int offset = i * 6;
    for (int j = 0; j < 6; ++j) {
      poly_msg.coef_x[offset + j] = cmat(0, j);
      poly_msg.coef_y[offset + j] = cmat(1, j);
      poly_msg.coef_z[offset + j] = cmat(2, j);
    }
  }
}

template <int D>
Piece<D> shiftPiece(const Piece<D>& piece, double offset) {
  typedef typename Piece<D>::CoefficientMat CoefficientMat;
  const double clamped_offset = std::max(0.0, std::min(offset, piece.getDuration()));
  Eigen::Matrix<double, 3, D + 1> asc;
  for (int power = 0; power <= D; ++power) {
    asc.col(power) = piece.getCoeffMat().col(D - power);
  }

  Eigen::Matrix<double, 3, D + 1> shifted_asc;
  shifted_asc.setZero();
  for (int new_power = 0; new_power <= D; ++new_power) {
    for (int old_power = new_power; old_power <= D; ++old_power) {
      shifted_asc.col(new_power) +=
          asc.col(old_power) * (std::tgamma(old_power + 1) /
                                (std::tgamma(new_power + 1) * std::tgamma(old_power - new_power + 1))) *
          std::pow(clamped_offset, old_power - new_power);
    }
  }

  CoefficientMat shifted;
  for (int power = 0; power <= D; ++power) {
    shifted.col(D - power) = shifted_asc.col(power);
  }
  return Piece<D>(std::max(1e-3, piece.getDuration() - clamped_offset), shifted);
}

template <int D>
Trajectory<D> sliceTrajectoryFromTime(const Trajectory<D>& traj, double t_from) {
  Trajectory<D> sliced;
  if (traj.getPieceNum() <= 0) return sliced;

  double t = std::max(0.0, std::min(t_from, traj.getTotalDuration()));
  int piece_idx = traj.locatePieceIdx(t);
  sliced.reserve(traj.getPieceNum() - piece_idx);
  sliced.emplace_back(shiftPiece<D>(traj[piece_idx], t));
  for (int i = piece_idx + 1; i < traj.getPieceNum(); ++i) {
    sliced.emplace_back(traj[i]);
  }
  return sliced;
}
}  // namespace

// SECTION interfaces for setup and query

FastPlannerManager::FastPlannerManager() {
}

FastPlannerManager::~FastPlannerManager() {
  std::cout << "des manager" << std::endl;
}

void FastPlannerManager::initPlanModules(ros::NodeHandle& nh) {
  /* read algorithm parameters */

  nh.param("manager/max_vel", pp_.max_vel_, -1.0);
  nh.param("manager/max_acc", pp_.max_acc_, -1.0);
  nh.param("manager/max_jerk", pp_.max_jerk_, -1.0);
  nh.param("manager/accept_vel", pp_.accept_vel_, pp_.max_vel_ + 0.5);
  nh.param("manager/accept_acc", pp_.accept_acc_, pp_.max_acc_ + 0.5);
  nh.param("manager/max_yawdot", pp_.max_yawdot_, -1.0);
  nh.param("manager/dynamic_environment", pp_.dynamic_, -1);
  nh.param("manager/clearance_threshold", pp_.clearance_, -1.0);
  nh.param("manager/local_segment_length", pp_.local_traj_len_, -1.0);
  nh.param("manager/control_points_distance", pp_.ctrl_pt_dist, -1.0);
  nh.param("manager/bspline_degree", pp_.bspline_degree_, 3);
  nh.param("manager/min_time", pp_.min_time_, false);
  nh.param("manager/minco_weight_time", pp_.minco_weight_time_, pp_.minco_weight_time_);
  nh.param("manager/minco_rho_collision", pp_.minco_rho_collision_, pp_.minco_rho_collision_);
  nh.param("manager/minco_rho_path", pp_.minco_rho_path_, pp_.minco_rho_path_);
  nh.param("manager/minco_rho_v", pp_.minco_rho_v_, pp_.minco_rho_v_);
  nh.param("manager/minco_rho_a", pp_.minco_rho_a_, pp_.minco_rho_a_);
  nh.param("manager/minco_safe_distance", pp_.minco_safe_distance_, pp_.minco_safe_distance_);
  nh.param("manager/minco_smooth_epsilon", pp_.minco_smooth_epsilon_,
           pp_.minco_smooth_epsilon_);
  nh.param("manager/minco_anchor_weight", pp_.minco_anchor_weight_, pp_.minco_anchor_weight_);
  nh.param("manager/minco_lbfgs_delta", pp_.minco_lbfgs_delta_, pp_.minco_lbfgs_delta_);
  nh.param("manager/minco_lbfgs_max_iter", pp_.minco_lbfgs_max_iter_,
           pp_.minco_lbfgs_max_iter_);
  nh.param("manager/explore_strict_horizon", pp_.explore_strict_horizon_,
           pp_.explore_strict_horizon_);
  ROS_WARN_STREAM("[planner_manager] manager/max_jerk=" << pp_.max_jerk_
                  << " is currently diagnostic only for exploration MINCO.");

  bool use_geometric_path, use_kinodynamic_path, use_topo_path, use_optimization,
      use_active_perception;
  nh.param("manager/use_geometric_path", use_geometric_path, false);
  nh.param("manager/use_kinodynamic_path", use_kinodynamic_path, false);
  nh.param("manager/use_topo_path", use_topo_path, false);
  nh.param("manager/use_optimization", use_optimization, false);
  nh.param("manager/use_active_perception", use_active_perception, false);

  local_data_.traj_id_ = 0;
  sdf_map_.reset(new SDFMap);
  sdf_map_->initMap(nh);
  edt_environment_.reset(new EDTEnvironment);
  edt_environment_->setMap(sdf_map_);

  if (use_geometric_path) {
    path_finder_.reset(new BubbleAstar);
    path_finder_->init(nh, edt_environment_);
  }

  if (use_kinodynamic_path) {
    kino_path_finder_.reset(new KinodynamicAstar);
    kino_path_finder_->setParam(nh);
    kino_path_finder_->setEnvironment(edt_environment_);
    kino_path_finder_->init();
  }

  if (use_optimization) {
    bspline_optimizers_.resize(10);
    for (int i = 0; i < 10; ++i) {
      bspline_optimizers_[i].reset(new BsplineOptimizer);
      bspline_optimizers_[i]->setParam(nh);
      bspline_optimizers_[i]->setEnvironment(edt_environment_);
    }
  }

  if (use_topo_path) {
    topo_prm_.reset(new TopologyPRM);
    topo_prm_->setEnvironment(edt_environment_);
    topo_prm_->init(nh);
  }

  if (use_active_perception) {
    frontier_finder_.reset(new FrontierFinder(edt_environment_, nh));
    heading_planner_.reset(new HeadingPlanner(nh));
    heading_planner_->setMap(sdf_map_);
    visib_util_.reset(new VisibilityUtil(nh));
    visib_util_->setEDTEnvironment(edt_environment_);
    plan_data_.view_cons_.idx_ = -1;
  }
}

void FastPlannerManager::setGlobalWaypoints(vector<Eigen::Vector3d>& waypoints) {
  plan_data_.global_waypoints_ = waypoints;
}

void FastPlannerManager::exportTrajToPolyMsg(traj_utils::PolyTraj& pos_msg,
                                             traj_utils::PolyTraj& yaw_msg,
                                             const ros::Time& start_time) {
  if (local_data_.use_minco_) {
    mincoTrajToRosMsg(local_data_.minco_traj_, local_data_.traj_id_, start_time, pos_msg);
    mincoYawTrajToRosMsg(local_data_.minco_yaw_traj_, local_data_.traj_id_, start_time, yaw_msg);
    return;
  }

  const PolynomialTraj pos_poly = bsplineToPolynomialTraj(local_data_.position_traj_);
  const PolynomialTraj yaw_poly = yawBsplineToPolynomialTraj(local_data_.yaw_traj_);
  polynomialTrajToRosMsg(pos_poly, local_data_.traj_id_, start_time, pos_msg);
  polynomialTrajToRosMsg(yaw_poly, local_data_.traj_id_, start_time, yaw_msg);
}

bool FastPlannerManager::reuseActiveTrajFromNow(traj_utils::PolyTraj& pos_msg,
                                                traj_utils::PolyTraj& yaw_msg,
                                                const ros::Time& start_time,
                                                double min_time_left) {
  if (!local_data_.use_minco_ || local_data_.minco_traj_.getPieceNum() <= 0 ||
      local_data_.minco_yaw_traj_.getPieceNum() <= 0 || local_data_.start_time_.toSec() <= 0.0) {
    return false;
  }

  const double t_cur = std::max(0.0, (start_time - local_data_.start_time_).toSec());
  const double time_left = local_data_.duration_ - t_cur;
  if (time_left <= min_time_left) return false;

  Trajectory<7> sliced_pos = sliceTrajectoryFromTime(local_data_.minco_traj_, t_cur);
  Trajectory<5> sliced_yaw = sliceTrajectoryFromTime(local_data_.minco_yaw_traj_, t_cur);
  if (sliced_pos.getPieceNum() <= 0 || sliced_yaw.getPieceNum() <= 0) return false;

  local_data_.traj_id_ += 1;

  mincoTrajToRosMsg(sliced_pos, local_data_.traj_id_, start_time, pos_msg);
  mincoYawTrajToRosMsg(sliced_yaw, local_data_.traj_id_, start_time, yaw_msg);
  return true;
}

bool FastPlannerManager::isPointSafeInExploreSpace(const Eigen::Vector3d& pt,
                                                   bool allow_unknown) const {
  if (!sdf_map_ || !sdf_map_->isInMap(pt) || !sdf_map_->isInBox(pt)) return false;
  if (sdf_map_->getInflateOccupancy(pt) == 1) return false;
  const int occ = sdf_map_->getOccupancy(pt);
  return allow_unknown ? occ != SDFMap::OCCUPIED : occ == SDFMap::FREE;
}

bool FastPlannerManager::isSegmentSafeInExploreSpace(const Eigen::Vector3d& p0,
                                                     const Eigen::Vector3d& p1, double step,
                                                     bool allow_unknown) const {
  if (!isPointSafeInExploreSpace(p0, allow_unknown) || !isPointSafeInExploreSpace(p1, allow_unknown))
    return false;

  const double sample_step = step > 0.0 ? step : defaultExploreStep(sdf_map_);
  const Eigen::Vector3d diff = p1 - p0;
  const double len = diff.norm();
  if (len < 1e-6) return true;

  for (double s = sample_step; s < len; s += sample_step) {
    if (!isPointSafeInExploreSpace(p0 + diff * (s / len), allow_unknown)) return false;
  }
  return true;
}

bool FastPlannerManager::projectToValidExplorePoint(const Eigen::Vector3d& raw_pt,
                                                    Eigen::Vector3d& projected_pt,
                                                    double max_radius) const {
  if (!sdf_map_) return false;

  Eigen::Vector3d box_min, box_max;
  sdf_map_->getBox(box_min, box_max);
  const double res = sdf_map_->getResolution();
  const double margin = std::max(1e-3, 0.5 * res);
  const double target_clearance =
      std::max({0.18, 0.9 * pp_.minco_safe_distance_, 2.0 * res});
  Eigen::Vector3d clamped = raw_pt;
  for (int i = 0; i < 3; ++i) {
    clamped[i] = std::max(box_min[i] + margin, std::min(box_max[i] - margin, clamped[i]));
  }

  auto improveCandidate = [&](const Eigen::Vector3d& seed, double search_radius,
                              Eigen::Vector3d& improved_pt) {
    if (!isPointSafeInExploreSpace(seed, false)) return false;
    Eigen::Vector3i center_idx;
    sdf_map_->posToIndex(seed, center_idx);
    const int max_step =
        std::max(1, static_cast<int>(std::ceil(std::max(0.1, search_radius) / std::max(1e-3, res))));
    improved_pt = seed;
    double best_score =
        evaluateExploreProjectionScore(sdf_map_, edt_environment_, seed, clamped, target_clearance);
    double best_dist2 = (seed - clamped).squaredNorm();

    for (int radius = 0; radius <= max_step; ++radius) {
      for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -radius; dy <= radius; ++dy) {
          for (int dz = -radius; dz <= radius; ++dz) {
            if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != radius) continue;
            const Eigen::Vector3i idx = center_idx + Eigen::Vector3i(dx, dy, dz);
            if (!sdf_map_->isInMap(idx) || !sdf_map_->isInBox(idx)) continue;
            if (sdf_map_->getInflateOccupancy(idx) == 1 || sdf_map_->getOccupancy(idx) != SDFMap::FREE)
              continue;

            Eigen::Vector3d candidate;
            sdf_map_->indexToPos(idx, candidate);
            const double score = evaluateExploreProjectionScore(
                sdf_map_, edt_environment_, candidate, clamped, target_clearance);
            const double dist2 = (candidate - clamped).squaredNorm();
            if (score > best_score + 1e-6 ||
                (std::abs(score - best_score) <= 1e-6 && dist2 < best_dist2)) {
              best_score = score;
              best_dist2 = dist2;
              improved_pt = candidate;
            }
          }
        }
      }
    }

    return true;
  };

  if (isPointSafeInExploreSpace(raw_pt, false)) {
    return improveCandidate(raw_pt, std::min(max_radius, std::max(0.6, 2.0 * target_clearance)),
                            projected_pt);
  }

  if (isPointSafeInExploreSpace(clamped, false)) {
    return improveCandidate(clamped, std::min(max_radius, std::max(0.6, 2.0 * target_clearance)),
                            projected_pt);
  }

  Eigen::Vector3i center_idx;
  sdf_map_->posToIndex(clamped, center_idx);
  const int max_step = std::max(1, static_cast<int>(std::ceil(max_radius / std::max(1e-3, res))));
  double best_score = -std::numeric_limits<double>::infinity();
  double best_dist2 = std::numeric_limits<double>::infinity();
  bool found = false;

  for (int radius = 0; radius <= max_step; ++radius) {
    for (int dx = -radius; dx <= radius; ++dx) {
      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dz = -radius; dz <= radius; ++dz) {
          if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != radius) continue;
          const Eigen::Vector3i idx = center_idx + Eigen::Vector3i(dx, dy, dz);
          if (!sdf_map_->isInMap(idx) || !sdf_map_->isInBox(idx)) continue;
          if (sdf_map_->getInflateOccupancy(idx) == 1 || sdf_map_->getOccupancy(idx) != SDFMap::FREE)
            continue;

          Eigen::Vector3d candidate;
          sdf_map_->indexToPos(idx, candidate);
          const double score = evaluateExploreProjectionScore(
              sdf_map_, edt_environment_, candidate, clamped, target_clearance);
          const double dist2 = (candidate - clamped).squaredNorm();
          if (!found || score > best_score + 1e-6 ||
              (std::abs(score - best_score) <= 1e-6 && dist2 < best_dist2)) {
            best_score = score;
            best_dist2 = dist2;
            projected_pt = candidate;
            found = true;
          }
        }
      }
    }
  }

  if (!found) return false;
  Eigen::Vector3d improved_pt = projected_pt;
  improveCandidate(projected_pt, std::min(max_radius, std::max(0.6, 2.0 * target_clearance)),
                   improved_pt);
  projected_pt = improved_pt;
  return true;
}

bool FastPlannerManager::sanitizeExplorePath(const vector<Eigen::Vector3d>& raw_path,
                                             vector<Eigen::Vector3d>& safe_path,
                                             double max_spacing) const {
  safe_path.clear();
  if (raw_path.empty()) return false;

  const double spacing = max_spacing > 0.0 ? max_spacing : std::max(pp_.ctrl_pt_dist * 0.5, defaultExploreStep(sdf_map_));
  const double target_clearance = std::max(0.18, 0.9 * pp_.minco_safe_distance_);
  Eigen::Vector3d prev_pt;
  bool has_prev = false;

  for (size_t i = 0; i < raw_path.size(); ++i) {
    Eigen::Vector3d safe_pt;
    if (!projectToValidExplorePoint(raw_path[i], safe_pt, std::max(2.0, 2.0 * spacing))) {
      ROS_ERROR_STREAM("Failed to project path waypoint into exploration space: "
                       << raw_path[i].transpose());
      return false;
    }
    pushPointAwayFromObstacles(safe_pt, safe_pt, target_clearance);

    if (!has_prev) {
      safe_path.push_back(safe_pt);
      prev_pt = safe_pt;
      has_prev = true;
      continue;
    }

    const Eigen::Vector3d diff = safe_pt - prev_pt;
    const double len = diff.norm();
    if (len < 1e-4) continue;

    const int pieces = std::max(1, static_cast<int>(std::ceil(len / spacing)));
    for (int k = 1; k <= pieces; ++k) {
      Eigen::Vector3d interp = prev_pt + diff * (static_cast<double>(k) / pieces);
      Eigen::Vector3d projected = interp;
      if (!isPointSafeInExploreSpace(projected, false)) {
        if (!projectToValidExplorePoint(interp, projected, std::max(2.0, 2.0 * spacing))) {
          ROS_ERROR_STREAM("Failed to project interpolated path point into exploration space: "
                           << interp.transpose());
          return false;
        }
      }
      Eigen::Vector3d inflated = projected;
      if (pushPointAwayFromObstacles(projected, inflated, target_clearance) &&
          isSegmentSafeInExploreSpace(safe_path.back(), inflated, spacing, false)) {
        projected = inflated;
      }

      if (!isSegmentSafeInExploreSpace(safe_path.back(), projected, spacing, false)) {
        ROS_ERROR_STREAM("Unsafe segment remains after path sanitization: "
                         << safe_path.back().transpose() << " -> " << projected.transpose());
        return false;
      }
      safe_path.push_back(projected);
    }

    prev_pt = safe_path.back();
  }

  if (safe_path.size() == 1) safe_path.push_back(safe_path.front());
  return safe_path.size() >= 2;
}

bool FastPlannerManager::pushPointAwayFromObstacles(const Eigen::Vector3d& seed,
                                                    Eigen::Vector3d& adjusted_pt,
                                                    double target_clearance,
                                                    double max_shift,
                                                    int max_iters) const {
  adjusted_pt = seed;
  if (!sdf_map_ || max_iters <= 0 || target_clearance <= 1e-3) return true;

  const double res = sdf_map_->getResolution();
  const double step_cap = std::max(0.08, 1.5 * res);
  const double shift_cap = std::max(0.15, max_shift);
  double accumulated_shift = 0.0;

  for (int iter = 0; iter < max_iters; ++iter) {
    double dist = 0.0;
    Eigen::Vector3d grad = Eigen::Vector3d::Zero();
    queryDistanceWithGrad(adjusted_pt, dist, grad);
    if (!std::isfinite(dist) || dist >= target_clearance) {
      return isPointSafeInExploreSpace(adjusted_pt, false);
    }

    const double grad_norm = grad.norm();
    if (grad_norm < 1e-4) break;

    const double remaining = std::max(0.0, target_clearance - dist);
    double step = std::min(step_cap, std::max(0.04, 0.75 * remaining));
    step = std::min(step, shift_cap - accumulated_shift);
    if (step <= 1e-3) break;

    Eigen::Vector3d candidate = adjusted_pt + grad / grad_norm * step;
    Eigen::Vector3d projected = candidate;
    if (!projectToValidExplorePoint(candidate, projected, std::max(0.4, 2.5 * res))) {
      break;
    }

    double new_dist = 0.0;
    Eigen::Vector3d new_grad = Eigen::Vector3d::Zero();
    queryDistanceWithGrad(projected, new_dist, new_grad);
    if (!std::isfinite(new_dist) || new_dist <= dist + 1e-4) break;

    accumulated_shift += (projected - adjusted_pt).norm();
    adjusted_pt = projected;
    if (accumulated_shift >= shift_cap - 1e-3) break;
  }

  return isPointSafeInExploreSpace(adjusted_pt, false);
}

bool FastPlannerManager::checkTrajCollision(double& distance) {
  double collision_time = 0.0;
  return checkTrajCollision(distance, collision_time);
}

bool FastPlannerManager::checkTrajCollision(double& distance, double& collision_time) {
  double t_now = (ros::Time::now() - local_data_.start_time_).toSec();
  if (!std::isfinite(t_now)) t_now = 0.0;
  t_now = std::max(0.0, t_now);
  collision_time = std::numeric_limits<double>::infinity();
  Eigen::Vector3d cur_pt;
  if (local_data_.use_minco_)
    cur_pt = local_data_.minco_traj_.getPos(std::min(t_now, local_data_.duration_));
  else
    cur_pt = local_data_.position_traj_.evaluateDeBoorT(std::min(t_now, local_data_.duration_));
  double radius = 0.0;
  Eigen::Vector3d fut_pt;
  double fut_t = 0.02;
  const double hard_collision_margin = std::max(0.02, 0.35 * pp_.minco_safe_distance_);

  while (radius < 6.0 && t_now + fut_t < local_data_.duration_) {
    if (local_data_.use_minco_)
      fut_pt = local_data_.minco_traj_.getPos(t_now + fut_t);
    else
      fut_pt = local_data_.position_traj_.evaluateDeBoorT(t_now + fut_t);
    double dist = 0.0;
    Eigen::Vector3d grad = Eigen::Vector3d::Zero();
    queryDistanceWithGrad(fut_pt, dist, grad);
    if (dist < hard_collision_margin) {
      distance = radius;
      collision_time = fut_t;
      std::cout << "collision at: " << fut_pt.transpose() << ", dist=" << dist << std::endl;
      return false;
    }
    radius = (fut_pt - cur_pt).norm();
    fut_t += 0.02;
  }

  return true;
}

bool FastPlannerManager::smoothedL1(const double& x, const double& mu, double& f,
                                    double& df) const {
  if (x < 0.0) return false;

  if (x > mu) {
    f = x - 0.5 * mu;
    df = 1.0;
    return true;
  }

  const double xdmu = x / mu;
  f = (mu - 0.5 * x) * xdmu * xdmu * xdmu;
  df = xdmu * xdmu * (3.0 * (mu - 0.5 * x) / mu - 0.5 * xdmu);
  return true;
}

void FastPlannerManager::queryDistanceWithGrad(const Eigen::Vector3d& pos, double& dist,
                                               Eigen::Vector3d& grad) const {
  if (!edt_environment_) {
    dist = 0.0;
    grad.setZero();
    return;
  }

  edt_environment_->evaluateEDTWithGrad(pos, -1.0, dist, grad);
  if (!std::isfinite(dist)) {
    dist = 0.0;
    grad.setZero();
  }
}

void logMincoTrajStats(const Trajectory<7>& traj, fast_planner::FastPlannerManager* manager,
                       double sample_dt) {
  if (manager == nullptr || traj.getPieceNum() <= 0) return;
  const double duration = traj.getTotalDuration();
  const MincoTrajStats stats = evaluateMincoTrajStats(traj, manager, sample_dt);
  ROS_WARN_STREAM("[explore minco] duration=" << duration << " max_vel=" << stats.max_vel
                  << " max_acc=" << stats.max_acc << " max_jerk=" << stats.max_jerk
                  << " min_obs_dist=" << stats.min_obs_dist << " at t=" << stats.min_obs_t);
}


bool FastPlannerManager::setupExploreMinco(const vector<Eigen::Vector3d>& safe_tour,
                                           const Eigen::Vector3d& cur_vel,
                                           const Eigen::Vector3d& cur_acc,
                                           const double& time_lb,
                                           const Eigen::Vector3d& terminal_vel,
                                           const bool stop_at_goal) {
  if (safe_tour.size() < 3) return false;

  explore_safe_tour_ = safe_tour;
  explore_ref_path_points_ = safe_tour;
  explore_ref_path_segments_.clear();
  explore_piece_num_ = static_cast<int>(safe_tour.size()) - 1;
  explore_inner_count_ = explore_piece_num_ - 1;
  explore_time_lb_ = time_lb;
  explore_stop_at_goal_ = stop_at_goal;
  explore_terminal_vel_ = computeConstrainedTerminalVelocity(safe_tour, terminal_vel, pp_);

  explore_ref_path_segments_.resize(explore_piece_num_);
  for (int i = 0; i < explore_piece_num_; ++i) {
    const Eigen::Vector3d& p0 = safe_tour[i];
    const Eigen::Vector3d& p1 = safe_tour[i + 1];
    vector<Eigen::Vector3d>& seg_pts = explore_ref_path_segments_[i];
    seg_pts.reserve(3);
    seg_pts.push_back(p0);
    seg_pts.push_back(0.5 * (p0 + p1));
    seg_pts.push_back(p1);
  }

  Eigen::Vector3d init_pos = safe_tour.front();
  Eigen::Vector3d init_vel = cur_vel;
  Eigen::Vector3d init_acc = cur_acc;
  Eigen::Vector3d init_jerk = Eigen::Vector3d::Zero();
  Eigen::Vector3d sampled_pos, sampled_vel, sampled_acc, sampled_jerk;
  if (sampleCurrentMincoState(local_data_, sampled_pos, sampled_vel, sampled_acc, sampled_jerk) &&
      (sampled_pos - safe_tour.front()).norm() < 1.0) {
    bool can_shift_start = isPointSafeInExploreSpace(sampled_pos, true);
    if (can_shift_start && safe_tour.size() > 1) {
      can_shift_start = isSegmentSafeInExploreSpace(sampled_pos, safe_tour[1], -1.0, true);
    }
    if (can_shift_start) {
      init_pos = sampled_pos;
      explore_safe_tour_.front() = init_pos;
      explore_ref_path_points_.front() = init_pos;
      if (!explore_ref_path_segments_.empty() && !explore_ref_path_segments_.front().empty()) {
        explore_ref_path_segments_.front().front() = init_pos;
      }
    }
    init_vel = sampled_vel;
    init_acc = sampled_acc;
    init_jerk = sampled_jerk;
  }
  explore_ini_state_ << init_pos, init_vel, init_acc, init_jerk;
  const Eigen::Vector3d fin_vel =
      explore_stop_at_goal_ ? Eigen::Vector3d::Zero() : explore_terminal_vel_;
  explore_fin_state_ << safe_tour.back(), fin_vel, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
  ROS_INFO_STREAM("[explore minco] terminal mode=" << (explore_stop_at_goal_ ? "stop" : "through")
                  << " fin_vel=" << fin_vel.transpose());

  explore_ref_inner_wps_.resize(3, explore_inner_count_);
  for (int i = 0; i < explore_inner_count_; ++i) {
    explore_ref_inner_wps_.col(i) = safe_tour[i + 1];
  }
  explore_inner_wps_ = explore_ref_inner_wps_;

  explore_times_.resize(explore_piece_num_);
  double total_time = 0.0;
  for (int i = 0; i < explore_piece_num_; ++i) {
    const double seg_len = (safe_tour[i + 1] - safe_tour[i]).norm();
    explore_times_(i) = std::max(0.12, (seg_len + 0.1) / std::max(0.1, pp_.max_vel_ * 0.8));
    total_time += explore_times_(i);
  }
  if (time_lb > 0.0 && total_time < time_lb) {
    explore_times_ *= time_lb / total_time;
  }

  explore_max_vel_sq_ = pp_.max_vel_ * pp_.max_vel_;
  explore_max_acc_sq_ = pp_.max_acc_ * pp_.max_acc_;
  explore_minco_.setConditions(explore_ini_state_, explore_fin_state_, explore_piece_num_);
  explore_minco_.setParameters(explore_inner_wps_, explore_times_);
  return true;
}

bool FastPlannerManager::setupExploreLongMinco(const vector<PathSegmentWithYaw>& segments,
                                               const Eigen::Vector3d& cur_vel,
                                               const Eigen::Vector3d& cur_acc,
                                               const double& time_lb) {
  if (segments.empty()) return false;

  explore_long_ref_path_points_.clear();
  explore_long_ref_path_segments_.clear();
  explore_long_target_yaws_.clear();
  explore_long_viewpoint_piece_indices_.clear();
  explore_long_viewpoint_arrival_times_.clear();
  explore_long_opt_indi_.clear();
  explore_long_time_lb_ = time_lb;

  int total_wps = 0;
  for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
    const auto& segment = segments[seg_idx];
    if (segment.path.size() > 2) total_wps += static_cast<int>(segment.path.size()) - 2;
    if (seg_idx + 1 < segments.size()) total_wps += 1;
  }

  explore_long_all_wps_.resize(3, total_wps);
  int wp_idx = 0;
  int piece_idx = 0;
  for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
    const auto& segment = segments[seg_idx];
    vector<Eigen::Vector3d> seg_pts = segment.path;
    if (seg_pts.empty() || (seg_pts.back() - segment.viewpoint).norm() > 1e-3) {
      seg_pts.push_back(segment.viewpoint);
    }
    explore_long_ref_path_segments_.push_back(seg_pts);
    explore_long_ref_path_points_.insert(explore_long_ref_path_points_.end(), seg_pts.begin(),
                                         seg_pts.end());

    for (size_t i = 1; i + 1 < segment.path.size(); ++i) {
      explore_long_all_wps_.col(wp_idx) = segment.path[i];
      explore_long_opt_indi_.push_back(true);
      ++wp_idx;
      ++piece_idx;
    }

    if (seg_idx + 1 < segments.size()) {
      explore_long_all_wps_.col(wp_idx) = segment.viewpoint;
      explore_long_opt_indi_.push_back(false);
      ++wp_idx;
      ++piece_idx;
    }

    double raw_yaw = segment.yaw;
    if (!explore_long_target_yaws_.empty()) {
      raw_yaw = unwrapTowards(explore_long_target_yaws_.back(), raw_yaw);
    }
    explore_long_target_yaws_.push_back(raw_yaw);
    explore_long_viewpoint_piece_indices_.push_back(piece_idx);
  }

  explore_long_piece_num_ = total_wps + 1;
  const int viewpt_count = total_wps - std::count(explore_long_opt_indi_.begin(),
                                                  explore_long_opt_indi_.end(), true);
  explore_long_waypt_count_ = std::count(explore_long_opt_indi_.begin(), explore_long_opt_indi_.end(), true);
  explore_long_way_wps_.resize(3, explore_long_waypt_count_);
  explore_long_view_wps_.resize(3, viewpt_count);
  explore_long_select_waypt_.resize(total_wps, explore_long_waypt_count_);
  explore_long_select_viewpt_.resize(total_wps, viewpt_count);
  explore_long_select_waypt_.setZero();
  explore_long_select_viewpt_.setZero();

  int way_idx = 0;
  int view_idx = 0;
  for (int i = 0; i < total_wps; ++i) {
    if (explore_long_opt_indi_[i]) {
      explore_long_select_waypt_(i, way_idx) = 1.0;
      explore_long_way_wps_.col(way_idx) = explore_long_all_wps_.col(i);
      ++way_idx;
    } else {
      explore_long_select_viewpt_(i, view_idx) = 1.0;
      explore_long_view_wps_.col(view_idx) = explore_long_all_wps_.col(i);
      ++view_idx;
    }
  }

  Eigen::Vector3d start_pos = !segments.front().path.empty() ? segments.front().path.front()
                                                             : segments.front().viewpoint;
  Eigen::Vector3d init_vel = cur_vel;
  Eigen::Vector3d init_acc = cur_acc;
  Eigen::Vector3d init_jerk = Eigen::Vector3d::Zero();
  Eigen::Vector3d sampled_pos, sampled_vel, sampled_acc, sampled_jerk;
  if (sampleCurrentMincoState(local_data_, sampled_pos, sampled_vel, sampled_acc, sampled_jerk) &&
      (sampled_pos - start_pos).norm() < 1.0) {
    Eigen::Vector3d next_ref = segments.back().viewpoint;
    if (total_wps > 0) {
      next_ref = explore_long_all_wps_.col(0);
    }
    bool can_shift_start = isPointSafeInExploreSpace(sampled_pos, true) &&
                           isSegmentSafeInExploreSpace(sampled_pos, next_ref, -1.0, true);
    if (can_shift_start) {
      start_pos = sampled_pos;
      if (!explore_long_ref_path_points_.empty()) {
        explore_long_ref_path_points_.front() = start_pos;
      }
      if (!explore_long_ref_path_segments_.empty() &&
          !explore_long_ref_path_segments_.front().empty()) {
        explore_long_ref_path_segments_.front().front() = start_pos;
      }
    }
    init_vel = sampled_vel;
    init_acc = sampled_acc;
    init_jerk = sampled_jerk;
  }
  explore_long_ini_state_ << start_pos, init_vel, init_acc, init_jerk;
  explore_long_fin_state_ << segments.back().viewpoint, Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

  explore_long_times_.resize(explore_long_piece_num_);
  Eigen::Vector3d prev = explore_long_ini_state_.col(0);
  if (total_wps > 0) {
    for (int i = 0; i < total_wps; ++i) {
      const Eigen::Vector3d cur = explore_long_all_wps_.col(i);
      explore_long_times_(i) = std::max(0.12, (cur - prev).norm() / std::max(0.1, pp_.max_vel_ * 0.8));
      prev = cur;
    }
  }
  explore_long_times_(explore_long_piece_num_ - 1) =
      std::max(0.12, (explore_long_fin_state_.col(0) - prev).norm() / std::max(0.1, pp_.max_vel_ * 0.8));

  const double total_time = explore_long_times_.sum();
  if (time_lb > 0.0 && total_time < time_lb) {
    explore_long_times_ *= time_lb / std::max(1e-3, total_time);
  }

  if (total_wps > 0) {
    explore_long_fused_wps_ =
        explore_long_way_wps_ * explore_long_select_waypt_.transpose() +
        explore_long_view_wps_ * explore_long_select_viewpt_.transpose();
  } else {
    explore_long_fused_wps_.resize(3, 0);
  }

  explore_long_minco_.setConditions(explore_long_ini_state_, explore_long_fin_state_,
                                    explore_long_piece_num_);
  explore_long_minco_.setParameters(explore_long_fused_wps_, explore_long_times_);
  ROS_INFO_STREAM("[explore long] setup with pieces=" << explore_long_piece_num_
                  << " waypoints=" << total_wps
                  << " frozen_viewpoints="
                  << (explore_long_target_yaws_.empty() ? 0 : explore_long_target_yaws_.size() - 1));
  return true;
}

void FastPlannerManager::computeExploreConstraintCostGrad(double& cost, Eigen::MatrixX3d& gdC,
                                                          Eigen::VectorXd& gdT) {
  const int K = 64;
  const double smooth_eps = pp_.minco_smooth_epsilon_;
  const auto& coeffs = explore_minco_.getCoeffs();

  cost = 0.0;
  gdC.setZero(8 * explore_piece_num_, 3);
  gdT.setZero(explore_piece_num_);

  for (int i = 0; i < explore_piece_num_; ++i) {
    const double T = explore_times_(i);
    const double step = T / K;

    for (int j = 0; j <= K; ++j) {
      const double node = (j == 0 || j == K) ? 0.5 : 1.0;
      const double s = j * step;
      const double s2 = s * s;
      const double s3 = s2 * s;
      const double s4 = s3 * s;
      const double s5 = s4 * s;
      const double s6 = s5 * s;
      const double s7 = s6 * s;

      Eigen::Matrix<double, 8, 1> beta0, beta1, beta2;
      beta0 << 1.0, s, s2, s3, s4, s5, s6, s7;
      beta1 << 0.0, 1.0, 2.0 * s, 3.0 * s2, 4.0 * s3, 5.0 * s4, 6.0 * s5, 7.0 * s6;
      beta2 << 0.0, 0.0, 2.0, 6.0 * s, 12.0 * s2, 20.0 * s3, 30.0 * s4, 42.0 * s5;

      const Eigen::Matrix<double, 8, 3> c = coeffs.block<8, 3>(8 * i, 0);
      const Eigen::Vector3d pos = c.transpose() * beta0;
      const Eigen::Vector3d vel = c.transpose() * beta1;
      const Eigen::Vector3d acc = c.transpose() * beta2;

      double pena = 0.0, pena_d = 0.0;

      const double vel_violation = vel.squaredNorm() - explore_max_vel_sq_;
      if (smoothedL1(vel_violation, smooth_eps, pena, pena_d)) {
        const Eigen::Vector3d grad_vel = pp_.minco_rho_v_ * pena_d * 2.0 * vel;
        cost += pp_.minco_rho_v_ * pena * node * step;
        gdC.block<8, 3>(8 * i, 0) += node * step * beta1 * grad_vel.transpose();
      }

      const double acc_violation = acc.squaredNorm() - explore_max_acc_sq_;
      if (smoothedL1(acc_violation, smooth_eps, pena, pena_d)) {
        const Eigen::Vector3d grad_acc = pp_.minco_rho_a_ * pena_d * 2.0 * acc;
        cost += pp_.minco_rho_a_ * pena * node * step;
        gdC.block<8, 3>(8 * i, 0) += node * step * beta2 * grad_acc.transpose();
      }

      double dist = 0.0;
      Eigen::Vector3d grad_obs = Eigen::Vector3d::Zero();
      queryDistanceWithGrad(pos, dist, grad_obs);
      const double coll_violation = pp_.minco_safe_distance_ - dist;
      if (smoothedL1(coll_violation, smooth_eps, pena, pena_d)) {
        const Eigen::Vector3d grad_coll = -pp_.minco_rho_collision_ * pena_d * grad_obs;
        cost += pp_.minco_rho_collision_ * pena * node * step;
        gdC.block<8, 3>(8 * i, 0) += node * step * beta0 * grad_coll.transpose();
      }

      if (!explore_ref_path_points_.empty() && pp_.minco_rho_path_ > 0.0) {
        const vector<Eigen::Vector3d>& ref_pts =
            (i < static_cast<int>(explore_ref_path_segments_.size()) &&
             !explore_ref_path_segments_[i].empty())
                ? explore_ref_path_segments_[i]
                : explore_ref_path_points_;
        double best_sq = std::numeric_limits<double>::infinity();
        Eigen::Vector3d best_pt = ref_pts.front();
        for (const auto& ref_pt : ref_pts) {
          const double sq = (pos - ref_pt).squaredNorm();
          if (sq < best_sq) {
            best_sq = sq;
            best_pt = ref_pt;
          }
        }
        const Eigen::Vector3d diff = pos - best_pt;
        cost += 0.5 * pp_.minco_rho_path_ * best_sq * node * step;
        gdC.block<8, 3>(8 * i, 0) +=
            node * step * beta0 * (pp_.minco_rho_path_ * diff).transpose();
      }
    }
  }

  if (explore_time_lb_ > 0.0) {
    const double total_t = explore_times_.sum();
    if (total_t < explore_time_lb_) {
      const double diff = explore_time_lb_ - total_t;
      const double rho_lb = 100.0;
      cost += rho_lb * diff * diff;
      gdT.array() -= 2.0 * rho_lb * diff;
    }
  }
}

void FastPlannerManager::computeExploreYawCostGrad(const Eigen::VectorXd& T, double& cost,
                                                   Eigen::VectorXd& gdT) {
  cost = 0.0;
  gdT.setZero(T.size());

  if (!std::isfinite(explore_target_yaw_) || T.size() <= 0) return;

  Eigen::Vector3d start_yaw = Eigen::Vector3d::Zero();
  if (!sampleCurrentYawState(local_data_, start_yaw)) {
    start_yaw.x() = inferYawFromPath(explore_safe_tour_);
  }

  minco::MINCO_S3NU yaw_minco;
  Eigen::Matrix3d ini_state, fin_state;
  const double yaw_start = wrapAngle(start_yaw.x());
  const double yaw_end = unwrapTowards(yaw_start, explore_target_yaw_);
  ini_state << Eigen::Vector3d(yaw_start, 0.0, 0.0), Eigen::Vector3d(start_yaw.y(), 0.0, 0.0),
      Eigen::Vector3d(start_yaw.z(), 0.0, 0.0);
  fin_state << Eigen::Vector3d(yaw_end, 0.0, 0.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

  Eigen::Matrix3Xd inner_points(3, std::max(0, static_cast<int>(T.size()) - 1));
  for (int k = 1; k < T.size(); ++k) {
    const double alpha = static_cast<double>(k) / static_cast<double>(T.size());
    inner_points.col(k - 1) << yaw_start + alpha * (yaw_end - yaw_start), 0.0, 0.0;
  }

  yaw_minco.setConditions(ini_state, fin_state, T.size());
  yaw_minco.setParameters(inner_points, T);

  const double max_yaw_rate = pp_.max_yawdot_ > 1e-3 ? pp_.max_yawdot_ : 1.0;
  const double max_yaw_rate_sq = max_yaw_rate * max_yaw_rate;
  const double rho_yaw = std::max(1.0, pp_.minco_rho_v_);
  const double smooth_eps = std::max(1e-3, pp_.minco_smooth_epsilon_);

  Eigen::MatrixX3d gdC(6 * T.size(), 3);
  Eigen::VectorXd gdT_minco(T.size());
  gdC.setZero();
  gdT_minco.setZero();

  const auto& coeffs = yaw_minco.getCoeffs();
  static constexpr int K = 16;
  for (int i = 0; i < T.size(); ++i) {
    const double duration = T(i);
    const double step = duration / K;

    for (int j = 0; j <= K; ++j) {
      const double node = (j == 0 || j == K) ? 0.5 : 1.0;
      const double s = j * step;
      const double s2 = s * s;
      const double s3 = s2 * s;
      const double s4 = s3 * s;

      Eigen::Matrix<double, 6, 1> beta1;
      beta1 << 0.0, 1.0, 2.0 * s, 3.0 * s2, 4.0 * s3, 5.0 * s4;

      const Eigen::Matrix<double, 6, 3> c = coeffs.block<6, 3>(6 * i, 0);
      const Eigen::Vector3d yaw_vel_vec = c.transpose() * beta1;
      const double yaw_vel = yaw_vel_vec.x();

      double pena = 0.0, pena_d = 0.0;
      const double yaw_violation = yaw_vel * yaw_vel - max_yaw_rate_sq;
      if (smoothedL1(yaw_violation, smooth_eps, pena, pena_d)) {
        cost += rho_yaw * pena * node * step;
        const Eigen::Vector3d grad_vel = rho_yaw * pena_d * 2.0 * yaw_vel_vec;
        gdC.block<6, 3>(6 * i, 0) += node * step * beta1 * grad_vel.transpose();
      }
    }
  }

  Eigen::Matrix3Xd gradP;
  yaw_minco.propogateGrad(gdC, gdT_minco, gradP, gdT);
}

void FastPlannerManager::computeExploreLongConstraintCostGrad(double& cost, Eigen::MatrixX3d& gdC,
                                                              Eigen::VectorXd& gdT) {
  const int K = 64;
  const double smooth_eps = pp_.minco_smooth_epsilon_;
  const auto& coeffs = explore_long_minco_.getCoeffs();

  cost = 0.0;
  gdC.setZero(8 * explore_long_piece_num_, 3);
  gdT.setZero(explore_long_piece_num_);

  for (int i = 0; i < explore_long_piece_num_; ++i) {
    const double T = explore_long_times_(i);
    const double step = T / K;
    int seg_idx = static_cast<int>(explore_long_ref_path_segments_.size()) - 1;
    for (int k = 0; k < static_cast<int>(explore_long_viewpoint_piece_indices_.size()); ++k) {
      if (i < explore_long_viewpoint_piece_indices_[k]) {
        seg_idx = k;
        break;
      }
    }
    const vector<Eigen::Vector3d>& ref_pts =
        (seg_idx >= 0 && seg_idx < static_cast<int>(explore_long_ref_path_segments_.size()) &&
         !explore_long_ref_path_segments_[seg_idx].empty())
            ? explore_long_ref_path_segments_[seg_idx]
            : explore_long_ref_path_points_;

    for (int j = 0; j <= K; ++j) {
      const double node = (j == 0 || j == K) ? 0.5 : 1.0;
      const double s = j * step;
      const double s2 = s * s;
      const double s3 = s2 * s;
      const double s4 = s3 * s;
      const double s5 = s4 * s;
      const double s6 = s5 * s;
      const double s7 = s6 * s;

      Eigen::Matrix<double, 8, 1> beta0, beta1, beta2;
      beta0 << 1.0, s, s2, s3, s4, s5, s6, s7;
      beta1 << 0.0, 1.0, 2.0 * s, 3.0 * s2, 4.0 * s3, 5.0 * s4, 6.0 * s5, 7.0 * s6;
      beta2 << 0.0, 0.0, 2.0, 6.0 * s, 12.0 * s2, 20.0 * s3, 30.0 * s4, 42.0 * s5;

      const Eigen::Matrix<double, 8, 3> c = coeffs.block<8, 3>(8 * i, 0);
      const Eigen::Vector3d pos = c.transpose() * beta0;
      const Eigen::Vector3d vel = c.transpose() * beta1;
      const Eigen::Vector3d acc = c.transpose() * beta2;

      double pena = 0.0, pena_d = 0.0;
      const double vel_violation = vel.squaredNorm() - pp_.max_vel_ * pp_.max_vel_;
      if (smoothedL1(vel_violation, smooth_eps, pena, pena_d)) {
        cost += pp_.minco_rho_v_ * pena * node * step;
        const Eigen::Vector3d grad_vel = pp_.minco_rho_v_ * pena_d * 2.0 * vel;
        gdC.block<8, 3>(8 * i, 0) += node * step * beta1 * grad_vel.transpose();
      }

      const double acc_violation = acc.squaredNorm() - pp_.max_acc_ * pp_.max_acc_;
      if (smoothedL1(acc_violation, smooth_eps, pena, pena_d)) {
        cost += pp_.minco_rho_a_ * pena * node * step;
        const Eigen::Vector3d grad_acc = pp_.minco_rho_a_ * pena_d * 2.0 * acc;
        gdC.block<8, 3>(8 * i, 0) += node * step * beta2 * grad_acc.transpose();
      }

      double dist = 0.0;
      Eigen::Vector3d grad_obs = Eigen::Vector3d::Zero();
      queryDistanceWithGrad(pos, dist, grad_obs);
      const double coll_violation = pp_.minco_safe_distance_ - dist;
      if (smoothedL1(coll_violation, smooth_eps, pena, pena_d)) {
        cost += pp_.minco_rho_collision_ * pena * node * step;
        const Eigen::Vector3d grad_coll = -pp_.minco_rho_collision_ * pena_d * grad_obs;
        gdC.block<8, 3>(8 * i, 0) += node * step * beta0 * grad_coll.transpose();
      }

      if (!ref_pts.empty() && pp_.minco_rho_path_ > 0.0) {
        double best_sq = std::numeric_limits<double>::infinity();
        Eigen::Vector3d best_pt = ref_pts.front();
        for (const auto& ref_pt : ref_pts) {
          const double sq = (pos - ref_pt).squaredNorm();
          if (sq < best_sq) {
            best_sq = sq;
            best_pt = ref_pt;
          }
        }
        const Eigen::Vector3d diff = pos - best_pt;
        cost += 0.5 * pp_.minco_rho_path_ * best_sq * node * step;
        gdC.block<8, 3>(8 * i, 0) +=
            node * step * beta0 * (pp_.minco_rho_path_ * diff).transpose();
      }
    }
  }

  if (explore_long_time_lb_ > 0.0) {
    const double total_t = explore_long_times_.sum();
    if (total_t < explore_long_time_lb_) {
      const double diff = explore_long_time_lb_ - total_t;
      const double rho_lb = 100.0;
      cost += rho_lb * diff * diff;
      gdT.array() -= 2.0 * rho_lb * diff;
    }
  }
}

void FastPlannerManager::computeExploreLongYawCostGrad(const Eigen::VectorXd& T, double& cost,
                                                       Eigen::VectorXd& gdT) {
  cost = 0.0;
  gdT.setZero(T.size());
  if (explore_long_target_yaws_.empty() || explore_long_viewpoint_piece_indices_.empty()) return;

  Eigen::Vector3d start_yaw = Eigen::Vector3d::Zero();
  if (!sampleCurrentYawState(local_data_, start_yaw)) {
    start_yaw.x() = wrapAngle(explore_long_target_yaws_.front());
  }

  const int num_viewpoints = static_cast<int>(explore_long_target_yaws_.size());
  Eigen::VectorXd yaw_times(num_viewpoints);
  std::vector<double> arrival_times(num_viewpoints, 0.0);
  double prev_arrival = 0.0;
  for (int k = 0; k < num_viewpoints; ++k) {
    double arrival = 0.0;
    const int piece_end = explore_long_viewpoint_piece_indices_[k];
    for (int i = 0; i < piece_end && i < T.size(); ++i) arrival += T(i);
    arrival_times[k] = arrival;
    yaw_times(k) = std::max(0.01, arrival - prev_arrival);
    prev_arrival = arrival;
  }
  arrival_times.back() = T.sum();
  yaw_times(num_viewpoints - 1) =
      std::max(0.01, arrival_times.back() - (num_viewpoints > 1 ? arrival_times[num_viewpoints - 2] : 0.0));

  std::vector<double> continuous_yaws(num_viewpoints);
  double accumulated_yaw = start_yaw.x();
  for (int i = 0; i < num_viewpoints; ++i) {
    accumulated_yaw = unwrapTowards(accumulated_yaw, explore_long_target_yaws_[i]);
    continuous_yaws[i] = accumulated_yaw;
  }

  Eigen::Matrix3d ini_state, fin_state;
  ini_state << Eigen::Vector3d(start_yaw.x(), 0.0, 0.0), Eigen::Vector3d(start_yaw.y(), 0.0, 0.0),
      Eigen::Vector3d(start_yaw.z(), 0.0, 0.0);
  fin_state << Eigen::Vector3d(continuous_yaws.back(), 0.0, 0.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero();

  Eigen::Matrix3Xd inner_points(3, std::max(0, num_viewpoints - 1));
  for (int i = 0; i < num_viewpoints - 1; ++i) {
    inner_points.col(i) << continuous_yaws[i], 0.0, 0.0;
  }

  minco::MINCO_S3NU yaw_minco;
  yaw_minco.setConditions(ini_state, fin_state, num_viewpoints);
  yaw_minco.setParameters(inner_points, yaw_times);

  const double max_yaw_rate = pp_.max_yawdot_ > 1e-3 ? pp_.max_yawdot_ : 1.0;
  const double max_yaw_rate_sq = max_yaw_rate * max_yaw_rate;
  const double rho_yaw = std::max(1.0, pp_.minco_rho_v_);
  Eigen::MatrixX3d gdC(6 * num_viewpoints, 3);
  Eigen::VectorXd gdT_yaw(num_viewpoints);
  gdC.setZero();
  gdT_yaw.setZero();

  const auto& coeffs = yaw_minco.getCoeffs();
  static constexpr int K = 16;
  for (int i = 0; i < num_viewpoints; ++i) {
    const double duration = yaw_times(i);
    const double step = duration / K;
    for (int j = 0; j <= K; ++j) {
      const double node = (j == 0 || j == K) ? 0.5 : 1.0;
      const double s = j * step;
      const double s2 = s * s;
      const double s3 = s2 * s;
      const double s4 = s3 * s;
      Eigen::Matrix<double, 6, 1> beta1;
      beta1 << 0.0, 1.0, 2.0 * s, 3.0 * s2, 4.0 * s3, 5.0 * s4;
      const Eigen::Matrix<double, 6, 3> c = coeffs.block<6, 3>(6 * i, 0);
      const Eigen::Vector3d yaw_vel_vec = c.transpose() * beta1;
      const double yaw_vel = yaw_vel_vec.x();

      double pena = 0.0, pena_d = 0.0;
      const double violation = yaw_vel * yaw_vel - max_yaw_rate_sq;
      if (smoothedL1(violation, std::max(1e-3, pp_.minco_smooth_epsilon_), pena, pena_d)) {
        cost += rho_yaw * pena * node * step;
        const Eigen::Vector3d grad_vel = rho_yaw * pena_d * 2.0 * yaw_vel_vec;
        gdC.block<6, 3>(6 * i, 0) += node * step * beta1 * grad_vel.transpose();
      }
    }
  }

  Eigen::Matrix3Xd gradP;
  Eigen::VectorXd gradT_minco;
  yaw_minco.propogateGrad(gdC, gdT_yaw, gradP, gradT_minco);

  for (int i = 0; i < T.size(); ++i) {
    int vp_idx = num_viewpoints - 1;
    for (int k = 0; k < num_viewpoints; ++k) {
      if (i < explore_long_viewpoint_piece_indices_[k]) {
        vp_idx = k;
        break;
      }
    }
    gdT(i) += gradT_minco(vp_idx);
  }
}

void FastPlannerManager::computeExploreLongArrivalTimes(const Eigen::VectorXd& T) {
  explore_long_viewpoint_arrival_times_.clear();
  for (size_t vp_idx = 0; vp_idx < explore_long_viewpoint_piece_indices_.size(); ++vp_idx) {
    double arrival = 0.0;
    const int piece_end = explore_long_viewpoint_piece_indices_[vp_idx];
    for (int i = 0; i < piece_end && i < T.size(); ++i) arrival += T(i);
    explore_long_viewpoint_arrival_times_.push_back(arrival);
  }
  if (!explore_long_viewpoint_arrival_times_.empty()) {
    explore_long_viewpoint_arrival_times_.back() = T.sum();
  }
}

bool FastPlannerManager::buildExploreLongYawTraj(const Eigen::Vector3d& start_yaw,
                                                 Trajectory<5>& yaw_traj) {
  if (explore_long_target_yaws_.empty() || explore_long_viewpoint_arrival_times_.empty()) return false;

  Eigen::Vector3d yaw_state = start_yaw;
  sampleCurrentYawState(local_data_, yaw_state);
  const int num_viewpoints = static_cast<int>(explore_long_target_yaws_.size());
  Eigen::VectorXd yaw_times(num_viewpoints);
  for (int i = 0; i < num_viewpoints; ++i) {
    const double prev = (i == 0) ? 0.0 : explore_long_viewpoint_arrival_times_[i - 1];
    yaw_times(i) = std::max(0.01, explore_long_viewpoint_arrival_times_[i] - prev);
  }

  std::vector<double> continuous_yaws(num_viewpoints);
  double accumulated_yaw = yaw_state.x();
  for (int i = 0; i < num_viewpoints; ++i) {
    accumulated_yaw = unwrapTowards(accumulated_yaw, explore_long_target_yaws_[i]);
    continuous_yaws[i] = accumulated_yaw;
  }

  Eigen::Matrix3d ini_state, fin_state;
  ini_state << Eigen::Vector3d(yaw_state.x(), 0.0, 0.0), Eigen::Vector3d(yaw_state.y(), 0.0, 0.0),
      Eigen::Vector3d(yaw_state.z(), 0.0, 0.0);
  fin_state << Eigen::Vector3d(continuous_yaws.back(), 0.0, 0.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero();

  Eigen::Matrix3Xd inner_points(3, std::max(0, num_viewpoints - 1));
  for (int i = 0; i < num_viewpoints - 1; ++i) {
    inner_points.col(i) << continuous_yaws[i], 0.0, 0.0;
  }

  minco::MINCO_S3NU yaw_minco;
  yaw_minco.setConditions(ini_state, fin_state, num_viewpoints);
  yaw_minco.setParameters(inner_points, yaw_times);
  yaw_minco.getTrajectory(yaw_traj);
  return true;
}

double FastPlannerManager::innerCallbackExplore(void* ptrObj, const Eigen::VectorXd& x,
                                                Eigen::VectorXd& grad) {
  FastPlannerManager& obj = *(FastPlannerManager*)ptrObj;

  Eigen::Map<const Eigen::VectorXd> tau(x.data(), obj.explore_piece_num_);
  Eigen::Map<Eigen::VectorXd> grad_tau(grad.data(), obj.explore_piece_num_);

  Eigen::VectorXd T;
  forwardTLocal(tau, T);
  obj.explore_times_ = T;

  if (obj.explore_inner_count_ > 0) {
    Eigen::Map<const Eigen::Matrix3Xd> wps(x.data() + obj.explore_piece_num_, 3,
                                           obj.explore_inner_count_);
    obj.explore_inner_wps_ = wps;

    Eigen::Vector3d box_min, box_max;
    obj.sdf_map_->getBox(box_min, box_max);
    const double margin = std::max(1e-3, 0.5 * obj.sdf_map_->getResolution());
    for (int i = 0; i < obj.explore_inner_count_; ++i) {
      obj.explore_inner_wps_.col(i) =
          obj.explore_inner_wps_.col(i)
              .cwiseMax(box_min + margin * Eigen::Vector3d::Ones())
              .cwiseMin(box_max - margin * Eigen::Vector3d::Ones());
    }
  }

  obj.explore_minco_.setParameters(obj.explore_inner_wps_, T);

  double snap_cost = 0.0;
  obj.explore_minco_.getEnergy(snap_cost);

  Eigen::MatrixX3d gdC_snap;
  obj.explore_minco_.getEnergyPartialGradByCoeffs(gdC_snap);
  Eigen::VectorXd gdT_snap;
  obj.explore_minco_.getEnergyPartialGradByTimes(gdT_snap);

  double constraint_cost = 0.0;
  Eigen::MatrixX3d gdC_constrain;
  Eigen::VectorXd gdT_constrain;
  obj.computeExploreConstraintCostGrad(constraint_cost, gdC_constrain, gdT_constrain);

  double yaw_cost = 0.0;
  Eigen::VectorXd gdT_yaw(T.size());
  gdT_yaw.setZero();
  obj.computeExploreYawCostGrad(T, yaw_cost, gdT_yaw);
  constraint_cost += yaw_cost;
  gdT_constrain += gdT_yaw;

  const double time_cost = obj.pp_.minco_weight_time_ * T.sum();

  Eigen::MatrixX3d gdC = gdC_snap + gdC_constrain;
  Eigen::VectorXd gdT = gdT_snap + gdT_constrain;
  gdT.array() += obj.pp_.minco_weight_time_;

  Eigen::Matrix3Xd gradP;
  Eigen::VectorXd gradT;
  obj.explore_minco_.propogateGrad(gdC, gdT, gradP, gradT);

  Eigen::VectorXd grad_tau_vec;
  backwardGradTLocal(tau, gradT, grad_tau_vec);
  grad_tau = grad_tau_vec;

  if (obj.explore_inner_count_ > 0) {
    Eigen::Map<Eigen::Matrix3Xd> grad_wps(grad.data() + obj.explore_piece_num_, 3,
                                          obj.explore_inner_count_);
    grad_wps = gradP;
  }

  return snap_cost + constraint_cost + time_cost;
}

double FastPlannerManager::innerCallbackExploreLong(void* ptrObj, const Eigen::VectorXd& x,
                                                    Eigen::VectorXd& grad) {
  FastPlannerManager& obj = *(FastPlannerManager*)ptrObj;
  Eigen::Map<const Eigen::VectorXd> tau(x.data(), obj.explore_long_piece_num_);
  Eigen::Map<Eigen::VectorXd> grad_tau(grad.data(), obj.explore_long_piece_num_);

  Eigen::VectorXd T;
  forwardTLocal(tau, T);
  obj.explore_long_times_ = T;

  if (obj.explore_long_waypt_count_ > 0) {
    Eigen::Map<const Eigen::Matrix3Xd> wps(x.data() + obj.explore_long_piece_num_, 3,
                                           obj.explore_long_waypt_count_);
    obj.explore_long_way_wps_ = wps;
    Eigen::Vector3d box_min, box_max;
    obj.sdf_map_->getBox(box_min, box_max);
    const double margin = std::max(1e-3, 0.5 * obj.sdf_map_->getResolution());
    for (int i = 0; i < obj.explore_long_waypt_count_; ++i) {
      obj.explore_long_way_wps_.col(i) =
          obj.explore_long_way_wps_.col(i)
              .cwiseMax(box_min + margin * Eigen::Vector3d::Ones())
              .cwiseMin(box_max - margin * Eigen::Vector3d::Ones());
    }
  }

  if (obj.explore_long_all_wps_.cols() > 0) {
    obj.explore_long_fused_wps_ =
        obj.explore_long_way_wps_ * obj.explore_long_select_waypt_.transpose() +
        obj.explore_long_view_wps_ * obj.explore_long_select_viewpt_.transpose();
  } else {
    obj.explore_long_fused_wps_.resize(3, 0);
  }

  obj.explore_long_minco_.setParameters(obj.explore_long_fused_wps_, T);

  double snap_cost = 0.0;
  obj.explore_long_minco_.getEnergy(snap_cost);
  Eigen::MatrixX3d gdC_snap;
  obj.explore_long_minco_.getEnergyPartialGradByCoeffs(gdC_snap);
  Eigen::VectorXd gdT_snap;
  obj.explore_long_minco_.getEnergyPartialGradByTimes(gdT_snap);

  double constraint_cost = 0.0;
  Eigen::MatrixX3d gdC_constrain;
  Eigen::VectorXd gdT_constrain;
  obj.computeExploreLongConstraintCostGrad(constraint_cost, gdC_constrain, gdT_constrain);

  double yaw_cost = 0.0;
  Eigen::VectorXd gdT_yaw(T.size());
  gdT_yaw.setZero();
  obj.computeExploreLongYawCostGrad(T, yaw_cost, gdT_yaw);
  constraint_cost += yaw_cost;
  gdT_constrain += gdT_yaw;

  const double time_cost = obj.pp_.minco_weight_time_ * T.sum();

  Eigen::MatrixX3d gdC = gdC_snap + gdC_constrain;
  Eigen::VectorXd gdT = gdT_snap + gdT_constrain;
  gdT.array() += obj.pp_.minco_weight_time_;

  Eigen::Matrix3Xd gradP;
  Eigen::VectorXd gradT;
  obj.explore_long_minco_.propogateGrad(gdC, gdT, gradP, gradT);

  Eigen::VectorXd grad_tau_vec;
  backwardGradTLocal(tau, gradT, grad_tau_vec);
  grad_tau = grad_tau_vec;

  if (obj.explore_long_waypt_count_ > 0) {
    Eigen::Map<Eigen::Matrix3Xd> grad_wps(grad.data() + obj.explore_long_piece_num_, 3,
                                          obj.explore_long_waypt_count_);
    grad_wps = gradP * obj.explore_long_select_waypt_;
  }

  return snap_cost + constraint_cost + time_cost;
}

bool FastPlannerManager::planExploreTrajLong(const vector<PathSegmentWithYaw>& segments,
                                             const Eigen::Vector3d& cur_vel,
                                             const Eigen::Vector3d& cur_acc,
                                             const double& time_lb) {
  if (segments.empty()) return false;
  vector<PathSegmentWithYaw> safe_segments = segments;
  for (int attempt = 0; attempt < 4; ++attempt) {
    if (!setupExploreLongMinco(safe_segments, cur_vel, cur_acc, time_lb)) {
      ROS_ERROR("Failed to setup long exploration MINCO.");
      return false;
    }

    const Eigen::VectorXd base_times = explore_long_times_;
    const Eigen::Matrix3Xd base_way_wps = explore_long_way_wps_;
    const int opt_dim = explore_long_piece_num_ + 3 * explore_long_waypt_count_;
    lbfgs::lbfgs_parameter_t params;
    params.mem_size = 128;
    params.past = 3;
    params.g_epsilon = 0.0;
    params.min_step = 1.0e-32;
    params.delta = pp_.minco_lbfgs_delta_;
    params.max_linesearch = 256;
    params.max_iterations = pp_.minco_lbfgs_max_iter_;

    Eigen::VectorXd x(opt_dim);
    bool accepted = false;
    bool used_fallback = false;
    const std::array<double, 3> time_scales = {1.0, 1.35, 1.8};
    for (size_t retry = 0; retry < time_scales.size(); ++retry) {
      explore_long_times_ = base_times * time_scales[retry];
      explore_long_way_wps_ = base_way_wps;
      if (explore_long_all_wps_.cols() > 0) {
        explore_long_fused_wps_ =
            explore_long_way_wps_ * explore_long_select_waypt_.transpose() +
            explore_long_view_wps_ * explore_long_select_viewpt_.transpose();
      } else {
        explore_long_fused_wps_.resize(3, 0);
      }
      explore_long_minco_.setParameters(explore_long_fused_wps_, explore_long_times_);

      Eigen::VectorXd tau;
      backwardTLocal(explore_long_times_, tau);
      x.head(explore_long_piece_num_) = tau;
      if (explore_long_waypt_count_ > 0) {
        Eigen::Map<Eigen::VectorXd> waypt_vec(explore_long_way_wps_.data(),
                                              3 * explore_long_waypt_count_);
        x.tail(3 * explore_long_waypt_count_) = waypt_vec;
      }

      double final_cost = 0.0;
      const int result =
          lbfgs::lbfgs_optimize(x, final_cost, &FastPlannerManager::innerCallbackExploreLong, nullptr,
                                nullptr, this, params);
      const bool ok = lbfgsReturnedUsableSolution(result);
      if (ok) {
        accepted = true;
        break;
      }

      ROS_WARN_STREAM("Explore long MINCO optimization returned " << result << ": "
                                                                  << lbfgs::lbfgs_strerror(result)
                                                                  << ", retry_scale=" << time_scales[retry]);
    }

    if (accepted) {
      Eigen::Map<const Eigen::VectorXd> final_tau(x.data(), explore_long_piece_num_);
      Eigen::VectorXd final_T;
      forwardTLocal(final_tau, final_T);
      explore_long_times_ = final_T;
      if (explore_long_waypt_count_ > 0) {
        Eigen::Map<const Eigen::Matrix3Xd> final_wps(x.data() + explore_long_piece_num_, 3,
                                                     explore_long_waypt_count_);
        explore_long_way_wps_ = final_wps;
        explore_long_fused_wps_ =
            explore_long_way_wps_ * explore_long_select_waypt_.transpose() +
            explore_long_view_wps_ * explore_long_select_viewpt_.transpose();
      }
      explore_long_minco_.setParameters(explore_long_fused_wps_, explore_long_times_);
    } else {
      used_fallback = true;
      explore_long_times_ = base_times * time_scales.back();
      explore_long_way_wps_ = base_way_wps;
      if (explore_long_all_wps_.cols() > 0) {
        explore_long_fused_wps_ =
            explore_long_way_wps_ * explore_long_select_waypt_.transpose() +
            explore_long_view_wps_ * explore_long_select_viewpt_.transpose();
      } else {
        explore_long_fused_wps_.resize(3, 0);
      }
      explore_long_minco_.setParameters(explore_long_fused_wps_, explore_long_times_);
      ROS_WARN_STREAM("Explore long MINCO fell back to conservative timing on attempt "
                      << attempt + 1 << ", time_scale=" << time_scales.back());
    }

    explore_long_minco_.getTrajectory(local_data_.minco_traj_);
    local_data_.duration_ = local_data_.minco_traj_.getTotalDuration();
    local_data_.start_pos_ = explore_long_ini_state_.col(0);
    local_data_.use_minco_ = true;

    const MincoTrajStats long_stats_before_validate =
        evaluateMincoTrajStats(local_data_.minco_traj_, this, defaultExploreStep(sdf_map_));
    const double long_time_stretch = computeExploreTimeStretch(long_stats_before_validate, pp_);
    if (long_time_stretch > 1.0) {
      explore_long_times_ *= long_time_stretch;
      explore_long_minco_.setParameters(explore_long_fused_wps_, explore_long_times_);
      explore_long_minco_.getTrajectory(local_data_.minco_traj_);
      local_data_.duration_ = local_data_.minco_traj_.getTotalDuration();
      ROS_WARN_STREAM("[explore long] stretched timing by " << long_time_stretch
                      << " before validation to reduce dynamic limit violations.");
    }

    if (!validateActiveTrajInExploreSpace()) {
      ROS_WARN_STREAM("Long exploration trajectory leaves exploration space on attempt "
                      << attempt + 1 << ".");
      subdividePathSegments(safe_segments);
      continue;
    }

    computeExploreLongArrivalTimes(explore_long_times_);
    Eigen::Vector3d start_yaw = Eigen::Vector3d::Zero();
    if (!sampleCurrentYawState(local_data_, start_yaw)) start_yaw.setZero();
    if (!buildExploreLongYawTraj(start_yaw, local_data_.minco_yaw_traj_)) {
      ROS_ERROR("Failed to build long exploration yaw trajectory.");
      return false;
    }

    local_data_.traj_id_ += 1;
    logMincoTrajStats(local_data_.minco_traj_, this, defaultExploreStep(sdf_map_));
    if (used_fallback) {
      ROS_WARN("[explore long] using fallback trajectory after optimizer failure.");
    }
    for (size_t i = 0; i < explore_long_viewpoint_arrival_times_.size(); ++i) {
      ROS_INFO_STREAM("[explore long] viewpoint " << i
                      << " piece_end=" << explore_long_viewpoint_piece_indices_[i]
                      << " arrival=" << explore_long_viewpoint_arrival_times_[i]);
    }
    return true;
  }

  ROS_ERROR("Long exploration trajectory cannot be constrained within exploration space.");
  return false;
}

bool FastPlannerManager::solveMincoPositionTraj(const vector<Eigen::Vector3d>& tour,
                                                const Eigen::Vector3d& cur_vel,
                                                const Eigen::Vector3d& cur_acc,
                                                const double& time_lb,
                                                const double target_yaw,
                                                const Eigen::Vector3d& terminal_vel,
                                                const bool stop_at_goal) {
  if (tour.size() < 2) return false;

  vector<Eigen::Vector3d> safe_tour;
  if (!sanitizeExplorePath(tour, safe_tour)) return false;

  if (safe_tour.size() == 2) {
    safe_tour.insert(safe_tour.begin() + 1, 0.5 * (safe_tour.front() + safe_tour.back()));
  }

  for (int attempt = 0; attempt < 4; ++attempt) {
    if (!setupExploreMinco(safe_tour, cur_vel, cur_acc, time_lb, terminal_vel, stop_at_goal))
      return false;
    explore_target_yaw_ = target_yaw;

    const int opt_dim = explore_piece_num_ + 3 * explore_inner_count_;
    Eigen::VectorXd x(opt_dim);
    Eigen::VectorXd tau;
    backwardTLocal(explore_times_, tau);
    x.head(explore_piece_num_) = tau;
    if (explore_inner_count_ > 0) {
      Eigen::Map<Eigen::VectorXd> wps(explore_ref_inner_wps_.data(), 3 * explore_inner_count_);
      x.tail(3 * explore_inner_count_) = wps;
    }

    lbfgs::lbfgs_parameter_t params;
    params.mem_size = 128;
    params.past = 3;
    params.g_epsilon = 0.0;
    params.min_step = 1.0e-32;
    params.delta = pp_.minco_lbfgs_delta_;
    params.max_linesearch = 256;
    params.max_iterations = pp_.minco_lbfgs_max_iter_;

    double final_cost = 0.0;
    const int result =
        lbfgs::lbfgs_optimize(x, final_cost, &FastPlannerManager::innerCallbackExplore, nullptr,
                              nullptr, this, params);
    if (!lbfgsReturnedUsableSolution(result)) {
      ROS_WARN_STREAM("Explore MINCO optimization returned " << result << ": "
                                                             << lbfgs::lbfgs_strerror(result));
      subdividePath(safe_tour);
      vector<Eigen::Vector3d> repaired_tour;
      if (!sanitizeExplorePath(safe_tour, repaired_tour,
                               std::max(0.15, pp_.ctrl_pt_dist * 0.35))) {
        break;
      }
      if (repaired_tour.size() == 2) {
        repaired_tour.insert(repaired_tour.begin() + 1,
                             0.5 * (repaired_tour.front() + repaired_tour.back()));
      }
      safe_tour.swap(repaired_tour);
      continue;
    }

    explore_minco_.getTrajectory(local_data_.minco_traj_);
    local_data_.duration_ = local_data_.minco_traj_.getTotalDuration();
    local_data_.start_pos_ = safe_tour.front();
    local_data_.use_minco_ = true;

    const MincoTrajStats explore_stats_before_validate =
        evaluateMincoTrajStats(local_data_.minco_traj_, this, defaultExploreStep(sdf_map_));
    const double explore_time_stretch = computeExploreTimeStretch(explore_stats_before_validate, pp_);
    if (explore_time_stretch > 1.0) {
      explore_times_ *= explore_time_stretch;
      explore_minco_.setParameters(explore_inner_wps_, explore_times_);
      explore_minco_.getTrajectory(local_data_.minco_traj_);
      local_data_.duration_ = local_data_.minco_traj_.getTotalDuration();
      ROS_WARN_STREAM("[explore minco] stretched timing by " << explore_time_stretch
                      << " before validation to reduce dynamic limit violations.");
    }

    if (validateActiveTrajInExploreSpace()) {
      logMincoTrajStats(local_data_.minco_traj_, this, defaultExploreStep(sdf_map_));
      return true;
    }

    ROS_WARN_STREAM("Optimized MINCO trajectory leaves exploration space on attempt "
                    << attempt + 1 << ".");
    subdividePath(safe_tour);
    vector<Eigen::Vector3d> repaired_tour;
    if (!sanitizeExplorePath(safe_tour, repaired_tour,
                             std::max(0.15, pp_.ctrl_pt_dist * 0.35))) {
      break;
    }
    if (repaired_tour.size() == 2) {
      repaired_tour.insert(repaired_tour.begin() + 1,
                           0.5 * (repaired_tour.front() + repaired_tour.back()));
    }
    safe_tour.swap(repaired_tour);
  }

  ROS_ERROR("MINCO position trajectory cannot be constrained within exploration space.");
  return false;
}

bool FastPlannerManager::solveMincoYawTraj(const Eigen::Vector3d& start_yaw, const double& end_yaw,
                                           bool lookfwd, const double& relax_time) {
  (void)lookfwd;
  (void)relax_time;

  if (local_data_.use_minco_ && std::isfinite(end_yaw) && local_data_.minco_traj_.getPieceNum() > 0) {
    Eigen::Vector3d coupled_start_yaw = start_yaw;
    sampleCurrentYawState(local_data_, coupled_start_yaw);
    const Eigen::VectorXd times = local_data_.minco_traj_.getDurations();
    if (buildCoupledMincoYawTraj(times, coupled_start_yaw, end_yaw, local_data_.minco_yaw_traj_)) {
      return true;
    }
  }

  double yaw_sp = start_yaw(0);
  double yaw_sv = start_yaw(1);
  double yaw_sa = start_yaw(2);
  Eigen::Vector3d sampled_yaw_state;
  bool has_prev_traj = sampleCurrentYawState(local_data_, sampled_yaw_state);
  if (has_prev_traj) {
    yaw_sp = sampled_yaw_state.x();
    yaw_sv = sampled_yaw_state.y();
    yaw_sa = sampled_yaw_state.z();
  }
  if (!has_prev_traj) {
    yaw_sp = start_yaw(0);
    yaw_sv = start_yaw(1);
    yaw_sa = start_yaw(2);
  }

  angleLimite(yaw_sp);
  double yaw_diff = end_yaw - yaw_sp;
  angleLimite(yaw_diff);
  if (!std::isfinite(yaw_diff)) return false;

  const double max_yaw_rate = pp_.max_yawdot_ > 1e-3 ? pp_.max_yawdot_ : 1.0;
  const double min_duration = std::max(0.05, local_data_.duration_);
  const double yaw_time = std::max(min_duration, std::fabs(yaw_diff) / max_yaw_rate);

  if (std::fabs(yaw_diff) < 1e-3) {
    minco::MINCO_S3NU minco;
    Eigen::Matrix3d ini_state, fin_state;
    ini_state << Eigen::Vector3d(yaw_sp, 0.0, 0.0), Eigen::Vector3d(yaw_sv, 0.0, 0.0),
        Eigen::Vector3d(yaw_sa, 0.0, 0.0);
    fin_state << Eigen::Vector3d(yaw_sp, 0.0, 0.0), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero();
    Eigen::Matrix3Xd inner_points(3, 1);
    inner_points << yaw_sp, 0.0, 0.0;
    Eigen::VectorXd times(2);
    times << yaw_time * 0.5, yaw_time * 0.5;
    minco.setConditions(ini_state, fin_state, 2);
    minco.setParameters(inner_points, times);
    minco.getTrajectory(local_data_.minco_yaw_traj_);
    return true;
  }

  vector<double> wp;
  wp.push_back(yaw_sp);
  static constexpr double yaw_piece_nominal = 0.3;
  for (double t = yaw_piece_nominal; t < yaw_time + yaw_piece_nominal; t += yaw_piece_nominal) {
    if (t > yaw_time) {
      wp.push_back(yaw_sp + yaw_diff);
      break;
    }
    const double alpha = t / yaw_time;
    wp.push_back(yaw_sp + alpha * yaw_diff);
  }

  if (wp.size() < 3) {
    wp.insert(wp.begin() + 1, yaw_sp + 0.5 * yaw_diff);
  }

  const double yaw_ep = wp.back();
  Eigen::Matrix3d ini_state, fin_state;
  ini_state << Eigen::Vector3d(yaw_sp, 0.0, 0.0), Eigen::Vector3d(yaw_sv, 0.0, 0.0),
      Eigen::Vector3d(yaw_sa, 0.0, 0.0);
  fin_state << Eigen::Vector3d(yaw_ep, 0.0, 0.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero();

  int piece_num = static_cast<int>(wp.size()) - 1;
  Eigen::Matrix3Xd inner_points(3, piece_num - 1);
  for (int k = 1; k < piece_num; ++k) {
    inner_points.col(k - 1) << wp[k], 0.0, 0.0;
  }

  Eigen::VectorXd times(piece_num);
  times.setConstant(yaw_piece_nominal);
  double assigned = yaw_piece_nominal * piece_num;
  if (assigned > yaw_time) {
    times.array() *= yaw_time / assigned;
  } else {
    times(piece_num - 1) += yaw_time - assigned;
  }
  for (int i = 0; i < piece_num; ++i) {
    times(i) = std::max(0.03, times(i));
  }

  minco::MINCO_S3NU minco;
  minco.setConditions(ini_state, fin_state, piece_num);
  minco.setParameters(inner_points, times);
  minco.getTrajectory(local_data_.minco_yaw_traj_);
  return true;
}

// !SECTION

// SECTION kinodynamic replanning

bool FastPlannerManager::kinodynamicReplan(const Eigen::Vector3d& start_pt,
    const Eigen::Vector3d& start_vel, const Eigen::Vector3d& start_acc,
    const Eigen::Vector3d& end_pt, const Eigen::Vector3d& end_vel, const double& time_lb) {
  std::cout << "[Kino replan]: start: " << start_pt.transpose() << ", " << start_vel.transpose()
            << ", " << start_acc.transpose() << ", goal:" << end_pt.transpose() << ", "
            << end_vel.transpose() << endl;

  if ((start_pt - end_pt).norm() < 1e-2) {
    cout << "Close goal" << endl;
    return false;
  }

  Eigen::Vector3d init_pos = start_pt;
  Eigen::Vector3d init_vel = start_vel;
  Eigen::Vector3d init_acc = start_acc;

  // Kinodynamic path searching

  auto t1 = ros::Time::now();

  kino_path_finder_->reset();
  int status = kino_path_finder_->search(start_pt, start_vel, start_acc, end_pt, end_vel, true);
  if (status == KinodynamicAstar::NO_PATH) {
    ROS_ERROR("search 1 fail");
    // Retry
    kino_path_finder_->reset();
    status = kino_path_finder_->search(start_pt, start_vel, start_acc, end_pt, end_vel, false);
    if (status == KinodynamicAstar::NO_PATH) {
      cout << "[Kino replan]: Can't find path." << endl;
      return false;
    }
  }
  plan_data_.kino_path_ = kino_path_finder_->getKinoTraj(0.01);

  double t_search = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Parameterize path to B-spline
  double ts = pp_.ctrl_pt_dist / pp_.max_vel_;
  vector<Eigen::Vector3d> point_set, start_end_derivatives;
  kino_path_finder_->getSamples(ts, point_set, start_end_derivatives);

  // std::cout << "point set:" << std::endl;
  // for (auto pt : point_set) std::cout << pt.transpose() << std::endl;
  // std::cout << "derivative:" << std::endl;
  // for (auto dr : start_end_derivatives) std::cout << dr.transpose() << std::endl;

  Eigen::MatrixXd ctrl_pts;
  NonUniformBspline::parameterizeToBspline(
      ts, point_set, start_end_derivatives, pp_.bspline_degree_, ctrl_pts);
  NonUniformBspline init(ctrl_pts, pp_.bspline_degree_, ts);

  // B-spline-based optimization
  int cost_function = BsplineOptimizer::NORMAL_PHASE;
  if (pp_.min_time_) cost_function |= BsplineOptimizer::MINTIME;
  vector<Eigen::Vector3d> start, end;
  init.getBoundaryStates(2, 0, start, end);
  bspline_optimizers_[0]->setBoundaryStates(start, end);
  if (time_lb > 0) bspline_optimizers_[0]->setTimeLowerBound(time_lb);

  bspline_optimizers_[0]->optimize(ctrl_pts, ts, cost_function, 1, 1);
  local_data_.position_traj_.setUniformBspline(ctrl_pts, pp_.bspline_degree_, ts);
  local_data_.use_minco_ = false;

  vector<Eigen::Vector3d> start2, end2;
  local_data_.position_traj_.getBoundaryStates(2, 0, start2, end2);
  std::cout << "State error: (" << (start2[0] - start[0]).norm() << ", "
            << (start2[1] - start[1]).norm() << ", " << (start2[2] - start[2]).norm() << ")"
            << std::endl;

  double t_opt = (ros::Time::now() - t1).toSec();
  ROS_WARN("Kino t: %lf, opt: %lf", t_search, t_opt);

  // t1 = ros::Time::now();

  // // Adjust time and refine

  // double dt;
  // for (int i = 0; i < 2; ++i)
  // {
  //   NonUniformBspline pos = NonUniformBspline(ctrl_pts, pp_.bspline_degree_, ts);
  //   pos.setPhysicalLimits(pp_.max_vel_, pp_.max_acc_);
  //   pos.lengthenTime(min(1.01, pos.checkRatio()));
  //   double duration = pos.getTimeSum();
  //   dt = duration / double(pos.getControlPoint().rows() - pp_.bspline_degree_);

  //   point_set.clear();
  //   for (double time = 0.0; time <= duration + 1e-4; time += dt)
  //     point_set.push_back(pos.evaluateDeBoorT(time));
  //   NonUniformBspline::parameterizeToBspline(dt, point_set, start_end_derivatives,
  //   pp_.bspline_degree_, ctrl_pts);
  //   bspline_optimizers_[0]->optimize(ctrl_pts, dt, cost_function, 1, 1);
  // }
  // local_data_.position_traj_.setUniformBspline(ctrl_pts, pp_.bspline_degree_, dt);

  // iterative time adjustment

  // double to = pos.getTimeSum();
  // pos.setPhysicalLimits(pp_.max_vel_, pp_.max_acc_);
  // bool feasible = pos.checkFeasibility(false);

  // int iter_num = 0;
  // while (!feasible && ros::ok()) {

  //   feasible = pos.reallocateTime();

  //   if (++iter_num >= 3) break;
  // }

  // // pos.checkFeasibility(true);
  // // cout << "[Main]: iter num: " << iter_num << endl;

  // double tn = pos.getTimeSum();

  // cout << "[kino replan]: Reallocate ratio: " << tn / to << endl;
  // if (tn / to > 3.0) ROS_ERROR("reallocate error.");

  // t_adjust = (ros::Time::now() - t1).toSec();

  // // save planned results

  // local_data_.position_traj_ = pos;

  // double t_total = t_search + t_opt + t_adjust;
  // cout << "[kino replan]: time: " << t_total << ", search: " << t_search << ",
  // optimize: " << t_opt
  //      << ", adjust time:" << t_adjust << endl;

  // pp_.time_search_   = t_search;
  // pp_.time_optimize_ = t_opt;
  // pp_.time_adjust_   = t_adjust;

  // int rd = rand() % 2;
  // if (rd == 0) {
  //   updateTrajInfo();
  //   return true;
  // } else
  //   return false;

  updateTrajInfo();
  return true;
}

bool FastPlannerManager::planExploreTraj(const vector<Eigen::Vector3d>& tour,
    const Eigen::Vector3d& cur_vel, const Eigen::Vector3d& cur_acc, const double& time_lb) {
  return planExploreTraj(tour, cur_vel, cur_acc, time_lb,
                         std::numeric_limits<double>::quiet_NaN(), Eigen::Vector3d::Zero(), true);
}

bool FastPlannerManager::planExploreTraj(const vector<Eigen::Vector3d>& tour,
    const Eigen::Vector3d& cur_vel, const Eigen::Vector3d& cur_acc, const double& time_lb,
    const double target_yaw, const Eigen::Vector3d& terminal_vel, const bool stop_at_goal) {
  if (tour.empty()) {
    ROS_ERROR("Empty path to traj planner");
    return false;
  }
  if (!solveMincoPositionTraj(tour, cur_vel, cur_acc, time_lb, target_yaw, terminal_vel,
                              stop_at_goal)) {
    ROS_ERROR("MINCO position solve failed.");
    return false;
  }
  local_data_.traj_id_ += 1;
  return true;
}

bool FastPlannerManager::validateActiveTrajInExploreSpace(double sample_step) {
  const double step = sample_step > 0.0 ? sample_step : defaultExploreStep(sdf_map_);
  if (local_data_.duration_ <= 0.0) return false;
  const ExploreAcceptanceLimits limits = getExploreAcceptanceLimits(pp_);
  const double strict_horizon =
      std::max(step, std::min(local_data_.duration_, std::max(0.5, pp_.explore_strict_horizon_)));
  const double soft_safe_margin = std::max(0.05, 0.5 * pp_.minco_safe_distance_);
  const double hard_safe_margin = std::max(0.01, 0.08 * pp_.minco_safe_distance_);
  int soft_violation_num = 0;
  double min_dist = std::numeric_limits<double>::infinity();
  double min_dist_t = 0.0;
  Eigen::Vector3d min_dist_pt = Eigen::Vector3d::Zero();
  double min_dist_strict = std::numeric_limits<double>::infinity();
  double min_dist_strict_t = 0.0;
  Eigen::Vector3d min_dist_strict_pt = Eigen::Vector3d::Zero();
  double max_vel = 0.0;
  double max_acc = 0.0;
  double max_jerk = 0.0;

  for (double t = 0.0; t <= local_data_.duration_ + 1e-6; t += step) {
    const double tc = std::min(local_data_.duration_, t);
    Eigen::Vector3d pt;
    if (local_data_.use_minco_)
      pt = local_data_.minco_traj_.getPos(tc);
    else
      pt = local_data_.position_traj_.evaluateDeBoorT(tc);
    if (local_data_.use_minco_) {
      max_vel = std::max(max_vel, local_data_.minco_traj_.getVel(tc).norm());
      max_acc = std::max(max_acc, local_data_.minco_traj_.getAcc(tc).norm());
      max_jerk = std::max(max_jerk, local_data_.minco_traj_.getJer(tc).norm());
    }
    double dist = 0.0;
    Eigen::Vector3d grad = Eigen::Vector3d::Zero();
    queryDistanceWithGrad(pt, dist, grad);
    if (dist < min_dist) {
      min_dist = dist;
      min_dist_t = tc;
      min_dist_pt = pt;
    }
    if (tc <= strict_horizon && dist < min_dist_strict) {
      min_dist_strict = dist;
      min_dist_strict_t = tc;
      min_dist_strict_pt = pt;
    }
    if (dist < hard_safe_margin) {
      ROS_WARN_STREAM("Trajectory sample too close to obstacle at t=" << tc << ", dist=" << dist
                                                                      << ", pt=" << pt.transpose());
      return false;
    }
    if (dist < soft_safe_margin) ++soft_violation_num;
  }

  const int soft_violation_limit = std::max(2, static_cast<int>(std::ceil(local_data_.duration_ / step * 0.2)));
  if (soft_violation_num > soft_violation_limit) {
    ROS_WARN_STREAM("Trajectory stays near obstacles for too long: " << soft_violation_num
                                                                     << " samples within "
                                                                     << soft_safe_margin
                                                                     << "m, but no hard collision was found."
                                                                     << " Closest sample: dist=" << min_dist
                                                                     << " at t=" << min_dist_t << ", pt="
                                                                     << min_dist_pt.transpose());
  }
  if (!std::isfinite(min_dist_strict)) {
    min_dist_strict = min_dist;
    min_dist_strict_t = min_dist_t;
    min_dist_strict_pt = min_dist_pt;
  }
  if (local_data_.use_minco_ && min_dist_strict < limits.reject_safe_margin) {
    ROS_WARN_STREAM("Rejecting MINCO trajectory: min_obs_dist=" << min_dist_strict
                                                                << " below release threshold "
                                                                << limits.reject_safe_margin << " at t="
                                                                << min_dist_strict_t
                                                                << " within strict horizon " << strict_horizon
                                                                << ", pt=" << min_dist_strict_pt.transpose());
    return false;
  }
  if (local_data_.use_minco_ && min_dist < limits.reject_safe_margin &&
      min_dist_t > strict_horizon + 1e-6) {
    ROS_WARN_STREAM("Accepting exploration trajectory with tail clearance " << min_dist
                    << " below release threshold " << limits.reject_safe_margin
                    << " because the violation is beyond strict horizon " << strict_horizon
                    << " at t=" << min_dist_t << ", pt=" << min_dist_pt.transpose());
  }
  if (local_data_.use_minco_ && max_vel > limits.vel) {
    ROS_WARN_STREAM("Rejecting MINCO trajectory: max_vel=" << max_vel
                                                           << " exceeds limit " << limits.vel);
    return false;
  }
  if (local_data_.use_minco_ && max_acc > limits.acc) {
    ROS_WARN_STREAM("Rejecting MINCO trajectory: max_acc=" << max_acc
                                                           << " exceeds limit " << limits.acc);
    return false;
  }
  if (local_data_.use_minco_ && max_jerk > limits.jerk) {
    ROS_WARN_STREAM("Rejecting MINCO trajectory: max_jerk=" << max_jerk
                                                            << " exceeds limit " << limits.jerk);
    return false;
  }
  return true;
}

// !SECTION

// SECTION topological replanning

bool FastPlannerManager::planGlobalTraj(const Eigen::Vector3d& start_pos) {
  plan_data_.clearTopoPaths();

  // Generate global reference trajectory
  vector<Eigen::Vector3d> points = plan_data_.global_waypoints_;
  if (points.size() == 0) std::cout << "no global waypoints!" << std::endl;

  points.insert(points.begin(), start_pos);

  // Insert intermediate points if two waypoints are too far
  vector<Eigen::Vector3d> inter_points;
  const double dist_thresh = 4.0;

  for (int i = 0; i < points.size() - 1; ++i) {
    inter_points.push_back(points.at(i));
    double dist = (points.at(i + 1) - points.at(i)).norm();
    if (dist > dist_thresh) {
      int id_num = floor(dist / dist_thresh) + 1;
      for (int j = 1; j < id_num; ++j) {
        Eigen::Vector3d inter_pt =
            points.at(i) * (1.0 - double(j) / id_num) + points.at(i + 1) * double(j) / id_num;
        inter_points.push_back(inter_pt);
      }
    }
  }
  inter_points.push_back(points.back());

  // At least 3 waypoints are required to solve the problem
  if (inter_points.size() == 2) {
    Eigen::Vector3d mid = (inter_points[0] + inter_points[1]) * 0.5;
    inter_points.insert(inter_points.begin() + 1, mid);
  }

  int pt_num = inter_points.size();
  Eigen::MatrixXd pos(pt_num, 3);
  for (int i = 0; i < pt_num; ++i) pos.row(i) = inter_points[i];

  Eigen::Vector3d zero(0, 0, 0);
  Eigen::VectorXd time(pt_num - 1);
  for (int i = 0; i < pt_num - 1; ++i)
    time(i) = (pos.row(i + 1) - pos.row(i)).norm() / (pp_.max_vel_ * 0.5);

  time(0) += pp_.max_vel_ / (2 * pp_.max_acc_);
  time(time.rows() - 1) += pp_.max_vel_ / (2 * pp_.max_acc_);

  PolynomialTraj gl_traj;
  PolynomialTraj::waypointsTraj(pos, zero, zero, zero, zero, time, gl_traj);

  auto time_now = ros::Time::now();
  global_data_.setGlobalTraj(gl_traj, time_now);

  // truncate a local trajectory

  double dt, duration;
  Eigen::MatrixXd ctrl_pts = paramLocalTraj(0.0, dt, duration);
  NonUniformBspline bspline(ctrl_pts, pp_.bspline_degree_, dt);

  std::cout << "ctrl pt: " << ctrl_pts.rows() << std::endl;

  global_data_.setLocalTraj(bspline, 0.0, duration, 0.0);
  local_data_.position_traj_ = bspline;
  local_data_.start_time_ = time_now;
  ROS_INFO("global trajectory generated.");

  updateTrajInfo();

  return true;
}

bool FastPlannerManager::topoReplan(bool collide) {
  ros::Time t1, t2;

  /* truncate a new local segment for replanning */
  ros::Time time_now = ros::Time::now();
  double t_now = (time_now - global_data_.global_start_time_).toSec();
  double local_traj_dt, local_traj_duration;

  Eigen::MatrixXd ctrl_pts = paramLocalTraj(t_now, local_traj_dt, local_traj_duration);
  NonUniformBspline init_traj(ctrl_pts, pp_.bspline_degree_, local_traj_dt);
  local_data_.start_time_ = time_now;

  std::cout << "dt: " << local_traj_dt << ", dur: " << local_traj_duration << std::endl;

  if (!collide) {
    // No collision detected, but we can further refine the trajectory
    refineTraj(init_traj);
    double time_change = init_traj.getTimeSum() - local_traj_duration;
    local_data_.position_traj_ = init_traj;
    global_data_.setLocalTraj(
        local_data_.position_traj_, t_now, local_traj_duration + time_change + t_now, time_change);
    // local_data_.position_traj_ = init_traj;
    // global_data_.setLocalTraj(init_traj, t_now, local_traj_duration + t_now, 0.0);
  } else {
    // Find topologically distinctive path and guide optimization in parallel
    plan_data_.initial_local_segment_ = init_traj;
    vector<Eigen::Vector3d> colli_start, colli_end, start_pts, end_pts;
    findCollisionRange(colli_start, colli_end, start_pts, end_pts);

    if (colli_start.size() == 1 && colli_end.size() == 0) {
      ROS_WARN("Init traj ends in obstacle, no replanning.");
      local_data_.position_traj_ = init_traj;
      global_data_.setLocalTraj(init_traj, t_now, local_traj_duration + t_now, 0.0);
    } else {
      // Call topological replanning when local segment is in collision
      /* Search topological distinctive paths */
      ROS_INFO("[Topo]: ---------");
      plan_data_.clearTopoPaths();
      list<GraphNode::Ptr> graph;
      vector<vector<Eigen::Vector3d>> raw_paths, filtered_paths, select_paths;
      topo_prm_->findTopoPaths(colli_start.front(), colli_end.back(), start_pts, end_pts, graph,
          raw_paths, filtered_paths, select_paths);

      if (select_paths.size() == 0) {
        ROS_WARN("No path.");
        return false;
      }
      plan_data_.addTopoPaths(graph, raw_paths, filtered_paths, select_paths);

      /* Optimize trajectory using different topo guiding paths */
      ROS_INFO("[Optimize]: ---------");
      t1 = ros::Time::now();

      plan_data_.topo_traj_pos1_.resize(select_paths.size());
      plan_data_.topo_traj_pos2_.resize(select_paths.size());
      vector<thread> optimize_threads;
      for (int i = 0; i < select_paths.size(); ++i) {
        optimize_threads.emplace_back(&FastPlannerManager::optimizeTopoBspline, this, t_now,
            local_traj_duration, select_paths[i], i);
        // optimizeTopoBspline(t_now, local_traj_duration,
        // select_paths[i], origin_len, i);
      }
      for (int i = 0; i < select_paths.size(); ++i) optimize_threads[i].join();

      double t_opt = (ros::Time::now() - t1).toSec();
      cout << "[planner]: optimization time: " << t_opt << endl;

      NonUniformBspline best_traj;
      selectBestTraj(best_traj);
      refineTraj(best_traj);
      double time_change = best_traj.getTimeSum() - local_traj_duration;

      local_data_.position_traj_ = best_traj;
      global_data_.setLocalTraj(local_data_.position_traj_, t_now,
          local_traj_duration + time_change + t_now, time_change);
    }
  }
  updateTrajInfo();

  double tr = (ros::Time::now() - time_now).toSec();
  ROS_WARN("Replan time: %lf", tr);

  return true;
}

void FastPlannerManager::selectBestTraj(NonUniformBspline& traj) {
  // sort by jerk
  vector<NonUniformBspline>& trajs = plan_data_.topo_traj_pos2_;
  sort(trajs.begin(), trajs.end(),
      [](NonUniformBspline& tj1, NonUniformBspline& tj2) { return tj1.getJerk() < tj2.getJerk(); });
  traj = trajs[0];
}

void FastPlannerManager::refineTraj(NonUniformBspline& best_traj) {
  ros::Time t1 = ros::Time::now();
  plan_data_.no_visib_traj_ = best_traj;

  int cost_function = BsplineOptimizer::NORMAL_PHASE;
  if (pp_.min_time_) cost_function |= BsplineOptimizer::MINTIME;

  // ViewConstraint view_cons;
  // visib_util_->calcViewConstraint(best_traj, view_cons);
  // plan_data_.view_cons_ = view_cons;
  // if (view_cons.idx_ >= 0)
  // {
  //   cost_function |= BsplineOptimizer::VIEWCONS;
  //   bspline_optimizers_[0]->setViewConstraint(view_cons);
  // }

  // Refine selected best traj
  Eigen::MatrixXd ctrl_pts = best_traj.getControlPoint();
  double dt = best_traj.getKnotSpan();
  vector<Eigen::Vector3d> start1, end1;
  best_traj.getBoundaryStates(2, 0, start1, end1);

  bspline_optimizers_[0]->setBoundaryStates(start1, end1);
  bspline_optimizers_[0]->optimize(ctrl_pts, dt, cost_function, 2, 2);
  best_traj.setUniformBspline(ctrl_pts, pp_.bspline_degree_, dt);

  vector<Eigen::Vector3d> start2, end2;
  best_traj.getBoundaryStates(2, 2, start2, end2);
  for (int i = 0; i < 3; ++i)
    std::cout << "error start: " << (start1[i] - start2[i]).norm() << std::endl;
  for (int i = 0; i < 1; ++i)
    std::cout << "error end  : " << (end1[i] - end2[i]).norm() << std::endl;
}

void FastPlannerManager::updateTrajInfo() {
  local_data_.use_minco_ = false;
  local_data_.velocity_traj_ = local_data_.position_traj_.getDerivative();
  local_data_.acceleration_traj_ = local_data_.velocity_traj_.getDerivative();

  local_data_.start_pos_ = local_data_.position_traj_.evaluateDeBoorT(0.0);
  local_data_.duration_ = local_data_.position_traj_.getTimeSum();

  local_data_.traj_id_ += 1;
}

void FastPlannerManager::reparamBspline(NonUniformBspline& bspline, double ratio,
    Eigen::MatrixXd& ctrl_pts, double& dt, double& time_inc) {
  int prev_num = bspline.getControlPoint().rows();
  double time_origin = bspline.getTimeSum();

  int seg_num = bspline.getControlPoint().rows() - pp_.bspline_degree_;
  ratio = min(1.01, ratio);

  bspline.lengthenTime(ratio);
  double duration = bspline.getTimeSum();
  dt = duration / double(seg_num);
  time_inc = duration - time_origin;

  vector<Eigen::Vector3d> point_set;
  for (double time = 0.0; time <= duration + 1e-4; time += dt)
    point_set.push_back(bspline.evaluateDeBoorT(time));
  NonUniformBspline::parameterizeToBspline(
      dt, point_set, plan_data_.local_start_end_derivative_, pp_.bspline_degree_, ctrl_pts);
  // ROS_WARN("prev: %d, new: %d", prev_num, ctrl_pts.rows());
}

void FastPlannerManager::optimizeTopoBspline(
    double start_t, double duration, vector<Eigen::Vector3d> guide_path, int traj_id) {
  auto t1 = ros::Time::now();

  // Re-parameterize B-spline according to the length of guide path
  int seg_num = topo_prm_->pathLength(guide_path) / pp_.ctrl_pt_dist;
  seg_num = max(6, seg_num);  // Min number required for optimizing
  double dt = duration / double(seg_num);
  Eigen::MatrixXd ctrl_pts = reparamLocalTraj(start_t, duration, dt);

  NonUniformBspline tmp_traj(ctrl_pts, pp_.bspline_degree_, dt);
  vector<Eigen::Vector3d> start, end;
  tmp_traj.getBoundaryStates(2, 0, start, end);

  // std::cout << "ctrl pt num: " << ctrl_pts.rows() << std::endl;

  // Discretize the guide path and align it with B-spline control points
  vector<Eigen::Vector3d> tmp_pts, guide_pts;
  if (pp_.bspline_degree_ == 3 || pp_.bspline_degree_ == 5) {
    topo_prm_->pathToGuidePts(guide_path, int(ctrl_pts.rows()) - 2, tmp_pts);
    guide_pts.insert(guide_pts.end(), tmp_pts.begin() + 2, tmp_pts.end() - 2);
    if (guide_pts.size() != int(ctrl_pts.rows()) - 6) ROS_WARN("Incorrect guide for 3 degree");
  } else if (pp_.bspline_degree_ == 4) {
    topo_prm_->pathToGuidePts(guide_path, int(2 * ctrl_pts.rows()) - 7, tmp_pts);
    for (int i = 0; i < tmp_pts.size(); ++i) {
      if (i % 2 == 1 && i >= 5 && i <= tmp_pts.size() - 6) guide_pts.push_back(tmp_pts[i]);
    }
    if (guide_pts.size() != int(ctrl_pts.rows()) - 8) ROS_WARN("Incorrect guide for 4 degree");
  }

  // std::cout << "guide pt num: " << guide_pt.size() << std::endl;

  double tm1 = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // First phase, path-guided optimization
  bspline_optimizers_[traj_id]->setBoundaryStates(start, end);
  bspline_optimizers_[traj_id]->setGuidePath(guide_pts);
  bspline_optimizers_[traj_id]->optimize(ctrl_pts, dt, BsplineOptimizer::GUIDE_PHASE, 0, 1);
  plan_data_.topo_traj_pos1_[traj_id] = NonUniformBspline(ctrl_pts, pp_.bspline_degree_, dt);

  double tm2 = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Second phase, smooth+safety+feasibility
  int cost_func = BsplineOptimizer::NORMAL_PHASE;
  // if (pp_.min_time_)
  //   cost_func |= BsplineOptimizer::MINTIME;
  bspline_optimizers_[traj_id]->setBoundaryStates(start, end);
  bspline_optimizers_[traj_id]->optimize(ctrl_pts, dt, cost_func, 1, 1);
  plan_data_.topo_traj_pos2_[traj_id] = NonUniformBspline(ctrl_pts, pp_.bspline_degree_, dt);

  double tm3 = (ros::Time::now() - t1).toSec();
  // ROS_INFO("optimization %d cost %lf, %lf, %lf seconds.", traj_id, tm1, tm2, tm3);
}

Eigen::MatrixXd FastPlannerManager::paramLocalTraj(double start_t, double& dt, double& duration) {
  vector<Eigen::Vector3d> point_set;
  vector<Eigen::Vector3d> start_end_derivative;
  global_data_.getTrajInfoInSphere(start_t, pp_.local_traj_len_, pp_.ctrl_pt_dist, point_set,
      start_end_derivative, dt, duration);

  Eigen::MatrixXd ctrl_pts;
  NonUniformBspline::parameterizeToBspline(
      dt, point_set, start_end_derivative, pp_.bspline_degree_, ctrl_pts);
  plan_data_.local_start_end_derivative_ = start_end_derivative;

  return ctrl_pts;
}

Eigen::MatrixXd FastPlannerManager::reparamLocalTraj(
    const double& start_t, const double& duration, const double& dt) {
  vector<Eigen::Vector3d> point_set;
  vector<Eigen::Vector3d> start_end_derivative;

  global_data_.getTrajInfoInDuration(start_t, duration, dt, point_set, start_end_derivative);
  plan_data_.local_start_end_derivative_ = start_end_derivative;

  /* parameterization of B-spline */
  Eigen::MatrixXd ctrl_pts;
  NonUniformBspline::parameterizeToBspline(
      dt, point_set, start_end_derivative, pp_.bspline_degree_, ctrl_pts);
  // cout << "ctrl pts:" << ctrl_pts.rows() << endl;

  return ctrl_pts;
}

void FastPlannerManager::findCollisionRange(vector<Eigen::Vector3d>& colli_start,
    vector<Eigen::Vector3d>& colli_end, vector<Eigen::Vector3d>& start_pts,
    vector<Eigen::Vector3d>& end_pts) {
  bool last_safe = true, safe;
  double t_m, t_mp;
  NonUniformBspline* initial_traj = &plan_data_.initial_local_segment_;
  initial_traj->getTimeSpan(t_m, t_mp);

  /* find range of collision */
  double t_s = -1.0;
  double t_e = t_mp;
  for (double tc = t_m; tc <= t_mp + 1e-4; tc += 0.05) {
    Eigen::Vector3d ptc = initial_traj->evaluateDeBoor(tc);
    safe = edt_environment_->evaluateCoarseEDT(ptc, -1.0) < topo_prm_->clearance_ ? false : true;

    if (last_safe && !safe) {
      colli_start.push_back(initial_traj->evaluateDeBoor(tc - 0.05));
      if (t_s < 0.0) t_s = tc - 0.05;
    } else if (!last_safe && safe) {
      colli_end.push_back(ptc);
      t_e = tc;
    }

    last_safe = safe;
  }

  if (colli_start.size() == 0) return;

  if (colli_start.size() == 1 && colli_end.size() == 0) return;

  /* find start and end safe segment */
  double dt = initial_traj->getKnotSpan();
  int sn = std::max(1, static_cast<int>(ceil((t_s - t_m) / dt)));
  dt = (t_s - t_m) / sn;

  for (double tc = t_m; tc <= t_s + 1e-4; tc += dt) {
    start_pts.push_back(initial_traj->evaluateDeBoor(tc));
  }

  dt = initial_traj->getKnotSpan();
  sn = std::max(1, static_cast<int>(ceil((t_mp - t_e) / dt)));
  dt = (t_mp - t_e) / sn;
  // std::cout << "dt: " << dt << std::endl;
  // std::cout << "sn: " << sn << std::endl;
  // std::cout << "t_m: " << t_m << std::endl;
  // std::cout << "t_mp: " << t_mp << std::endl;
  // std::cout << "t_s: " << t_s << std::endl;
  // std::cout << "t_e: " << t_e << std::endl;

  if (dt > 1e-4) {
    for (double tc = t_e; tc <= t_mp + 1e-4; tc += dt) {
      end_pts.push_back(initial_traj->evaluateDeBoor(tc));
    }
  } else {
    end_pts.push_back(initial_traj->evaluateDeBoor(t_mp));
  }
}

// !SECTION

void FastPlannerManager::planYaw(const Eigen::Vector3d& start_yaw) {
  auto t1 = ros::Time::now();
  // calculate waypoints of heading

  auto& pos = local_data_.position_traj_;
  double duration = pos.getTimeSum();

  double dt_yaw = 0.3;
  int seg_num = ceil(duration / dt_yaw);
  dt_yaw = duration / seg_num;

  const double forward_t = 2.0;
  double last_yaw = start_yaw(0);
  vector<Eigen::Vector3d> waypts;
  vector<int> waypt_idx;

  // seg_num -> seg_num - 1 points for constraint excluding the boundary states

  for (int i = 0; i < seg_num; ++i) {
    double tc = i * dt_yaw;
    Eigen::Vector3d pc = pos.evaluateDeBoorT(tc);
    double tf = min(duration, tc + forward_t);
    Eigen::Vector3d pf = pos.evaluateDeBoorT(tf);
    Eigen::Vector3d pd = pf - pc;

    Eigen::Vector3d waypt;
    if (pd.norm() > 1e-6) {
      waypt(0) = atan2(pd(1), pd(0));
      waypt(1) = waypt(2) = 0.0;
      calcNextYaw(last_yaw, waypt(0));
    } else {
      waypt = waypts.back();
    }
    last_yaw = waypt(0);
    waypts.push_back(waypt);
    waypt_idx.push_back(i);
  }

  // calculate initial control points with boundary state constraints

  Eigen::MatrixXd yaw(seg_num + 3, 1);
  yaw.setZero();

  Eigen::Matrix3d states2pts;
  states2pts << 1.0, -dt_yaw, (1 / 3.0) * dt_yaw * dt_yaw, 1.0, 0.0, -(1 / 6.0) * dt_yaw * dt_yaw,
      1.0, dt_yaw, (1 / 3.0) * dt_yaw * dt_yaw;
  yaw.block(0, 0, 3, 1) = states2pts * start_yaw;

  Eigen::Vector3d end_v = local_data_.velocity_traj_.evaluateDeBoorT(duration - 0.1);
  Eigen::Vector3d end_yaw(atan2(end_v(1), end_v(0)), 0, 0);
  calcNextYaw(last_yaw, end_yaw(0));
  yaw.block(seg_num, 0, 3, 1) = states2pts * end_yaw;

  // solve
  bspline_optimizers_[1]->setWaypoints(waypts, waypt_idx);
  int cost_func = BsplineOptimizer::SMOOTHNESS | BsplineOptimizer::WAYPOINTS |
                  BsplineOptimizer::START | BsplineOptimizer::END;

  vector<Eigen::Vector3d> start = { Eigen::Vector3d(start_yaw[0], 0, 0),
    Eigen::Vector3d(start_yaw[1], 0, 0), Eigen::Vector3d(start_yaw[2], 0, 0) };
  vector<Eigen::Vector3d> end = { Eigen::Vector3d(end_yaw[0], 0, 0),
    Eigen::Vector3d(end_yaw[1], 0, 0), Eigen::Vector3d(end_yaw[2], 0, 0) };
  bspline_optimizers_[1]->setBoundaryStates(start, end);
  bspline_optimizers_[1]->optimize(yaw, dt_yaw, cost_func, 1, 1);

  // update traj info
  local_data_.yaw_traj_.setUniformBspline(yaw, pp_.bspline_degree_, dt_yaw);
  local_data_.yawdot_traj_ = local_data_.yaw_traj_.getDerivative();
  local_data_.yawdotdot_traj_ = local_data_.yawdot_traj_.getDerivative();

  vector<double> path_yaw;
  for (int i = 0; i < waypts.size(); ++i) path_yaw.push_back(waypts[i][0]);
  plan_data_.path_yaw_ = path_yaw;
  plan_data_.dt_yaw_ = dt_yaw;
  plan_data_.dt_yaw_path_ = dt_yaw;

  std::cout << "yaw time: " << (ros::Time::now() - t1).toSec() << std::endl;
}

void FastPlannerManager::planYawExplore(const Eigen::Vector3d& start_yaw, const double& end_yaw,
    bool lookfwd, const double& relax_time) {
  if (local_data_.use_minco_) {
    solveMincoYawTraj(start_yaw, end_yaw, lookfwd, relax_time);
    return;
  }

  const int seg_num = 12;
  double dt_yaw = local_data_.duration_ / seg_num;  // time of B-spline segment
  Eigen::Vector3d start_yaw3d = start_yaw;
  std::cout << "dt_yaw: " << dt_yaw << ", start yaw: " << start_yaw3d.transpose()
            << ", end: " << end_yaw << std::endl;

  while (start_yaw3d[0] < -M_PI) start_yaw3d[0] += 2 * M_PI;
  while (start_yaw3d[0] > M_PI) start_yaw3d[0] -= 2 * M_PI;
  double last_yaw = start_yaw3d[0];

  // Yaw traj control points
  Eigen::MatrixXd yaw(seg_num + 3, 1);
  yaw.setZero();

  // Initial state
  Eigen::Matrix3d states2pts;
  states2pts << 1.0, -dt_yaw, (1 / 3.0) * dt_yaw * dt_yaw, 1.0, 0.0, -(1 / 6.0) * dt_yaw * dt_yaw,
      1.0, dt_yaw, (1 / 3.0) * dt_yaw * dt_yaw;
  yaw.block<3, 1>(0, 0) = states2pts * start_yaw3d;

  // Add waypoint constraints if look forward is enabled
  vector<Eigen::Vector3d> waypts;
  vector<int> waypt_idx;
  if (lookfwd) {
    const double forward_t = 2.0;
    const int relax_num = relax_time / dt_yaw;
    for (int i = 1; i < seg_num - relax_num; ++i) {
      double tc = i * dt_yaw;
      Eigen::Vector3d pc = local_data_.position_traj_.evaluateDeBoorT(tc);
      double tf = min(local_data_.duration_, tc + forward_t);
      Eigen::Vector3d pf = local_data_.position_traj_.evaluateDeBoorT(tf);
      Eigen::Vector3d pd = pf - pc;
      Eigen::Vector3d waypt;
      if (pd.norm() > 1e-6) {
        waypt(0) = atan2(pd(1), pd(0));
        waypt(1) = waypt(2) = 0.0;
        calcNextYaw(last_yaw, waypt(0));
      } else
        waypt = waypts.back();

      last_yaw = waypt(0);
      waypts.push_back(waypt);
      waypt_idx.push_back(i);
    }
  }
  // Final state
  Eigen::Vector3d end_yaw3d(end_yaw, 0, 0);
  calcNextYaw(last_yaw, end_yaw3d(0));
  yaw.block<3, 1>(seg_num, 0) = states2pts * end_yaw3d;

  // Debug rapid change of yaw
  if (fabs(start_yaw3d[0] - end_yaw3d[0]) >= M_PI) {
    ROS_ERROR("Yaw change rapidly!");
    std::cout << "start yaw: " << start_yaw3d[0] << ", " << end_yaw3d[0] << std::endl;
  }

  // // Interpolate start and end value for smoothness
  // for (int i = 1; i < seg_num; ++i)
  // {
  //   double tc = i * dt_yaw;
  //   Eigen::Vector3d waypt = (1 - double(i) / seg_num) * start_yaw3d + double(i) / seg_num *
  //   end_yaw3d;
  //   std::cout << "i: " << i << ", wp: " << waypt[0] << ", ";
  //   calcNextYaw(last_yaw, waypt(0));
  // }
  // std::cout << "" << std::endl;

  auto t1 = ros::Time::now();

  // Call B-spline optimization solver
  int cost_func = BsplineOptimizer::SMOOTHNESS | BsplineOptimizer::START | BsplineOptimizer::END |
                  BsplineOptimizer::WAYPOINTS;
  vector<Eigen::Vector3d> start = { Eigen::Vector3d(start_yaw3d[0], 0, 0),
    Eigen::Vector3d(start_yaw3d[1], 0, 0), Eigen::Vector3d(start_yaw3d[2], 0, 0) };
  vector<Eigen::Vector3d> end = { Eigen::Vector3d(end_yaw3d[0], 0, 0), Eigen::Vector3d(0, 0, 0) };
  bspline_optimizers_[1]->setBoundaryStates(start, end);
  bspline_optimizers_[1]->setWaypoints(waypts, waypt_idx);
  bspline_optimizers_[1]->optimize(yaw, dt_yaw, cost_func, 1, 1);

  // std::cout << "2: " << (ros::Time::now() - t1).toSec() << std::endl;

  // Update traj info
  local_data_.yaw_traj_.setUniformBspline(yaw, 3, dt_yaw);
  local_data_.yawdot_traj_ = local_data_.yaw_traj_.getDerivative();
  local_data_.yawdotdot_traj_ = local_data_.yawdot_traj_.getDerivative();
  plan_data_.dt_yaw_ = dt_yaw;

  // plan_data_.path_yaw_ = path;
  // plan_data_.dt_yaw_path_ = dt_yaw * subsp;
}

void FastPlannerManager::calcNextYaw(const double& last_yaw, double& yaw) {
  // round yaw to [-PI, PI]
  double round_last = last_yaw;
  while (round_last < -M_PI) {
    round_last += 2 * M_PI;
  }
  while (round_last > M_PI) {
    round_last -= 2 * M_PI;
  }

  double diff = yaw - round_last;
  if (fabs(diff) <= M_PI) {
    yaw = last_yaw + diff;
  } else if (diff > M_PI) {
    yaw = last_yaw + diff - 2 * M_PI;
  } else if (diff < -M_PI) {
    yaw = last_yaw + diff + 2 * M_PI;
  }
}

void FastPlannerManager::angleLimite(double& angle) {
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
}

}  // namespace fast_planner
