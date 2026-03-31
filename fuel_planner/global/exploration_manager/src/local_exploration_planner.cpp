#include <exploration_manager/local_exploration_planner.h>
#include <exploration_manager/fast_exploration_manager.h>

#include <active_perception/graph_node.h>
#include <exploration_manager/expl_data.h>
#include <plan_env/edt_environment.h>
#include <plan_env/raycast.h>
#include <plan_env/sdf_map.h>
#include <plan_manage/planner_manager.h>

#include <algorithm>
#include <cmath>

namespace fast_planner {
namespace {
double wrapYaw(double yaw) {
  while (yaw > M_PI) yaw -= 2.0 * M_PI;
  while (yaw < -M_PI) yaw += 2.0 * M_PI;
  return yaw;
}

double unwrapYawToward(double reference, double target) {
  return reference + wrapYaw(target - reference);
}

void ensureSegmentHasInteriorWaypoint(std::vector<Eigen::Vector3d>& segment) {
  if (segment.size() == 2) {
    segment.insert(segment.begin() + 1, 0.5 * (segment.front() + segment.back()));
  }
}

Eigen::Vector3d computeThroughGoalVelocity(const std::vector<Eigen::Vector3d>& path,
                                           const Eigen::Vector3d& cur_vel,
                                           const fast_planner::PlanParameters& pp) {
  if (path.size() < 2) return Eigen::Vector3d::Zero();

  Eigen::Vector3d dir_sum = Eigen::Vector3d::Zero();
  double tail_len = 0.0;
  for (int i = static_cast<int>(path.size()) - 1; i > 0; --i) {
    const Eigen::Vector3d diff = path[i] - path[i - 1];
    const double seg_len = diff.norm();
    if (seg_len <= 1e-3) continue;
    dir_sum += diff;
    tail_len += seg_len;
    if (tail_len >= 3.0) break;
  }
  if (dir_sum.squaredNorm() < 1e-6) return Eigen::Vector3d::Zero();

  const Eigen::Vector3d dir = dir_sum.normalized();
  const double lookahead_len = BubbleAstar::pathLength(path);

  const double cur_forward_speed = std::max(0.0, cur_vel.dot(dir));
  const double nominal_speed = std::max(0.6, pp.max_vel_ * 0.55);
  const double decel_limited_speed =
      std::sqrt(std::max(0.0, 2.0 * std::max(0.1, pp.max_acc_) * std::max(0.5, lookahead_len)));
  const double target_speed =
      std::min(pp.max_vel_ * 0.8, std::max(cur_forward_speed * 0.7, nominal_speed));
  return dir * std::min(target_speed, decel_limited_speed);
}

void appendSegmentPath(const std::vector<Eigen::Vector3d>& segment, std::vector<Eigen::Vector3d>& stitched) {
  if (segment.empty()) return;
  if (stitched.empty()) {
    stitched = segment;
    return;
  }
  const size_t start_idx = (segment.front() - stitched.back()).norm() < 1e-3 ? 1 : 0;
  stitched.insert(stitched.end(), segment.begin() + start_idx, segment.end());
}

void truncatePathByLength(const double max_len, std::vector<Eigen::Vector3d>& path) {
  if (path.size() < 2 || max_len <= 0.0) return;

  double len = 0.0;
  std::vector<Eigen::Vector3d> truncated = { path.front() };
  for (size_t i = 1; i < path.size(); ++i) {
    const Eigen::Vector3d& prev = truncated.back();
    const Eigen::Vector3d& cur = path[i];
    const double seg_len = (cur - prev).norm();
    if (seg_len <= 1e-6) continue;
    if (len + seg_len <= max_len) {
      truncated.push_back(cur);
      len += seg_len;
      continue;
    }
    const double remain = std::max(0.0, max_len - len);
    if (remain > 1e-3) {
      truncated.push_back(prev + (cur - prev).normalized() * remain);
    }
    break;
  }
  if (truncated.size() == 1) truncated.push_back(path.back());
  path.swap(truncated);
}
}  // namespace

void LocalExplorationPlanner::initialize(
    const shared_ptr<FastPlannerManager>& planner_manager,
    const shared_ptr<EDTEnvironment>& edt_environment, const shared_ptr<SDFMap>& sdf_map,
    const shared_ptr<ExplorationData>& exploration_data, const double relax_time) {
  planner_manager_ = planner_manager;
  edt_environment_ = edt_environment;
  sdf_map_ = sdf_map;
  ed_ = exploration_data;
  relax_time_ = relax_time;
}

int LocalExplorationPlanner::planToViewpoint(
    const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d& yaw,
    const vector<PathSegmentWithYaw>& path_segments) {
  ros::Time t1 = ros::Time::now();
  vector<Vector3d> stitched_path;
  vector<PathSegmentWithYaw> safe_segments;
  if (!sanitizePathSegments(pos, path_segments, safe_segments, stitched_path)) return FAIL;
  int backend_result = solveMincoBackend(vel, acc, yaw, safe_segments, stitched_path);
  if (backend_result != SUCCEED) return backend_result;

  double traj_plan_time = (ros::Time::now() - t1).toSec();
  ROS_WARN("Traj: %lf, yaw: %lf", traj_plan_time, 0.0);
  return SUCCEED;
}

bool LocalExplorationPlanner::sanitizePathSegments(const Vector3d& raw_start,
                                                   const vector<PathSegmentWithYaw>& raw_segments,
                                                   vector<PathSegmentWithYaw>& safe_segments,
                                                   vector<Vector3d>& stitched_path) {
  if (raw_segments.empty()) {
    ROS_ERROR("Empty path segments for local exploration planner");
    return false;
  }

  Vector3d search_start = raw_start;
  if (!findSearchStart(raw_start, search_start)) {
    ROS_ERROR_STREAM("Failed to project exploration start into box/free space: "
                     << raw_start.transpose());
    return false;
  }

  stitched_path.clear();
  safe_segments.clear();
  Vector3d seg_start = search_start;
  auto segmentIsSafe = [&](const vector<Vector3d>& path) {
    if (path.size() < 2) return false;
    if (!planner_manager_->isPointSafeInExploreSpace(path.front(), false) ||
        !planner_manager_->isPointSafeInExploreSpace(path.back(), false)) {
      return false;
    }
    for (size_t j = 1; j < path.size(); ++j) {
      if (!planner_manager_->isSegmentSafeInExploreSpace(path[j - 1], path[j], -1.0, false)) {
        return false;
      }
    }
    return true;
  };
  auto rebuildStrictSegment = [&](const Vector3d& start, const Vector3d& goal,
                                  vector<Vector3d>& safe_segment) {
    if (!planner_manager_->path_finder_) return false;
    planner_manager_->path_finder_->reset();
    const bool prev_optimistic_unknown = planner_manager_->path_finder_->getOptimisticUnknown();
    planner_manager_->path_finder_->setOptimisticUnknown(false);
    const int search_status = planner_manager_->path_finder_->search(start, goal);
    planner_manager_->path_finder_->setOptimisticUnknown(prev_optimistic_unknown);
    if (search_status != BubbleAstar::REACH_END) {
      return false;
    }
    return planner_manager_->sanitizeExplorePath(planner_manager_->path_finder_->getPath(), safe_segment);
  };
  for (size_t i = 0; i < raw_segments.size(); ++i) {
    Vector3d safe_goal;
    if (!planner_manager_->projectToValidExplorePoint(raw_segments[i].viewpoint, safe_goal, 3.0)) {
      ROS_ERROR_STREAM("Failed to project exploration goal into box/free space: "
                       << raw_segments[i].viewpoint.transpose());
      return false;
    }

    vector<Vector3d> safe_segment;
    if (!planner_manager_->sanitizeExplorePath(raw_segments[i].path, safe_segment)) {
      ROS_WARN_STREAM("Path segment " << i
                      << " is invalid after optimistic sanitization, retrying with strict local search.");
      if (!rebuildStrictSegment(seg_start, safe_goal, safe_segment)) {
        ROS_ERROR_STREAM("Path segment " << i << " is invalid in exploration space.");
        return false;
      }
    }
    if (safe_segment.empty()) {
      ROS_ERROR_STREAM("Path segment " << i << " is empty after sanitization.");
      return false;
    }

    safe_segment.front() = seg_start;
    safe_segment.back() = safe_goal;

    vector<Vector3d> shortened_segment = safe_segment;
    shortenPath(shortened_segment);
    ensureSegmentHasInteriorWaypoint(shortened_segment);

    if (segmentIsSafe(shortened_segment)) {
      safe_segment.swap(shortened_segment);
    } else {
      ensureSegmentHasInteriorWaypoint(safe_segment);
      if (!segmentIsSafe(safe_segment)) {
        ROS_WARN_STREAM("Sanitized path segment " << i
                        << " is still unsafe after endpoint alignment, retrying with strict local search.");
        if (!rebuildStrictSegment(seg_start, safe_goal, safe_segment)) {
          ROS_ERROR_STREAM("Failed to recover safe path segment to viewpoint " << i);
          return false;
        }
        safe_segment.front() = seg_start;
        safe_segment.back() = safe_goal;
        ensureSegmentHasInteriorWaypoint(safe_segment);
        if (!segmentIsSafe(safe_segment)) {
          ROS_ERROR_STREAM("Strict local search still produced unsafe path segment to viewpoint " << i);
          return false;
        }
      }
    }

    PathSegmentWithYaw safe_seg;
    safe_seg.path = safe_segment;
    safe_seg.viewpoint = safe_goal;
    safe_seg.yaw = raw_segments[i].yaw;
    safe_segments.push_back(safe_seg);

    appendSegmentPath(safe_segment, stitched_path);
    seg_start = safe_goal;
  }

  vector<Vector3d> fallback_path = stitched_path;
  truncatePathByLength(12.0, fallback_path);
  shortenPath(fallback_path);
  ensureSegmentHasInteriorWaypoint(fallback_path);
  ed_->path_lookahead_ = fallback_path;
  ed_->path_next_goal_ = fallback_path;
  ed_->lookahead_path_segments_ = safe_segments;
  ed_->lookahead_goals_.clear();
  ed_->lookahead_yaws_.clear();
  for (const auto& seg : safe_segments) {
    ed_->lookahead_goals_.push_back(seg.viewpoint);
    ed_->lookahead_yaws_.push_back(seg.yaw);
  }
  if (!safe_segments.empty()) ed_->next_goal_ = safe_segments.front().viewpoint;
  return true;
}

int LocalExplorationPlanner::solveMincoBackend(
    const Vector3d& vel, const Vector3d& acc, const Vector3d& yaw,
    const vector<PathSegmentWithYaw>& safe_segments, const vector<Vector3d>& stitched_path) {
  if (safe_segments.empty()) {
    ROS_ERROR("Empty lookahead path segments in solveMincoBackend");
    return FAIL;
  }

  const double target_yaw = unwrapYawToward(yaw[0], safe_segments.front().yaw);
  const double diff = fabs(target_yaw - yaw[0]);
  const double time_lb = min(diff, 2 * M_PI - diff) / ViewNode::yd_;
  ROS_INFO_STREAM("[explore local] long_viewpoint_mode segments=" << safe_segments.size());

  if (planner_manager_->planExploreTrajLong(safe_segments, vel, acc, time_lb)) {
    ed_->lookahead_arrival_times_ = planner_manager_->getExploreViewpointArrivalTimes();
  } else {
    const double lookahead_len = BubbleAstar::pathLength(stitched_path);
    const bool has_followup_goal = safe_segments.size() > 1;
    const bool stop_at_goal = !has_followup_goal && lookahead_len < 1.0;
    const Eigen::Vector3d terminal_vel =
        stop_at_goal ? Eigen::Vector3d::Zero()
                     : computeThroughGoalVelocity(stitched_path, vel, planner_manager_->pp_);
    ROS_WARN_STREAM("[explore local] fallback_path_mode lookahead_len=" << lookahead_len
                    << " stop_at_goal=" << stop_at_goal
                    << " terminal_vel=" << terminal_vel.transpose());
    if (!planner_manager_->planExploreTraj(stitched_path, vel, acc, time_lb, target_yaw,
                                           terminal_vel, stop_at_goal)) {
      return FAIL;
    }
    planner_manager_->planYawExplore(yaw, target_yaw, true, relax_time_);
    ed_->lookahead_arrival_times_.clear();
  }

  if (planner_manager_->local_data_.duration_ < time_lb - 0.1) {
    ROS_ERROR("Lower bound not satified!");
  }
  return SUCCEED;
}

bool LocalExplorationPlanner::findSearchStart(const Vector3d& raw_start,
                                              Vector3d& search_start) const {
  const bool ok = planner_manager_->projectToValidExplorePoint(raw_start, search_start, 3.0);
  if (ok && (search_start - raw_start).norm() > 1e-3) {
    ROS_WARN_STREAM("Exploration start snapped into exploration space from "
                    << raw_start.transpose() << " to " << search_start.transpose());
  }
  return ok;
}

void LocalExplorationPlanner::shortenPath(vector<Vector3d>& path) const {
  if (path.empty()) {
    ROS_ERROR("Empty path to shorten");
    return;
  }

  const double dist_thresh = 3.0;
  const double near_unknown_block_radius =
      std::max(1.0, 2.5 * planner_manager_->pp_.minco_safe_distance_);
  const Eigen::Vector3d path_start = path.front();
  vector<Vector3d> short_tour = { path.front() };
  for (size_t i = 1; i + 1 < path.size(); ++i) {
    if ((path[i] - short_tour.back()).norm() > dist_thresh) {
      short_tour.push_back(path[i]);
    } else {
      ViewNode::caster_->input(short_tour.back(), path[i + 1]);
      Eigen::Vector3i idx;
      while (ViewNode::caster_->nextId(idx) && ros::ok()) {
        Eigen::Vector3d ckpt;
        edt_environment_->sdf_map_->indexToPos(idx, ckpt);
        const bool unknown_near_start =
            edt_environment_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN &&
            (ckpt - path_start).norm() <= near_unknown_block_radius;
        if (!edt_environment_->sdf_map_->isInBox(idx) ||
            edt_environment_->sdf_map_->getInflateOccupancy(idx) == 1 || unknown_near_start) {
          short_tour.push_back(path[i]);
          break;
        }
      }
    }
  }

  if ((path.back() - short_tour.back()).norm() > 1e-3) short_tour.push_back(path.back());
  if (short_tour.size() == 2) {
    short_tour.insert(short_tour.begin() + 1, 0.5 * (short_tour[0] + short_tour[1]));
  }
  path = short_tour;
}

}  // namespace fast_planner
