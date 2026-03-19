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
const char* occToString(int occ) {
  switch (occ) {
    case SDFMap::FREE:
      return "FREE";
    case SDFMap::OCCUPIED:
      return "OCCUPIED";
    case SDFMap::UNKNOWN:
      return "UNKNOWN";
    default:
      return "OUT_OF_MAP";
  }
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
    const Vector3d& next_pos, const double next_yaw) {
  ros::Time t1 = ros::Time::now();
  Vector3d safe_goal;
  if (!planner_manager_->projectToValidExplorePoint(next_pos, safe_goal, 3.0)) {
    ROS_ERROR_STREAM("Failed to project exploration goal into box/free space: "
                     << next_pos.transpose());
    return FAIL;
  }

  if (!planGeometricPathFrontend(pos, safe_goal)) return FAIL;
  int backend_result = solveMincoBackend(vel, acc, yaw, safe_goal, next_yaw);
  if (backend_result != SUCCEED) return backend_result;

  double traj_plan_time = (ros::Time::now() - t1).toSec();
  ROS_WARN("Traj: %lf, yaw: %lf", traj_plan_time, 0.0);
  return SUCCEED;
}

bool LocalExplorationPlanner::planGeometricPathFrontend(const Vector3d& raw_start,
                                                        const Vector3d& goal) {
  Vector3d search_start = raw_start;
  if (!findSearchStart(raw_start, search_start)) {
    ROS_ERROR_STREAM("Failed to project exploration start into box/free space: "
                     << raw_start.transpose());
    return false;
  }

  planner_manager_->path_finder_->reset();
  if (planner_manager_->path_finder_->search(search_start, goal) != Astar::REACH_END) {
    const bool start_in_map = sdf_map_->isInMap(search_start);
    const bool raw_start_in_map = sdf_map_->isInMap(raw_start);
    const bool goal_in_map = sdf_map_->isInMap(goal);
    Eigen::Vector3i start_idx(-1, -1, -1);
    Eigen::Vector3i goal_idx(-1, -1, -1);
    if (start_in_map) sdf_map_->posToIndex(search_start, start_idx);
    if (goal_in_map) sdf_map_->posToIndex(goal, goal_idx);

    const int raw_occ = raw_start_in_map ? sdf_map_->getOccupancy(raw_start) : -1;
    const int raw_inflate = raw_start_in_map ? sdf_map_->getInflateOccupancy(raw_start) : -1;
    const int goal_occ = goal_in_map ? sdf_map_->getOccupancy(goal) : -1;
    const int goal_inflate = goal_in_map ? sdf_map_->getInflateOccupancy(goal) : -1;

    ROS_ERROR_STREAM(
        "A* search failed."
        << " start=" << search_start.transpose()
        << " idx=" << start_idx.transpose()
        << " in_map=" << start_in_map
        << " occ=" << occToString(start_in_map ? sdf_map_->getOccupancy(search_start) : -1)
        << " inflate_occ=" << (start_in_map ? sdf_map_->getInflateOccupancy(search_start) : -1)
        << " raw_start=" << raw_start.transpose()
        << " raw_occ=" << occToString(raw_occ)
        << " raw_inflate_occ=" << raw_inflate
        << " goal=" << goal.transpose()
        << " idx=" << goal_idx.transpose()
        << " in_map=" << goal_in_map
        << " occ=" << occToString(goal_occ)
        << " inflate_occ=" << goal_inflate);
    ROS_ERROR("No path to next viewpoint");
    return false;
  }

  ed_->path_next_goal_ = planner_manager_->path_finder_->getPath();
  vector<Vector3d> safe_path;
  if (!planner_manager_->sanitizeExplorePath(ed_->path_next_goal_, safe_path)) {
    ROS_ERROR("Path frontend produced a path outside exploration space.");
    return false;
  }
  ed_->path_next_goal_ = safe_path;
  shortenPath(ed_->path_next_goal_);
  return true;
}

int LocalExplorationPlanner::solveMincoBackend(
    const Vector3d& vel, const Vector3d& acc, const Vector3d& yaw, const Vector3d& next_pos,
    const double next_yaw) {
  const double diff = fabs(next_yaw - yaw[0]);
  const double time_lb = min(diff, 2 * M_PI - diff) / ViewNode::yd_;
  const double radius_far = 5.0;
  const double radius_close = 1.5;
  const double len = Astar::pathLength(ed_->path_next_goal_);

  if (len < radius_close) {
    ed_->next_goal_ = next_pos;
    if (!planner_manager_->planExploreTraj(ed_->path_next_goal_, vel, acc, time_lb, next_yaw))
      return FAIL;
  } else if (len > radius_far) {
    double len2 = 0.0;
    vector<Eigen::Vector3d> truncated_path = { ed_->path_next_goal_.front() };
    for (int i = 1; i < ed_->path_next_goal_.size() && len2 < radius_far; ++i) {
      auto cur_pt = ed_->path_next_goal_[i];
      len2 += (cur_pt - truncated_path.back()).norm();
      truncated_path.push_back(cur_pt);
    }
    ed_->next_goal_ = truncated_path.back();
    if (!planner_manager_->planExploreTraj(truncated_path, vel, acc, time_lb, next_yaw))
      return FAIL;
  } else {
    ed_->next_goal_ = next_pos;
    if (!planner_manager_->planExploreTraj(ed_->path_next_goal_, vel, acc, time_lb, next_yaw))
      return FAIL;
  }

  if (planner_manager_->local_data_.duration_ < time_lb - 0.1) {
    ROS_ERROR("Lower bound not satified!");
  }

  planner_manager_->planYawExplore(yaw, next_yaw, true, relax_time_);
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
  vector<Vector3d> short_tour = { path.front() };
  for (int i = 1; i < path.size() - 1; ++i) {
    if ((path[i] - short_tour.back()).norm() > dist_thresh) {
      short_tour.push_back(path[i]);
    } else {
      ViewNode::caster_->input(short_tour.back(), path[i + 1]);
      Eigen::Vector3i idx;
      while (ViewNode::caster_->nextId(idx) && ros::ok()) {
        if (!edt_environment_->sdf_map_->isInBox(idx) ||
            edt_environment_->sdf_map_->getInflateOccupancy(idx) == 1 ||
            edt_environment_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
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
