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
#include <limits>

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

  if (!planGeometricPathFrontend(pos, next_pos)) return FAIL;
  int backend_result = solveMincoBackend(vel, acc, yaw, next_pos, next_yaw);
  if (backend_result != SUCCEED) return backend_result;

  double traj_plan_time = (ros::Time::now() - t1).toSec();
  ROS_WARN("Traj: %lf, yaw: %lf", traj_plan_time, 0.0);
  return SUCCEED;
}

bool LocalExplorationPlanner::planGeometricPathFrontend(const Vector3d& raw_start,
                                                        const Vector3d& goal) {
  Vector3d search_start = raw_start;
  findSearchStart(raw_start, search_start);

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
    planner_manager_->planExploreTraj(ed_->path_next_goal_, vel, acc, time_lb);
  } else if (len > radius_far) {
    double len2 = 0.0;
    vector<Eigen::Vector3d> truncated_path = { ed_->path_next_goal_.front() };
    for (int i = 1; i < ed_->path_next_goal_.size() && len2 < radius_far; ++i) {
      auto cur_pt = ed_->path_next_goal_[i];
      len2 += (cur_pt - truncated_path.back()).norm();
      truncated_path.push_back(cur_pt);
    }
    ed_->next_goal_ = truncated_path.back();
    planner_manager_->planExploreTraj(truncated_path, vel, acc, time_lb);
  } else {
    ed_->next_goal_ = next_pos;
    planner_manager_->planExploreTraj(ed_->path_next_goal_, vel, acc, time_lb);
  }

  if (planner_manager_->local_data_.duration_ < time_lb - 0.1) {
    ROS_ERROR("Lower bound not satified!");
  }

  planner_manager_->planYawExplore(yaw, next_yaw, true, relax_time_);
  return SUCCEED;
}

bool LocalExplorationPlanner::findSearchStart(const Vector3d& raw_start,
                                              Vector3d& search_start) const {
  search_start = raw_start;
  if (!sdf_map_ || !sdf_map_->isInMap(raw_start)) return false;
  if (sdf_map_->getOccupancy(raw_start) != SDFMap::UNKNOWN) return true;

  Eigen::Vector3i start_idx;
  sdf_map_->posToIndex(raw_start, start_idx);

  const double resolution = sdf_map_->getResolution();
  const int max_radius = std::max(1, static_cast<int>(std::ceil(2.0 / resolution)));
  double best_dist2 = std::numeric_limits<double>::infinity();
  bool found = false;

  for (int radius = 0; radius <= max_radius; ++radius) {
    for (int dx = -radius; dx <= radius; ++dx) {
      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dz = -radius; dz <= radius; ++dz) {
          if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != radius) continue;

          const Eigen::Vector3i idx = start_idx + Eigen::Vector3i(dx, dy, dz);
          if (!sdf_map_->isInMap(idx)) continue;
          if (sdf_map_->getOccupancy(idx) != SDFMap::FREE) continue;
          if (sdf_map_->getInflateOccupancy(idx) != 0) continue;

          Vector3d candidate;
          sdf_map_->indexToPos(idx, candidate);
          const double dist2 = (candidate - raw_start).squaredNorm();
          if (dist2 < best_dist2) {
            best_dist2 = dist2;
            search_start = candidate;
            found = true;
          }
        }
      }
    }

    if (found) {
      ROS_WARN_STREAM("Exploration start snapped from unknown cell " << raw_start.transpose()
                                                                     << " to "
                                                                     << search_start.transpose());
      return true;
    }
  }

  return false;
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
        if (edt_environment_->sdf_map_->getInflateOccupancy(idx) == 1 ||
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
