#include <exploration_manager/fast_exploration_manager.h>
#include <exploration_manager/global_exploration_planner.h>
#include <exploration_manager/local_exploration_planner.h>

#include <active_perception/graph_node.h>
#include <plan_env/sdf_map.h>
#include <plan_env/edt_environment.h>
#include <plan_env/raycast.h>
#include <active_perception/frontier_finder.h>
#include <plan_manage/planner_manager.h>
#include <exploration_manager/expl_data.h>

#include <fstream>

using namespace Eigen;

namespace fast_planner {

// SECTION interfaces for setup and query

FastExplorationManager::FastExplorationManager() {
}

FastExplorationManager::~FastExplorationManager() {
  ViewNode::astar_.reset();
  ViewNode::caster_.reset();
  ViewNode::map_.reset();
}

void FastExplorationManager::initialize(ros::NodeHandle& nh) {
  planner_manager_.reset(new FastPlannerManager);
  planner_manager_->initPlanModules(nh);
  edt_environment_ = planner_manager_->edt_environment_;
  sdf_map_ = edt_environment_->sdf_map_;
  frontier_finder_.reset(new FrontierFinder(edt_environment_, nh));

  ed_.reset(new ExplorationData);
  ep_.reset(new ExplorationParam);

  nh.param("exploration/refine_local", ep_->refine_local_, true);
  nh.param("exploration/refined_num", ep_->refined_num_, -1);
  nh.param("exploration/refined_radius", ep_->refined_radius_, -1.0);
  nh.param("exploration/top_view_num", ep_->top_view_num_, -1);
  nh.param("exploration/max_decay", ep_->max_decay_, -1.0);
  nh.param("exploration/tsp_dir", ep_->tsp_dir_, string("null"));
  nh.param("exploration/relax_time", ep_->relax_time_, 1.0);

  nh.param("exploration/vm", ViewNode::vm_, -1.0);
  nh.param("exploration/am", ViewNode::am_, -1.0);
  nh.param("exploration/yd", ViewNode::yd_, -1.0);
  nh.param("exploration/ydd", ViewNode::ydd_, -1.0);
  nh.param("exploration/w_dir", ViewNode::w_dir_, -1.0);

  ViewNode::astar_.reset(new Astar);
  ViewNode::astar_->init(nh, edt_environment_);
  ViewNode::map_ = sdf_map_;

  double resolution_ = sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  sdf_map_->getRegion(origin, size);
  ViewNode::caster_.reset(new RayCaster);
  ViewNode::caster_->setParams(resolution_, origin);

  planner_manager_->path_finder_->lambda_heu_ = 1.0;
  // planner_manager_->path_finder_->max_search_time_ = 0.05;
  planner_manager_->path_finder_->max_search_time_ = 1.0;

  // Initialize TSP par file
  ofstream par_file(ep_->tsp_dir_ + "/single.par");
  par_file << "PROBLEM_FILE = " << ep_->tsp_dir_ << "/single.tsp\n";
  par_file << "GAIN23 = NO\n";
  par_file << "OUTPUT_TOUR_FILE =" << ep_->tsp_dir_ << "/single.txt\n";
  par_file << "RUNS = 1\n";

  // Analysis
  // ofstream fout;
  // fout.open("/home/boboyu/Desktop/RAL_Time/frontier.txt");
  // fout.close();

  global_planner_.reset(new GlobalExplorationPlanner);
  global_planner_->initialize(frontier_finder_, planner_manager_, ed_, ep_);

  local_planner_.reset(new LocalExplorationPlanner);
  local_planner_->initialize(planner_manager_, edt_environment_, sdf_map_, ed_, ep_->relax_time_);
}

int FastExplorationManager::planExploreMotion(
    const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d& yaw) {
  const ros::Time planning_start = ros::Time::now();
  Vector3d next_pos;
  double next_yaw;
  int global_result = global_planner_->computeNextViewpoint(pos, vel, yaw, next_pos, next_yaw);
  if (global_result != SUCCEED) return global_result;

  int local_result = local_planner_->planToViewpoint(pos, vel, acc, yaw, next_pos, next_yaw);
  if (local_result != SUCCEED) return local_result;

  double total = (ros::Time::now() - planning_start).toSec();
  ROS_WARN("Total time: %lf", total);
  ROS_ERROR_COND(total > 0.1, "Total time too long!!!");

  return SUCCEED;
}

}  // namespace fast_planner
