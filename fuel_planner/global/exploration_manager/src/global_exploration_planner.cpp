#include <exploration_manager/global_exploration_planner.h>
#include <exploration_manager/fast_exploration_manager.h>

#include <active_perception/frontier_finder.h>
#include <active_perception/graph_node.h>
#include <active_perception/graph_search.h>
#include <active_perception/perception_utils.h>
#include <exploration_manager/expl_data.h>
#include <cmath>
#include <fstream>
#include <lkh_tsp_solver/lkh_interface.h>
#include <plan_manage/planner_manager.h>

namespace fast_planner {

void GlobalExplorationPlanner::initialize(
    const shared_ptr<FrontierFinder>& frontier_finder,
    const shared_ptr<FastPlannerManager>& planner_manager,
    const shared_ptr<ExplorationData>& exploration_data,
    const shared_ptr<ExplorationParam>& exploration_param) {
  frontier_finder_ = frontier_finder;
  planner_manager_ = planner_manager;
  ed_ = exploration_data;
  ep_ = exploration_param;
}

int GlobalExplorationPlanner::computeNextViewpoint(
    const Vector3d& pos, const Vector3d& vel, const Vector3d& yaw, Vector3d& next_pos,
    double& next_yaw) {
  ros::Time t1 = ros::Time::now();
  ed_->views_.clear();
  ed_->global_tour_.clear();

  frontier_finder_->searchFrontiers();

  double frontier_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  frontier_finder_->computeFrontiersToVisit();
  frontier_finder_->getFrontiers(ed_->frontiers_);
  frontier_finder_->getFrontierBoxes(ed_->frontier_boxes_);
  frontier_finder_->getDormantFrontiers(ed_->dead_frontiers_);

  if (ed_->frontiers_.empty()) {
    ROS_WARN("No coverable frontier.");
    return NO_FRONTIER;
  }

  frontier_finder_->getTopViewpointsInfo(pos, ed_->points_, ed_->yaws_, ed_->averages_);
  for (int i = 0; i < ed_->points_.size(); ++i) {
    ed_->views_.push_back(
        ed_->points_[i] + 2.0 * Vector3d(cos(ed_->yaws_[i]), sin(ed_->yaws_[i]), 0));
  }

  double view_time = (ros::Time::now() - t1).toSec();
  ROS_WARN(
      "Frontier: %zu, t: %lf, viewpoint: %zu, t: %lf", ed_->frontiers_.size(), frontier_time,
      ed_->points_.size(), view_time);

  if (ed_->points_.size() > 1) {
    vector<int> indices;
    findGlobalTour(pos, vel, yaw, indices);

    if (ep_->refine_local_) {
      t1 = ros::Time::now();

      ed_->refined_ids_.clear();
      ed_->unrefined_points_.clear();
      int knum = min(int(indices.size()), ep_->refined_num_);
      for (int i = 0; i < knum; ++i) {
        auto tmp = ed_->points_[indices[i]];
        ed_->unrefined_points_.push_back(tmp);
        ed_->refined_ids_.push_back(indices[i]);
        if ((tmp - pos).norm() > ep_->refined_radius_ && ed_->refined_ids_.size() >= 2) break;
      }

      ed_->n_points_.clear();
      vector<vector<double>> n_yaws;
      frontier_finder_->getViewpointsInfo(
          pos, ed_->refined_ids_, ep_->top_view_num_, ep_->max_decay_, ed_->n_points_, n_yaws);

      ed_->refined_points_.clear();
      ed_->refined_views_.clear();
      vector<double> refined_yaws;
      refineLocalTour(pos, vel, yaw, ed_->n_points_, n_yaws, ed_->refined_points_, refined_yaws);
      next_pos = ed_->refined_points_[0];
      next_yaw = refined_yaws[0];

      for (int i = 0; i < ed_->refined_points_.size(); ++i) {
        Vector3d view = ed_->refined_points_[i] +
                        2.0 * Vector3d(cos(refined_yaws[i]), sin(refined_yaws[i]), 0);
        ed_->refined_views_.push_back(view);
      }
      ed_->refined_views1_.clear();
      ed_->refined_views2_.clear();
      for (int i = 0; i < ed_->refined_points_.size(); ++i) {
        vector<Vector3d> v1, v2;
        frontier_finder_->percep_utils_->setPose(ed_->refined_points_[i], refined_yaws[i]);
        frontier_finder_->percep_utils_->getFOV(v1, v2);
        ed_->refined_views1_.insert(ed_->refined_views1_.end(), v1.begin(), v1.end());
        ed_->refined_views2_.insert(ed_->refined_views2_.end(), v2.begin(), v2.end());
      }
      double local_time = (ros::Time::now() - t1).toSec();
      ROS_WARN("Local refine time: %lf", local_time);
    } else {
      next_pos = ed_->points_[indices[0]];
      next_yaw = ed_->yaws_[indices[0]];
    }
  } else if (ed_->points_.size() == 1) {
    frontier_finder_->updateFrontierCostMatrix();
    ed_->global_tour_ = { pos, ed_->points_[0] };
    ed_->refined_tour_.clear();
    ed_->refined_views1_.clear();
    ed_->refined_views2_.clear();

    if (ep_->refine_local_) {
      ed_->refined_ids_ = { 0 };
      ed_->unrefined_points_ = { ed_->points_[0] };
      ed_->n_points_.clear();
      vector<vector<double>> n_yaws;
      frontier_finder_->getViewpointsInfo(
          pos, { 0 }, ep_->top_view_num_, ep_->max_decay_, ed_->n_points_, n_yaws);

      double min_cost = 100000.0;
      int min_cost_id = -1;
      vector<Vector3d> tmp_path;
      for (int i = 0; i < ed_->n_points_[0].size(); ++i) {
        auto tmp_cost = ViewNode::computeCost(
            pos, ed_->n_points_[0][i], yaw[0], n_yaws[0][i], vel, yaw[1], tmp_path);
        if (tmp_cost < min_cost) {
          min_cost = tmp_cost;
          min_cost_id = i;
        }
      }
      next_pos = ed_->n_points_[0][min_cost_id];
      next_yaw = n_yaws[0][min_cost_id];
      ed_->refined_points_ = { next_pos };
      ed_->refined_views_ = { next_pos + 2.0 * Vector3d(cos(next_yaw), sin(next_yaw), 0) };
    } else {
      next_pos = ed_->points_[0];
      next_yaw = ed_->yaws_[0];
    }
  } else {
    ROS_ERROR("Empty destination.");
    return FAIL;
  }

  return SUCCEED;
}

void GlobalExplorationPlanner::findGlobalTour(
    const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw,
    vector<int>& indices) {
  auto t1 = ros::Time::now();

  Eigen::MatrixXd cost_mat;
  frontier_finder_->updateFrontierCostMatrix();
  frontier_finder_->getFullCostMatrix(cur_pos, cur_vel, cur_yaw, cost_mat);
  const int dimension = cost_mat.rows();

  double mat_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  std::ofstream prob_file(ep_->tsp_dir_ + "/single.tsp");
  string prob_spec = "NAME : single\nTYPE : ATSP\nDIMENSION : " + std::to_string(dimension) +
                     "\nEDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\n"
                     "EDGE_WEIGHT_SECTION\n";

  prob_file << prob_spec;

  const int scale = 100;
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = cost_mat(i, j) * scale;
      prob_file << int_cost << " ";
    }
    prob_file << "\n";
  }

  prob_file << "EOF";
  prob_file.close();

  solveTSPLKH((ep_->tsp_dir_ + "/single.par").c_str());

  std::ifstream res_file(ep_->tsp_dir_ + "/single.txt");
  string res;
  while (getline(res_file, res)) {
    if (res.compare("TOUR_SECTION") == 0) break;
  }

  while (getline(res_file, res)) {
    int id = stoi(res);
    if (id == 1) continue;
    if (id == -1) break;
    indices.push_back(id - 2);
  }

  res_file.close();

  frontier_finder_->getPathForTour(cur_pos, indices, ed_->global_tour_);

  double tsp_time = (ros::Time::now() - t1).toSec();
  ROS_WARN("Cost mat: %lf, TSP: %lf", mat_time, tsp_time);
}

void GlobalExplorationPlanner::refineLocalTour(
    const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d& cur_yaw,
    const vector<vector<Vector3d>>& n_points, const vector<vector<double>>& n_yaws,
    vector<Vector3d>& refined_pts, vector<double>& refined_yaws) {
  GraphSearch<ViewNode> g_search;
  vector<ViewNode::Ptr> last_group, cur_group;

  ViewNode::Ptr first(new ViewNode(cur_pos, cur_yaw[0]));
  first->vel_ = cur_vel;
  g_search.addNode(first);
  last_group.push_back(first);
  ViewNode::Ptr final_node;

  for (int i = 0; i < n_points.size(); ++i) {
    for (int j = 0; j < n_points[i].size(); ++j) {
      ViewNode::Ptr node(new ViewNode(n_points[i][j], n_yaws[i][j]));
      g_search.addNode(node);
      for (auto nd : last_group) g_search.addEdge(nd->id_, node->id_);
      cur_group.push_back(node);

      if (i == n_points.size() - 1) {
        final_node = node;
        break;
      }
    }
    last_group = cur_group;
    cur_group.clear();
  }

  vector<ViewNode::Ptr> path;
  g_search.DijkstraSearch(first->id_, final_node->id_, path);

  for (int i = 1; i < path.size(); ++i) {
    refined_pts.push_back(path[i]->pos_);
    refined_yaws.push_back(path[i]->yaw_);
  }

  ed_->refined_tour_.clear();
  ed_->refined_tour_.push_back(cur_pos);
  ViewNode::astar_->lambda_heu_ = 1.0;
  ViewNode::astar_->setResolution(0.2);
  for (auto pt : refined_pts) {
    vector<Vector3d> local_path;
    if (ViewNode::searchPath(ed_->refined_tour_.back(), pt, local_path)) {
      ed_->refined_tour_.insert(ed_->refined_tour_.end(), local_path.begin(), local_path.end());
    } else {
      ed_->refined_tour_.push_back(pt);
    }
  }
  ViewNode::astar_->lambda_heu_ = 10000;
}

}  // namespace fast_planner
