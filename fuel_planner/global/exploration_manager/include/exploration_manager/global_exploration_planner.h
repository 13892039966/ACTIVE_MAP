#ifndef _GLOBAL_EXPLORATION_PLANNER_H_
#define _GLOBAL_EXPLORATION_PLANNER_H_

#include <Eigen/Eigen>
#include <memory>
#include <vector>

using Eigen::Vector3d;
using std::shared_ptr;
using std::vector;

namespace fast_planner {
class FrontierFinder;
class FastPlannerManager;
struct ExplorationData;
struct ExplorationParam;

class GlobalExplorationPlanner {
public:
  typedef shared_ptr<GlobalExplorationPlanner> Ptr;

  void initialize(const shared_ptr<FrontierFinder>& frontier_finder,
                  const shared_ptr<FastPlannerManager>& planner_manager,
                  const shared_ptr<ExplorationData>& exploration_data,
                  const shared_ptr<ExplorationParam>& exploration_param);

  int computeNextViewpoint(const Vector3d& pos, const Vector3d& vel, const Vector3d& yaw,
                           Vector3d& next_pos, double& next_yaw);

private:
  void findGlobalTour(const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw,
                      vector<int>& indices);

  void refineLocalTour(const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d& cur_yaw,
                       const vector<vector<Vector3d>>& n_points, const vector<vector<double>>& n_yaws,
                       vector<Vector3d>& refined_pts, vector<double>& refined_yaws);

  shared_ptr<FrontierFinder> frontier_finder_;
  shared_ptr<FastPlannerManager> planner_manager_;
  shared_ptr<ExplorationData> ed_;
  shared_ptr<ExplorationParam> ep_;
};

}  // namespace fast_planner

#endif
