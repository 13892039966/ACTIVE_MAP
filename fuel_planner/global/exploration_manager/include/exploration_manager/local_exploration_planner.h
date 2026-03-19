#ifndef _LOCAL_EXPLORATION_PLANNER_H_
#define _LOCAL_EXPLORATION_PLANNER_H_

#include <Eigen/Eigen>
#include <memory>
#include <vector>

using Eigen::Vector3d;
using std::shared_ptr;
using std::vector;

namespace fast_planner {
class EDTEnvironment;
class FastPlannerManager;
class SDFMap;
struct ExplorationData;

class LocalExplorationPlanner {
public:
  typedef shared_ptr<LocalExplorationPlanner> Ptr;

  void initialize(const shared_ptr<FastPlannerManager>& planner_manager,
                  const shared_ptr<EDTEnvironment>& edt_environment,
                  const shared_ptr<SDFMap>& sdf_map,
                  const shared_ptr<ExplorationData>& exploration_data,
                  const double relax_time);

  int planToViewpoint(const Vector3d& pos, const Vector3d& vel, const Vector3d& acc,
                      const Vector3d& yaw, const Vector3d& next_pos, const double next_yaw);

private:
  bool planGeometricPathFrontend(const Vector3d& raw_start, const Vector3d& goal);
  int solveMincoBackend(const Vector3d& vel, const Vector3d& acc, const Vector3d& yaw,
                        const Vector3d& next_pos, const double next_yaw);
  bool findSearchStart(const Vector3d& raw_start, Vector3d& search_start) const;
  void shortenPath(vector<Vector3d>& path) const;

  shared_ptr<FastPlannerManager> planner_manager_;
  shared_ptr<EDTEnvironment> edt_environment_;
  shared_ptr<SDFMap> sdf_map_;
  shared_ptr<ExplorationData> ed_;
  double relax_time_ = 1.0;
};

}  // namespace fast_planner

#endif
