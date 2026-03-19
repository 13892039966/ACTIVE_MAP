#ifndef _PLANNER_MANAGER_H_
#define _PLANNER_MANAGER_H_

#include <bspline_opt/bspline_optimizer.h>
#include <bspline/non_uniform_bspline.h>
#include <gcopter/lbfgs.hpp>
#include <gcopter/minco.hpp>
#include <gcopter/trajectory.hpp>

#include <path_searching/astar2.h>
#include <path_searching/kinodynamic_astar.h>
#include <path_searching/topo_prm.h>

#include <plan_env/edt_environment.h>

#include <active_perception/frontier_finder.h>
#include <active_perception/heading_planner.h>

#include <plan_manage/plan_container.hpp>

#include <Eigen/Sparse>
#include <limits>
#include <ros/ros.h>
#include <traj_utils/PolyTraj.h>

namespace fast_planner {
// Fast Planner Manager
// Key algorithms of mapping and planning are called

class FastPlannerManager {
  // SECTION stable
public:
  FastPlannerManager();
  ~FastPlannerManager();

  /* main planning interface */
  bool kinodynamicReplan(const Eigen::Vector3d& start_pt, const Eigen::Vector3d& start_vel,
                         const Eigen::Vector3d& start_acc, const Eigen::Vector3d& end_pt,
                         const Eigen::Vector3d& end_vel, const double& time_lb = -1);
  bool planExploreTraj(const vector<Eigen::Vector3d>& tour, const Eigen::Vector3d& cur_vel,
                       const Eigen::Vector3d& cur_acc, const double& time_lb);
  bool planExploreTraj(const vector<Eigen::Vector3d>& tour, const Eigen::Vector3d& cur_vel,
                       const Eigen::Vector3d& cur_acc, const double& time_lb = -1,
                       const double target_yaw = std::numeric_limits<double>::quiet_NaN());
  bool planGlobalTraj(const Eigen::Vector3d& start_pos);
  bool topoReplan(bool collide);

  void planYaw(const Eigen::Vector3d& start_yaw);
  void planYawExplore(const Eigen::Vector3d& start_yaw, const double& end_yaw, bool lookfwd,
                      const double& relax_time);

  void initPlanModules(ros::NodeHandle& nh);
  void setGlobalWaypoints(vector<Eigen::Vector3d>& waypoints);
  void exportTrajToPolyMsg(traj_utils::PolyTraj& pos_msg, traj_utils::PolyTraj& yaw_msg,
                           const ros::Time& start_time);

  bool checkTrajCollision(double& distance);
  void calcNextYaw(const double& last_yaw, double& yaw);
  void angleLimite(double& angle);
  bool isPointSafeInExploreSpace(const Eigen::Vector3d& pt, bool allow_unknown = false) const;
  bool isSegmentSafeInExploreSpace(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
                                   double step = -1.0, bool allow_unknown = false) const;
  bool projectToValidExplorePoint(const Eigen::Vector3d& raw_pt, Eigen::Vector3d& projected_pt,
                                  double max_radius = 2.0) const;
  bool sanitizeExplorePath(const vector<Eigen::Vector3d>& raw_path, vector<Eigen::Vector3d>& safe_path,
                           double max_spacing = -1.0) const;

  PlanParameters pp_;
  LocalTrajData local_data_;
  GlobalTrajData global_data_;
  MidPlanData plan_data_;
  EDTEnvironment::Ptr edt_environment_;
  unique_ptr<Astar> path_finder_;
  unique_ptr<TopologyPRM> topo_prm_;

private:
  /* main planning algorithms & modules */
  shared_ptr<SDFMap> sdf_map_;

  unique_ptr<KinodynamicAstar> kino_path_finder_;
  vector<BsplineOptimizer::Ptr> bspline_optimizers_;

  bool solveMincoPositionTraj(const vector<Eigen::Vector3d>& tour, const Eigen::Vector3d& cur_vel,
                              const Eigen::Vector3d& cur_acc, const double& time_lb,
                              const double target_yaw);
  bool solveMincoYawTraj(const Eigen::Vector3d& start_yaw, const double& end_yaw, bool lookfwd,
                         const double& relax_time);
  bool validateActiveTrajInExploreSpace(double sample_step = -1.0);
  bool setupExploreMinco(const vector<Eigen::Vector3d>& safe_tour, const Eigen::Vector3d& cur_vel,
                         const Eigen::Vector3d& cur_acc, const double& time_lb);
  void computeExploreConstraintCostGrad(double& cost, Eigen::MatrixX3d& gdC,
                                        Eigen::VectorXd& gdT);
  void computeExploreYawCostGrad(const Eigen::VectorXd& T, double& cost, Eigen::VectorXd& gdT);
  void queryDistanceWithGrad(const Eigen::Vector3d& pos, double& dist, Eigen::Vector3d& grad) const;
  bool smoothedL1(const double& x, const double& mu, double& f, double& df) const;
  static double innerCallbackExplore(void* ptrObj, const Eigen::VectorXd& x, Eigen::VectorXd& grad);

  void updateTrajInfo();

  // topology guided optimization

  void findCollisionRange(vector<Eigen::Vector3d>& colli_start, vector<Eigen::Vector3d>& colli_end,
                          vector<Eigen::Vector3d>& start_pts, vector<Eigen::Vector3d>& end_pts);

  void optimizeTopoBspline(double start_t, double duration, vector<Eigen::Vector3d> guide_path,
                           int traj_id);
  Eigen::MatrixXd paramLocalTraj(double start_t, double& dt, double& duration);
  Eigen::MatrixXd reparamLocalTraj(const double& start_t, const double& duration, const double& dt);

  void selectBestTraj(NonUniformBspline& traj);
  void refineTraj(NonUniformBspline& best_traj);
  void reparamBspline(NonUniformBspline& bspline, double ratio, Eigen::MatrixXd& ctrl_pts, double& dt,
                      double& time_inc);

  // Heading planning

  // !SECTION stable

  // SECTION developing

public:
  typedef shared_ptr<FastPlannerManager> Ptr;

  void planYawActMap(const Eigen::Vector3d& start_yaw);
  void test();
  void searchFrontier(const Eigen::Vector3d& p);

private:
  unique_ptr<FrontierFinder> frontier_finder_;
  unique_ptr<HeadingPlanner> heading_planner_;
  unique_ptr<VisibilityUtil> visib_util_;

  minco::MINCO_S4NU explore_minco_;
  Eigen::Matrix<double, 3, 4> explore_ini_state_;
  Eigen::Matrix<double, 3, 4> explore_fin_state_;
  Eigen::Matrix3Xd explore_ref_inner_wps_;
  Eigen::Matrix3Xd explore_inner_wps_;
  Eigen::VectorXd explore_times_;
  vector<Eigen::Vector3d> explore_ref_path_points_;
  vector<vector<Eigen::Vector3d>> explore_ref_path_segments_;
  vector<Eigen::Vector3d> explore_safe_tour_;
  int explore_piece_num_ = 0;
  int explore_inner_count_ = 0;
  double explore_time_lb_ = -1.0;
  double explore_target_yaw_ = std::numeric_limits<double>::quiet_NaN();
  double explore_max_vel_sq_ = 0.0;
  double explore_max_acc_sq_ = 0.0;

  static inline void forwardTLocal(const Eigen::VectorXd& tau, Eigen::VectorXd& T) {
    const int sizeTau = tau.size();
    T.resize(sizeTau);
    for (int i = 0; i < sizeTau; i++) {
      T(i) = tau(i) > 0.0 ? ((0.5 * tau(i) + 1.0) * tau(i) + 1.0)
                          : 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);
    }
  }

  static inline void backwardTLocal(const Eigen::VectorXd& T, Eigen::VectorXd& tau) {
    const int sizeT = T.size();
    tau.resize(sizeT);
    for (int i = 0; i < sizeT; i++) {
      tau(i) = T(i) > 1.0 ? (sqrt(2.0 * T(i) - 1.0) - 1.0)
                          : (1.0 - sqrt(2.0 / T(i) - 1.0));
    }
  }

  static inline void backwardGradTLocal(const Eigen::VectorXd& tau, const Eigen::VectorXd& gradT,
                                        Eigen::VectorXd& gradTau) {
    const int sizeTau = tau.size();
    gradTau.resize(sizeTau);
    for (int i = 0; i < sizeTau; i++) {
      gradTau(i) = tau(i) > 0.0 ? gradT(i) * (tau(i) + 1.0)
                                : gradT(i) * (1.0 - tau(i)) /
                                      std::pow((0.5 * tau(i) - 1.0) * tau(i) + 1.0, 2);
    }
  }

  // Benchmark method, local exploration
public:
  bool localExplore(Eigen::Vector3d start_pt, Eigen::Vector3d start_vel, Eigen::Vector3d start_acc,
                    Eigen::Vector3d end_pt);

  // !SECTION
};
}  // namespace fast_planner

#endif
