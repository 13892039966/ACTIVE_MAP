#ifndef _PLANNER_MANAGER_H_
#define _PLANNER_MANAGER_H_

#include <path_searching/bubble_astar.h>

#include <plan_manage/plan_container.hpp>
#include <ros/ros.h>
#include <traj_utils/PolyTraj.h>
#include <observation_quality_manager/global_planner.h>  // For PathSegmentWithView
#include <observation_quality_manager/observation_quality_manager.h>  // For FreeRegion, VoxelHash
#include <lidar_map/lidar_map.h>
#include <random>
#include <Eigen/Sparse>
#include "gcopter/lbfgs.hpp"
#include "gcopter/firi.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/gcopter.hpp"
#include "gcopter/sfc_gen.hpp"
#include "gcopter/trajectory.hpp"
#include "gcopter/voxel_map.hpp"
#include "misc/visualizer.hpp"

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <pointcloud_topo/graph.h>
#include <pointcloud_topo/graph_visualizer.hpp>
#include <pointcloud_topo/parallel_bubble_astar.h>
#include <tf/tf.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

namespace fast_planner {

// PlanResult - planExploreTraj 函数的返回值
enum PlanResult { PLAN_FAIL = 0, PLAN_SUCCEED = 1 };

// ViewPoint5D - 带朝向的5D视点 (x, y, z, yaw, pitch)
struct ViewPoint5D {
  Eigen::Vector3f position;  // 视点位置 (世界坐标)
  float yaw;                 // 偏航角 [-π, π], 0 = +X方向
  float pitch;               // 俯仰角 [pitch_min, pitch_max], 0 = 水平
  float score;               // 该视点的累计得分 (调试用)

  ViewPoint5D() : position(Eigen::Vector3f::Zero()), yaw(0.0f), pitch(0.0f), score(0.0f) {}
  ViewPoint5D(const Eigen::Vector3f& pos, float y = 0.0f, float p = 0.0f, float s = 0.0f)
    : position(pos), yaw(y), pitch(p), score(s) {}
};

// Fast Planner Manager
// Key algorithms of mapping and planning are called
struct GcopterConfig {
  std::string mapTopic;
  std::string targetTopic;
  double dilateRadiusSoft, dilateRadiusHard;
  double timeoutRRT;
  double maxVelMag;
  double maxBdrMag;
  double maxTiltAngle;
  double minThrust;
  double maxThrust;
  double vehicleMass;
  double gravAcc;
  double horizDrag;
  double vertDrag;
  double parasDrag;
  double speedEps;
  double weightT;
  double WeightSafeT;
  std::vector<double> chiVec;
  double smoothingEps;
  int integralIntervs;
  double relCostTol;
  double corridor_size;
  double yaw_max_vel;
  double yaw_rho_vis;
  double yaw_time_fwd;
  double rho_collision;          // Collision penalty weight
  double safe_distance;          // Safety margin from obstacles (m)
  double collision_gradient_eps; // Finite difference epsilon (m)
  double rho_path;               // Path adherence weight (grid A* path)
  double max_pitch;              // Pitch upper bound (rad)
  double min_pitch;              // Pitch lower bound (rad)

  void init(const ros::NodeHandle &nh_priv) {
    nh_priv.getParam("DilateRadiusSoft", dilateRadiusSoft);
    nh_priv.getParam("DilateRadiusHard", dilateRadiusHard);
    nh_priv.getParam("MaxVelMag", maxVelMag);
    nh_priv.getParam("maxBdrMag", maxBdrMag);
    nh_priv.getParam("MaxTiltAngle", maxTiltAngle);
    nh_priv.getParam("MinThrust", minThrust);
    nh_priv.getParam("MaxThrust", maxThrust);
    nh_priv.getParam("VehicleMass", vehicleMass);
    nh_priv.getParam("GravAcc", gravAcc);
    nh_priv.getParam("HorizDrag", horizDrag);
    nh_priv.getParam("VertDrag", vertDrag);
    nh_priv.getParam("ParasDrag", parasDrag);
    nh_priv.getParam("SpeedEps", speedEps);
    nh_priv.getParam("WeightT", weightT);
    nh_priv.getParam("WeightSafeT", WeightSafeT);
    nh_priv.getParam("ChiVec", chiVec);
    nh_priv.getParam("SmoothingEps", smoothingEps);
    nh_priv.getParam("IntegralIntervs", integralIntervs);
    nh_priv.getParam("RelCostTol", relCostTol);
    nh_priv.getParam("MaxCorridorSize", corridor_size);
    nh_priv.getParam("yaw_rho_vis", yaw_rho_vis);
    nh_priv.getParam("yaw_max_vel", yaw_max_vel);
    nh_priv.getParam("yaw_time_fwd", yaw_time_fwd);
    nh_priv.getParam("rho_collision", rho_collision);
    nh_priv.getParam("safe_distance", safe_distance);
    nh_priv.getParam("collision_gradient_eps", collision_gradient_eps);
    nh_priv.getParam("rho_path", rho_path);
    nh_priv.getParam("MAX_PITCH", max_pitch);
    nh_priv.getParam("MIN_PITCH", min_pitch);
  }
};

class FastPlannerManager {
  // SECTION stable
public:
  typedef shared_ptr<FastPlannerManager> Ptr;
  FastPlannerManager();
  ~FastPlannerManager();
  void printTimeCost(double time_threhold, double time_cost, string printInfo);

  int planExploreTraj(const vector<Eigen::Vector3f> &path, const ViewPoint5D &target_viewpoint, bool is_static);

  int planLongExploreTraj(const std::vector<PathSegmentWithView> &path_segments,
                          ObservationQualityManager::Ptr oqm,
                          bool is_static);

  bool flyToSafeRegion(bool is_static);
  void polyTraj2ROSMsg(traj_utils::PolyTraj &poly_msg, const ros::Time &start_time);
  void polyYawTraj2ROSMsg(traj_utils::PolyTraj &poly_msg, const ros::Time &start_time);
  void polyPitchTraj2ROSMsg(traj_utils::PolyTraj &poly_msg, const ros::Time &start_time);

  void initPlanModules(ros::NodeHandle &nh, ParallelBubbleAstar::Ptr &parallel_path_finder,
                       TopoGraph::Ptr &graph);

  bool checkTrajCollision(double &collision_time);
  bool checkTrajVelocity();

  bool YawTrajOpt(double &start_yaw, double &end_yaw, bool is_static, bool use_shorten_path);
  bool YawTrajwithoutOpt(double &start_yaw, double &end_yaw, bool is_static, bool use_shorten_path);
  bool PitchTrajOpt(double &start_pitch, double &end_pitch, bool is_static);

  // New trajectory planning functions (separated position/yaw/pitch)
  bool PositionTrajPlan(const vector<Eigen::Vector3f> &path,
                        const ViewPoint5D &target_viewpoint,
                        bool is_static,
                        double &duration,
                        double &end_yaw,
                        Trajectory<7> &traj);
  bool YawTrajPlan(double start_yaw, double end_yaw,
                   bool is_static,
                   double duration,
                   Trajectory<5> &traj);
  bool PitchTrajPlan(double start_pitch, double end_pitch,
                     bool is_static,
                     double duration,
                     Trajectory<5> &traj);
  void goalCallback(const geometry_msgs::PoseStampedConstPtr &msg);
  void posCallback(const nav_msgs::OdometryConstPtr &msg);
  bool YawInterpolationwithoutOpt(double &start, double &end, vector<double> &newYaw,
                                  vector<double> &newDur, double &CompT);
  void YawLookforward(const Trajectory<5> &pos_traj, double &start, double &end,
                      vector<double> &newYaw, vector<double> &newDur, double &CompT);
  void YawLookforwardwithoutOpt(double &start, double &end, vector<double> &newYaw,
                                vector<double> &newDur, double &CompT, bool use_short_path);
  void angleLimite(double &angle);
  void calculateTimelb(const vector<Eigen::Vector3d> &path2next_goal,
                                 const double &current_yaw, const double &goal_yaw, double &time_lb);

  // 获取每个viewpoint的到达时间偏移量（相对于轨迹开始时间）
  const std::vector<double>& getViewpointArrivalTimes() const { return viewpoint_arrival_times_; }

  double start_yaw, end_yaw;
  double is_static_yaw = false;

  ros::Subscriber goal_sub;
  ros::Subscriber pos_sub;
  ros::Publisher yaw_state_pub;
  ros::Publisher yaw_waypoints_pub;
  ros::Publisher waypoints_opt_pub;  // Visualization for optimizable/fixed waypoints
  ros::Publisher waypoints_grad_pub;  // Visualization for waypoint gradients (arrows)
  ros::Publisher esdf_pub_;          // ESDF visualization publisher
  ros::Publisher path_segments_pub_; // Visualization for input path segments

  void visualizeTrajWithCamera();
  void visualizePathSegments(const std::vector<PathSegmentWithView>& path_segments);
  void visualizeWaypointsOptStatus(const Eigen::Matrix3Xd& wps, const std::vector<bool>& opt_indi);
  void visualizeWaypointsGradients(const Eigen::Matrix3Xd& wps, const Eigen::Matrix3Xd& grads);
  void publishESDF();  // ESDF visualization

  minco::MINCO_S3NU yaw_traj_opt_;
  LocalTrajData local_data_;
  double max_traj_len_;
  LIOInterface::Ptr lidar_map_interface_;
  unique_ptr<Visualizer> gcopter_viz_;
  unique_ptr<GcopterConfig> gcopter_config_;
  BubbleAstar::Ptr bubble_path_finder_;
  ParallelBubbleAstar::Ptr parallel_path_finder_;
  TopoGraph::Ptr topo_graph_;
  GraphVisualizer::Ptr graph_visualizer_;
  FastSearcher::Ptr fast_searcher_;
  bool use_mid360;
  double max_ray_length;
  double fov_up, fov_down;
  double lidar_pitch;

private:
  /* main planning algorithms & modules */
  shared_ptr<SDFMap> sdf_map_;

  // Helper functions for simple trajectory generation (no optimization)
  bool GenerateSimplePositionTraj(const Eigen::Vector3d &start,
                                  const Eigen::Vector3d &end,
                                  double duration,
                                  Trajectory<7> &traj);
  bool GenerateSimpleAngleTraj(double start_angle, double start_vel, double start_acc,
                               double end_angle, double end_vel, double end_acc,
                               double duration,
                               Trajectory<5> &traj);

  // ==========================================
  // Long trajectory optimization state (FC-Planner style)
  // ==========================================
  Eigen::SparseMatrix<double> select_waypt_;    // Selection matrix for optimizable points
  Eigen::SparseMatrix<double> select_viewpt_;   // Selection matrix for frozen viewpoints
  Eigen::Matrix3Xd opt_wps_;                    // All intermediate waypoints [3 x N-1]
  Eigen::Matrix3Xd way_wps_;                    // Optimizable waypoints only
  Eigen::Matrix3Xd view_wps_;                   // Frozen viewpoints only
  Eigen::Matrix3Xd fused_wps_;                  // Combined result after optimization
  std::vector<bool> opt_indi_;                  // true=waypoint(optimizable), false=viewpoint(frozen)
  int piece_nums_;
  int waypt_count_;
  Eigen::Matrix<double, 3, 4> opt_inistate_;    // [P, V, A, J] at start (3x4: each column is P, V, A, J)
  Eigen::Matrix<double, 3, 4> opt_finstate_;    // [P, V, A, J] at end
  Eigen::VectorXd opt_times_;
  minco::MINCO_S4NU minco_long_;
  double max_v_squared_, max_a_squared_;
  std::vector<double> viewpoint_arrival_times_;
  std::vector<double> target_yaws_;
  std::vector<double> target_pitches_;
  std::vector<int> viewpoint_piece_indices_;    // Which piece each viewpoint is at the end of
  std::vector<Eigen::Vector3d> ref_path_points_; // Concatenated front-end path points (for adherence cost)
  std::vector<std::vector<Eigen::Vector3d>> ref_path_segments_; // Per-segment path points

  // 联合优化所需的起始角度 (在优化开始前保存)
  double opt_start_yaw_;
  double opt_start_pitch_;

  // ==========================================
  // ESDF for smooth collision gradients
  // ==========================================
  ObservationQualityManager::Ptr oqm_;          // Observation quality manager pointer
  Eigen::Vector3i esdf_bound_min_;              // Local ESDF grid bounds (voxel indices)
  Eigen::Vector3i esdf_bound_max_;
  Eigen::Vector3i esdf_size_;                   // Grid dimensions
  std::vector<float> esdf_distance_buffer_;     // Final distance values (positive for FREE, negative for UNKNOWN)
  std::vector<float> esdf_neg_distance_buffer_; // Negative distance buffer (distance from FREE into UNKNOWN/OCCUPIED)
  std::vector<float> esdf_tmp_buffer1_;         // Temp buffer for distance transform
  std::vector<float> esdf_tmp_buffer2_;         // Temp buffer for distance transform
  bool esdf_valid_;                             // Whether ESDF is computed

  // Helper functions for long trajectory planning
  void prepareLongTrajWaypoints(const std::vector<PathSegmentWithView>& path_segments,
                                bool is_static,
                                Eigen::Matrix3Xd& wps,
                                Eigen::Matrix<double, 3, 4>& iniState,
                                Eigen::Matrix<double, 3, 4>& finState,
                                std::vector<bool>& opt_indi,
                                std::vector<double>& target_yaws,
                                std::vector<double>& target_pitches,
                                std::vector<int>& viewpoint_piece_indices);
  void setupSelectionMatrices(const Eigen::Matrix3Xd& all_wps, const std::vector<bool>& opt_indi);
  void initializeLongTimeAllocation();
  void computeConstraintCostGrad(double& cost, Eigen::MatrixX3d& gdC, Eigen::VectorXd& gdT);
  void computeYawPitchConstraint(const Eigen::VectorXd& T, double& cost, Eigen::VectorXd& gradT);
  void queryDistanceWithGrad(const Eigen::Vector3d& pos, double& dist, Eigen::Vector3d& grad);
  void computeViewpointArrivalTimes(const Eigen::VectorXd& T);
  bool generateYawPitchTrajByTime(double start_yaw, double start_pitch,
                                   bool is_static,
                                   Trajectory<5>& yaw_traj, Trajectory<5>& pitch_traj);
  static double innerCallbackLong(void* ptrObj, const Eigen::VectorXd& x, Eigen::VectorXd& grad);
  static bool smoothedL1(const double& x, const double& mu, double& f, double& df);

  // ESDF construction and query
  void buildLocalESDF(const std::vector<PathSegmentWithView>& path_segments);
  void fillESDF1D(const std::function<float(int)>& get_val,
                  const std::function<void(int, float)>& set_val,
                  int start, int end);
  double getDistanceESDF(const Eigen::Vector3i& local_idx) const;
  double getDistWithGradESDF(const Eigen::Vector3d& pos, Eigen::Vector3d& grad);
  inline int esdfAddress(int x, int y, int z) const {
    return x + y * esdf_size_.x() + z * esdf_size_.x() * esdf_size_.y();
  }

  // Time parameterization helpers (copied from GCOPTER since they're private there)
  static inline void forwardT_local(const Eigen::VectorXd &tau, Eigen::VectorXd &T) {
    const int sizeTau = tau.size();
    T.resize(sizeTau);
    for (int i = 0; i < sizeTau; i++) {
      T(i) = tau(i) > 0.0 ? ((0.5 * tau(i) + 1.0) * tau(i) + 1.0)
                          : 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);
    }
  }

  template <typename EIGENVEC>
  static inline void backwardT_local(const Eigen::VectorXd &T, EIGENVEC &tau) {
    const int sizeT = T.size();
    tau.resize(sizeT);
    for (int i = 0; i < sizeT; i++) {
      tau(i) = T(i) > 1.0 ? (sqrt(2.0 * T(i) - 1.0) - 1.0) : (1.0 - sqrt(2.0 / T(i) - 1.0));
    }
  }

  template <typename EIGENVEC>
  static inline void backwardGradT_local(const Eigen::VectorXd &tau, const Eigen::VectorXd &gradT,
                                         EIGENVEC &gradTau) {
    const int sizeTau = tau.size();
    gradTau.resize(sizeTau);
    double denSqrt;
    for (int i = 0; i < sizeTau; i++) {
      if (tau(i) > 0) {
        gradTau(i) = gradT(i) * (tau(i) + 1.0);
      } else {
        denSqrt = (0.5 * tau(i) - 1.0) * tau(i) + 1.0;
        gradTau(i) = gradT(i) * (1.0 - tau(i)) / (denSqrt * denSqrt);
      }
    }
  }

  // topology guided optimization

  void findCollisionRange(vector<Eigen::Vector3d> &colli_start, vector<Eigen::Vector3d> &colli_end,
                          vector<Eigen::Vector3d> &start_pts, vector<Eigen::Vector3d> &end_pts);

  Eigen::MatrixXd paramLocalTraj(double start_t, double &dt, double &duration);
  Eigen::MatrixXd reparamLocalTraj(const double &start_t, const double &duration, const double &dt);

public:
  void planYawActMap(const Eigen::Vector3d &start_yaw);
  void test();
  void searchFrontier(const Eigen::Vector3d &p);

private:
  // Benchmark method, local exploration
public:
  bool localExplore(Eigen::Vector3d start_pt, Eigen::Vector3d start_vel, Eigen::Vector3d start_acc,
                    Eigen::Vector3d end_pt);

  // !SECTION
};
} // namespace fast_planner

#endif
