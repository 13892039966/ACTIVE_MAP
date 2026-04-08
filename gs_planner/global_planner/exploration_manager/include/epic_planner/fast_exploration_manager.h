/***
 * @Author: ning-zelin && zl.ning@qq.com
 * @Date: 2023-12-21 21:31:51
 * @LastEditTime: 2024-03-10 11:39:33
 * @Description:
 * @
 * @Copyright (c) 2023 by ning-zelin, All Rights Reserved.
 */

#ifndef _EXPLORATION_MANAGER_H_
#define _EXPLORATION_MANAGER_H_

#include <Eigen/Eigen>
#include <observation_quality_manager/observation_quality_manager.h>
#include <observation_quality_manager/global_planner.h>
#include <memory>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <pointcloud_topo/graph.h>
#include <plan_manage/planner_manager.h>
#include <ros/ros.h>
#include <vector>
using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

namespace fast_planner {
class EDTEnvironment;
class SDFMap;
class FastPlannerManager;
struct ExplorationParam;
struct ExplorationData;
struct ViewPose;

enum EXPL_RESULT { NO_FRONTIER, FAIL, START_FAIL, SUCCEED };

class FastExplorationManager {
public:
  typedef shared_ptr<FastExplorationManager> Ptr;
  FastExplorationManager();
  ~FastExplorationManager();
  shared_ptr<ExplorationData> ed_;
  shared_ptr<ExplorationParam> ep_;

  // ObservationQualityManager 替代 FrontierManager
  ::ObservationQualityManager::Ptr oqm_;

  double goal_yaw;
  double goal_pitch;

  shared_ptr<FastPlannerManager> planner_manager_;


  void initialize(ros::NodeHandle &nh,
                  ::ObservationQualityManager::Ptr oqm,
                  FastPlannerManager::Ptr planner_manager);

  int planGlobalPath(const Vector3d &pos, const Vector3d &vel, float odom_yaw, float odom_pitch);

  void goalCallback(const geometry_msgs::PoseStampedConstPtr &msg);
  void updateGoalNode();
};

} // namespace fast_planner

#endif