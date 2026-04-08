/***
 * @Author: ning-zelin && zl.ning@qq.com
 * @Date: 2024-02-29 16:54:46
 * @LastEditTime: 2024-03-11 13:22:44
 * @Description:
 * @
 * @Copyright (c) 2024 by ning-zelin, All Rights Reserved.
 */

#include <epic_planner/expl_data.h>
#include <epic_planner/fast_exploration_fsm.h>
#include <epic_planner/fast_exploration_manager.h>
#include <plan_manage/planner_manager.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>
#include <traj_utils/planning_visualization.h>
using Eigen::Vector3d;
using Eigen::Vector4d;
bool debug_planner;
typedef visualization_msgs::Marker Marker;
typedef visualization_msgs::MarkerArray MarkerArray;

void FastExplorationFSM::FSMCallback(const ros::TimerEvent &e) {
  pubState();
  switch (state_) {
  case INIT: {
    if (!fd_->have_odom_) {
      ROS_WARN_THROTTLE(1.0, "no raw odom.");
      return;
    }
    if (!fd_->have_cloud_odom_) {
      ROS_WARN_THROTTLE(1.0, "waiting for synced cloud+odom.");
      return;
    }
    transitState(WAIT_TRIGGER, "FSM");
    break;
  }

  case WAIT_TRIGGER: {
    ROS_WARN_THROTTLE(1.0, "wait for trigger.");
    break;
  }

  case FINISH: {
    // stopTraj();
    double collision_time = 0.0;
    bool safe = planner_manager_->checkTrajCollision(collision_time);
    if (!safe) {
      stopTraj();
    }

    // Export FREE voxels to PCD (only on first entry)
    static bool finish_first_entry = true;
    if (finish_first_entry) {
      exportFreeVoxelsToPCD();
      finish_first_entry = false;
    }

    ROS_WARN_THROTTLE(1.0, "Finished.");
    break;
  }

  case PLAN_TRAJ: {
    if (!fd_->trigger_)
      return;
    if (planner_manager_->topo_graph_->odom_node_->neighbors_.empty())
      return;
    ros::Time start = ros::Time::now();
    // 要报min-step的case
    LocalTrajData *info = &planner_manager_->local_data_;
    double t_cur = (ros::Time::now() - info->start_time_).toSec();
    double time_to_end = info->duration_ - t_cur;
    if (expl_manager_->ed_->global_tour_.size() == 2) {
      Eigen::Vector3f goal = expl_manager_->ed_->global_tour_[1];
      if ((goal - fd_->odom_pos_).norm() < 1e-1) {
        transitState(FINISH, "fsm");
        return;
      }
    }
    ros::Time tplan = ros::Time::now();
    exec_timer_.stop();
    int res = callExplorationPlanner();
    exec_timer_.start();
    ROS_INFO("\033[31m call planner \033[0m: %.3f",
             (ros::Time::now() - tplan).toSec() * 1000.0);

    if (res == SUCCEED) {
      poly_yaw_traj_pub_.publish(fd_->newest_yaw_traj_);
      poly_pitch_traj_pub_.publish(fd_->newest_pitch_traj_);
      poly_traj_pub_.publish(fd_->newest_traj_);
      // 根据 pos_needed 设置 static_state
      fd_->static_state_ = !planner_manager_->local_data_.pos_needed_;
      if (fd_->use_bubble_a_star_) {
        transitState(EXEC_TRAJ,
                     "ParallelBubbleAstar plan success: new traj pub");
      } else {
        transitState(EXEC_TRAJ, "plan success: new traj pub");
      }
      fd_->use_bubble_a_star_ = false;
      fd_->half_resolution = false;

    } else if (res == NO_FRONTIER) {
      // if (planner_manager_->topo_graph_->global_view_points_.empty())
      transitState(FINISH, "PLAN_TRAJ: no frontier");
      fd_->static_state_ = true;
    } else if (res == FAIL) {
      // Still in PLAN_TRAJ state, keep replanning
      stopTraj();
      transitState(PLAN_TRAJ, "PLAN_TRAJ: plan failed", true);

    } else if (res == START_FAIL) {
      transitState(CAUTION, "PLAN_TRAJ: start failed", true);
    } else {
      cout << "330?" << endl;
    }
    break;
  }

  case EXEC_TRAJ: {
    // collision check
    double collision_time;
    bool safe = planner_manager_->checkTrajCollision(collision_time);
    if (!safe) {
      transitState(
          PLAN_TRAJ,
          "safetyCallback: not safe, time:" + to_string(collision_time), true);
      if (collision_time < fp_->replan_time_ + 0.2)
        stopTraj();
    } 


    break;
  }

  case CAUTION: {
    stopTraj();
    exec_timer_.stop();
    bool success = planner_manager_->flyToSafeRegion(fd_->static_state_);
    if (success) {
      traj_utils::PolyTraj poly_traj_msg;
      auto info = &planner_manager_->local_data_;
      planner_manager_->polyTraj2ROSMsg(poly_traj_msg, info->start_time_);
      fd_->newest_traj_ = poly_traj_msg;
      poly_traj_pub_.publish(fd_->newest_traj_);
      ros::Duration(0.2).sleep();
    }
    exec_timer_.start();
    double dis2occ =
        planner_manager_->lidar_map_interface_->getDisToOcc(fd_->odom_pos_);
    if (dis2occ > planner_manager_->gcopter_config_->dilateRadiusSoft)
      transitState(PLAN_TRAJ, "safe now");
    break;
  }
  case LAND: {
    stopTraj();
    exec_timer_.stop();
    global_path_update_timer_.stop();
    // 没电了！！再飞就会炸鸡，降落！！！
    while (1) {
      quadrotor_msgs::TakeoffLand land_msg;
      land_msg.takeoff_land_cmd = land_msg.LAND;
      land_pub_.publish(land_msg);
      ros::Duration(0.2).sleep();
      ROS_WARN_THROTTLE(1.0, "NO POWER. LAND!!");
    }

    break;
  }
  }
}

void FastExplorationFSM::init(ros::NodeHandle &nh,
                              FastExplorationManager::Ptr &explorer) {
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  /*  Fsm param  */
  nh.param("fsm/thresh_replan", fp_->replan_thresh_, -1.0);
  nh.param("fsm/replan_time", fp_->replan_time_, -1.0);
  nh.param("bubble_astar/resolution_astar", fp_->bubble_a_star_resolution, 0.1);
  nh.param("fsm/debug_planner", debug_planner, false);
  nh.param("fsm/emergency_replan_control_error",
           fp_->emergency_replan_control_error, 0.5);
  nh.param("fsm/replan_time_after_traj_start",
           fp_->replan_time_after_traj_start_, 1.5);
  nh.param("fsm/replan_time_before_traj_end", fp_->replan_time_before_traj_end_,
           1.5);
  /* Initialize main modules */
  // expl_manager_.reset(new FastExplorationManager);
  // expl_manager_->initialize(nh);
  expl_manager_ = explorer;
  planner_manager_ = expl_manager_->planner_manager_;

  // Initialize OQM visualization
  oqm_visualization_ = std::make_shared<observation_quality::OQMVisualization>(nh, expl_manager_->oqm_);

  state_ = EXPL_STATE::INIT;
  fd_->have_odom_ = false;
  fd_->have_cloud_odom_ = false;
  fd_->state_str_ = {"INIT",      "WAIT_TRIGGER", "PLAN_TRAJ", "CAUTION",
                     "EXEC_TRAJ", "FINISH",       "LAND"};
  fd_->static_state_ = true;
  fd_->trigger_ = false;
  fd_->use_bubble_a_star_ = false;
  battary_sub_ =
      nh.subscribe("/mavros/battery", 10, &FastExplorationFSM::battaryCallback,
                   this, ros::TransportHints().tcpNoDelay());

  /* Ros sub, pub and timer */
  // if (debug_planner) {
  //   exec_timer_ = nh.createTimer(ros::Duration(0.01),
  //   &FastExplorationFSM::PlannerDebugFSMCallback, this);
  // } else {
  exec_timer_ = nh.createTimer(ros::Duration(0.01),
                               &FastExplorationFSM::FSMCallback, this);
  // }
  global_path_update_timer_ = nh.createTimer(
      ros::Duration(0.2), &FastExplorationFSM::globalPathUpdateCallback, this);
  trigger_sub_ = nh.subscribe("/waypoint_generator/waypoints", 1,
                              &FastExplorationFSM::triggerCallback, this);
  replan_pub_ = nh.advertise<std_msgs::Empty>("/planning/replan", 10);

  heartbeat_pub_ = nh.advertise<std_msgs::Empty>("/planning/heartbeat", 10);
  land_pub_ =
      nh.advertise<quadrotor_msgs::TakeoffLand>("/px4ctrl/takeoff_land", 10);

  poly_traj_pub_ =
      nh.advertise<traj_utils::PolyTraj>("/planning/trajectory", 10);
  poly_yaw_traj_pub_ =
      nh.advertise<traj_utils::PolyTraj>("/planning/yaw_trajectory", 10);
  poly_pitch_traj_pub_ =
      nh.advertise<traj_utils::PolyTraj>("/planning/pitch_trajectory", 10);
  time_cost_pub_ = nh.advertise<std_msgs::Float32>("/time_cost", 10);
  static_pub_ = nh.advertise<std_msgs::Bool>("/planning/static", 10);
  state_pub_ = nh.advertise<visualization_msgs::Marker>("/planning/state", 10);

  string odom_topic, cloud_topic_colored, cloud_topic_nocolor;
  nh.getParam("odometry_topic", odom_topic);
  nh.getParam("cloud_topic_colored", cloud_topic_colored);
  nh.param<string>("cloud_topic_nocolor", cloud_topic_nocolor, std::string(""));
  single_cloud_mode_ =
      cloud_topic_nocolor.empty() || cloud_topic_nocolor == cloud_topic_colored;
  raw_odom_sub_ = nh.subscribe(odom_topic, 100,
                               &FastExplorationFSM::odometryCallback, this,
                               ros::TransportHints().tcpNoDelay());
  odom_sub_.reset(
      new message_filters::Subscriber<nav_msgs::Odometry>(nh, odom_topic, 500));

  if (single_cloud_mode_) {
    raw_cloud_sub_ = nh.subscribe(
        cloud_topic_colored, 100,
        &FastExplorationFSM::SingleCloudPointCloudCallback, this,
        ros::TransportHints().tcpNoDelay());
    ROS_WARN("FastExplorationFSM: single-cloud mode enabled, using %s for both colored and uncolored inputs",
             cloud_topic_colored.c_str());
  } else {
    cloud_colored_sub_.reset(
        new message_filters::Subscriber<sensor_msgs::PointCloud2>(
            nh, cloud_topic_colored, 100));
    cloud_nocolor_sub_.reset(
        new message_filters::Subscriber<sensor_msgs::PointCloud2>(
            nh, cloud_topic_nocolor, 100));
    sync_cloud_odom_.reset(new message_filters::Synchronizer<SyncPolicyCloudOdom>(
        SyncPolicyCloudOdom(10), *cloud_colored_sub_, *cloud_nocolor_sub_, *odom_sub_));
    sync_cloud_odom_->registerCallback(
        boost::bind(&FastExplorationFSM::CloudOdomCallback, this, _1, _2, _3));
  }

  // Initialize data recording for COLMAP
  nh.param("data_record/enable", enable_data_recording_, false);
  nh.param<string>("data_record/save_path", data_record_path_, "/tmp/colmap_data");
  nh.param("data_record/position_threshold", record_position_threshold_, 0.2);
  nh.param("data_record/angle_threshold", record_angle_threshold_, 10.0);
  nh.param("data_record/cam_fx", record_cam_fx_, 157.05f);
  nh.param("data_record/cam_fy", record_cam_fy_, 157.05f);
  nh.param("data_record/cam_cx", record_cam_cx_, 272.0f);
  nh.param("data_record/cam_cy", record_cam_cy_, 272.0f);
  nh.param("data_record/cam_width", record_cam_width_, 544);
  nh.param("data_record/cam_height", record_cam_height_, 544);

  record_frame_id_ = 0;
  first_record_frame_ = true;
  last_record_position_ = Eigen::Vector3f::Zero();
  last_record_orientation_ = Eigen::Quaternionf::Identity();

  if (enable_data_recording_) {
    // Clear previous data and create directories
    std::string cmd_clear = "rm -rf " + data_record_path_ + "/images " +
                            data_record_path_ + "/pointcloud_colored " +
                            data_record_path_ + "/pointcloud_nocolor " +
                            data_record_path_ + "/poses.txt " +
                            data_record_path_ + "/camera_intrinsics.txt";
    system(cmd_clear.c_str());

    std::string cmd_mkdir = "mkdir -p " + data_record_path_ + "/images " +
                            data_record_path_ + "/pointcloud_colored " +
                            data_record_path_ + "/pointcloud_nocolor";
    system(cmd_mkdir.c_str());

    ROS_WARN("Cleared previous data in: %s", data_record_path_.c_str());

    // Subscribe to RGB image topic
    string rgb_topic;
    nh.param<string>("data_record/rgb_topic", rgb_topic, "/camera/rgb");

    rgb_image_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh, rgb_topic, 1));
    record_odom_sub_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(nh, odom_topic, 5));

    if (single_cloud_mode_) {
      record_cloud_single_sub_.reset(
          new message_filters::Subscriber<sensor_msgs::PointCloud2>(
              nh, cloud_topic_colored, 1));
      sync_single_data_record_.reset(
          new message_filters::Synchronizer<SyncPolicySingleCloudDataRecord>(
              SyncPolicySingleCloudDataRecord(10), *rgb_image_sub_,
              *record_cloud_single_sub_, *record_odom_sub_));
      sync_single_data_record_->registerCallback(
          boost::bind(&FastExplorationFSM::singleCloudDataRecordCallback, this,
                      _1, _2, _3));
    } else {
      record_cloud_colored_sub_.reset(
          new message_filters::Subscriber<sensor_msgs::PointCloud2>(
              nh, cloud_topic_colored, 1));
      record_cloud_nocolor_sub_.reset(
          new message_filters::Subscriber<sensor_msgs::PointCloud2>(
              nh, cloud_topic_nocolor, 1));
      sync_data_record_.reset(new message_filters::Synchronizer<SyncPolicyDataRecord>(
          SyncPolicyDataRecord(10), *rgb_image_sub_, *record_cloud_colored_sub_,
          *record_cloud_nocolor_sub_, *record_odom_sub_));
      sync_data_record_->registerCallback(
          boost::bind(&FastExplorationFSM::dataRecordCallback, this, _1, _2, _3, _4));
    }

    ROS_INFO("Data recording enabled. Saving to: %s", data_record_path_.c_str());
    ROS_INFO("RGB topic: %s, Position threshold: %.2f m, Angle threshold: %.2f deg",
             rgb_topic.c_str(), record_position_threshold_, record_angle_threshold_);
  }

  // Initialize FREE voxel export
  nh.param("free_voxel_export/enable", enable_free_voxel_export_, true);
  nh.param<string>("free_voxel_export/save_path", free_voxel_export_path_, "/tmp/free_voxels");
  nh.param("free_voxel_export/downsample_resolution", free_voxel_downsample_resolution_, 1.0f);
  free_voxel_exported_ = false;

  if (enable_free_voxel_export_) {
    // Create output directory
    std::string cmd_mkdir = "mkdir -p " + free_voxel_export_path_;
    system(cmd_mkdir.c_str());

    ROS_INFO("FREE voxel export enabled. Will save to: %s", free_voxel_export_path_.c_str());
    ROS_INFO("Downsample resolution: %.2f m", free_voxel_downsample_resolution_);
  }
}

void FastExplorationFSM::battaryCallback(
    const sensor_msgs::BatteryStateConstPtr &msg) {
  // if(msg->voltage < 21.0){
  //   transitState(LAND, "battary low");
  // }
}

void FastExplorationFSM::updateTopoAndGlobalPath() {
  if (!(state_ == WAIT_TRIGGER || state_ == PLAN_TRAJ || state_ == EXEC_TRAJ ||
        state_ == FINISH)) {
    global_path_update_timer_.stop();
    // OQM 可视化由 OQM 模块自己处理
    global_path_update_timer_.start();
    return;
  }
  static int cnt = 0;
  cnt++;

  global_path_update_timer_.stop();
  ros::Time t2 = ros::Time::now();
  planner_manager_->topo_graph_->getRegionsToUpdate();
  planner_manager_->topo_graph_->updateSkeleton();

  ros::Time t3 = ros::Time::now();
  planner_manager_->topo_graph_->updateOdomNode(fd_->odom_pos_, fd_->odom_yaw_);
  planner_manager_->topo_graph_->updateHistoricalOdoms();

  if (planner_manager_->topo_graph_->odom_node_->neighbors_.empty()) {
    double time;
    if (planner_manager_->local_data_.traj_id_ > 1) {
      bool safe = planner_manager_->checkTrajCollision(time);
      if (!safe) {
        transitState(CAUTION, "odom_node no nbrs");
      } else {
        global_path_update_timer_.start();
        return;
      }
    } else {
      transitState(CAUTION, "odom_node no nbrs");
    }
    global_path_update_timer_.start();
    return;
  }
  if (planner_manager_->local_data_.traj_id_ > 1) {

    double curr_time =
        (ros::Time::now() - planner_manager_->local_data_.start_time_).toSec();
    double time;
    bool safe = planner_manager_->checkTrajCollision(time);
    double total_time = planner_manager_->local_data_.duration_;
    double time2end = total_time - curr_time;

    if (safe && curr_time < fp_->replan_time_after_traj_start_ &&
        time2end > fp_->replan_time_before_traj_end_) {
      global_path_update_timer_.start();
      return;
    }
  }
  cout << endl << endl;
  cout << "\033[1;33m------------- <" << cnt
       << "> Plan Global Path start---------------" << "\033[0m" << endl;
  planner_manager_->topo_graph_->log << "<" << cnt << ">" << endl;
  ros::Time t4 = ros::Time::now();

  ROS_INFO("update topo skeleton cost: %fms, update odom vertex cost:%fms ",
           (t3 - t2).toSec() * 1000, (t4 - t3).toSec() * 1000);
  Eigen::Vector3d vel = fd_->odom_vel_.cast<double>();
  Eigen::Vector3d odom = fd_->odom_pos_.cast<double>();
  int res = expl_manager_->planGlobalPath(odom, vel, fd_->odom_yaw_, fd_->odom_pitch_);
  ros::Time t5 = ros::Time::now();

  // Call OQM visualization
  if (oqm_visualization_) {
    oqm_visualization_->publishAll(fd_->odom_pos_);
  }

  cout << "\033[1;33m-------------Plan Global Path end-----------------"
       << "\033[0m" << endl
       << endl;

  planner_manager_->graph_visualizer_->vizBox(planner_manager_->topo_graph_);
  if(expl_manager_->ep_->view_graph_)
    planner_manager_->graph_visualizer_->vizGraph(planner_manager_->topo_graph_);
  std_msgs::Float32 time_cost;
  double time_cost_now = (t5 - t2).toSec() * 1000;
  time_cost.data = time_cost_now;
  time_cost_pub_.publish(time_cost);

  cout << "total time cost: " << time_cost_now << "ms" << endl;
  if (res == NO_FRONTIER && state_ != WAIT_TRIGGER) {
    transitState(FINISH, "planGlobalPath: no frontier");
  } else if (res == SUCCEED && state_ != WAIT_TRIGGER) {
    transitState(PLAN_TRAJ, "planGlobalPath: succeed");
  }

  // OQM 可视化由 OQM 模块自己处理

  global_path_update_timer_.start();
  cout << "viz&&print cost:" << (ros::Time::now() - t5).toSec() * 1000 << "ms"
       << endl;
}

void FastExplorationFSM::globalPathUpdateCallback(const ros::TimerEvent &e) {
  updateTopoAndGlobalPath();
}
