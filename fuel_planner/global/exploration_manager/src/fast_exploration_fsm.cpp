
#include <plan_manage/planner_manager.h>
#include <exploration_manager/fast_exploration_manager.h>
#include <traj_utils/planning_visualization.h>

#include <exploration_manager/fast_exploration_fsm.h>
#include <exploration_manager/expl_data.h>
#include <plan_env/edt_environment.h>
#include <plan_env/sdf_map.h>

using Eigen::Vector4d;

namespace fast_planner {
namespace {
double polyTrajTotalDuration(const traj_utils::PolyTraj& traj) {
  double total = 0.0;
  for (double dt : traj.duration) total += dt;
  return total;
}
}  // namespace

bool FastExplorationFSM::publishRemainingActiveTraj() {
  if (fd_->static_state_) return false;

  const ros::Time now = ros::Time::now();
  const double min_time_left = std::max(0.3, fp_->replan_thresh1_);
  if (!planner_manager_->reuseActiveTrajFromNow(fd_->newest_traj_, fd_->newest_yaw_traj_, now,
                                                min_time_left)) {
    return false;
  }

  const double reused_duration = polyTrajTotalDuration(fd_->newest_traj_);
  const double protect_dt =
      std::max(0.35, std::min(fp_->reuse_traj_safety_grace_, std::max(0.0, reused_duration - 0.05)));
  active_traj_reuse_protect_until_ = now + ros::Duration(protect_dt);
  ROS_WARN_STREAM("Planning failed, publish remaining active trajectory. min_time_left="
                  << min_time_left << " protect_dt=" << protect_dt);
  return true;
}

void FastExplorationFSM::stopCurrentTraj() {
  replan_pub_.publish(std_msgs::Empty());
  ros::Time time_now = ros::Time::now();
  auto* info = &planner_manager_->local_data_;
  if (info->start_time_.toSec() <= 0.0 || info->duration_ <= 0.0) {
    fd_->static_state_ = true;
    return;
  }

  const double elapsed = std::max(0.0, (time_now - info->start_time_).toSec());
  const double shortened =
      std::min(info->duration_,
               elapsed + std::max(0.0, fp_->replan_timeout_) + std::max(0.0, fp_->replan_time_));
  info->duration_ = shortened;
  if (info->duration_ <= elapsed + 1e-3) {
    fd_->static_state_ = true;
  }
}

void FastExplorationFSM::init(ros::NodeHandle& nh) {
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  /*  Fsm param  */
  nh.param("fsm/thresh_replan1", fp_->replan_thresh1_, -1.0);
  nh.param("fsm/thresh_replan2", fp_->replan_thresh2_, -1.0);
  nh.param("fsm/thresh_replan3", fp_->replan_thresh3_, -1.0);
  nh.param("fsm/replan_time", fp_->replan_time_, -1.0);
  nh.param("fsm/replan_time_after_traj_start", fp_->replan_time_after_traj_start_,
           std::max(0.3, fp_->replan_time_));
  nh.param("fsm/replan_time_before_traj_end", fp_->replan_time_before_traj_end_,
           std::max(0.5, fp_->replan_thresh1_));
  nh.param("fsm/replan_min_interval", fp_->replan_min_interval_, 0.8);
  nh.param("fsm/cluster_replan_min_progress", fp_->cluster_replan_min_progress_, 0.35);
  nh.param("fsm/replan_timeout", fp_->replan_timeout_, 0.0);
  nh.param("fsm/emergency_stop_time", fp_->emergency_stop_time_, 0.05);
  nh.param("fsm/reuse_traj_safety_grace", fp_->reuse_traj_safety_grace_, 1.2);
  nh.param("fsm/startup_free_radius_xy", fp_->startup_free_radius_xy_, 1.0);
  nh.param("fsm/startup_free_radius_z", fp_->startup_free_radius_z_, 1.0);
  nh.param("fsm/show_viewpoints", fp_->show_viewpoints_, false);
  nh.param("fsm/show_trajectory", fp_->show_trajectory_, false);
  nh.param("fsm/show_next_goal", fp_->show_next_goal_, false);

  /* Initialize main modules */
  expl_manager_.reset(new FastExplorationManager);
  expl_manager_->initialize(nh);
  visualization_.reset(new PlanningVisualization(nh));

  planner_manager_ = expl_manager_->planner_manager_;
  state_ = EXPL_STATE::INIT;
  fd_->have_odom_ = false;
  fd_->state_str_ = { "INIT", "WAIT_TRIGGER", "PLAN_TRAJ", "PUB_TRAJ", "EXEC_TRAJ", "FINISH" };
  fd_->static_state_ = true;
  fd_->trigger_ = false;
  last_replan_time_ = ros::Time(0);
  active_traj_reuse_protect_until_ = ros::Time(0);
  startup_free_space_initialized_ = false;

  /* Ros sub, pub and timer */
  exec_timer_ = nh.createTimer(ros::Duration(0.01), &FastExplorationFSM::FSMCallback, this);
  safety_timer_ = nh.createTimer(ros::Duration(0.05), &FastExplorationFSM::safetyCallback, this);
  frontier_timer_ = nh.createTimer(ros::Duration(0.2), &FastExplorationFSM::frontierCallback, this);

  trigger_sub_ =
      nh.subscribe("/waypoint_generator/waypoints", 1, &FastExplorationFSM::triggerCallback, this);
  odom_sub_ = nh.subscribe("/odom_world", 1, &FastExplorationFSM::odometryCallback, this);

  replan_pub_ = nh.advertise<std_msgs::Empty>("/planning/replan", 10);
  new_pub_ = nh.advertise<std_msgs::Empty>("/planning/new", 10);
  poly_traj_pub_ = nh.advertise<traj_utils::PolyTraj>("/planning/trajectory", 10);
  poly_yaw_traj_pub_ = nh.advertise<traj_utils::PolyTraj>("/planning/yaw_trajectory", 10);
}

void FastExplorationFSM::FSMCallback(const ros::TimerEvent& e) {
  ROS_INFO_STREAM_THROTTLE(1.0, "[FSM]: state: " << fd_->state_str_[int(state_)]);

  switch (state_) {
    case INIT: {
      // Wait for odometry ready
      if (!fd_->have_odom_) {
        ROS_WARN_THROTTLE(1.0, "no odom.");
        return;
      }
      // Go to wait trigger when odom is ok
      transitState(WAIT_TRIGGER, "FSM");
      break;
    }

    case WAIT_TRIGGER: {
      // Do nothing but wait for trigger
      ROS_WARN_THROTTLE(1.0, "wait for trigger.");
      break;
    }

    case FINISH: {
      ROS_INFO_THROTTLE(1.0, "finish exploration.");
      break;
    }

    case PLAN_TRAJ: {
      const ros::Time plan_begin = ros::Time::now();
      const ros::Time state_query_time = ros::Time::now();
      ros::Time traj_start_time = state_query_time;
      fd_->has_pending_traj_ = false;
      fd_->pending_traj_published_ = false;
      if (fd_->static_state_) {
        fd_->start_pt_ = fd_->odom_pos_;
        fd_->start_vel_ = fd_->odom_vel_;
        fd_->start_acc_.setZero();
        fd_->start_yaw_(0) = fd_->odom_yaw_;
        fd_->start_yaw_(1) = fd_->start_yaw_(2) = 0.0;
      } else {
        auto* info = &planner_manager_->local_data_;
        double t_r = std::max(0.0, (state_query_time - info->start_time_).toSec());
        if (info->use_minco_ && info->minco_traj_.getPieceNum() > 0) {
          t_r = std::min(t_r, info->duration_);
          fd_->start_pt_ = info->minco_traj_.getPos(t_r);
          fd_->start_vel_ = info->minco_traj_.getVel(t_r);
          fd_->start_acc_ = info->minco_traj_.getAcc(t_r);
          if (info->minco_yaw_traj_.getPieceNum() > 0) {
            const double yaw_t = std::min(t_r, info->minco_yaw_traj_.getTotalDuration());
            fd_->start_yaw_(0) = info->minco_yaw_traj_.getPos(yaw_t)[0];
            fd_->start_yaw_(1) = info->minco_yaw_traj_.getVel(yaw_t)[0];
            fd_->start_yaw_(2) = info->minco_yaw_traj_.getAcc(yaw_t)[0];
          } else {
            fd_->start_yaw_(0) = fd_->odom_yaw_;
            fd_->start_yaw_(1) = fd_->start_yaw_(2) = 0.0;
          }
        } else {
          t_r = std::min(t_r, info->duration_);
          fd_->start_pt_ = info->position_traj_.evaluateDeBoorT(t_r);
          fd_->start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_r);
          fd_->start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_r);
          fd_->start_yaw_(0) = info->yaw_traj_.evaluateDeBoorT(t_r)[0];
          fd_->start_yaw_(1) = info->yawdot_traj_.evaluateDeBoorT(t_r)[0];
          fd_->start_yaw_(2) = info->yawdotdot_traj_.evaluateDeBoorT(t_r)[0];
        }
      }
      int res = callExplorationPlanner(traj_start_time);
      const double plan_runtime = (ros::Time::now() - plan_begin).toSec();
      if (res == SUCCEED && !fd_->static_state_) {
        ROS_INFO_STREAM("[explore fsm] planned switch_time_offset="
                        << (traj_start_time - plan_begin).toSec()
                        << " plan_runtime=" << plan_runtime);
      }
      if (res == SUCCEED) {
        if (fd_->has_pending_traj_) {
          planner_manager_->local_data_ = fd_->pending_traj_;
          fd_->has_pending_traj_ = false;
        }
        fd_->pending_traj_published_ = false;
        poly_traj_pub_.publish(fd_->newest_traj_);
        poly_yaw_traj_pub_.publish(fd_->newest_yaw_traj_);
        fd_->static_state_ = false;
        transitState(EXEC_TRAJ, "FSM");

        thread vis_thread(&FastExplorationFSM::visualize, this);
        vis_thread.detach();
      } else if (res == NO_FRONTIER) {
        transitState(FINISH, "FSM");
        fd_->static_state_ = true;
        clearVisMarker();
      } else if (res == FAIL) {
        auto* info = &planner_manager_->local_data_;
        double time_left = 0.0;
        if (info->start_time_.toSec() > 0.0 && info->duration_ > 0.0) {
          time_left = info->duration_ - (ros::Time::now() - info->start_time_).toSec();
        }
        if (time_left > 0.05) {
          fd_->static_state_ = false;
          ROS_WARN_STREAM("plan fail, keep current trajectory while replanning. time_left="
                          << time_left);
        } else {
          ROS_WARN("plan fail");
          fd_->static_state_ = true;
        }
      }
      break;
    }

    case PUB_TRAJ: {
      if (fd_->has_pending_traj_ && !fd_->pending_traj_published_) {
        poly_traj_pub_.publish(fd_->newest_traj_);
        poly_yaw_traj_pub_.publish(fd_->newest_yaw_traj_);
        fd_->pending_traj_published_ = true;
        fd_->static_state_ = false;
      }

      double dt = (ros::Time::now() - fd_->newest_traj_.start_time).toSec();
      if (dt >= 0.0) {
        if (fd_->has_pending_traj_) {
          planner_manager_->local_data_ = fd_->pending_traj_;
          fd_->has_pending_traj_ = false;
        }
        fd_->pending_traj_published_ = false;
        transitState(EXEC_TRAJ, "FSM");
        thread vis_thread(&FastExplorationFSM::visualize, this);
        vis_thread.detach();
      }
      break;
    }

    case EXEC_TRAJ: {
      LocalTrajData* info = &planner_manager_->local_data_;
      const ros::Time now = ros::Time::now();
      double t_cur = (now - info->start_time_).toSec();
      const double duration = std::max(1e-3, info->duration_);
      const double progress = std::max(0.0, std::min(1.0, t_cur / duration));
      const bool replan_interval_ok =
          last_replan_time_.isZero() || (now - last_replan_time_).toSec() > fp_->replan_min_interval_;
      const bool in_roll_window =
          t_cur > fp_->replan_time_after_traj_start_ &&
          info->duration_ - t_cur > fp_->replan_time_before_traj_end_;

      // Replan if traj is almost fully executed
      double time_to_end = info->duration_ - t_cur;
      if (time_to_end < fp_->replan_thresh1_) {
        transitState(PLAN_TRAJ, "FSM");
        ROS_WARN("Replan: traj fully executed=================================");
        return;
      }
      // Replan if next frontier to be visited is covered
      if (in_roll_window && replan_interval_ok && progress > fp_->cluster_replan_min_progress_ &&
          t_cur > fp_->replan_thresh2_ &&
          expl_manager_->frontier_finder_->isFrontierCovered()) {
        last_replan_time_ = now;
        transitState(PLAN_TRAJ, "FSM");
        ROS_WARN_STREAM("Replan: cluster covered===================================== progress="
                        << progress << " dt_since_last="
                        << (last_replan_time_.isZero() ? -1.0 : (now - last_replan_time_).toSec()));
        return;
      }
      // Replan after some time
      if (in_roll_window && replan_interval_ok && t_cur > fp_->replan_thresh3_ && !classic_) {
        last_replan_time_ = now;
        transitState(PLAN_TRAJ, "FSM");
        ROS_WARN("Replan: rolling lookahead===================================");
      }
      break;
    }
  }
}

int FastExplorationFSM::callExplorationPlanner(const ros::Time& traj_start_time) {
  const LocalTrajData active_backup = planner_manager_->local_data_;
  const bool had_active =
      !fd_->static_state_ && active_backup.start_time_.toSec() > 0.0 && active_backup.duration_ > 0.0;

  int res = expl_manager_->planExploreMotion(fd_->start_pt_, fd_->start_vel_, fd_->start_acc_,
                                             fd_->start_yaw_);
  classic_ = false;

  // int res = expl_manager_->classicFrontier(fd_->start_pt_, fd_->start_yaw_[0]);
  // classic_ = true;

  // int res = expl_manager_->rapidFrontier(fd_->start_pt_, fd_->start_vel_, fd_->start_yaw_[0],
  // classic_);

  if (res == SUCCEED) {
    const ros::Time publish_start_time = ros::Time::now();
    auto planned = planner_manager_->local_data_;
    planned.start_time_ = publish_start_time;
    fd_->pending_traj_ = planned;
    fd_->has_pending_traj_ = true;
    fd_->pending_traj_published_ = false;

    planner_manager_->local_data_ = planned;
    planner_manager_->exportTrajToPolyMsg(fd_->newest_traj_, fd_->newest_yaw_traj_, publish_start_time);

    if (had_active) {
      planner_manager_->local_data_ = active_backup;
    }
  } else if (had_active) {
    planner_manager_->local_data_ = active_backup;
  }
  return res;
}

void FastExplorationFSM::visualize() {
  auto info = &planner_manager_->local_data_;
  auto ed_ptr = expl_manager_->ed_;

  // Draw updated box
  // Vector3d bmin, bmax;
  // planner_manager_->edt_environment_->sdf_map_->getUpdatedBox(bmin, bmax);
  // visualization_->drawBox((bmin + bmax) / 2.0, bmax - bmin, Vector4d(0, 1, 0, 0.3), "updated_box", 0,
  // 4);

  // Draw frontier
  static int last_ftr_num = 0;
  for (int i = 0; i < ed_ptr->frontiers_.size(); ++i) {
    visualization_->drawCubes(ed_ptr->frontiers_[i], 0.1,
                              visualization_->getColor(double(i) / ed_ptr->frontiers_.size(), 0.4),
                              "frontier", i, 4);
    // visualization_->drawBox(ed_ptr->frontier_boxes_[i].first, ed_ptr->frontier_boxes_[i].second,
    //                         Vector4d(0.5, 0, 1, 0.3), "frontier_boxes", i, 4);
  }
  for (int i = ed_ptr->frontiers_.size(); i < last_ftr_num; ++i) {
    visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
    // visualization_->drawBox(Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector4d(1, 0, 0, 0.3),
    // "frontier_boxes", i, 4);
  }
  last_ftr_num = ed_ptr->frontiers_.size();
  // for (int i = 0; i < ed_ptr->dead_frontiers_.size(); ++i)
  //   visualization_->drawCubes(ed_ptr->dead_frontiers_[i], 0.1, Vector4d(0, 0, 0, 0.5), "dead_frontier",
  //                             i, 4);
  // for (int i = ed_ptr->dead_frontiers_.size(); i < 5; ++i)
  //   visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 0.5), "dead_frontier", i, 4);

  // Draw global top viewpoints info
  if (fp_->show_viewpoints_) {
    visualization_->drawSpheres(ed_ptr->points_, 0.2, Vector4d(0, 0.5, 0, 1), "points", 0, 6);
    visualization_->drawLines(ed_ptr->global_tour_, 0.07, Vector4d(0, 0.5, 0, 1), "global_tour", 0, 6);
    visualization_->drawLines(ed_ptr->points_, ed_ptr->views_, 0.05, Vector4d(0, 1, 0.5, 1), "view", 0,
                              6);
    visualization_->drawLines(ed_ptr->points_, ed_ptr->averages_, 0.03, Vector4d(1, 0, 0, 1),
                              "point-average", 0, 6);
  } else {
    visualization_->drawSpheres({}, 0.2, Vector4d(0, 0.5, 0, 1), "points", 0, 6);
    visualization_->drawLines({}, 0.07, Vector4d(0, 0.5, 0, 1), "global_tour", 0, 6);
    visualization_->drawLines({}, {}, 0.05, Vector4d(0, 1, 0.5, 1), "view", 0, 6);
    visualization_->drawLines({}, {}, 0.03, Vector4d(1, 0, 0, 1), "point-average", 0, 6);
  }

  // Draw local refined viewpoints info
  if (fp_->show_viewpoints_) {
    visualization_->drawSpheres(ed_ptr->refined_points_, 0.2, Vector4d(0, 0, 1, 1), "refined_pts", 0, 6);
    visualization_->drawLines(ed_ptr->refined_points_, ed_ptr->refined_views_, 0.05,
                              Vector4d(0.5, 0, 1, 1), "refined_view", 0, 6);
    visualization_->drawLines(ed_ptr->refined_tour_, 0.07, Vector4d(0, 0, 1, 1), "refined_tour", 0, 6);
    visualization_->drawLines(ed_ptr->refined_views1_, ed_ptr->refined_views2_, 0.04,
                              Vector4d(0, 0, 0, 1), "refined_view", 1, 6);
    visualization_->drawLines(ed_ptr->refined_points_, ed_ptr->unrefined_points_, 0.05,
                              Vector4d(1, 1, 0, 1), "refine_pair", 0, 6);
    for (int i = 0; i < ed_ptr->n_points_.size(); ++i)
      visualization_->drawSpheres(
          ed_ptr->n_points_[i], 0.1,
          visualization_->getColor(ed_ptr->frontiers_.empty()
                                       ? 0.0
                                       : double(ed_ptr->refined_ids_[i]) / ed_ptr->frontiers_.size()),
          "n_points", i, 6);
    for (int i = ed_ptr->n_points_.size(); i < 15; ++i)
      visualization_->drawSpheres({}, 0.1, Vector4d(0, 0, 0, 1), "n_points", i, 6);
  } else {
    visualization_->drawSpheres({}, 0.2, Vector4d(0, 0, 1, 1), "refined_pts", 0, 6);
    visualization_->drawLines({}, {}, 0.05, Vector4d(0.5, 0, 1, 1), "refined_view", 0, 6);
    visualization_->drawLines({}, {}, 0.04, Vector4d(0, 0, 0, 1), "refined_view", 1, 6);
    visualization_->drawLines({}, 0.07, Vector4d(0, 0, 1, 1), "refined_tour", 0, 6);
    visualization_->drawLines({}, {}, 0.05, Vector4d(1, 1, 0, 1), "refine_pair", 0, 6);
    for (int i = 0; i < 15; ++i)
      visualization_->drawSpheres({}, 0.1, Vector4d(0, 0, 0, 1), "n_points", i, 6);
  }

  // Draw trajectory
  if (fp_->show_next_goal_) {
    if (!ed_ptr->lookahead_goals_.empty()) {
      visualization_->drawSpheres(ed_ptr->lookahead_goals_, 0.3, Vector4d(0, 1, 1, 1), "next_goal", 0, 6);
    } else if (!ed_ptr->path_next_goal_.empty()) {
      visualization_->drawSpheres({ ed_ptr->next_goal_ }, 0.3, Vector4d(0, 1, 1, 1), "next_goal", 0, 6);
    } else {
      visualization_->drawSpheres({}, 0.3, Vector4d(0, 1, 1, 1), "next_goal", 0, 6);
    }

    if (!ed_ptr->lookahead_path_segments_.empty()) {
      for (int i = 0; i < static_cast<int>(ed_ptr->lookahead_path_segments_.size()); ++i) {
        visualization_->drawLines(ed_ptr->lookahead_path_segments_[i].path, 0.05,
                                  Vector4d(0, 1, 1, 1), "next_goal", i + 1, 6);
      }
      for (int i = static_cast<int>(ed_ptr->lookahead_path_segments_.size()); i < 5; ++i) {
        visualization_->drawLines({}, 0.05, Vector4d(0, 1, 1, 1), "next_goal", i + 1, 6);
      }
    } else {
      visualization_->drawLines(ed_ptr->path_next_goal_, 0.05, Vector4d(0, 1, 1, 1), "next_goal", 1, 6);
      for (int i = 1; i < 5; ++i) {
        visualization_->drawLines({}, 0.05, Vector4d(0, 1, 1, 1), "next_goal", i + 1, 6);
      }
    }
  } else {
    visualization_->drawSpheres({}, 0.3, Vector4d(0, 1, 1, 1), "next_goal", 0, 6);
    for (int i = 0; i < 5; ++i) {
      visualization_->drawLines({}, 0.05, Vector4d(0, 1, 1, 1), "next_goal", i + 1, 6);
    }
  }
  if (fp_->show_trajectory_) {
    if (info->use_minco_) {
      vector<Vector3d> traj_pts;
      for (double t = 0.0; t <= info->duration_ + 1e-3; t += 0.05)
        traj_pts.push_back(info->minco_traj_.getPos(std::min(t, info->duration_)));
      visualization_->drawLines(traj_pts, 0.08, Vector4d(1.0, 0.0, 0.0, 1), "MINCO", 0, 0);
    } else {
      visualization_->drawBspline(info->position_traj_, 0.1, Vector4d(1.0, 0.0, 0.0, 1), false, 0.15,
                                  Vector4d(1, 1, 0, 1));
    }
  } else {
    visualization_->drawLines({}, 0.1, Vector4d(1.0, 0.0, 0.0, 1), "MINCO", 0, 0);
    for (int i = 1; i < 100; ++i) visualization_->drawSpheres({}, 0.1, Vector4d(1, 0, 0, 1), "MINCO", i, 0);
    for (int i = 50; i < 150; ++i) visualization_->drawSpheres({}, 0.15, Vector4d(1, 1, 0, 1), "MINCO", i, 0);
  }
}

void FastExplorationFSM::clearVisMarker() {
  visualization_->drawSpheres({}, 0.2, Vector4d(0, 0.5, 0, 1), "points", 0, 6);
  visualization_->drawLines({}, 0.07, Vector4d(0, 0.5, 0, 1), "global_tour", 0, 6);
  visualization_->drawLines({}, {}, 0.05, Vector4d(0, 1, 0.5, 1), "view", 0, 6);
  visualization_->drawLines({}, {}, 0.03, Vector4d(1, 0, 0, 1), "point-average", 0, 6);
  visualization_->drawSpheres({}, 0.2, Vector4d(0, 0, 1, 1), "refined_pts", 0, 6);
  visualization_->drawLines({}, {}, 0.05, Vector4d(0.5, 0, 1, 1), "refined_view", 0, 6);
  visualization_->drawLines({}, {}, 0.04, Vector4d(0, 0, 0, 1), "refined_view", 1, 6);
  visualization_->drawLines({}, 0.07, Vector4d(0, 0, 1, 1), "refined_tour", 0, 6);
  visualization_->drawLines({}, {}, 0.05, Vector4d(1, 1, 0, 1), "refine_pair", 0, 6);
  for (int i = 0; i < 15; ++i)
    visualization_->drawSpheres({}, 0.1, Vector4d(0, 0, 0, 1), "n_points", i, 6);
  visualization_->drawSpheres({}, 0.3, Vector4d(0, 1, 1, 1), "next_goal", 0, 6);
  for (int i = 0; i < 5; ++i)
    visualization_->drawLines({}, 0.05, Vector4d(0, 1, 1, 1), "next_goal", i + 1, 6);
}

void FastExplorationFSM::frontierCallback(const ros::TimerEvent& e) {
  if (state_ == WAIT_TRIGGER || state_ == FINISH) {
    auto ft = expl_manager_->frontier_finder_;
    auto ed = expl_manager_->ed_;
    ft->searchFrontiers();
    ft->computeFrontiersToVisit();
    ft->updateFrontierCostMatrix();

    ft->getFrontiers(ed->frontiers_);
    ft->getFrontierBoxes(ed->frontier_boxes_);

    // Draw frontier and bounding box
    for (int i = 0; i < ed->frontiers_.size(); ++i) {
      visualization_->drawCubes(ed->frontiers_[i], 0.1,
                                visualization_->getColor(double(i) / ed->frontiers_.size(), 0.4),
                                "frontier", i, 4);
      // visualization_->drawBox(ed->frontier_boxes_[i].first, ed->frontier_boxes_[i].second,
      // Vector4d(0.5, 0, 1, 0.3),
      //                         "frontier_boxes", i, 4);
    }
    for (int i = ed->frontiers_.size(); i < 50; ++i) {
      visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
      // visualization_->drawBox(Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector4d(1, 0, 0, 0.3),
      // "frontier_boxes", i, 4);
    }

    ed->points_.clear();
    ed->yaws_.clear();
    ed->averages_.clear();
    ed->views_.clear();
    ed->path_next_goal_.clear();
    ed->lookahead_goals_.clear();
    ed->lookahead_yaws_.clear();
    ed->lookahead_path_segments_.clear();
    ed->lookahead_arrival_times_.clear();
    ed->refined_points_.clear();
    ed->refined_views_.clear();
    ed->refined_views1_.clear();
    ed->refined_views2_.clear();
    ed->refined_tour_.clear();
    ed->unrefined_points_.clear();
    ed->n_points_.clear();
    ed->refined_ids_.clear();

    if ((fp_->show_viewpoints_ || fp_->show_next_goal_) && !ed->frontiers_.empty()) {
      ft->getTopViewpointsInfo(fd_->odom_pos_, ed->points_, ed->yaws_, ed->averages_);
      for (int i = 0; i < ed->points_.size(); ++i) {
        ed->views_.push_back(
            ed->points_[i] + 2.0 * Vector3d(cos(ed->yaws_[i]), sin(ed->yaws_[i]), 0));
      }

      if (!ed->points_.empty()) {
        ed->next_goal_ = ed->points_.front();
        planner_manager_->path_finder_->reset();
        if (planner_manager_->path_finder_->search(fd_->odom_pos_, ed->next_goal_) ==
            BubbleAstar::REACH_END) {
          ed->path_next_goal_ = planner_manager_->path_finder_->getPath();
        }
      }
    }

    if (fp_->show_viewpoints_) {
      visualization_->drawSpheres(ed->points_, 0.2, Vector4d(0, 0.5, 0, 1), "points", 0, 6);
      visualization_->drawLines(ed->points_, ed->views_, 0.05, Vector4d(0, 1, 0.5, 1), "view", 0, 6);
      visualization_->drawLines(ed->points_, ed->averages_, 0.03, Vector4d(1, 0, 0, 1),
                                "point-average", 0, 6);
    } else {
      visualization_->drawSpheres({}, 0.2, Vector4d(0, 0.5, 0, 1), "points", 0, 6);
      visualization_->drawLines({}, {}, 0.05, Vector4d(0, 1, 0.5, 1), "view", 0, 6);
      visualization_->drawLines({}, {}, 0.03, Vector4d(1, 0, 0, 1), "point-average", 0, 6);
    }
    if (fp_->show_next_goal_) {
      if (!ed->path_next_goal_.empty())
        visualization_->drawSpheres({ ed->next_goal_ }, 0.3, Vector4d(0, 1, 1, 1), "next_goal", 0, 6);
      else
        visualization_->drawSpheres({}, 0.3, Vector4d(0, 1, 1, 1), "next_goal", 0, 6);
      visualization_->drawLines(ed->path_next_goal_, 0.05, Vector4d(0, 1, 1, 1), "next_goal", 1, 6);
      for (int i = 1; i < 5; ++i)
        visualization_->drawLines({}, 0.05, Vector4d(0, 1, 1, 1), "next_goal", i + 1, 6);
    } else {
      visualization_->drawSpheres({}, 0.3, Vector4d(0, 1, 1, 1), "next_goal", 0, 6);
      for (int i = 0; i < 5; ++i)
        visualization_->drawLines({}, 0.05, Vector4d(0, 1, 1, 1), "next_goal", i + 1, 6);
    }
  }

  // if (!fd_->static_state_)
  // {
  //   static double astar_time = 0.0;
  //   static int astar_num = 0;
  //   auto t1 = ros::Time::now();

  //   planner_manager_->path_finder_->reset();
  //   planner_manager_->path_finder_->setResolution(0.4);
  //   if (planner_manager_->path_finder_->search(fd_->odom_pos_, Vector3d(-5, 0, 1)))
  //   {
  //     auto path = planner_manager_->path_finder_->getPath();
  //     visualization_->drawLines(path, 0.05, Vector4d(1, 0, 0, 1), "astar", 0, 6);
  //     auto visit = planner_manager_->path_finder_->getVisited();
  //     visualization_->drawCubes(visit, 0.3, Vector4d(0, 0, 1, 0.4), "astar-visit", 0, 6);
  //   }
  //   astar_num += 1;
  //   astar_time = (ros::Time::now() - t1).toSec();
  //   ROS_WARN("Average astar time: %lf", astar_time);
  // }
}

void FastExplorationFSM::triggerCallback(const nav_msgs::PathConstPtr& msg) {
  if (msg->poses[0].pose.position.z < -0.1) return;
  if (state_ != WAIT_TRIGGER) return;
  fd_->trigger_ = true;
  cout << "Triggered!" << endl;
  transitState(PLAN_TRAJ, "triggerCallback");
}

void FastExplorationFSM::safetyCallback(const ros::TimerEvent& e) {
  if (state_ == EXPL_STATE::EXEC_TRAJ || state_ == EXPL_STATE::PUB_TRAJ) {
    if (!active_traj_reuse_protect_until_.isZero() &&
        ros::Time::now() < active_traj_reuse_protect_until_) {
      return;
    }
    // Check safety and trigger replan if necessary
    double dist, collision_time;
    bool safe = planner_manager_->checkTrajCollision(dist, collision_time);
    if (!safe) {
      ROS_WARN("Replan: collision detected==================================");
      if (collision_time < std::max(0.01, fp_->emergency_stop_time_)) {
        stopCurrentTraj();
      } else {
        ROS_WARN_STREAM("[explore fsm] keep current trajectory during collision-triggered replanning. "
                        << "collision_time=" << collision_time
                        << " emergency_stop_time=" << fp_->emergency_stop_time_);
      }
      fd_->has_pending_traj_ = false;
      fd_->pending_traj_published_ = false;
      transitState(PLAN_TRAJ, "safetyCallback");
    }
  }
}

void FastExplorationFSM::odometryCallback(const nav_msgs::OdometryConstPtr& msg) {
  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  fd_->odom_vel_(0) = msg->twist.twist.linear.x;
  fd_->odom_vel_(1) = msg->twist.twist.linear.y;
  fd_->odom_vel_(2) = msg->twist.twist.linear.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

  if (!startup_free_space_initialized_ && expl_manager_ &&
      fp_->startup_free_radius_xy_ > 1e-3 && fp_->startup_free_radius_z_ > 1e-3) {
    expl_manager_->planner_manager_->edt_environment_->sdf_map_->carveFreeRegion(
        fd_->odom_pos_, fp_->startup_free_radius_xy_, fp_->startup_free_radius_z_);
    startup_free_space_initialized_ = true;
    ROS_WARN_STREAM("[FSM] carved startup free region around odom at "
                    << fd_->odom_pos_.transpose() << " with radius_xy="
                    << fp_->startup_free_radius_xy_ << " radius_z="
                    << fp_->startup_free_radius_z_);
  }

  fd_->have_odom_ = true;
}

void FastExplorationFSM::transitState(EXPL_STATE new_state, string pos_call) {
  int pre_s = int(state_);
  state_ = new_state;
  cout << "[" + pos_call + "]: from " + fd_->state_str_[pre_s] + " to " + fd_->state_str_[int(new_state)]
       << endl;
}
}  // namespace fast_planner
