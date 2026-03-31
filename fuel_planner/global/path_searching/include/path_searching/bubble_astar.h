#ifndef _BUBBLE_ASTAR_H_
#define _BUBBLE_ASTAR_H_

#include <Eigen/Eigen>

#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

#include <ros/ros.h>

#include <path_searching/matrix_hash.h>
#include <plan_env/edt_environment.h>

namespace fast_planner {

struct Bubble {
  double radius2 = 0.0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();

  void init(double radius2_, const Eigen::Vector3d& position_) {
    radius2 = radius2_;
    position = position_;
  }
};

class BubbleGridNode {
public:
  Eigen::Vector3i index = Eigen::Vector3i::Zero();
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  double g_score = 0.0;
  double f_score = 0.0;
  Bubble* safe_bubble = nullptr;
  BubbleGridNode* parent = nullptr;
};

typedef BubbleGridNode* BubbleGridNodePtr;
typedef Bubble* BubblePtr;

class BubbleNodeComparator {
public:
  bool operator()(BubbleGridNodePtr lhs, BubbleGridNodePtr rhs) const {
    return lhs->f_score > rhs->f_score;
  }
};

class BubbleAstar {
public:
  typedef std::shared_ptr<BubbleAstar> Ptr;

  BubbleAstar();
  ~BubbleAstar();

  enum { REACH_END = 1, NO_PATH = 2, START_FAIL = 3, END_FAIL = 4 };

  void init(ros::NodeHandle& nh, const EDTEnvironment::Ptr& env);
  void reset(double resolution = -1.0);
  int search(const Eigen::Vector3d& start_pt, const Eigen::Vector3d& end_pt);
  void setResolution(const double& res);
  void setOptimisticUnknown(const bool enabled);
  bool getOptimisticUnknown() const;
  static double pathLength(const std::vector<Eigen::Vector3d>& path);

  std::vector<Eigen::Vector3d> getPath();
  std::vector<Eigen::Vector3d> getVisited();
  double getEarlyTerminateCost();

  double lambda_heu_ = 1.0;
  double max_search_time_ = 1.0;

private:
  void backtrack(const BubbleGridNodePtr& end_node, const Eigen::Vector3d& end_pt);
  bool generateBubble(BubbleGridNodePtr& node, bool is_start = false);
  bool safeCheck(BubbleGridNodePtr node);
  bool pointBlocked(const Eigen::Vector3d& pos) const;
  bool pointValidForBubble(const Eigen::Vector3d& pos) const;
  void posToIndex(const Eigen::Vector3d& pt, Eigen::Vector3i& idx) const;
  double getDiagHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2) const;

  EDTEnvironment::Ptr edt_env_;

  std::vector<BubbleGridNodePtr> path_node_pool_;
  std::vector<BubblePtr> bubble_node_pool_;
  std::vector<BubblePtr> safe_area_;
  std::vector<Eigen::Vector3d> path_nodes_;
  std::vector<Eigen::Vector3d> visited_nodes_;

  std::priority_queue<BubbleGridNodePtr, std::vector<BubbleGridNodePtr>, BubbleNodeComparator>
      open_set_;
  std::unordered_map<Eigen::Vector3i, BubbleGridNodePtr, matrix_hash<Eigen::Vector3i>> open_set_map_;
  std::unordered_map<Eigen::Vector3i, int, matrix_hash<Eigen::Vector3i>> close_set_map_;
  std::unordered_map<Eigen::Vector3i, int, matrix_hash<Eigen::Vector3i>> dead_set_map_;

  Eigen::Vector3d origin_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d current_search_start_ = Eigen::Vector3d::Zero();
  double resolution_ = 0.2;
  double inv_resolution_ = 5.0;
  double safe_distance_ = 0.25;
  double min_bubble_radius_ = 0.05;
  double unknown_block_radius_ = 1.2;
  double tie_breaker_ = 1.001;
  bool optimistic_unknown_ = false;
  bool debug_ = false;
  int allocate_num_ = 200000;
  int use_node_num_ = 0;
  int bubble_used_ = 0;
  int iter_num_ = 0;
  double early_terminate_cost_ = 0.0;
};

}  // namespace fast_planner

#endif
