#include <path_searching/bubble_astar.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include <plan_env/sdf_map.h>

namespace fast_planner {

namespace {
double squaredDistance(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  return (a - b).squaredNorm();
}
}  // namespace

BubbleAstar::BubbleAstar() {
}

BubbleAstar::~BubbleAstar() {
  for (BubbleGridNodePtr node : path_node_pool_) delete node;
  for (BubblePtr bubble : bubble_node_pool_) delete bubble;
}

void BubbleAstar::init(ros::NodeHandle& nh, const EDTEnvironment::Ptr& env) {
  nh.param("bubble_astar/resolution_astar", resolution_, 0.2);
  nh.param("bubble_astar/lambda_heu", lambda_heu_, 1.0);
  nh.param("bubble_astar/max_search_time", max_search_time_, 1.0);
  nh.param("bubble_astar/allocate_num", allocate_num_, 200000);
  nh.param("bubble_astar/safe_distance", safe_distance_, -1.0);
  nh.param("bubble_astar/debug", debug_, false);
  nh.param("bubble_astar/optimistic_unknown", optimistic_unknown_, false);
  nh.param("bubble_astar/unknown_block_radius", unknown_block_radius_, 1.2);
  nh.param("bubble_astar/min_bubble_radius", min_bubble_radius_, 0.05);
  if (safe_distance_ < 0.0) {
    nh.param("manager/minco_safe_distance", safe_distance_, 0.25);
  }

  edt_env_ = env;
  inv_resolution_ = 1.0 / std::max(1e-3, resolution_);
  Eigen::Vector3d map_size = Eigen::Vector3d::Zero();
  edt_env_->sdf_map_->getRegion(origin_, map_size);

  path_node_pool_.resize(allocate_num_);
  bubble_node_pool_.resize(allocate_num_);
  for (int i = 0; i < allocate_num_; ++i) {
    path_node_pool_[i] = new BubbleGridNode;
    bubble_node_pool_[i] = new Bubble;
  }

  reset(resolution_);
}

void BubbleAstar::reset(double resolution) {
  if (resolution > 1e-6) {
    resolution_ = resolution;
    inv_resolution_ = 1.0 / resolution_;
  }

  safe_area_.clear();
  path_nodes_.clear();
  visited_nodes_.clear();
  open_set_map_.clear();
  close_set_map_.clear();
  dead_set_map_.clear();

  std::priority_queue<BubbleGridNodePtr, std::vector<BubbleGridNodePtr>, BubbleNodeComparator>()
      .swap(open_set_);

  use_node_num_ = 0;
  bubble_used_ = 0;
  iter_num_ = 0;
  early_terminate_cost_ = 0.0;
}

void BubbleAstar::setResolution(const double& res) {
  if (res <= 1e-6) return;
  resolution_ = res;
  inv_resolution_ = 1.0 / resolution_;
}

void BubbleAstar::setOptimisticUnknown(const bool enabled) {
  optimistic_unknown_ = enabled;
}

bool BubbleAstar::getOptimisticUnknown() const {
  return optimistic_unknown_;
}

double BubbleAstar::pathLength(const std::vector<Eigen::Vector3d>& path) {
  double length = 0.0;
  if (path.size() < 2) return length;
  for (size_t i = 1; i < path.size(); ++i) {
    length += (path[i] - path[i - 1]).norm();
  }
  return length;
}

std::vector<Eigen::Vector3d> BubbleAstar::getPath() {
  return path_nodes_;
}

std::vector<Eigen::Vector3d> BubbleAstar::getVisited() {
  return visited_nodes_;
}

double BubbleAstar::getEarlyTerminateCost() {
  return early_terminate_cost_;
}

bool BubbleAstar::pointBlocked(const Eigen::Vector3d& pos) const {
  auto map = edt_env_->sdf_map_;
  if (!map->isInMap(pos) || !map->isInBox(pos)) return true;
  if (map->getInflateOccupancy(pos) == 1) return true;
  const int occ = map->getOccupancy(pos);
  if (occ == SDFMap::OCCUPIED) return true;
  if (occ == SDFMap::UNKNOWN) {
    if (!optimistic_unknown_) return true;
    const double block_radius_sq = std::max(0.0, unknown_block_radius_) * std::max(0.0, unknown_block_radius_);
    if ((pos - current_search_start_).squaredNorm() <= block_radius_sq) return true;
  }
  return false;
}

bool BubbleAstar::pointValidForBubble(const Eigen::Vector3d& pos) const {
  return !pointBlocked(pos);
}

void BubbleAstar::posToIndex(const Eigen::Vector3d& pt, Eigen::Vector3i& idx) const {
  idx = ((pt - origin_) * inv_resolution_).array().floor().cast<int>();
}

double BubbleAstar::getDiagHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2) const {
  double dx = std::fabs(x1.x() - x2.x());
  double dy = std::fabs(x1.y() - x2.y());
  double dz = std::fabs(x1.z() - x2.z());
  double diag = std::min({dx, dy, dz});
  dx -= diag;
  dy -= diag;
  dz -= diag;

  double h = 0.0;
  if (dx < 1e-4) {
    h = std::sqrt(3.0) * diag + std::sqrt(2.0) * std::min(dy, dz) + std::fabs(dy - dz);
  } else if (dy < 1e-4) {
    h = std::sqrt(3.0) * diag + std::sqrt(2.0) * std::min(dx, dz) + std::fabs(dx - dz);
  } else {
    h = std::sqrt(3.0) * diag + std::sqrt(2.0) * std::min(dx, dy) + std::fabs(dx - dy);
  }
  return tie_breaker_ * h;
}

bool BubbleAstar::generateBubble(BubbleGridNodePtr& node, bool is_start) {
  if (bubble_used_ >= allocate_num_) return false;
  if (!pointValidForBubble(node->position)) return false;

  double dist = 0.0;
  Eigen::Vector3d grad = Eigen::Vector3d::Zero();
  edt_env_->evaluateEDTWithGrad(node->position, -1.0, dist, grad);
  if (!std::isfinite(dist)) return false;

  double bubble_radius = dist - safe_distance_;
  if (is_start && bubble_radius <= min_bubble_radius_) {
    bubble_radius = std::max(min_bubble_radius_, 0.5 * std::max(0.0, dist));
  }
  if (bubble_radius <= min_bubble_radius_) return false;

  node->safe_bubble = bubble_node_pool_[bubble_used_++];
  node->safe_bubble->init(bubble_radius * bubble_radius, node->position);
  safe_area_.push_back(node->safe_bubble);
  return true;
}

bool BubbleAstar::safeCheck(BubbleGridNodePtr node) {
  if (node->parent == nullptr || node->parent->safe_bubble == nullptr) {
    return generateBubble(node, true);
  }

  const BubblePtr parent_bubble = node->parent->safe_bubble;
  if (squaredDistance(node->position, parent_bubble->position) <= parent_bubble->radius2) {
    node->safe_bubble = parent_bubble;
    return true;
  }

  for (const BubblePtr bubble : safe_area_) {
    if (squaredDistance(node->position, bubble->position) <= bubble->radius2) {
      node->safe_bubble = bubble;
      return true;
    }
  }

  if (!generateBubble(node, false)) {
    dead_set_map_[node->index] = 1;
    close_set_map_[node->index] = 1;
    return false;
  }

  const double parent_radius = std::sqrt(std::max(0.0, parent_bubble->radius2));
  const double current_radius = std::sqrt(std::max(0.0, node->safe_bubble->radius2));
  const double center_dist = (node->safe_bubble->position - parent_bubble->position).norm();
  return parent_radius + current_radius + resolution_ >= center_dist;
}

int BubbleAstar::search(const Eigen::Vector3d& start_pt, const Eigen::Vector3d& end_pt) {
  reset(resolution_);
  current_search_start_ = start_pt;

  if (!edt_env_ || !edt_env_->sdf_map_) return NO_PATH;
  if (allocate_num_ < 3) return NO_PATH;

  BubbleGridNodePtr start_node = path_node_pool_[use_node_num_++];
  BubbleGridNodePtr end_node = path_node_pool_[use_node_num_++];
  *start_node = BubbleGridNode();
  *end_node = BubbleGridNode();

  start_node->position = start_pt;
  end_node->position = end_pt;
  posToIndex(start_pt, start_node->index);
  posToIndex(end_pt, end_node->index);

  if (!generateBubble(start_node, true)) return START_FAIL;
  if (!generateBubble(end_node, false)) return END_FAIL;

  start_node->g_score = 0.0;
  start_node->f_score = lambda_heu_ * getDiagHeu(start_pt, end_pt);
  open_set_.push(start_node);
  open_set_map_[start_node->index] = start_node;

  const ros::Time t_start = ros::Time::now();
  static const int kDirs[6][3] = {
      {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

  while (!open_set_.empty()) {
    BubbleGridNodePtr cur_node = open_set_.top();
    open_set_.pop();

    if (close_set_map_.find(cur_node->index) != close_set_map_.end()) continue;
    open_set_map_.erase(cur_node->index);
    close_set_map_[cur_node->index] = 1;
    visited_nodes_.push_back(cur_node->position);
    iter_num_ += 1;

    if (squaredDistance(cur_node->position, end_node->safe_bubble->position) <=
        end_node->safe_bubble->radius2) {
      end_node->parent = cur_node;
      backtrack(end_node, end_pt);
      return REACH_END;
    }

    if ((ros::Time::now() - t_start).toSec() > max_search_time_) {
      early_terminate_cost_ = cur_node->g_score + getDiagHeu(cur_node->position, end_pt);
      return NO_PATH;
    }

    for (const auto& dir : kDirs) {
      if (use_node_num_ >= allocate_num_) return NO_PATH;
      Eigen::Vector3d nbr_pos = cur_node->position;
      nbr_pos.x() += dir[0] * resolution_;
      nbr_pos.y() += dir[1] * resolution_;
      nbr_pos.z() += dir[2] * resolution_;
      if (!edt_env_->sdf_map_->isInBox(nbr_pos)) continue;

      BubbleGridNodePtr candidate = path_node_pool_[use_node_num_];
      *candidate = BubbleGridNode();
      candidate->position = nbr_pos;
      posToIndex(nbr_pos, candidate->index);
      if (close_set_map_.find(candidate->index) != close_set_map_.end()) continue;

      candidate->parent = cur_node;
      if (!safeCheck(candidate)) continue;

      const double tmp_g_score = (nbr_pos - cur_node->position).norm() + cur_node->g_score;
      auto open_it = open_set_map_.find(candidate->index);
      BubbleGridNodePtr neighbor = candidate;
      if (open_it == open_set_map_.end()) {
        use_node_num_ += 1;
        if (use_node_num_ >= allocate_num_) return NO_PATH;
      } else {
        if (tmp_g_score >= open_it->second->g_score) continue;
        neighbor = open_it->second;
      }

      neighbor->parent = cur_node;
      neighbor->position = nbr_pos;
      neighbor->index = candidate->index;
      neighbor->safe_bubble = candidate->safe_bubble;
      neighbor->g_score = tmp_g_score;
      neighbor->f_score = tmp_g_score + lambda_heu_ * getDiagHeu(nbr_pos, end_pt);
      open_set_.push(neighbor);
      open_set_map_[neighbor->index] = neighbor;
    }
  }

  return NO_PATH;
}

void BubbleAstar::backtrack(const BubbleGridNodePtr& end_node, const Eigen::Vector3d& end_pt) {
  std::vector<BubbleGridNodePtr> bubble_path;
  for (BubbleGridNodePtr node = end_node; node != nullptr; node = node->parent) {
    bubble_path.push_back(node);
  }
  std::reverse(bubble_path.begin(), bubble_path.end());

  std::vector<BubbleGridNodePtr> shortened;
  shortened.reserve(bubble_path.size());
  for (BubbleGridNodePtr node : bubble_path) {
    if (shortened.empty()) {
      shortened.push_back(node);
      continue;
    }

    if (squaredDistance(node->safe_bubble->position, shortened.back()->safe_bubble->position) < 1e-8) {
      continue;
    }

    if (shortened.size() > 1) {
      BubbleGridNodePtr prev2 = shortened[shortened.size() - 2];
      const double a = std::sqrt(std::max(0.0, node->safe_bubble->radius2));
      const double b = std::sqrt(std::max(0.0, prev2->safe_bubble->radius2));
      const double c = (node->safe_bubble->position - prev2->safe_bubble->position).norm();
      if (a + b > c) shortened.pop_back();
    }
    shortened.push_back(node);
  }

  path_nodes_.clear();
  for (BubbleGridNodePtr node : shortened) {
    path_nodes_.push_back(node->safe_bubble->position);
  }

  if (path_nodes_.empty()) {
    path_nodes_.push_back(end_pt);
  } else {
    path_nodes_.front() = bubble_path.front()->position;
    path_nodes_.back() = end_pt;
  }
}

}  // namespace fast_planner
