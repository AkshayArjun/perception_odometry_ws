#include "perception/kd_tree.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <numeric>
#include <utility>

namespace perception {

KdTree::KdTree(std::vector<Point> points) : points_(std::move(points)) {
  std::vector<std::size_t> indices(points_.size());
  std::iota(indices.begin(), indices.end(), 0);
  root_ = build(indices.begin(), indices.end(), 0);
}

std::unique_ptr<KdTree::Node> KdTree::build(std::vector<std::size_t>::iterator begin,
                                            std::vector<std::size_t>::iterator end,
                                            int depth) {

  // the base condition: if the range is empty, return nullptr
  if (begin >= end) {
    return nullptr;
  }

  int axis = depth % 3;
  auto mid = begin + std::distance(begin, end) / 2;

  std::nth_element(begin, mid, end, [this, axis](std::size_t a, std::size_t b)
  {
    return points_[a][axis] < points_[b][axis];
  });

  auto node = std::make_unique<Node>();
  node->index = *mid;
  node->axis = axis;
  node->left = build(begin, mid, depth + 1);
  node->right = build(mid + 1, end, depth + 1);

  return node;
}

std::size_t KdTree::nearest(const Point& query) const {
  std::size_t best_index = 0;
  double best_dist2 = std::numeric_limits<double>::max();
  nearestSearch(root_.get(), query, best_index, best_dist2);
  return best_index;
}

void KdTree::nearestSearch(const Node* node, const Point& query, std::size_t& best_index,
                           double& best_dist2) const {
  if (!node) {
    return;
  }

  double sq_dist = (points_[node->index] - query).squaredNorm();
  if (sq_dist < best_dist2) {
    best_dist2 = sq_dist;
    best_index = node->index;
  }
  
  double diff = query[node->axis] - points_[node->index][node->axis];
  const Node* first = diff < 0 ? node->left.get() : node->right.get();
  const Node* second = diff < 0 ? node->right.get() : node->left.get(); 

  nearestSearch(first, query, best_index, best_dist2);


  if (diff * diff < best_dist2) {
    nearestSearch(second, query, best_index, best_dist2);
  }
}

void KdTree::print(std::ostream& os) const {
  if (!root_) {
    os << "(empty tree)\n";
    return;
  }
  printNode(root_.get(), os, "", "root", true);
}

void KdTree::printNode(const Node* node, std::ostream& os, const std::string& prefix,
                       const std::string& label, bool last) const {
  static const char* kAxis[] = {"x", "y", "z"};
  const bool is_root = label == "root";
  const Point& p = points_[node->index];

  os << prefix << (is_root ? "" : (last ? "└── " : "├── ")) << label << " split "
     << kAxis[node->axis] << "=" << std::fixed << std::setprecision(2) << p[node->axis]
     << "  #" << node->index << " (" << p.x() << ", " << p.y() << ", " << p.z() << ")\n";

  const std::string child_prefix = prefix + (is_root ? "" : (last ? "    " : "│   "));
  if (node->left) printNode(node->left.get(), os, child_prefix, "L", !node->right);
  if (node->right) printNode(node->right.get(), os, child_prefix, "R", true);
}

}  // namespace perception
