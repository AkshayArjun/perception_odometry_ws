#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace perception {

// A static 3D kd-tree over a point cloud, used for nearest-neighbor lookups in ICP.
// Points are copied in at construction; queries return indices into the original point vector.
class KdTree {
 public:
  using Point = Eigen::Vector3d;

  explicit KdTree(std::vector<Point> points);

  // Index of the closest point to `query`. Tree must be non-empty.
  std::size_t nearest(const Point& query) const;

  std::size_t size() const { return points_.size(); }
  const Point& point(std::size_t i) const { return points_[i]; }

  // Draws the tree structure (split axis, point index, coordinates) for debugging.
  void print(std::ostream& os) const;

 private:
  struct Node {
    std::size_t index;  // point stored at this node
    int axis;           // splitting dimension (0=x, 1=y, 2=z)
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
  };

  std::unique_ptr<Node> build(std::vector<std::size_t>::iterator begin,
                              std::vector<std::size_t>::iterator end, int depth);

  // Recursive helper for nearest(): updates best_index / best_dist2 if a closer point is found.
  void nearestSearch(const Node* node, const Point& query, std::size_t& best_index,
                     double& best_dist2) const;

  void printNode(const Node* node, std::ostream& os, const std::string& prefix,
                 const std::string& label, bool last) const;

  std::vector<Point> points_;
  std::unique_ptr<Node> root_;
};

}  // namespace perception
