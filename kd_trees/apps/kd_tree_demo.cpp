#include "perception/kd_tree.hpp"

#include <chrono>
#include <iostream>
#include <random>

int main() {
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  std::vector<Eigen::Vector3d> cloud(100000);
  for (auto& p : cloud) p = {dist(rng), dist(rng), dist(rng)};

  auto t0 = std::chrono::steady_clock::now();
  perception::KdTree tree(cloud);
  auto t1 = std::chrono::steady_clock::now();
  std::cout << "Built tree over " << tree.size() << " points in "
            << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

  const Eigen::Vector3d query(1.0, 2.0, 3.0);
  const auto nn = tree.nearest(query);
  std::cout << "Nearest to " << query.transpose() << " is " << tree.point(nn).transpose()
            << " (dist " << (tree.point(nn) - query).norm() << ")\n";

}
