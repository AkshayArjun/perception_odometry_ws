// Builds a kd-tree over a few random points and prints its structure.
// Usage: kd_tree_print [num_points] [seed]
#include "perception/kd_tree.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
  const std::size_t n = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 15;
  const unsigned seed = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 0;

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);
  // Round to 1 decimal so the printout is easy to read.
  auto coord = [&] { return std::round(dist(rng) * 10.0) / 10.0; };

  std::vector<Eigen::Vector3d> cloud(n);
  for (auto& p : cloud) p = {coord(), coord(), coord()};

  std::cout << std::fixed << std::setprecision(2) << "Input points:\n";
  for (std::size_t i = 0; i < cloud.size(); ++i)
    std::cout << "  #" << i << " (" << cloud[i].x() << ", " << cloud[i].y() << ", "
              << cloud[i].z() << ")\n";

  perception::KdTree tree(cloud);
  std::cout << "\nTree:\n";
  tree.print(std::cout);

  if (tree.size() == 0) return 0;
  const Eigen::Vector3d query(coord(), coord(), coord());
  const auto nn = tree.nearest(query);
  std::cout << "\nQuery (" << query.x() << ", " << query.y() << ", " << query.z()
            << ") -> nearest #" << nn << " at dist " << (tree.point(nn) - query).norm() << '\n';
}
