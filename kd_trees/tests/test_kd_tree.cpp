// Checks the kd-tree against brute-force search on random clouds.
#include "perception/kd_tree.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK failed: " #cond "\n"; \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

using perception::KdTree;
using Point = KdTree::Point;

int main() {
  std::mt19937 rng(0);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);
  auto randomPoint = [&] { return Point(dist(rng), dist(rng), dist(rng)); };

  std::vector<Point> cloud(2000);
  for (auto& p : cloud) p = randomPoint();
  KdTree tree(cloud);

  for (int trial = 0; trial < 200; ++trial) {
    const Point q = randomPoint();

    // Brute-force distances, sorted.
    std::vector<std::pair<double, std::size_t>> brute;
    for (std::size_t i = 0; i < cloud.size(); ++i) brute.emplace_back((cloud[i] - q).norm(), i);
    std::sort(brute.begin(), brute.end());

    CHECK(tree.nearest(q) == brute[0].second);
  }

  CHECK(KdTree({}).size() == 0);
  std::cout << "All kd-tree tests passed\n";
}
