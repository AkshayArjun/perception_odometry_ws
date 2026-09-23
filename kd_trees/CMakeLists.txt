cmake_minimum_required(VERSION 3.16)
project(perception_odometry LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for clangd / VS Code IntelliSense

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ---- Core library: every reusable algorithm goes here ----
add_library(perception
  src/kd_tree.cpp
)
target_include_directories(perception PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(perception PUBLIC Eigen3::Eigen)
target_compile_options(perception PRIVATE -Wall -Wextra -Wpedantic)

# ---- Executables: experiments / demos ----
add_executable(kd_tree_demo apps/kd_tree_demo.cpp)
target_link_libraries(kd_tree_demo PRIVATE perception)
add_executable(kd_tree_print apps/kd_tree_print.cpp)
target_link_libraries(kd_tree_print PRIVATE perception)

# ---- Tests ----
enable_testing()
add_executable(test_kd_tree tests/test_kd_tree.cpp)
target_link_libraries(test_kd_tree PRIVATE perception)
add_test(NAME test_kd_tree COMMAND test_kd_tree)
