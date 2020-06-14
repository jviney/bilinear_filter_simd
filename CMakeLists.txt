cmake_minimum_required(VERSION 3.16)

project(bilinear_filter_simd LANGUAGES CXX)

add_executable(bilinear_filter_simd
  src/main.cpp
)

set(CMAKE_BUILD_TYPE Release)

target_include_directories(bilinear_filter_simd PRIVATE src/)

# Enable compiler warnings
set(WARNINGS -Wall -Wextra -Wpedantic)
target_compile_options(bilinear_filter_simd PRIVATE ${WARNINGS})

# C++17
set_target_properties(bilinear_filter_simd PROPERTIES CXX_STANDARD 17)
set_target_properties(bilinear_filter_simd PROPERTIES CXX_STANDARD_REQUIRED ON)

# Target native architecture
target_compile_options(bilinear_filter_simd PUBLIC -mavx2 -mavx)

# Use OpenCV
find_package(OpenCV 4 REQUIRED)
target_include_directories(bilinear_filter_simd PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(bilinear_filter_simd PRIVATE ${OpenCV_LIBRARIES})

# Benchmark
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
add_subdirectory(vendor/benchmark)
target_link_libraries(bilinear_filter_simd PRIVATE benchmark::benchmark)
