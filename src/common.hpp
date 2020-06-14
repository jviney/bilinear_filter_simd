#pragma once

#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>

#include "benchmark/benchmark.h"

struct BenchmarkInput {
  cv::Mat3b source_image;
  cv::Mat2f coords;
  cv::Size2i output_size;
};

static cv::Mat2f sampling_coordinates(cv::Size2i output_size, cv::Size2i input_size) {
  auto coords = cv::Mat2f(output_size);

  for (auto y = 0; y < output_size.height; y++) {
    for (auto x = 0; x < output_size.width; x++) {
      auto x_sample = (float(x) / float(output_size.width)) * input_size.width;
      auto y_sample = (float(y) / float(output_size.height)) * input_size.height;

      coords(y, x) = {y_sample, x_sample};
    }
  }

  return coords;
}

bool mats_equivalent(const cv::Mat3b& a, const cv::Mat3b& b) {
  if ((a.rows != b.rows) || (a.cols != b.cols)) {
    std::cout << "mats different size\n";
    return false;
  }

  for (auto y = 0; y < a.rows; y++) {
    for (auto x = 0; x < a.cols; x++) {
      cv::Vec3b px1 = a(y, x);
      cv::Vec3b px2 = b(y, x);

      auto b_diff = std::abs(px1[0] - px2[0]);
      auto g_diff = std::abs(px1[1] - px2[1]);
      auto r_diff = std::abs(px1[2] - px2[2]);

      // TODO: there are some very slight differences with the SIMD interpolations,
      // probably due to rounding.
      if (b_diff > 1 || g_diff > 1 || r_diff > 1) {
        std::cout << "pixels not equal at " << x << "x" << y << "\n";
        std::cout << int32_t(px1[0]) << " " << int32_t(px1[1]) << " " << int32_t(px1[2]) << "\n";
        std::cout << int32_t(px2[0]) << " " << int32_t(px2[1]) << " " << int32_t(px2[2]) << "\n";
        return false;
      }
    }
  }

  return true;
}
