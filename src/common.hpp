#pragma once

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "benchmark/benchmark.h"

#include "interpolate/types.hpp"

struct BenchmarkInput {
  cv::Mat3b source_image_mat;
  interpolate::BGRImage source_image;
  cv::Mat2f coords;
  cv::Size2i output_size;
};

static cv::Mat2f sampling_coordinates(cv::Size2i output_size, cv::Size2i input_size) {
  auto coords = cv::Mat2f(output_size);

  for (auto y = 0; y < output_size.height; y++) {
    for (auto x = 0; x < output_size.width; x++) {
      auto x_sample = (float(x) / float(output_size.width)) * input_size.width;
      auto y_sample = float(y) / float(output_size.height) * input_size.height;

      coords(y, x) = {y_sample, x_sample};
    }
  }

  // Rotate a bit
  auto angle = 3.14f / 10.0f;
  auto warp = cv::Mat1f(2, 3);
  warp(0, 0) = cos(angle);
  warp(0, 1) = -sin(angle);
  warp(0, 2) = 100.0f;
  warp(1, 0) = sin(angle);
  warp(1, 1) = cos(angle);
  warp(1, 2) = -output_size.height / 2.0f;

  cv::warpAffine(coords, coords, warp, coords.size());

  // This sampling coordinate will be clamped when fetching the pixel data for the next row, which
  // is out of bounds.
  coords(0, 0) = {float(input_size.height - 1), float(input_size.width - 1)};

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
      if (b_diff > 3 || g_diff > 3 || r_diff > 3) {
        std::cout << "pixels not equal at " << x << "x" << y << "\n";
        std::cout << int32_t(px1[0]) << " " << int32_t(px1[1]) << " " << int32_t(px1[2]) << "\n";
        std::cout << int32_t(px2[0]) << " " << int32_t(px2[1]) << " " << int32_t(px2[2]) << "\n";
        return false;
      }
    }
  }

  return true;
}
