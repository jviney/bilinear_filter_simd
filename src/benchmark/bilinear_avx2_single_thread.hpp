#pragma once

#include "common.hpp"
#include "interpolate/bilinear_avx2.hpp"

cv::Mat3b bilinear_avx2_single_thread(const BenchmarkInput& input) {
  auto output_image = cv::Mat3b(input.output_size);

  for (auto y = 0; y < output_image.rows; y++) {
    for (auto x = 0; x < output_image.cols; x += 2) {
      auto* pixels = output_image.ptr<cv::Vec3b>(y, x);
      auto* px_coords = input.coords.ptr<cv::Vec2f>(y, x);
      interpolate::bilinear::avx2::interpolate(input.source_image, px_coords[0][1], px_coords[0][0],
                                               px_coords[1][1], px_coords[1][0], pixels);
    }
  }

  return output_image;
}

static void BM_bilinear_avx2_single_thread(benchmark::State& state, const BenchmarkInput& input) {
  for (auto _ : state) {
    bilinear_avx2_single_thread(input);
  }
}
