#pragma once

#include "common.hpp"
#include "interpolate/bilinear_sse4.hpp"

static cv::Mat3b bilinear_sse4_single_thread(const BenchmarkInput& input) {
  auto output_image = cv::Mat3b(input.output_size);

  auto* last_output_pixel =
      output_image.ptr<cv::Vec3b>(output_image.rows - 1, output_image.cols - 1);

  for (auto y = 0; y < output_image.rows; y++) {
    auto* output_row_start = output_image.ptr<cv::Vec3b>(y);

    for (auto x = 0; x < output_image.cols; x++) {
      const auto& px_coords = input.coords(y, x);
      auto* output_pixel = output_row_start + x;

      auto is_last_output_pixel = (output_pixel == last_output_pixel);

      interpolate::bilinear::sse4::interpolate(input.source_image, px_coords[1], px_coords[0],
                                               output_pixel, !is_last_output_pixel);
    }
  }

  return output_image;
}

static void BM_bilinear_sse4_single_thread(benchmark::State& state, const BenchmarkInput& input) {
  for (auto _ : state) {
    bilinear_sse4_single_thread(input);
  }
}
