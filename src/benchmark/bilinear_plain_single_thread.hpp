#pragma once

#include "common.hpp"
#include "interpolate/bilinear_plain.hpp"

cv::Mat3b bilinear_plain_single_thread(const BenchmarkInput& input) {
  auto output_image = cv::Mat3b(input.output_size);

  for (auto y = 0; y < output_image.rows; y++) {
    for (auto x = 0; x < output_image.cols; x++) {
      const auto& px_coords = input.coords(y, x);
      output_image(y, x) =
          interpolate::bilinear::plain::interpolate(input.source_image, px_coords[1], px_coords[0]);
    }
  }

  return output_image;
}

static void BM_bilinear_plain_single_thread(benchmark::State& state, const BenchmarkInput& input) {
  for (auto _ : state) {
    bilinear_plain_single_thread(input);
  }
}
