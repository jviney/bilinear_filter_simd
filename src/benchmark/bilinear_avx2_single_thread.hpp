#pragma once

#include "common.hpp"
#include "interpolate/bilinear_avx2.hpp"

cv::Mat3b bilinear_avx2_single_thread(const BenchmarkInput& input) {
  auto output_image = cv::Mat3b(input.output_size);

  static constexpr auto step = 4;

  for (auto y = 0; y < output_image.rows; y++) {
    const auto* px_coords_row = input.coords.ptr<cv::Vec2f>(y);
    auto* output_pixels_row = output_image.ptr<cv::Vec3b>(y);

    for (auto x = 0; x < output_image.cols; x += step) {
      const auto* px_coords = px_coords_row + x;
      auto* output_pixels = output_pixels_row + x;

      interpolate::bilinear::avx2::interpolate(
          input.source_image, reinterpret_cast<const interpolate::InputCoords*>(px_coords),
          reinterpret_cast<interpolate::BGRPixel*>(output_pixels));
    }
  }

  return output_image;
}

static void BM_bilinear_avx2_single_thread(benchmark::State& state, const BenchmarkInput& input) {
  for (auto _ : state) {
    bilinear_avx2_single_thread(input);
  }
}
