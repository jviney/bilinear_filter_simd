#pragma once

#include "common.hpp"
#include "interpolate/bilinear_plain.hpp"

cv::Mat3b bilinear_plain_single_thread(const BenchmarkInput& input) {
  auto output_image = cv::Mat3b(input.output_size);

  for (auto y = 0; y < output_image.rows; y++) {
    const auto* px_coords_row = input.coords.ptr<cv::Vec2f>(y);
    auto* output_px_row = output_image.ptr<cv::Vec3b>(y);

    for (int x = 0; x < output_image.cols; x += 4) {
      const auto* px_coords = reinterpret_cast<const interpolate::InputCoords*>(px_coords_row + x);
      auto* output_pixels = reinterpret_cast<interpolate::BGRPixel*>(output_px_row + x);

      interpolate::bilinear::plain::interpolate_multiple<4>(input.source_image, output_pixels,
                                                            px_coords);
    }
  }

  return output_image;
}

static void BM_bilinear_plain_single_thread(benchmark::State& state, const BenchmarkInput& input) {
  for (auto _ : state) {
    bilinear_plain_single_thread(input);
  }
}
