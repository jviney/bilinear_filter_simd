#pragma once

#include "common.hpp"
#include "interpolate/bilinear_sse4.hpp"

class InterpolateSSE4MultiThread : public cv::ParallelLoopBody
{
public:
  InterpolateSSE4MultiThread(const interpolate::BGRImage& input_image, const cv::Mat2f& coords,
                             cv::Mat3b& output_image)
      : input_image_(input_image), coords_(coords), output_image_(output_image) {}

  virtual void operator()(const cv::Range& range) const override {
    auto* last_output_pixel = output_image_.ptr<cv::Vec3b>(range.end - 1, output_image_.cols - 1);

    for (auto y = range.start; y < range.end; y++) {
      const auto* px_coords_row = coords_.ptr<cv::Vec2f>(y);
      auto* output_px_row = output_image_.ptr<cv::Vec3b>(y);

      for (auto x = 0; x < output_image_.cols; x += 2) {
        const auto* px_coords = px_coords_row + x;
        auto* output_pixels = output_px_row + x;

        auto is_last_output_pixel = (output_pixels + 1 == last_output_pixel);

        interpolate::bilinear::sse4::interpolate(
            input_image_, reinterpret_cast<const interpolate::InputCoords*>(px_coords),
            reinterpret_cast<interpolate::BGRPixel*>(output_pixels), !is_last_output_pixel);
      }
    }
  }

private:
  const interpolate::BGRImage& input_image_;
  const cv::Mat2f coords_;
  cv::Mat3b& output_image_;
};

cv::Mat3b bilinear_sse4_multi_thread(const BenchmarkInput& input) {
  auto output_image = cv::Mat3b(input.output_size);
  auto parallel_executor =
      InterpolateSSE4MultiThread(input.source_image, input.coords, output_image);

  cv::parallel_for_(cv::Range(0, output_image.rows), parallel_executor);

  return output_image;
}

static void BM_bilinear_sse4_multi_thread(benchmark::State& state, const BenchmarkInput& input) {
  for (auto _ : state) {
    bilinear_sse4_multi_thread(input);
  }
}
