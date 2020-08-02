#pragma once

#include "common.hpp"
#include "interpolate/bilinear_plain.hpp"

class InterpolatePlainMultiThread : public cv::ParallelLoopBody
{
public:
  InterpolatePlainMultiThread(const cv::Mat3b& input_image, const cv::Mat2f& coords,
                              cv::Mat3b& output_image)
      : input_image_(input_image), coords_(coords), output_image_(output_image) {}

  virtual void operator()(const cv::Range& range) const override {
    for (auto y = range.start; y < range.end; y++) {
      const auto* px_coords_row = coords_.ptr<cv::Vec2f>(y);
      auto* output_row = output_image_.ptr<cv::Vec3b>(y);

      for (auto x = 0; x < output_image_.cols; x += 4) {
        const auto* px_coords = px_coords_row + x;
        auto* output_pixel = output_row + x;

        interpolate::bilinear::plain::interpolate_multiple<4>(
            input_image_, reinterpret_cast<interpolate::BGRPixel*>(output_pixel),
            reinterpret_cast<const interpolate::InputCoords*>(px_coords));
      }
    }
  }

private:
  const cv::Mat3b input_image_;
  const cv::Mat2f coords_;
  cv::Mat3b& output_image_;
};

cv::Mat3b bilinear_plain_multi_thread(const BenchmarkInput& input) {
  auto output_image = cv::Mat3b(input.output_size);
  auto parallel_executor =
      InterpolatePlainMultiThread(input.source_image, input.coords, output_image);

  cv::parallel_for_(cv::Range(0, output_image.rows), parallel_executor);

  return output_image;
}

static void BM_bilinear_plain_multi_thread(benchmark::State& state, const BenchmarkInput& input) {
  for (auto _ : state) {
    bilinear_plain_multi_thread(input);
  }
}
