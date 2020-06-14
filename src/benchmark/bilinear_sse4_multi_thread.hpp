#pragma once

#include "common.hpp"
#include "interpolate/bilinear_sse4.hpp"

class InterpolateSSE4MultiThread : public cv::ParallelLoopBody
{
public:
  InterpolateSSE4MultiThread(const cv::Mat3b& input_image, const cv::Mat2f& coords,
                             cv::Mat3b& output_image)
      : input_image_(input_image), coords_(coords), output_image_(output_image) {}

  virtual void operator()(const cv::Range& range) const override {
    for (auto y = range.start; y < range.end; y++) {
      for (auto x = 0; x < output_image_.cols; x++) {
        auto* output = output_image_.ptr<cv::Vec3b>(y, x);
        auto& px_coords = coords_(y, x);

        *output =
            interpolate::bilinear::sse4::interpolate(input_image_, px_coords[1], px_coords[0]);
      }
    }
  }

private:
  const cv::Mat3b input_image_;
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
