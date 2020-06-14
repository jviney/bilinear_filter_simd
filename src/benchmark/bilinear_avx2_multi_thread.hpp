#pragma once

#include "common.hpp"
#include "interpolate/bilinear_avx2.hpp"

class InterpolateAVX2MultiThread : public cv::ParallelLoopBody
{
public:
  InterpolateAVX2MultiThread(const cv::Mat3b& input_image, const cv::Mat2f& coords,
                             cv::Mat3b& output_image)
      : input_image_(input_image), coords_(coords), output_image_(output_image) {
    if (output_image.cols % step != 0) {
      throw std::runtime_error("output frame width must be multiple of 2");
    }
  }

  virtual void operator()(const cv::Range& range) const override {
    for (auto y = range.start; y < range.end; y++) {
      for (auto x = 0; x < output_image_.cols; x += step) {
        auto* output = output_image_.ptr<cv::Vec3b>(y, x);
        auto* px_coords = coords_.ptr<cv::Vec2f>(y, x);

        interpolate::bilinear::avx2::interpolate(input_image_, px_coords[0][1], px_coords[0][0],
                                                 px_coords[1][1], px_coords[1][0], output);
      }
    }
  }

private:
  static constexpr auto step = 2;

  const cv::Mat3b input_image_;
  const cv::Mat2f coords_;
  cv::Mat3b& output_image_;
};

cv::Mat3b bilinear_avx2_multi_thread(const BenchmarkInput& input) {
  auto output_image = cv::Mat3b(input.output_size);
  auto parallel_executor =
      InterpolateAVX2MultiThread(input.source_image, input.coords, output_image);

  cv::parallel_for_(cv::Range(0, output_image.rows), parallel_executor);

  return output_image;
}

static void BM_bilinear_avx2_multi_thread(benchmark::State& state, const BenchmarkInput& input) {
  for (auto _ : state) {
    bilinear_avx2_multi_thread(input);
  }
}
