#pragma once

#include "common.hpp"
#include "interpolate/bilinear_avx512.hpp"

class InterpolateAVX512MultiThread : public cv::ParallelLoopBody
{
public:
  InterpolateAVX512MultiThread(const interpolate::BGRImage& input_image, const cv::Mat2f& coords,
                               cv::Mat3b& output_image)
      : input_image_(input_image), coords_(coords), output_image_(output_image) {
    if (output_image.cols % step != 0) {
      throw std::runtime_error("output frame width must be multiple of 4");
    }
  }

  virtual void operator()(const cv::Range& range) const override {
    for (auto y = range.start; y < range.end; y++) {
      auto* px_coords_row = coords_.ptr<cv::Vec2f>(y);
      auto* output_pixels_row = output_image_.ptr<cv::Vec3b>(y);

      for (auto x = 0; x < output_image_.cols; x += step) {
        auto px_coords = px_coords_row + x;
        auto* output_pixels = output_pixels_row + x;

        interpolate::bilinear::avx512::interpolate(
            input_image_, reinterpret_cast<const interpolate::InputCoords*>(px_coords),
            reinterpret_cast<interpolate::BGRPixel*>(output_pixels));
      }
    }
  }

private:
  static constexpr auto step = 8;

  const interpolate::BGRImage input_image_;
  const cv::Mat2f coords_;
  cv::Mat3b& output_image_;
};

cv::Mat3b bilinear_avx512_multi_thread(const BenchmarkInput& input) {
  auto output_image = cv::Mat3b(input.output_size);
  auto parallel_executor =
      InterpolateAVX512MultiThread(input.source_image, input.coords, output_image);

  cv::parallel_for_(cv::Range(0, output_image.rows), parallel_executor);

  return output_image;
}

static void BM_bilinear_avx512_multi_thread(benchmark::State& state, const BenchmarkInput& input) {
  for (auto _ : state) {
    bilinear_avx512_multi_thread(input);
  }
}
