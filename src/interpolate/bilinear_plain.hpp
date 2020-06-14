#pragma once

#include "common.hpp"

namespace interpolate::bilinear::plain
{

static inline cv::Vec3b interpolate(const cv::Mat3b& image, float x, float y) {
  auto px = int(x);    // floor x
  auto py = int(y);    // floor y

  // Four neighbouring pixels
  const int stride = image.step / 3;
  const cv::Vec3b* pixel = image.ptr<cv::Vec3b>(py, px);

  const cv::Vec3b& p1 = pixel[0];
  const cv::Vec3b& p2 = pixel[1];
  const cv::Vec3b& p3 = pixel[stride];
  const cv::Vec3b& p4 = pixel[stride + 1];

  // Calculate the weights for each pixel
  float fx = x - px;
  float fy = y - py;
  float fx1 = 1.0f - fx;
  float fy1 = 1.0f - fy;

  // Using int for the weights is a bit faster than using floats
  int w1 = fx1 * fy1 * 256.0f;
  int w2 = fx * fy1 * 256.0f;
  int w3 = fx1 * fy * 256.0f;
  int w4 = fx * fy * 256.0f;

  // Calculate the weighted sum of pixels (for each color channel)
  int outr = p1[0] * w1 + p2[0] * w2 + p3[0] * w3 + p4[0] * w4;
  int outg = p1[1] * w1 + p2[1] * w2 + p3[1] * w3 + p4[1] * w4;
  int outb = p1[2] * w1 + p2[2] * w2 + p3[2] * w3 + p4[2] * w4;

  return {uint8_t(outr >> 8), uint8_t(outg >> 8), uint8_t(outb >> 8)};
}

}    // namespace interpolate::bilinear::plain
