#pragma once

#include "interpolate/types.hpp"

namespace interpolate::bilinear::plain
{

static inline interpolate::BGRPixel interpolate(const interpolate::BGRImage& image,
                                                const interpolate::InputCoords& input_coords) {
  auto px = int(input_coords.x);    // floor x
  auto py = int(input_coords.y);    // floor y

  // Four neighbouring pixels
  const auto* pixel = image.ptr(py, px);

  const auto& p1 = pixel[0];
  const auto& p2 = pixel[1];
  const auto& p3 = pixel[image.stride];
  const auto& p4 = pixel[image.stride + 1];

  // Calculate the weights for each pixel
  float fx = input_coords.x - px;
  float fy = input_coords.y - py;
  float fx1 = 1.0f - fx;
  float fy1 = 1.0f - fy;

  // Using int for the weights is a bit faster than using floats
  int w1 = fx1 * fy1 * 256.0f;
  int w2 = fx * fy1 * 256.0f;
  int w3 = fx1 * fy * 256.0f;
  int w4 = fx * fy * 256.0f;

  // Calculate the weighted sum of pixels (for each color channel)
  int outr = p1.r * w1 + p2.r * w2 + p3.r * w3 + p4.r * w4;
  int outg = p1.g * w1 + p2.g * w2 + p3.g * w3 + p4.g * w4;
  int outb = p1.b * w1 + p2.b * w2 + p3.b * w3 + p4.b * w4;

  return {uint8_t(outb >> 8), uint8_t(outg >> 8), uint8_t(outr >> 8)};
}

template <int N>
static inline void interpolate_multiple(const interpolate::BGRImage& image,
                                        interpolate::BGRPixel* output,
                                        const interpolate::InputCoords* input_coords) {
  interpolate::BGRPixel output_pixels[N];

  for (int i = 0; i < N; i++) {
    auto px = int(input_coords[i].x);    // floor x
    auto py = int(input_coords[i].y);    // floor y

    // Four neighbouring pixels
    const auto* pixel = image.ptr(py, px);
    const auto& p1 = pixel[0];
    const auto& p2 = pixel[1];
    const auto& p3 = pixel[image.stride];
    const auto& p4 = pixel[image.stride + 1];

    // Calculate the weights for each pixel
    float fx = input_coords[i].x - px;
    float fy = input_coords[i].y - py;
    float fx1 = 1.0f - fx;
    float fy1 = 1.0f - fy;

    // Using int for the weights is a bit faster than using floats
    int w1 = fx1 * fy1 * 256.0f;
    int w2 = fx * fy1 * 256.0f;
    int w3 = fx1 * fy * 256.0f;
    int w4 = fx * fy * 256.0f;

    // Calculate the weighted sum of pixels (for each color channel)
    int outr = p1.r * w1 + p2.r * w2 + p3.r * w3 + p4.r * w4;
    int outg = p1.g * w1 + p2.g * w2 + p3.g * w3 + p4.g * w4;
    int outb = p1.b * w1 + p2.b * w2 + p3.b * w3 + p4.b * w4;

    output_pixels[i] = {uint8_t(outb >> 8), uint8_t(outg >> 8), uint8_t(outr >> 8)};
  }

  memcpy(output, output_pixels, sizeof(interpolate::BGRPixel) * N);
}

}    // namespace interpolate::bilinear::plain
