#pragma once

#include "common.hpp"
#include <immintrin.h>

namespace interpolate::bilinear::sse4
{

// SSE bilinear interpolation implementation based on
// http://fastcpp.blogspot.com/2011/06/bilinear-pixel-interpolation-using-sse.html
// and altered to work with OpenCV types and BGR24 image format.

static const __m128 ONE = _mm_set1_ps(1.0f);
static const __m128 TWO_FIFTY_SIX = _mm_set1_ps(256.0f);

// Returns the weightings of the four neighbouring pixels
static inline __m128 calc_weights(float x, float y) {
  // 0 0 y x
  __m128 initial = _mm_unpacklo_ps(_mm_set_ss(x), _mm_set_ss(y));

  __m128 floored = _mm_floor_ps(initial);
  __m128 fractional = _mm_sub_ps(initial, floored);
  __m128 one_minus_fractional = _mm_sub_ps(ONE, fractional);

  // y (1-y) x (1-x)
  __m128 weights_x = _mm_unpacklo_ps(one_minus_fractional, fractional);

  // x (1-x) x (1-x)
  weights_x = _mm_movelh_ps(weights_x, weights_x);

  // y y (1-y) (1-y)
  __m128 weights_y = _mm_shuffle_ps(one_minus_fractional, fractional, _MM_SHUFFLE(1, 1, 1, 1));

  // Multiply to get per pixel weightings
  __m128 weights = _mm_mul_ps(weights_x, weights_y);

  return weights;
}

static inline cv::Vec3b interpolate(const cv::Mat3b& img, float x, float y) {
  const int stride = img.step / 3;
  const cv::Vec3b* p0 = img.ptr<cv::Vec3b>(y, x);

  // Load 4 pixels for interpolation
  __m128i p1 = _mm_loadl_epi64((const __m128i*) &p0[0]);
  __m128i p2 = _mm_loadl_epi64((const __m128i*) &p0[1]);
  __m128i p3 = _mm_loadl_epi64((const __m128i*) &p0[stride]);
  __m128i p4 = _mm_loadl_epi64((const __m128i*) &p0[stride + 1]);

  // Combine pixels 1 and 2, and 3 and 4
  // _ _ p2 p1
  __m128i p12 = _mm_unpacklo_epi32(p1, p2);
  // _ _ p4 p3
  __m128i p34 = _mm_unpacklo_epi32(p3, p4);

  // Convert to 16 bpc
  // _ r g b _ r g b
  p12 = _mm_unpacklo_epi8(p12, _mm_setzero_si128());
  p34 = _mm_unpacklo_epi8(p34, _mm_setzero_si128());

  __m128 weights = calc_weights(x, y);

  // Convert weights to range 0-256
  weights = _mm_mul_ps(weights, TWO_FIFTY_SIX);

  // Convert weights to 16 bit ints
  __m128i weight_i = _mm_packs_epi32(_mm_cvtps_epi32(weights), _mm_setzero_si128());

  // Prepare weights
  __m128i w12 = _mm_shufflelo_epi16(weight_i, _MM_SHUFFLE(1, 1, 0, 0));
  __m128i w34 = _mm_shufflelo_epi16(weight_i, _MM_SHUFFLE(3, 3, 2, 2));
  // w2 w2 w2 w2 w1 w1 w1 w1
  w12 = _mm_unpacklo_epi16(w12, w12);
  // w4 w4 w4 w4 w3 w3 w3 w3
  w34 = _mm_unpacklo_epi16(w34, w34);

  // Multiply each pixel with its weight
  __m128i out_12 = _mm_mullo_epi16(p12, w12);
  __m128i out_34 = _mm_mullo_epi16(p34, w34);

  // Sum the results
  __m128i out_1234 = _mm_add_epi16(out_12, out_34);
  __m128i out_high = _mm_shuffle_epi32(out_1234, _MM_SHUFFLE(3, 2, 3, 2));
  __m128i out = _mm_add_epi16(out_1234, out_high);

  // Divide by 256
  out = _mm_srli_epi16(out, 8);

  // Convert to 8 bpc
  out = _mm_packus_epi16(out, _mm_setzero_si128());

  // Extract the channels to create a cv::Vec3b
  int all_chans = _mm_cvtsi128_si32(out);
  uint8_t chan1 = all_chans >> 0;
  uint8_t chan2 = all_chans >> 8;
  uint8_t chan3 = all_chans >> 16;
  return {chan1, chan2, chan3};
}

}    // namespace interpolate::bilinear::sse4
