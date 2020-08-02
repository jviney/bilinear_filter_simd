#pragma once

#include "common.hpp"
#include <immintrin.h>
#include "interpolate/types.hpp"

namespace interpolate::bilinear::sse4
{

// SSE bilinear interpolation implementation based on
// http://fastcpp.blogspot.com/2011/06/bilinear-pixel-interpolation-using-sse.html
// and altered to work with OpenCV types and BGR24 image format.

static const __m128 ONE = _mm_set1_ps(1.0f);
static const __m128 TWO_FIFTY_SIX = _mm_set1_ps(256.0f);

// Returns the weightings of the four neighbouring pixels
static inline __m128i calc_weights(float x, float y) {
  // _ _ y x
  const __m128 initial = _mm_set_ps(0, 0, y, x);

  const __m128 floored = _mm_floor_ps(initial);
  const __m128 fractional = _mm_sub_ps(initial, floored);
  const __m128 one_minus_fractional = _mm_sub_ps(ONE, fractional);

  // (1-y) y (1-x) x
  const __m128 combined = _mm_unpacklo_ps(fractional, one_minus_fractional);

  // x (1-x) x (1-x)
  const __m128 weights_x = _mm_shuffle_ps(combined, combined, _MM_SHUFFLE(0, 1, 0, 1));

  // y y (1-y) (1-y)
  const __m128 weights_y = _mm_shuffle_ps(combined, combined, _MM_SHUFFLE(2, 2, 3, 3));

  // Multiply to get per pixel weightings
  __m128 weights = _mm_mul_ps(weights_x, weights_y);

  // * 256
  weights = _mm_mul_ps(weights, TWO_FIFTY_SIX);

  // Convert weights to 16 bit ints
  const __m128i weights_i = _mm_packs_epi32(_mm_cvtps_epi32(weights), _mm_setzero_si128());

  return weights_i;
}

static constexpr uint8_t ZEROED = 128;

static inline void interpolate(const cv::Mat3b& img, const interpolate::InputCoords& input_coords,
                               interpolate::BGRPixel* output_pixel, bool can_write_next_pixel) {

  const int stride = img.step / 3;
  const cv::Vec3b* p0 = img.ptr<cv::Vec3b>(input_coords.y, input_coords.x);

  // We are only using the lower 48 bits of each load.

  // Load 4 pixels for interpolation.
  // Shuffle 24bpp around to use 64bpp with to 16 bpc
  // _ r g b _ r g b
  __m128i p12 = _mm_shuffle_epi8(_mm_loadl_epi64((const __m128i*) p0),
                                 _mm_set_epi8(ZEROED, ZEROED, ZEROED, 5, ZEROED, 4, ZEROED, 3,
                                              ZEROED, ZEROED, ZEROED, 2, ZEROED, 1, ZEROED, 0));

  // _ r g b _ r g b
  __m128i p34 = _mm_shuffle_epi8(_mm_loadl_epi64((const __m128i*) (p0 + stride)),
                                 _mm_set_epi8(ZEROED, ZEROED, ZEROED, 5, ZEROED, 4, ZEROED, 3,
                                              ZEROED, ZEROED, ZEROED, 2, ZEROED, 1, ZEROED, 0));

  __m128i weights = calc_weights(input_coords.x, input_coords.y);

  // Prepare weights
  __m128i w12 = _mm_shufflelo_epi16(weights, _MM_SHUFFLE(1, 1, 0, 0));
  __m128i w34 = _mm_shufflelo_epi16(weights, _MM_SHUFFLE(3, 3, 2, 2));
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

  // Faster to write 4 bytes instead of 3 when allowed
  if (can_write_next_pixel) {
    memcpy(output_pixel, &all_chans, 4);
  } else {
    memcpy(output_pixel, &all_chans, sizeof(interpolate::BGRPixel));
  }
}

}    // namespace interpolate::bilinear::sse4
