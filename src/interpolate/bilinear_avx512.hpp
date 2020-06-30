#pragma once

#include "common.hpp"
#include <immintrin.h>

namespace interpolate::bilinear::avx512
{

static const __m512 ONE = _mm512_set1_ps(1.0f);
static const __m512 TWO_FIFTY_SIX = _mm512_set1_ps(256.0f);

// Calculate the weights for the 4 surrounding pixels of two independent xy pairs.
// Returns weights as 16 bit ints.
// The upper lane contains the weights the the second xy pair, and the lower the first pair.
// The upper and lower half of each lane are identical.
// Eg: w4 w3 w2 w1 w4 w3 w2 w1 (second pair)  |  w4 w3 w2 w1 w4 w3 w2 w1 (first pair)
static inline __m512i calculate_weights(float x1, float y1, float x2, float y2, float x3, float y3,
                                        float x4, float y4) {
  __m512 initial = _mm512_set_ps(0, 0, y4, x4, 0, 0, y3, x3, 0, 0, y2, x2, 0, 0, y1, x1);

  __m512 floored = _mm512_floor_ps(initial);
  __m512 fractional = _mm512_sub_ps(initial, floored);
  __m512 one_minus_fractional = _mm512_sub_ps(ONE, fractional);

  //
  // x weights
  //

  // Each 128 bit lane: y (1-y) x (1-x)
  __m512 x = _mm512_unpacklo_ps(one_minus_fractional, fractional);

  // Each 128 bit lane: x (1-x) x (1-x)
  x = _mm512_shuffle_ps(x, x, 0x44);

  //
  // y weights
  //

  // Each 128 bit lane: y y (1-y) (1-y)
  __m512 y = _mm512_shuffle_ps(one_minus_fractional, fractional, _MM_SHUFFLE(1, 1, 1, 1));

  // Multiply to get final weight
  __m512 weights = _mm512_mul_ps(x, y);

  // Convert to range 0-256
  weights = _mm512_mul_ps(weights, TWO_FIFTY_SIX);

  // Convert to 32 bit ints
  __m512i weights_i = _mm512_cvtps_epi32(weights);

  // Convert to 16 bit ints
  // Each 128 bit lane: 0 0 0 0 w4 w3 w2 w1
  weights_i = _mm512_packs_epi32(weights_i, _mm512_setzero_si512());

  // Copy lower half of each lane to the upper half
  // Each 128 bit lane: w4 w3 w2 w1 w4 w3 w2 w1
  weights_i = _mm512_unpacklo_epi64(weights_i, weights_i);

  return weights_i;
}

static inline void interpolate_four_pixels(__m512i p_bg, __m512i p_r0, __m512i weights,
                                           cv::Vec3b* img_data) {
  // Multiply with the pixel data and sum adjacent pairs to 32 bit ints
  // Each 128 bit lane: g g b b
  __m512i r_bg = _mm512_madd_epi16(p_bg, weights);
  // Each 128 bit lane: _ _ r r
  __m512i r_r0 = _mm512_madd_epi16(p_r0, weights);

  // Add adjacent pairs again. 32 bpc.
  // Each 128 bit lane: _ r g b
  __m512i out = _mm512_hadd_epi32(r_bg, r_r0);

  // Divide by 256 to get back into correct range.
  out = _mm512_srli_epi32(out, 8);

  // Convert from 32bpc => 16bpc => 8bpc
  out = _mm512_packus_epi32(out, _mm512_setzero_si512());
  out = _mm512_packus_epi16(out, _mm512_setzero_si512());

  // Store pixel data
  int32_t stored[16];
  _mm512_store_si512((__m512i*) stored, out);

  // Write pixel data back to image
  int32_t pixel_1 = stored[0];
  memcpy((void*) img_data, &pixel_1, 4);

  int32_t pixel_2 = stored[4];
  memcpy((void*) img_data, &pixel_2, 4);

  int32_t pixel_3 = stored[8];
  memcpy((void*) img_data, &pixel_3, 4);

  int32_t pixel_4 = stored[12];
  *(img_data + 3) = {uint8_t(pixel_4), uint8_t(pixel_4 >> 8), uint8_t(pixel_4 >> 16)};
}

static constexpr uint8_t ZEROED = 128;

// Mask to shuffle the blue and green channels from packed 24bpp to 64bpp (16bpc) in each lane.
// Upper and lower lanes of input should contain independent sets of 4 pixels.
// Eg:
// (12 other bits) rgb rgb (12 other bits) rgb rgb  |  (12 other bits) rgb rgb (12 other bits) rgb
// rgb Becomes g g g g b b b b  |  g g g g b b b b
#define MASK_SHUFFLE_BG_SINGLE \
  ZEROED, 12, ZEROED, 9, ZEROED, 4, ZEROED, 1, ZEROED, 11, ZEROED, 8, ZEROED, 3, ZEROED, 0

static const __m512i MASK_SHUFFLE_BG = _mm512_set_epi8(
    MASK_SHUFFLE_BG_SINGLE, MASK_SHUFFLE_BG_SINGLE, MASK_SHUFFLE_BG_SINGLE, MASK_SHUFFLE_BG_SINGLE);

// Do the same with the red channel. The upper half of each lane is not used.
#define MASK_SHUFFLE_R0_SINGLE                                                                    \
  ZEROED, ZEROED, ZEROED, ZEROED, ZEROED, ZEROED, ZEROED, ZEROED, ZEROED, 13, ZEROED, 10, ZEROED, \
      5, ZEROED, 2

static const __m512i MASK_SHUFFLE_R0 = _mm512_set_epi8(
    MASK_SHUFFLE_R0_SINGLE, MASK_SHUFFLE_R0_SINGLE, MASK_SHUFFLE_R0_SINGLE, MASK_SHUFFLE_R0_SINGLE);

// Bilinear interpolation of 4 adjacent output pixels with the supplied coordinates using AVX512.
static inline void interpolate(const cv::Mat3b& img, float x1, float y1, float x2, float y2,
                               float x3, float y3, float x4, float y4, cv::Vec3b* out) {
  const int stride = img.step / 3;

  const cv::Vec3b* p0_0 = img.ptr<cv::Vec3b>(y1, x1);
  const cv::Vec3b* p1_0 = img.ptr<cv::Vec3b>(y2, x2);
  const cv::Vec3b* p2_0 = img.ptr<cv::Vec3b>(y3, x3);
  const cv::Vec3b* p3_0 = img.ptr<cv::Vec3b>(y4, x4);

  __m512i pixels = _mm512_set_epi64(*((int64_t*) &p3_0[stride]), *((int64_t*) &p3_0[0]),
                                    *((int64_t*) &p2_0[stride]), *((int64_t*) &p2_0[0]),
                                    *((int64_t*) &p1_0[stride]), *((int64_t*) &p1_0[0]),
                                    *((int64_t*) &p0_0[stride]), *((int64_t*) &p0_0[0]));

  __m512i pixels_bg = _mm512_shuffle_epi8(pixels, MASK_SHUFFLE_BG);
  __m512i pixels_r0 = _mm512_shuffle_epi8(pixels, MASK_SHUFFLE_R0);

  __m512i weights = calculate_weights(x1, y1, x2, y2);

  interpolate_four_pixels(pixels_bg, pixels_r0, weights, out);
}

}    // namespace interpolate::bilinear::avx512
