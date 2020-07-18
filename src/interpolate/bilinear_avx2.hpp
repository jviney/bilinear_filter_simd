#pragma once

#include "common.hpp"
#include <immintrin.h>

namespace interpolate::bilinear::avx2
{

static const __m256 ONE = _mm256_set1_ps(1.0f);
static const __m256 TWO_FIFTY_SIX = _mm256_set1_ps(256.0f);

// Calculate the weights for the 4 surrounding pixels of two independent xy pairs.
// Returns weights as 16 bit ints.
// The upper lane contains the weights the the second xy pair, and the lower the first pair.
// The upper and lower half of each lane are identical.
// Eg: w4 w3 w2 w1 w4 w3 w2 w1 (second pair)  |  w4 w3 w2 w1 w4 w3 w2 w1 (first pair)
static inline __m256i calculate_weights(float x1, float y1, float x2, float y2) {
  __m256 initial = _mm256_set_ps(0, 0, y2, x2, 0, 0, y1, x1);

  __m256 floored = _mm256_floor_ps(initial);
  __m256 fractional = _mm256_sub_ps(initial, floored);
  __m256 one_minus_fractional = _mm256_sub_ps(ONE, fractional);

  //
  // x weights
  //

  // y (1-y) x (1-x)  |  y (1-y) x (1-x)
  __m256 x = _mm256_unpacklo_ps(one_minus_fractional, fractional);

  // x (1-x) x (1-x)  |  x (1-x) x (1-x)
  x = _mm256_shuffle_ps(x, x, 0x44);

  //
  // y weights
  //

  // y y (1-y) (1-y)  |  y y (1-y) (1-y)
  __m256 y = _mm256_shuffle_ps(one_minus_fractional, fractional, _MM_SHUFFLE(1, 1, 1, 1));

  // Multiply to get final weight
  __m256 weights = _mm256_mul_ps(x, y);

  // Convert to range 0-256
  weights = _mm256_mul_ps(weights, TWO_FIFTY_SIX);

  // Convert to 32 bit ints
  __m256i weights_i = _mm256_cvtps_epi32(weights);

  // Convert to 16 bit ints
  // 0 0 0 0 w4 w3 w2 w1  |  0 0 0 0 w4 w3 w2 w1
  weights_i = _mm256_packs_epi32(weights_i, _mm256_setzero_si256());

  // Copy lower half of each lane to the upper half
  // w4 w3 w2 w1 w4 w3 w2 w1  |  w4 w3 w2 w1 w4 w3 w2 w1
  weights_i = _mm256_unpacklo_epi64(weights_i, weights_i);

  return weights_i;
}

static inline void interpolate_two_pixels(__m256i p_bg, __m256i p_r0, __m256i weights,
                                          cv::Vec3b* output_pixels, bool can_write_third_pixel) {
  // Multiply with the pixel data and sum adjacent pairs to 32 bit ints
  // g g b b | g g b b
  __m256i r_bg = _mm256_madd_epi16(p_bg, weights);
  // _ _ r r | _ _ r r
  __m256i r_r0 = _mm256_madd_epi16(p_r0, weights);

  // Add adjacent pairs again. 32 bpc.
  // _ r g b  |  _ r g b
  __m256i out = _mm256_hadd_epi32(r_bg, r_r0);

  // Divide by 256 to get back into correct range.
  out = _mm256_srli_epi32(out, 8);

  // Convert from 32bpc => 16bpc => 8bpc
  out = _mm256_packus_epi32(out, _mm256_setzero_si256());
  out = _mm256_packus_epi16(out, _mm256_setzero_si256());

  // Store pixel data
  int32_t stored[8];
  _mm256_store_si256((__m256i*) stored, out);

  // Write pixel data back to image.
  // Faster to write 4 bytes instead of three when valid.
  // Always valid for first pixel, because we are about to overwrite the second pixel anyway.
  // Valid for second pixel only if we have been told so.
  *reinterpret_cast<int32_t*>(output_pixels) = stored[0];

  if (can_write_third_pixel) {
    *reinterpret_cast<int32_t*>(output_pixels + 1) = stored[4];
  } else {
    *(output_pixels + 1) = *reinterpret_cast<cv::Vec3b*>(&stored[4]);
  }
}

static constexpr uint8_t ZEROED = 128;

// Mask to shuffle the blue and green channels from packed 24bpp to 64bpp (16bpc) in each lane.
// Upper and lower lanes of input should contain independent sets of 4 pixels.
// Eg:
// (12 other bits) rgb rgb (12 other bits) rgb rgb  |  (12 other bits) rgb rgb (12 other bits) rgb
// rgb Becomes g g g g b b b b  |  g g g g b b b b
static const __m128i MASK_SHUFFLE_BG_HALF = _mm_set_epi8(
    // green
    ZEROED, 12, ZEROED, 9, ZEROED, 4, ZEROED, 1,
    // blue
    ZEROED, 11, ZEROED, 8, ZEROED, 3, ZEROED, 0);

static const __m256i MASK_SHUFFLE_BG = _mm256_set_m128i(MASK_SHUFFLE_BG_HALF, MASK_SHUFFLE_BG_HALF);

// Do the same with the red channel. The upper half of each lane is not used.
static const __m128i MASK_SHUFFLE_R0_HALF = _mm_set_epi8(
    // unused
    ZEROED, ZEROED, ZEROED, ZEROED, ZEROED, ZEROED, ZEROED, ZEROED,
    // red
    ZEROED, 13, ZEROED, 10, ZEROED, 5, ZEROED, 2);

static const __m256i MASK_SHUFFLE_R0 = _mm256_set_m128i(MASK_SHUFFLE_R0_HALF, MASK_SHUFFLE_R0_HALF);

// Bilinear interpolation of 2 adjacent output pixels with the supplied coordinates using AVX2.
static inline void interpolate(const cv::Mat3b& img, float x1, float y1, float x2, float y2,
                               cv::Vec3b* output_pixels, bool can_write_third_pixel = false) {
  const int stride = img.step / 3;

  const cv::Vec3b* p0_0 = img.ptr<cv::Vec3b>(y1, x1);
  const cv::Vec3b* p1_0 = img.ptr<cv::Vec3b>(y2, x2);

  __m256i pixels = _mm256_set_epi64x(*((int64_t*) &p1_0[stride]), *((int64_t*) &p1_0[0]),
                                     *((int64_t*) &p0_0[stride]), *((int64_t*) &p0_0[0]));

  __m256i pixels_bg = _mm256_shuffle_epi8(pixels, MASK_SHUFFLE_BG);
  __m256i pixels_r0 = _mm256_shuffle_epi8(pixels, MASK_SHUFFLE_R0);

  __m256i weights = calculate_weights(x1, y1, x2, y2);

  interpolate_two_pixels(pixels_bg, pixels_r0, weights, output_pixels, can_write_third_pixel);
}

}    // namespace interpolate::bilinear::avx2
