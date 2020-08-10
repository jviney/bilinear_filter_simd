#pragma once

#include "interpolate/types.hpp"
#include <immintrin.h>

namespace interpolate::bilinear::avx2
{

static const __m256i WEIGHTS_Y_SHUFFLE =
    _mm256_set_epi8(11, 10, 11, 10, 9, 8, 9, 8, 3, 2, 3, 2, 1, 0, 1, 0,
                    // Repeated
                    11, 10, 11, 10, 9, 8, 9, 8, 3, 2, 3, 2, 1, 0, 1, 0);

// Calculate the weights for the 4 surrounding pixels of 4 independent xy pairs.
// Returns weights as 16 bit ints.
// Eg: w4 w3 w2 w1 (x4/y4)   w4 w3 w2 w1 (x3/y3)   |  w4 w3 w2 w1 (x2/y2)  w4 w3 w2 w1 (x1/y1)
static inline __m256i calculate_weights(const float sample_coords[8]) {
  const __m256 initial =
      _mm256_castsi256_ps(_mm256_stream_load_si256((const __m256i*) sample_coords));

  const __m256 floored = _mm256_floor_ps(initial);
  const __m256 fractional = _mm256_sub_ps(initial, floored);

  // Convert fractional parts to 32 bit ints in range 0-256
  // x4 y4 x3 y3  |  x2 y2 x1 y1
  __m256i lower = _mm256_cvtps_epi32(_mm256_mul_ps(fractional, _mm256_set1_ps(256.0f)));

  // Convert to 16 bit ints
  // 0 0 0 0 x4 y4 x3 y3  |  0 0 0 0 x2 y2 x1 y1
  lower = _mm256_packs_epi32(lower, _mm256_set1_epi32(0));

  // Get the 1-fractional from the 16 bit result
  // 256 256 256 256 1-x4 1-y4 1-x3 1-y3  |  256 256 256 256 1-x2 1-y2 1-x1 1-y1
  const __m256i upper = _mm256_sub_epi16(_mm256_set1_epi16(256), lower);

  // ...y4 ...y3  |  ...y2  1-x1 x1  1-y1 y1
  const __m256i combined = _mm256_unpacklo_epi16(upper, lower);

  // ...(1-x4)  ...(1-x3)  |  ...(1-x2)  x1 (1-x1) x1 (1-x1)
  const __m256i weights_x = _mm256_shuffle_epi32(combined, _MM_SHUFFLE(3, 3, 1, 1));

  // ...(1-y4)  ...(1-y3)  |  ...(1-y2)  y1 y1 (1-y1) (1-y1)
  // Shuffle 16 bit numbers as 8 bits because there is no _mm256_shuffle_epi16
  const __m256i weights_y = _mm256_shuffle_epi8(combined, WEIGHTS_Y_SHUFFLE);

  // Multiply to get final per pixel weights. Divide by 256 to get back into correct range.
  // ...(x4/y4)  ... (x3/y3)  |  ... (x2/y2)  w4 w3 w2 w1 (x1/y1)
  __m256i weights = _mm256_srli_epi16(_mm256_mullo_epi16(weights_x, weights_y), 8);

  // If both weights were 256, the result is 65536 which is all 0s in the lower 16 bits.
  // Find the weights this happened to, and replace them with 256.
  const __m256i weights_hi = _mm256_mulhi_epi16(weights_x, weights_y);
  const __m256i weights_hi_mask = _mm256_cmpgt_epi16(weights_hi, _mm256_setzero_si256());
  weights = _mm256_blendv_epi8(weights, _mm256_set1_epi16(256), weights_hi_mask);

  return weights;
}

static constexpr uint8_t Z = 128;

// Mask to shuffle the blue and green channels from packed 24bpp to 64bpp (16bpc) in each lane.
// Upper and lower lanes of input should contain independent sets of 4 pixels.
// Eg:
// (12 other bits) rgb rgb (12 other bits) rgb rgb  |  (12 other bits) rgb rgb (12 other bits) rgb
// rgb Becomes g g g g b b b b  |  g g g g b b b b
static const __m128i MASK_SHUFFLE_BG_HALF = _mm_set_epi8(
    // green
    Z, 12, Z, 9, Z, 4, Z, 1,
    // blue
    Z, 11, Z, 8, Z, 3, Z, 0);

static const __m256i MASK_SHUFFLE_BG = _mm256_set_m128i(MASK_SHUFFLE_BG_HALF, MASK_SHUFFLE_BG_HALF);

// Do the same with the red channel. The upper half of each lane is not used.
static const __m128i MASK_SHUFFLE_R0_HALF = _mm_set_epi8(
    // unused
    Z, Z, Z, Z, Z, Z, Z, Z,
    // red
    Z, 13, Z, 10, Z, 5, Z, 2);

static const __m256i MASK_SHUFFLE_R0 = _mm256_set_m128i(MASK_SHUFFLE_R0_HALF, MASK_SHUFFLE_R0_HALF);

static inline __m256i interpolate_two_pixels(const interpolate::BGRImage& image,
                                             const interpolate::InputCoords input_coords[3],
                                             __m256i weights) {
  // Load pixel data
  const auto* p0_0 = image.ptr(input_coords[0].y, input_coords[0].x);
  const auto* p1_0 = image.ptr(input_coords[2].y, input_coords[2].x);

  const __m256i pixels = _mm256_set_epi64x(*((int64_t*) (p1_0 + image.stride)), *((int64_t*) p1_0),
                                           *((int64_t*) (p0_0 + image.stride)), *((int64_t*) p0_0));

  const __m256i pixels_bg = _mm256_shuffle_epi8(pixels, MASK_SHUFFLE_BG);
  const __m256i pixels_r0 = _mm256_shuffle_epi8(pixels, MASK_SHUFFLE_R0);

  // Multiply with the pixel data and sum adjacent pairs to 32 bit ints
  // g g b b | g g b b
  const __m256i result_bg = _mm256_madd_epi16(pixels_bg, weights);
  // _ _ r r | _ _ r r
  const __m256i result_r0 = _mm256_madd_epi16(pixels_r0, weights);

  // Add adjacent pairs again. 32 bpc.
  // _ r g b  |  _ r g b
  __m256i result = _mm256_hadd_epi32(result_bg, result_r0);

  // Divide by 256 to get back into correct range.
  result = _mm256_srli_epi32(result, 8);

  // Convert from 32bpc => 16bpc => 8bpc
  result = _mm256_packus_epi32(result, _mm256_setzero_si256());
  result = _mm256_packus_epi16(result, _mm256_setzero_si256());

  return result;
}

static inline void write_output_pixels(__m256i pixels_13, __m256i pixels_24,
                                       interpolate::BGRPixel output_pixels[4]) {
  // Unpack to get adjacent pixel data in lower 64 bits of each lane
  // _ _ 4 3  |  _ _ 2 1
  __m256i combined = _mm256_unpacklo_epi32(pixels_13, pixels_24);

  // Then pack everything into 1 lane
  // _ _ _ _  |  4 3 2 1
  combined = _mm256_permute4x64_epi64(combined, _MM_SHUFFLE(3, 3, 2, 0));

  // Shuffle around to get packed 24bpp at the bottom of the lower lane
  combined = _mm256_shuffle_epi8(combined,
                                 _mm256_set_epi8(
                                     // Top lane not used
                                     Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z,
                                     // Bottom lane - 32 unused bits at the top
                                     Z, Z, Z, Z,
                                     // Packed pixel data
                                     14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0));

  // Write out the lower 12 bytes
  uint8_t interpolated_pixels[32];
  _mm256_store_si256((__m256i*) interpolated_pixels, combined);
  memcpy(output_pixels, interpolated_pixels, 12);
}

// Bilinear interpolation of 4 adjacent output pixels with the supplied coordinates using AVX2.
static inline void interpolate(const interpolate::BGRImage& image,
                               const interpolate::InputCoords input_coords[4],
                               interpolate::BGRPixel output_pixels[4]) {
  // Calculate weights for 4 pixels
  const __m256i weights = calculate_weights(&input_coords[0].y);

  // Prepare weights for pixels 1 and 3, and interpolate
  const __m256i weights_13 = _mm256_unpacklo_epi64(weights, weights);
  const __m256i pixels_13 = interpolate_two_pixels(image, input_coords, weights_13);

  // Same for pixels 2 and 4
  const __m256i weights_24 = _mm256_unpackhi_epi64(weights, weights);
  const __m256i pixels_24 = interpolate_two_pixels(image, input_coords + 1, weights_24);

  write_output_pixels(pixels_13, pixels_24, output_pixels);
}

}    // namespace interpolate::bilinear::avx2
