#pragma once

#include <immintrin.h>

#include "interpolate/types.hpp"

namespace interpolate::bilinear::avx512
{

//
// Weights
//

static const __m512i WEIGHTS_Y_SHUFFLE =
    _mm512_set_epi8(11, 10, 11, 10, 9, 8, 9, 8, 3, 2, 3, 2, 1, 0, 1, 0,
                    // Repeated
                    11, 10, 11, 10, 9, 8, 9, 8, 3, 2, 3, 2, 1, 0, 1, 0,
                    // Repeated
                    11, 10, 11, 10, 9, 8, 9, 8, 3, 2, 3, 2, 1, 0, 1, 0,
                    // Repeated
                    11, 10, 11, 10, 9, 8, 9, 8, 3, 2, 3, 2, 1, 0, 1, 0);

static inline __m512i calculate_weights(const float sample_coords[16]) {
  const __m512 initial = _mm512_load_ps(sample_coords);

  const __m512 floored = _mm512_floor_ps(initial);
  const __m512 fractional = _mm512_sub_ps(initial, floored);

  // Convert fractional parts to 32 bit ints in range 0-256
  // ... | x2 y2 x1 y1
  __m512i lower = _mm512_cvtps_epi32(_mm512_mul_ps(fractional, _mm512_set1_ps(256.0f)));

  // Convert to 16 bit ints
  // ... | 0 0 0 0 x2 y2 x1 y1
  lower = _mm512_packs_epi32(lower, _mm512_set1_epi32(0));

  // Subtract each value from 256
  // ... | 256 256 256 256 1-x2 1-y2 1-x1 1-y1
  const __m512i upper = _mm512_sub_epi16(_mm512_set1_epi16(256), lower);

  // Combine all the weights into a single vector
  // ... | 1-y2  1-x1 x1  1-y1 y1
  const __m512i combined = _mm512_unpacklo_epi16(upper, lower);

  // x weights
  // ... | ...(1-x2)  x1 (1-x1) x1 (1-x1)
  __m512i weights_x = _mm512_shuffle_epi32(combined, _MM_PERM_DDBB);

  // y weights
  // ... | ...(1-y2)  y1 y1 (1-y1) (1-y1)
  __m512i weights_y = _mm512_shuffle_epi8(combined, WEIGHTS_Y_SHUFFLE);

  // Multiply to get final per pixel weights. Divide by 256 to get back into correct range.
  // ... | ... (x2/y2)  w4 w3 w2 w1 (x1/y1)
  __m512i weights = _mm512_mullo_epi16(weights_x, weights_y);
  weights = _mm512_srli_epi16(weights, 8);

  // If both weights were 256, the result is 65536 which is all 0s in the lower 16 bits.
  // Find the weights this happened to, and replace them with 256.
  __m512i weights_hi = _mm512_mulhi_epi16(weights_x, weights_y);
  __mmask32 weights_hi_mask = _mm512_cmpgt_epi16_mask(weights_hi, _mm512_setzero_si512());
  weights = _mm512_mask_blend_epi16(weights_hi_mask, weights, _mm512_set1_epi16(256));

  return weights;
}

// Masks to shuffle initial pixel data from packed 24bpp to 64bpp (16bpc) in each lane.
// Eg, a 128 bit lane with the following data:
// (16 other bits) rgb rgb (16 other bits) rgb rgb
// Becomes:
// g g g g b b b b

// Blue and green channels.
#define MASK_SHUFFLE_BG_SINGLE_LANE -1, 12, -1, 9, -1, 4, -1, 1, -1, 11, -1, 8, -1, 3, -1, 0
static const __m512i MASK_SHUFFLE_BG =
    _mm512_set_epi8(MASK_SHUFFLE_BG_SINGLE_LANE, MASK_SHUFFLE_BG_SINGLE_LANE,
                    MASK_SHUFFLE_BG_SINGLE_LANE, MASK_SHUFFLE_BG_SINGLE_LANE);

// Red channel. The upper half of each 128 bit lane is not used.
#define MASK_SHUFFLE_R0_SINGLE_LANE -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, -1, 10, -1, 5, -1, 2
static const __m512i MASK_SHUFFLE_R0 =
    _mm512_set_epi8(MASK_SHUFFLE_R0_SINGLE_LANE, MASK_SHUFFLE_R0_SINGLE_LANE,
                    MASK_SHUFFLE_R0_SINGLE_LANE, MASK_SHUFFLE_R0_SINGLE_LANE);

//
// Interpolation
//

static inline __m512i interpolate_four_pixels(const interpolate::BGRImage& image,
                                              const interpolate::InputCoords input_coords[7],
                                              __m512i weights) {
  // Load pixel data
  const auto* p1 = image.ptr(input_coords[0].y, input_coords[0].x);
  const auto* p2 = image.ptr(input_coords[2].y, input_coords[2].x);
  const auto* p3 = image.ptr(input_coords[4].y, input_coords[4].x);
  const auto* p4 = image.ptr(input_coords[6].y, input_coords[6].x);

  __m512i pixels = _mm512_set_epi64(*((int64_t*) (p4 + image.stride)), *((int64_t*) p4),
                                    *((int64_t*) (p3 + image.stride)), *((int64_t*) p3),
                                    *((int64_t*) (p2 + image.stride)), *((int64_t*) p2),
                                    *((int64_t*) (p1 + image.stride)), *((int64_t*) p1));

  __m512i pixels_bg = _mm512_shuffle_epi8(pixels, MASK_SHUFFLE_BG);
  __m512i pixels_r0 = _mm512_shuffle_epi8(pixels, MASK_SHUFFLE_R0);

  // Multiply with the pixel data and sum adjacent pairs to 32 bit ints
  // ... | g g b b
  __m512i result_bg = _mm512_madd_epi16(pixels_bg, weights);
  // ... | _ g _ b
  result_bg = _mm512_add_epi32(result_bg, _mm512_srli_epi64(result_bg, 32));
  // ... | _ _ g b
  result_bg = _mm512_shuffle_epi32(result_bg, _MM_PERM_DDCA);

  // ... | _ _ r r
  __m512i result_r0 = _mm512_madd_epi16(pixels_r0, weights);
  // ... | _ _ _ r
  result_r0 = _mm512_add_epi32(result_r0, _mm512_srli_epi64(result_r0, 32));

  // Add adjacent pairs again. 32 bpc.
  // ... | _ r g b
  __m512i out = _mm512_unpacklo_epi64(result_bg, result_r0);

  // Divide by 256 to get back into correct range.
  out = _mm512_srli_epi32(out, 8);

  // Convert from 32bpc => 16bpc => 8bpc
  out = _mm512_packus_epi32(out, _mm512_setzero_si512());
  out = _mm512_packus_epi16(out, _mm512_setzero_si512());

  return out;
}

// Slightly faster than memcpy
static inline void memcpy_12(uint8_t* dst, const uint8_t* src) {
  *((uint64_t*) dst) = *((uint64_t*) src);
  *((uint32_t*) (dst + 8)) = *((uint32_t*) (src + 8));
}

static inline void write_output_pixels(__m512i pixels_1357, __m512i pixels_2468,
                                       interpolate::BGRPixel output_pixels[8]) {

  // Unpack to get adjacent pixel data in lower 64 bits of each lane
  // _ _ 8 7 | _ _ 6 5 | _ _ 4 3 | _ _ 2 1
  __m512i combined = _mm512_unpacklo_epi32(pixels_1357, pixels_2468);

  // If AVX512 VBMI was available, _mm512_permutexvar_epi8 could pack
  // all the pixels into the lower 24 bytes in one instruction.

  // (unused) | 8 7 6 5 | (unused) | 4 3 2 1
  combined = _mm512_permutex_epi64(combined, _MM_SHUFFLE(3, 3, 2, 0));

  // Pack the pixels into the lower 96 bits of lanes 1 and 3
  // (unused) | _ 8 7 6 5 | (unused) | _ 4 3 2 1
  combined = _mm512_shuffle_epi8(combined,
                                 _mm512_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1,    // unused
                                                 -1, -1, -1, -1, -1, -1, -1, -1,    // unused
                                                 -1, -1, -1, -1,                    // unused
                                                 30, 29, 28, 26, 25, 24,            // 8, 7
                                                 22, 21, 20, 18, 17, 16,            // 6, 5
                                                 -1, -1, -1, -1, -1, -1, -1, -1,    // unused
                                                 -1, -1, -1, -1, -1, -1, -1, -1,    // unused
                                                 -1, -1, -1, -1,                    // unused
                                                 14, 13, 12, 10, 9, 8,              // 4, 3
                                                 6, 5, 4, 2, 1, 0                   // 2, 1
                                                 ));

  // Store pixel data
  alignas(64) uint8_t stored[64];
  _mm512_store_si512((__m512i*) stored, combined);

  // Write pixel data back to image.
  memcpy_12((uint8_t*) output_pixels, stored);
  memcpy_12((uint8_t*) (output_pixels + 4), stored + 32);
}

// Bilinear interpolation of 8 adjacent output pixels with the supplied coordinates using AVX512.
static inline void interpolate(const interpolate::BGRImage& image,
                               const interpolate::InputCoords input_coords[8],
                               interpolate::BGRPixel output_pixels[8]) {

  const __m512i weights = calculate_weights(&input_coords[0].y);

  const __m512i weights_1357 = _mm512_unpacklo_epi64(weights, weights);
  const __m512i pixels_1357 = interpolate_four_pixels(image, input_coords, weights_1357);

  const __m512i weights_2468 = _mm512_unpackhi_epi64(weights, weights);
  const __m512i pixels_2468 = interpolate_four_pixels(image, input_coords + 1, weights_2468);

  write_output_pixels(pixels_1357, pixels_2468, output_pixels);
}

}    // namespace interpolate::bilinear::avx512
