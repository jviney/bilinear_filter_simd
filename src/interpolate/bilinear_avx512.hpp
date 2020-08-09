#pragma once

#include <immintrin.h>

namespace interpolate::bilinear::avx512
{

//
// Weights
//

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
  __m512i weights_y = _mm512_shufflelo_epi16(combined, _MM_SHUFFLE(1, 1, 0, 0));
  weights_y = _mm512_shufflehi_epi16(weights_y, _MM_SHUFFLE(1, 1, 0, 0));

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

// zeroed, used to extend 8 bit channels to 16 bit
static constexpr uint8_t Z = 128;

// Blue and green channels.
#define MASK_SHUFFLE_BG_SINGLE_LANE Z, 12, Z, 9, Z, 4, Z, 1, Z, 11, Z, 8, Z, 3, Z, 0
static const __m512i MASK_SHUFFLE_BG =
    _mm512_set_epi8(MASK_SHUFFLE_BG_SINGLE_LANE, MASK_SHUFFLE_BG_SINGLE_LANE,
                    MASK_SHUFFLE_BG_SINGLE_LANE, MASK_SHUFFLE_BG_SINGLE_LANE);

// Red channel. The upper half of each 128 bit lane is not used.
#define MASK_SHUFFLE_R0_SINGLE_LANE Z, Z, Z, Z, Z, Z, Z, Z, Z, 13, Z, 10, Z, 5, Z, 2
static const __m512i MASK_SHUFFLE_R0 =
    _mm512_set_epi8(MASK_SHUFFLE_R0_SINGLE_LANE, MASK_SHUFFLE_R0_SINGLE_LANE,
                    MASK_SHUFFLE_R0_SINGLE_LANE, MASK_SHUFFLE_R0_SINGLE_LANE);

//
// Interpolation
//

static inline void interpolate_four_pixels(const interpolate::BGRImage& image,
                                           const interpolate::InputCoords input_coords[7],
                                           __m512i weights, interpolate::BGRPixel* output_pixels,
                                           bool first_three_pixels_can_overwrite,
                                           bool fourth_pixel_can_overwrite) {
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

  // Store pixel data
  int32_t stored[16];
  _mm512_store_si512((__m512i*) stored, out);

  // Write pixel data back to image.
  // Faster to write 4 bytes instead of three when valid.
  memcpy(output_pixels, &stored[0], first_three_pixels_can_overwrite ? 4 : 3);
  memcpy(output_pixels + 2, &stored[4], first_three_pixels_can_overwrite ? 4 : 3);
  memcpy(output_pixels + 4, &stored[8], first_three_pixels_can_overwrite ? 4 : 3);
  memcpy(output_pixels + 6, &stored[12], fourth_pixel_can_overwrite ? 4 : 3);
}

// Bilinear interpolation of 8 adjacent output pixels with the supplied coordinates using AVX512.
static inline void interpolate(const interpolate::BGRImage& image,
                               const interpolate::InputCoords input_coords[8],
                               interpolate::BGRPixel output_pixels[8], bool can_write_ninth_pixel) {

  const __m512i weights = calculate_weights(&input_coords[0].y);

  const __m512i weights_1357 = _mm512_unpacklo_epi64(weights, weights);
  interpolate_four_pixels(image, input_coords, weights_1357, output_pixels, true, true);

  const __m512i weights_2468 = _mm512_unpackhi_epi64(weights, weights);
  interpolate_four_pixels(image, input_coords + 1, weights_2468, output_pixels + 1, false,
                          can_write_ninth_pixel);
}

}    // namespace interpolate::bilinear::avx512
