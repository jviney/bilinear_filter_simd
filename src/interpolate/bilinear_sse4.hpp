#pragma once

#include <immintrin.h>
#include "interpolate/types.hpp"

namespace interpolate::bilinear::sse4
{

// SSE bilinear interpolation implementation based on
// http://fastcpp.blogspot.com/2011/06/bilinear-pixel-interpolation-using-sse.html
// and altered to work with BGR24 image format.

// Calculate the interpolation weights for 2 pixels.
// Returns weights as 16 bit ints.
// (px2) w4 w3 w2 w1  (px1) w4 w3 w2 w1
static inline __m128i calc_weights(float x1, float y1, float x2, float y2) {
  const __m128 initial = _mm_set_ps(y2, x2, y1, x1);

  const __m128 floored = _mm_floor_ps(initial);
  const __m128 fractional = _mm_sub_ps(initial, floored);

  // Convert fractional parts to 32 bit ints in range 0-256
  // y2 x2 y1 x1
  __m128i lower = _mm_cvtps_epi32(_mm_mul_ps(fractional, _mm_set1_ps(256.0f)));

  // Convert to 16 bit ints
  // 0 0 0 0   y2 x2 y1 x1
  lower = _mm_packs_epi32(lower, _mm_set1_epi32(0));

  // Get the 1-fractional from the 16 bit result
  // 256 256 256 256   1-y2 1-x2 1-y1 1-x1
  const __m128i upper = _mm_sub_epi16(_mm_set1_epi16(256), lower);

  // Combine so we have all the parts in one value to shuffle
  // y2 (1-y2) x2 (1-x2)   y1 (1-y1) x1 (1-x1)
  const __m128i combined = _mm_unpacklo_epi16(upper, lower);

  // x2 (1-x2) x2 (1-x2)   x1 (1-x1) x1 (1-x1)
  const __m128i weights_x = _mm_shuffle_epi32(combined, _MM_SHUFFLE(2, 2, 0, 0));

  // y2 y2 (1-y2) (1-y2)   y1 y1 (1-y1) (1-y1)
  __m128i weights_y = _mm_shufflelo_epi16(combined, _MM_SHUFFLE(3, 3, 2, 2));
  weights_y = _mm_shufflehi_epi16(weights_y, _MM_SHUFFLE(3, 3, 2, 2));

  // Multiply to get per pixel weights. Divide by 256 to get back into correct range.
  __m128i weights = _mm_srli_epi16(_mm_mullo_epi16(weights_x, weights_y), 8);

  // If both weights were 256, the result is 65536 which is all 0s in the lower 16 bits.
  // Find the weights this happened to, and replace them with 256.
  __m128i weights_hi = _mm_mulhi_epi16(weights_x, weights_y);
  __m128i weights_hi_mask = _mm_cmpgt_epi16(weights_hi, _mm_setzero_si128());
  weights = _mm_blendv_epi8(weights, _mm_set1_epi16(256), weights_hi_mask);

  return weights;
}

static constexpr uint8_t ZEROED = 128;

static inline void interpolate_one_pixel(const interpolate::BGRImage& image,
                                         const interpolate::InputCoords& input_coords, __m128i w12,
                                         __m128i w34, interpolate::BGRPixel* output_pixel,
                                         bool can_write_next_pixel) {

  const auto* p0 = image.ptr(input_coords.y, input_coords.x);

  // Load 4 pixels for interpolation.
  // We are only using the lower 48 bits of each load.
  // Shuffle 24bpp around to use 64bpp with 16bpc
  // _ r g b _ r g b
  __m128i p12 = _mm_shuffle_epi8(_mm_loadl_epi64((const __m128i*) p0),
                                 _mm_set_epi8(ZEROED, ZEROED, ZEROED, 5, ZEROED, 4, ZEROED, 3,
                                              ZEROED, ZEROED, ZEROED, 2, ZEROED, 1, ZEROED, 0));

  // _ r g b _ r g b
  __m128i p34 = _mm_shuffle_epi8(_mm_loadl_epi64((const __m128i*) (p0 + image.stride)),
                                 _mm_set_epi8(ZEROED, ZEROED, ZEROED, 5, ZEROED, 4, ZEROED, 3,
                                              ZEROED, ZEROED, ZEROED, 2, ZEROED, 1, ZEROED, 0));

  // Multiply each pixel with its weight
  __m128i out_12 = _mm_mullo_epi16(p12, w12);
  __m128i out_34 = _mm_mullo_epi16(p34, w34);

  // Sum the results
  __m128i out_1234 = _mm_add_epi16(out_12, out_34);
  __m128i out_high = _mm_shuffle_epi32(out_1234, _MM_SHUFFLE(3, 2, 3, 2));
  __m128i out = _mm_add_epi16(out_1234, out_high);

  // Divide by 256
  out = _mm_srli_epi16(out, 8);

  // Convert to 8bpc
  out = _mm_packus_epi16(out, _mm_setzero_si128());

  // Extract the channels to create a BGRPixel
  int all_chans = _mm_cvtsi128_si32(out);

  // Faster to write 4 bytes instead of 3 when allowed
  memcpy(output_pixel, &all_chans, can_write_next_pixel ? 4 : 3);
}

static inline void interpolate(const interpolate::BGRImage& image,
                               const interpolate::InputCoords input_coords[2],
                               interpolate::BGRPixel output_pixels[2], bool can_write_next_pixel) {

  // Calculate the weights for 2 pixels
  __m128i weights =
      calc_weights(input_coords[0].x, input_coords[0].y, input_coords[1].x, input_coords[1].y);

  // Prepare weights for pixel 1
  __m128i pixel1_w12 = _mm_shufflelo_epi16(weights, _MM_SHUFFLE(1, 1, 0, 0));
  __m128i pixel1_w34 = _mm_shufflelo_epi16(weights, _MM_SHUFFLE(3, 3, 2, 2));
  // w2 w2 w2 w2 w1 w1 w1 w1
  pixel1_w12 = _mm_unpacklo_epi16(pixel1_w12, pixel1_w12);
  // w4 w4 w4 w4 w3 w3 w3 w3
  pixel1_w34 = _mm_unpacklo_epi16(pixel1_w34, pixel1_w34);

  // Prepare weights for pixel 2
  __m128i pixel2_w12 = _mm_shufflehi_epi16(weights, _MM_SHUFFLE(1, 1, 0, 0));
  __m128i pixel2_w34 = _mm_shufflehi_epi16(weights, _MM_SHUFFLE(3, 3, 2, 2));
  // w2 w2 w2 w2 w1 w1 w1 w1
  pixel2_w12 = _mm_unpackhi_epi16(pixel2_w12, pixel2_w12);
  // w4 w4 w4 w4 w3 w3 w3 w3
  pixel2_w34 = _mm_unpackhi_epi16(pixel2_w34, pixel2_w34);

  interpolate_one_pixel(image, input_coords[0], pixel1_w12, pixel1_w34, output_pixels, true);
  interpolate_one_pixel(image, input_coords[1], pixel2_w12, pixel2_w34, output_pixels + 1,
                        can_write_next_pixel);
}

}    // namespace interpolate::bilinear::sse4
