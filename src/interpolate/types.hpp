#pragma once

#include <stdint.h>

namespace interpolate
{

struct InputCoords {
  float y;
  float x;
};

struct BGRPixel {
  uint8_t b;
  uint8_t g;
  uint8_t r;
};

class BGRImage
{
public:
  int rows;
  int cols;
  int step;
  BGRPixel* data;    // non-owner

  BGRImage(){};
  BGRImage(int rows, int cols, int step, BGRPixel* data)
      : rows(rows), cols(cols), step(step), data(data) {}

  inline const BGRPixel* ptr(int row, int col) const {
    return (const BGRPixel*) (((const uint8_t*) data) + (row * step) + (col * 3));
  }

  inline const BGRPixel* ptr_below(const BGRPixel* ptr) const {
    return (const BGRPixel*) (((const uint8_t*) ptr) + step);
  }
};

}    // namespace interpolate
