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
  uintptr_t data_end;

  BGRImage(){};
  BGRImage(int rows, int cols, int step, BGRPixel* data)
      : rows(rows),
        cols(cols),
        step(step),
        data(data),
        data_end(((uintptr_t) data) + rows * step) {}

  inline const BGRPixel* ptr(int row, int col) const {
    int offset = (row * step) + (col * 3);
    return (const BGRPixel*) (((const uint8_t*) data) + offset);
  }

  inline const BGRPixel* ptr_below(const BGRPixel* ptr) const {
    auto end = ((uintptr_t) ptr) + step;

    if (end < data_end) [[likely]] {
      return (const BGRPixel*) end;
    } else {
      return ptr;
    }
  }
};

}    // namespace interpolate
