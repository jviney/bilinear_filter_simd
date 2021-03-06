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
  int stride;
  BGRPixel* data;    // non-owner

  BGRImage(){};
  BGRImage(int rows, int cols, int step, BGRPixel* data)
      : rows(rows), cols(cols), step(step), stride(step / 3), data(data) {}

  const BGRPixel* ptr(int row, int col) const { return data + row * stride + col; }
};

}    // namespace interpolate
