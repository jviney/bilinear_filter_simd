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

}    // namespace interpolate
