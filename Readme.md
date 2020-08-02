# Bilinear image filters with SSE and AVX2

Demonstrates performance characteristics of bilinear image filtering using
SIMD (SSE and AVX2) and multithreading.

Requires a CPU with AVX2 support.

## Dependencies

* C++17 (GCC >= 8)
* OpenCV 4
* CMake

## Build and run

```
git submodule update --init --recursive --jobs 4
mkdir -p build
cd build
cmake ..
make -j$(nproc)
./bilinear_filter_simd
```

Displays benchmark numbers for different algorithms. Multithreaded AVX2 is the fastest.

## Benchmark results

```
2020-07-18 14:12:07
Running ./bilinear_filter_simd
Run on (12 X 4600 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 256 KiB (x6)
  L3 Unified 12288 KiB (x1)
Load Average: 3.22, 1.50, 0.86
---------------------------------------------------------------------------------
Benchmark                                       Time             CPU   Iterations
---------------------------------------------------------------------------------
No SIMD - single thread/min_time:2.000       10.4 ms         10.4 ms          269
No SIMD - multi thread/min_time:2.000        2.02 ms         2.02 ms         1382
SSE4 - single thread/min_time:2.000          7.04 ms         7.04 ms          394
SSE4 - multi thread/min_time:2.000           1.60 ms         1.60 ms         1749
AVX2 - single thread/min_time:2.000          5.81 ms         5.81 ms          480
AVX2 - multi thread/min_time:2.000           1.32 ms         1.32 ms         2127
```
