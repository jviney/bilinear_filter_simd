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

```
2020-06-14 00:58:30
Running ./bilinear_filter_simd
Run on (4 X 3100 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x2)
  L1 Instruction 32 KiB (x2)
  L2 Unified 256 KiB (x2)
  L3 Unified 4096 KiB (x1)
Load Average: 2.21, 2.64, 2.35
--------------------------------------------------------------------------------------------
Benchmark                                                  Time             CPU   Iterations
--------------------------------------------------------------------------------------------
BM_interpolate_plain_single_thread/min_time:2.000       26.6 ms         26.4 ms          101
BM_interpolate_plain_multi_thread/min_time:2.000        5.45 ms         4.87 ms          576
BM_interpolate_sse4_single_thread/min_time:2.000        24.2 ms         24.0 ms          118
BM_interpolate_sse4_multi_thread/min_time:2.000         4.04 ms         3.56 ms          822
BM_interpolate_avx2_single_thread/min_time:2.000        13.1 ms         13.0 ms          217
BM_interpolate_avx2_multi_thread/min_time:2.000         2.49 ms         2.31 ms         1196
```
