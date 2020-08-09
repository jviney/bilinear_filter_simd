# Bilinear image filter using SSE, AVX2 and AVX512

Demonstrates bilinear image filtering using SIMD (SSE, AVX2 and AVX512) and multithreading.

Requires a CPU with AVX2 support. AVX512 will be used if it is available.

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

Displays benchmark numbers for different algorithms. Multithreaded AVX512 is the fastest.

## Benchmark results

```
2020-08-09 11:34:16
Intel(R) Xeon(R) Platinum 8124M CPU @ 3.00GHz
Input image size: 3840x2160
Output image size: 1280x720
OpenCV: numberOfCPUS=16 getNumThreads=16
Running ./bilinear_filter_simd
Run on (16 X 3400.97 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 1024 KiB (x8)
  L3 Unified 25344 KiB (x1)
Load Average: 0.00, 0.04, 0.13
---------------------------------------------------------------------------------
Benchmark                                       Time             CPU   Iterations
---------------------------------------------------------------------------------
No SIMD - single thread/min_time:2.000       13.3 ms         13.3 ms          215
No SIMD - multi thread/min_time:2.000        1.25 ms         1.25 ms         2227
SSE4 - single thread/min_time:2.000          10.7 ms         10.7 ms          267
SSE4 - multi thread/min_time:2.000          0.987 ms        0.986 ms         2856
AVX2 - single thread/min_time:2.000          8.14 ms         8.14 ms          360
AVX2 - multi thread/min_time:2.000          0.868 ms        0.868 ms         3253
AVX512 - single thread/min_time:2.000        7.33 ms         7.33 ms          373
AVX512 - multi thread/min_time:2.000        0.822 ms        0.822 ms         3430
```

```
2020-08-09 23:39:59
Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
Running ./bilinear_filter_simd
Run on (12 X 4600 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 256 KiB (x6)
  L3 Unified 12288 KiB (x1)
Load Average: 0.05, 0.06, 0.01
---------------------------------------------------------------------------------
Benchmark                                       Time             CPU   Iterations
---------------------------------------------------------------------------------
No SIMD - single thread/min_time:2.000       7.64 ms         7.64 ms          367
No SIMD - multi thread/min_time:2.000        1.69 ms         1.69 ms         1675
SSE4 - single thread/min_time:2.000          5.26 ms         5.26 ms          532
SSE4 - multi thread/min_time:2.000           1.39 ms         1.39 ms         2010
AVX2 - single thread/min_time:2.000          4.26 ms         4.26 ms          658
AVX2 - multi thread/min_time:2.000           1.34 ms         1.34 ms         2082
```
