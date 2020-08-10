## Benchmark Results

### 2020-08-10

Combined storage of output pixels into a single memcpy for SSE4 and AVX2.

```
Input image size: 3840x2160
Output image size: 1280x720
OpenCV: numberOfCPUS=12 getNumThreads=12
2020-08-10 22:20:51
Running ./bilinear_filter_simd
Run on (12 X 4600 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 256 KiB (x6)
  L3 Unified 12288 KiB (x1)
Load Average: 3.25, 2.08, 1.04
---------------------------------------------------------------------------------
Benchmark                                       Time             CPU   Iterations
---------------------------------------------------------------------------------
No SIMD - single thread/min_time:2.000       7.58 ms         7.58 ms          371
No SIMD - multi thread/min_time:2.000        1.66 ms         1.64 ms         1696
SSE4 - single thread/min_time:2.000          5.01 ms         5.01 ms          555
SSE4 - multi thread/min_time:2.000           1.37 ms         1.37 ms         2056
AVX2 - single thread/min_time:2.000          4.07 ms         4.07 ms          689
AVX2 - multi thread/min_time:2.000           1.33 ms         1.33 ms         2098
```

### 2020-08-09

Implemented AVX512. Benchmarks from a c5.4xlarge EC2 instance.

```
Input image size: 3840x2160
Output image size: 1280x720
OpenCV: numberOfCPUS=16 getNumThreads=16
2020-08-09 11:34:16
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

### 2020-08-05

After optimising data loading for weight calculation in SSE4/AVX2 implementations.

```
2020-08-05 22:00:27
Running ./bilinear_filter_simd
Run on (12 X 4600 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 256 KiB (x6)
  L3 Unified 12288 KiB (x1)
Load Average: 0.27, 0.21, 0.14
---------------------------------------------------------------------------------
Benchmark                                       Time             CPU   Iterations
---------------------------------------------------------------------------------
No SIMD - single thread/min_time:2.000       7.48 ms         7.48 ms          375
SSE4 - single thread/min_time:2.000          5.21 ms         5.21 ms          538
AVX2 - single thread/min_time:2.000          4.23 ms         4.23 ms          661
```

### 2020-08-03

After applying same technique to AVX2 and calculating 4 weights at once with 16 bit ints.
Not much difference.

```
2020-08-03 21:10:56
Running ./bilinear_filter_simd
Run on (12 X 4600 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 256 KiB (x6)
  L3 Unified 12288 KiB (x1)
Load Average: 5.54, 2.48, 1.00
---------------------------------------------------------------------------------
Benchmark                                       Time             CPU   Iterations
---------------------------------------------------------------------------------
No SIMD - single thread/min_time:2.000       7.58 ms         7.58 ms          369
No SIMD - multi thread/min_time:2.000        1.68 ms         1.68 ms         1661
SSE4 - single thread/min_time:2.000          5.52 ms         5.52 ms          505
SSE4 - multi thread/min_time:2.000           1.42 ms         1.42 ms         1947
AVX2 - single thread/min_time:2.000          4.63 ms         4.63 ms          603
AVX2 - multi thread/min_time:2.000           1.33 ms         1.33 ms         2110
```

### 2020-08-02

After calculating two weights at once with SSE4 using 16 bit ints:

```
2020-08-03 00:35:04
Running ./bilinear_filter_simd
Run on (12 X 4600 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 256 KiB (x6)
  L3 Unified 12288 KiB (x1)
Load Average: 0.49, 0.40, 0.34
---------------------------------------------------------------------------------
Benchmark                                       Time             CPU   Iterations
---------------------------------------------------------------------------------
No SIMD - single thread/min_time:2.000       7.60 ms         7.60 ms          346
No SIMD - multi thread/min_time:2.000        1.68 ms         1.68 ms         1657
SSE4 - single thread/min_time:2.000          5.49 ms         5.49 ms          511
SSE4 - multi thread/min_time:2.000           1.42 ms         1.42 ms         1972
AVX2 - single thread/min_time:2.000          4.67 ms         4.67 ms          596
AVX2 - multi thread/min_time:2.000           1.37 ms         1.37 ms         2049
```

After introducing `BGRImage` type:

```
2020-08-02 16:27:20
Running ./bilinear_filter_simd
Run on (12 X 4600 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 256 KiB (x6)
  L3 Unified 12288 KiB (x1)
Load Average: 1.53, 1.62, 1.10
---------------------------------------------------------------------------------
Benchmark                                       Time             CPU   Iterations
---------------------------------------------------------------------------------
No SIMD - single thread/min_time:2.000       7.59 ms         7.59 ms          369
No SIMD - multi thread/min_time:2.000        1.68 ms         1.68 ms         1665
SSE4 - single thread/min_time:2.000          5.88 ms         5.88 ms          473
SSE4 - multi thread/min_time:2.000           1.47 ms         1.47 ms         1898
AVX2 - single thread/min_time:2.000          4.64 ms         4.64 ms          601
AVX2 - multi thread/min_time:2.000           1.35 ms         1.35 ms         2068
```

Before introducing `BGRImage` type:

```
2020-08-02 15:33:51
Running ./bilinear_filter_simd
Run on (12 X 4600 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 256 KiB (x6)
  L3 Unified 12288 KiB (x1)
Load Average: 0.04, 0.09, 0.48
---------------------------------------------------------------------------------
Benchmark                                       Time             CPU   Iterations
---------------------------------------------------------------------------------
No SIMD - single thread/min_time:2.000       8.12 ms         8.12 ms          346
No SIMD - multi thread/min_time:2.000        1.72 ms         1.72 ms         1623
SSE4 - single thread/min_time:5.000          6.18 ms         6.18 ms          453
SSE4 - multi thread/min_time:2.000           1.50 ms         1.50 ms         1866
AVX2 - single thread/min_time:2.000          5.29 ms         5.29 ms          528
AVX2 - multi thread/min_time:2.000           1.36 ms         1.36 ms         2050
```

### 2020-07-18

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

### 2020-06-14

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
