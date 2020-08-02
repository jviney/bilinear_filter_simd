## Benchmark Results

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
