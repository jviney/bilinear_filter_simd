#include "common.hpp"

#include "benchmark/bilinear_plain_single_thread.hpp"
#include "benchmark/bilinear_plain_multi_thread.hpp"
#include "benchmark/bilinear_sse4_single_thread.hpp"
#include "benchmark/bilinear_sse4_multi_thread.hpp"
#include "benchmark/bilinear_avx2_single_thread.hpp"
#include "benchmark/bilinear_avx2_multi_thread.hpp"

BenchmarkInput create_benchmark_input() {
  auto benchmark_input = BenchmarkInput();
  benchmark_input.source_image = cv::imread("../assets/155603.jpg");
  benchmark_input.output_size = cv::Size2i(1280, 720);
  benchmark_input.coords =
      sampling_coordinates(benchmark_input.output_size, benchmark_input.source_image.size());

  return benchmark_input;
}

void compare_mats(const cv::Mat3b& gold_standard, std::string name, const cv::Mat3b& comparison) {
  if (!mats_equivalent(gold_standard, comparison)) {
    std::cout << name << " output image not the same\n";
    std::exit(1);
  }
}

void validate_implementations(BenchmarkInput& benchmark_input) {
  auto gold_standard = bilinear_plain_single_thread(benchmark_input);

  compare_mats(gold_standard, "plain multi thread", bilinear_plain_multi_thread(benchmark_input));
  compare_mats(gold_standard, "sse4 single thread", bilinear_sse4_single_thread(benchmark_input));
  compare_mats(gold_standard, "sse4 multi thread", bilinear_sse4_multi_thread(benchmark_input));
  compare_mats(gold_standard, "avx2 single thread", bilinear_avx2_single_thread(benchmark_input));
  compare_mats(gold_standard, "avx2 multi thread", bilinear_avx2_multi_thread(benchmark_input));
}

void register_benchmarks(BenchmarkInput& benchmark_input) {
  auto benchmarks = std::vector<benchmark::internal::Benchmark*>();

  benchmarks.push_back(benchmark::RegisterBenchmark(
      "No SIMD - single thread", BM_bilinear_plain_single_thread, benchmark_input));
  benchmarks.push_back(benchmark::RegisterBenchmark(
      "No SIMD - multi thread", BM_bilinear_plain_multi_thread, benchmark_input));
  benchmarks.push_back(benchmark::RegisterBenchmark(
      "SSE4 - single thread", BM_bilinear_sse4_single_thread, benchmark_input));
  benchmarks.push_back(benchmark::RegisterBenchmark(
      "SSE4 - multi thread", BM_bilinear_sse4_multi_thread, benchmark_input));
  benchmarks.push_back(benchmark::RegisterBenchmark(
      "AVX2 - single thread", BM_bilinear_avx2_single_thread, benchmark_input));
  benchmarks.push_back(benchmark::RegisterBenchmark(
      "AVX2 - multi thread", BM_bilinear_avx2_multi_thread, benchmark_input));

  for (auto bm : benchmarks) {
    bm->Unit(benchmark::kMillisecond);
    bm->MinTime(2.0);
  }
}

int main(int argc, char** argv) {
  auto benchmark_input = create_benchmark_input();

  validate_implementations(benchmark_input);
  register_benchmarks(benchmark_input);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
