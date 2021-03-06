#include "common.hpp"

#include "benchmark/bilinear_plain_single_thread.hpp"
#include "benchmark/bilinear_plain_multi_thread.hpp"
#include "benchmark/bilinear_sse4_single_thread.hpp"
#include "benchmark/bilinear_sse4_multi_thread.hpp"
#include "benchmark/bilinear_avx2_single_thread.hpp"
#include "benchmark/bilinear_avx2_multi_thread.hpp"

#ifdef __AVX512F__
#include "benchmark/bilinear_avx512_single_thread.hpp"
#include "benchmark/bilinear_avx512_multi_thread.hpp"
#endif

BenchmarkInput create_benchmark_input() {
  auto benchmark_input = BenchmarkInput();

  auto source_image = cv::imread("../assets/155603.jpg");
  benchmark_input.source_image_mat = source_image;

  benchmark_input.source_image = interpolate::BGRImage(
      source_image.rows, source_image.cols, source_image.step,
      reinterpret_cast<interpolate::BGRPixel*>(source_image.ptr<cv::Vec3b>(0, 0)));

  benchmark_input.output_size = cv::Size2i(1280, 720);
  benchmark_input.coords =
      sampling_coordinates(benchmark_input.output_size, benchmark_input.source_image_mat.size());

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

#ifdef __AVX512F__
  compare_mats(gold_standard, "avx512 single thread",
               bilinear_avx512_single_thread(benchmark_input));
  compare_mats(gold_standard, "avx512 multi thread", bilinear_avx512_multi_thread(benchmark_input));
#endif
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

#ifdef __AVX512F__
  benchmarks.push_back(benchmark::RegisterBenchmark(
      "AVX512 - single thread", BM_bilinear_avx512_single_thread, benchmark_input));
  benchmarks.push_back(benchmark::RegisterBenchmark(
      "AVX512 - multi thread", BM_bilinear_avx512_multi_thread, benchmark_input));
#endif

  for (auto bm : benchmarks) {
    bm->Unit(benchmark::kMillisecond);
    bm->MinTime(2.0);
  }
}

int main(int argc, char** argv) {
  auto benchmark_input = create_benchmark_input();

  validate_implementations(benchmark_input);
  register_benchmarks(benchmark_input);

  printf("Input image size: %dx%d\n", benchmark_input.source_image.cols,
         benchmark_input.source_image.rows);
  printf("Output image size: %dx%d\n", benchmark_input.output_size.width,
         benchmark_input.output_size.height);
  printf("OpenCV: numberOfCPUS=%d getNumThreads=%d\n", cv::getNumberOfCPUs(), cv::getNumThreads());

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
