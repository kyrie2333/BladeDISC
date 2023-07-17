#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#define GpuElapse(start, stop, elapsed, total) \
  cudaEventRecord(stop, 0);                    \
  cudaEventSynchronize(stop);                  \
  cudaEventElapsedTime(&elapsed, start, stop); \
  total += elapsed;

/// Result structure
struct Result {

  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};

// implement GEMM : int8 * int8 = fp16

// datatype for input
using ElementInputA = int8_t;
using ElementInputB = int8_t;
// using ElementOutput = int32_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = int32_t;
using ElementComputeEpilogue = ElementAccumulator;
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAop = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 32>; 
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

constexpr int NumStage = 2;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA,
    LayoutInputA,
    ElementInputB,
    LayoutInputB,
    ElementOutput,
    LayoutOutput,
    ElementAccumulator,
    MMAop,
    SmArch,
    ShapeMMAThreadBlock,
    ShapeMMAWarp,
    ShapeMMAOp,
    EpilogueOp,
    SwizzleThreadBlock,
    NumStage>;

int run(int M, int N, int K) {

  const int length_m = M;
  const int length_n = N;
  const int length_k = K;

  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  // initialize data
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());

  // fill data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(4),
      ElementInputA(-4),
      0); 
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(4),
      ElementInputB(-4),
      0); 
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);
  cutlass::reference::host::TensorFill(tensor_d.host_view());
  cutlass::reference::host::TensorFill(tensor_ref_d.host_view());

  // copy data from host to device
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);
  
  int split_k_slices = 1;

  typename Gemm::Arguments arguments{problem_size,
                                    tensor_a.device_ref(), 
                                    tensor_b.device_ref(),
                                    tensor_c.device_ref(), 
                                    tensor_d.device_ref(), 
                                    {alpha, beta},
                                    split_k_slices};

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return -1;
  }

  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    return -1;
  }

  // ----- profling starts -----
  Result result;

  for (int warmup = 0; warmup < 100; ++warmup) {
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      return -1;
    }
  }

  cudaEvent_t events[2];
  for (auto & event : events) {
    result.error = cudaEventCreate(&event);
  }

  result.error = cudaEventRecord(events[0]);
  for (int iter = 0; iter < 100; ++iter) {
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      return -1;
    }
  }
  result.error = cudaEventRecord(events[1]);
  result.error = cudaEventSynchronize(events[1]);

  float elapsed_time_ms = 0;
  result.error = cudaEventElapsedTime(&elapsed_time_ms, events[0], events[1]);
  result.runtime_ms = double(elapsed_time_ms) / 100.0;

  for (auto event : events) {
    result.error = cudaEventDestroy(event);
  }

  // ----- profling ends -----


  // create instantiation for device reference
  cutlass::reference::device::Gemm<ElementInputA,
                                   LayoutInputA,
                                   ElementInputB,
                                   LayoutInputB,
                                   ElementOutput,
                                   LayoutOutput,
                                   ElementComputeEpilogue,
                                   ElementComputeEpilogue>
      gemm_device;

  gemm_device(problem_size,
              alpha,
              tensor_a.device_ref(),
              tensor_b.device_ref(),
              beta,
              tensor_c.device_ref(),
              tensor_ref_d.device_ref());
  
  cudaDeviceSynchronize();

  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  bool passed = cutlass::reference::host::TensorEquals(
      tensor_d.host_view(), 
      tensor_ref_d.host_view());

  if (passed) {
    std::cout << "GEMM(M, K, N) " << length_m << " " << length_k << " " << length_n << " ";
    std::cout << "Runtime: " << result.runtime_ms * 1000.0 << " us" << std::endl;
  }

  // std::cout << "GEMM " << (passed ? "passed" : "FAILED") << std::endl;
  // std::cout << (tensor_d.host_view().at({0,0})) << std::endl;

  return 0;
}


int main(int argc, char** argv) {

  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " M K N" << std::endl;
    return -1;
  }

  int M = atoi(argv[1]);
  int K = atoi(argv[2]);
  int N = atoi(argv[3]);
  
  return run(M, N, K);
  
}