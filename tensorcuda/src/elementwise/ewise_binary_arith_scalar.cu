#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include <libgpuc_cuda.hpp>

// TODO implement stride and split
/// Adds two tensors
template <typename O, typename I1, typename I2>
__global__ void addScalar(O *out, const I1 *in1, const I2 inp2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in1[i] + inp2;
  }
}

template <typename O, typename I1, typename I2>
__global__ void subScalar(O *out, const I1 *in1, const I2 inp2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in1[i] - inp2;
  }
}

template <typename O, typename I1, typename I2>
__global__ void subLhsScalar(O *out, const I1 *in1, const I2 inp2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = inp2 - in1[i];
  }
}

template <typename O, typename I1, typename I2>
__global__ void mulScalar(O *out, const I1 *in1, const I2 in2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in1[i] * in2;
  }
}

template <typename O, typename I1, typename I2>
__global__ void divScalar(O *out, const I1 *in1, const I2 in2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in1[i] / in2;
  }
}

template <typename O, typename I1, typename I2>
__global__ void divLhsScalar(O *out, const I1 *in1, const I2 in2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in2 / in1[i];
  }
  ADD;
}

typedef enum : uint8_t {
  BinaryArithOp_Add,
  BinaryArithOp_Sub,
  BinaryArithOp_Mul,
  BinaryArithOp_Div,
  BinaryArithOp_SubLhs,
  BinaryArithOp_DivLhs
} BinaryArithOp;

extern const char *libtcCudaBinaryArith_f64_f64_f64(
    libtcCudaStream &stream, void *out, void *in1, void *in2, uint64_t n,
    BinaryArithOp op
) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernelStrided(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  cudaError_t err;
  if (op == BinaryArithOp_Add) {
    err = cudaLaunchKernelEx(
        &config, addScalar<double, double, double>, (double *)out,
        (double *)in1, *(double *)in2, n
    );
  } else if (op == BinaryArithOp_Sub) {
    err = cudaLaunchKernelEx(
        &config, subScalar<double, double, double>, (double *)out,
        (double *)in1, *(double *)in2, n
    );
  } else if (op == BinaryArithOp_Mul) {
    err = cudaLaunchKernelEx(
        &config, mulScalar<double, double, double>, (double *)out,
        (double *)in1, *(double *)in2, n
    );
  } else if (op == BinaryArithOp_Div) {
    err = cudaLaunchKernelEx(
        &config, divScalar<double, double, double>, (double *)out,
        (double *)in1, *(double *)in2, n
    );
  } else if (op == BinaryArithOp_SubLhs) {
    err = cudaLaunchKernelEx(
        &config, subLhsScalar<double, double, double>, (double *)out,
        (double *)in1, *(double *)in2, n
    );
  } else if (op == BinaryArithOp_DivLhs) {
    err = cudaLaunchKernelEx(
        &config, divLhsScalar<double, double, double>, (double *)out,
        (double *)in1, *(double *)in2, n
    );
  } else {
    return "Invalid operation";
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}