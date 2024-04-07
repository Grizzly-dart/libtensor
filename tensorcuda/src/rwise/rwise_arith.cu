#include <cuda_runtime.h>

#include <cstdint>
#include <reducers.hpp>
#include <string>
#include <tensorcuda.hpp>

template <typename O, typename I1, typename I2>
void rwise_plus(O *out, I1 *inp1, I2 *inp2, uint32_t numCols) {
  uint32_t row = blockIdx.x;
  uint32_t numRows = gridDim.x;
  uint32_t batch = blockIdx.y;

  out = out + numRows * numCols * batch;
  inp1 = inp1 + numRows * numCols * batch;

  for (uint32_t col = threadIdx.x; col < numCols; col += blockDim.x) {
    out[row * numCols + col] = inp1[row * numCols + col] + inp2[row];
  }
}

const char *tcuRwisePlus(
    tcuStream &stream, void *out, void *inp1, void* inp2, Dim3 size
) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  cudaDeviceProp props;
  err = cudaGetDeviceProperties(&props, stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }

  cudaLaunchConfig_t config = {
      .stream = stream.stream,
  };
  if(size.c < props.maxThreadsPerBlock) {
    config.blockDim.x = size.c;
  } else {
    config.blockDim.x = props.maxThreadsPerBlock;
  }
  config.gridDim.x = size.r;
  config.gridDim.y = size.c;

  // TODO
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}