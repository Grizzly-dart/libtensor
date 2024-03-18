#include <cuda_runtime.h>
#include <stdio.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
#include <string>

#define TILE_SIZE 32
#define BLOCK_ROWS 8

template <typename T>
__global__ void transpose2d(T *out, const T *inp, Dim2 inSize) {
  int outCol = blockIdx.x * TILE_SIZE + threadIdx.x;
  int outRow = blockIdx.y * TILE_SIZE + threadIdx.y;

  int batch = blockIdx.z;
  inp += batch * inSize.r * inSize.c;
  out += batch * inSize.r * inSize.c;

  // TILE_SIZE+1 to avoid shared memory bank conflicts
  __shared__ T tile[TILE_SIZE][TILE_SIZE + 1];

  // coalesced read from global mem to shared mem row-by-row
  for (int rowOffset = 0; rowOffset < TILE_SIZE; rowOffset += BLOCK_ROWS) {
    uint32_t row = outRow + rowOffset;
    T val = inp[outCol * inSize.c + row];
    tile[threadIdx.y + rowOffset][threadIdx.x] = val;
  }

  __syncthreads();

  Dim2 outSize{inSize.c, inSize.r};

  // coalesced write from shared mem to global mem row-by-row
  for (int rowOffset = 0; rowOffset < TILE_SIZE; rowOffset += BLOCK_ROWS) {
    uint32_t row = outRow + rowOffset;
    if (row < outSize.r && outCol < outSize.c) {
      out[row * outSize.c + outCol] = tile[threadIdx.y + rowOffset][threadIdx.x];
    }
  }
}

const char *libtcCudaTranspose2d(
    libtcCudaStream &stream, double *out, double *inp, Dim3 inSize
) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }

  Dim2 outSize{inSize.c, inSize.r};

  cudaLaunchConfig_t config = {
      .stream = stream.stream,
  };
  config.blockDim = {TILE_SIZE, BLOCK_ROWS, 1};
  config.gridDim.x = (outSize.c + TILE_SIZE - 1) / TILE_SIZE;
  config.gridDim.y = (outSize.r + TILE_SIZE - 1) / TILE_SIZE;
  config.gridDim.z = inSize.ch;

  err =
      cudaLaunchKernelEx(&config, transpose2d<double>, out, inp, inSize.toDim2());
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}