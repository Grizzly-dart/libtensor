#include <cuda_runtime.h>
#include <memory.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
#include <string>

void* libtcCudaAlloc(uint64_t size, int32_t device) {
  auto err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
  void* ret;
  cudaMalloc(&ret, size);
  return ret;
}

void libtcCudaFree(void* ptr, int32_t device) {
  auto err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
  cudaFree(ptr);
}

void libtcCudaMemcpy(void* dst, void* src, uint64_t size, int32_t device) {
  auto err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
  err = cudaMemcpy(dst, src, size, cudaMemcpyDefault);
  if (err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
}

void* libtcRealloc(void* ptr, uint64_t size) {
  return realloc(ptr, size);
}

void libtcMemcpy(void* dst, void* src, uint64_t size) {
  memcpy(dst, src, size);
}

libtcDeviceProps libtcGetDeviceProps(int32_t device) {
  cudaDeviceProp props;
  auto err = cudaGetDeviceProperties(&props, device);
  if (err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
  return libtcDeviceProps{
    totalGlobalMem : props.totalGlobalMem,
    totalConstMem : props.totalConstMem,
    sharedMemPerBlock : props.sharedMemPerBlock,
    reservedSharedMemPerBlock : props.reservedSharedMemPerBlock,
    sharedMemPerMultiprocessor : props.sharedMemPerMultiprocessor,
    warpSize : static_cast<uint32_t>(props.warpSize),
    multiProcessorCount : static_cast<uint32_t>(props.multiProcessorCount),
    maxThreadsPerMultiProcessor : static_cast<uint32_t>(props.maxThreadsPerMultiProcessor),
    maxThreadsPerBlock : static_cast<uint32_t>(props.maxThreadsPerBlock),
    maxBlocksPerMultiProcessor : static_cast<uint32_t>(props.maxBlocksPerMultiProcessor),
    l2CacheSize : static_cast<uint32_t>(props.l2CacheSize),
    memPitch : static_cast<uint32_t>(props.memPitch),
    memoryBusWidth : static_cast<uint32_t>(props.memoryBusWidth),
    pciBusID : static_cast<uint32_t>(props.pciBusID),
    pciDeviceID : static_cast<uint32_t>(props.pciDeviceID),
    pciDomainID : static_cast<uint32_t>(props.pciDomainID),
  };
}