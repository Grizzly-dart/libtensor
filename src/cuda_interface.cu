#include <cuda_runtime.h>
#include <memory.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
#include <string>

const char* libtcCudaCreateStream(libtcCudaStream& ret, int32_t device) {
  auto err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  cudaStream_t stream;
  err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  ret.stream = stream;
  ret.device = device;
  return nullptr;
}

const char* libtcCudaDestroyStream(libtcCudaStream& ret) {
  auto err = cudaSetDevice(ret.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  err = cudaStreamDestroy(static_cast<cudaStream_t>(ret.stream));
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

const char* libtcCudaAlloc(libtcCudaStream& stream, void** mem, uint64_t size) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  err = cudaMallocAsync(mem, size, stream.stream);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

const char* libtcCudaFree(libtcCudaStream& stream, void* ptr) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  err = cudaFreeAsync(ptr, stream.stream);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

const char* libtcCudaMemcpy(libtcCudaStream& stream, void* dst, void* src, uint64_t size) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream.stream);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

const char* libtcCudaGetMemInfo(libtcCudaMemInfo& memInfo, int32_t device) {
  auto err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  err = cudaMemGetInfo(&memInfo.free, &memInfo.total);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

const char* libtcCudaGetDeviceProps(libtcDeviceProps& ret, int32_t device) {
  cudaDeviceProp props;
  auto err = cudaGetDeviceProperties(&props, device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  ret.totalGlobalMem = props.totalGlobalMem;
  ret.totalConstMem = props.totalConstMem;
  ret.sharedMemPerBlock = props.sharedMemPerBlock;
  ret.reservedSharedMemPerBlock = props.reservedSharedMemPerBlock;
  ret.sharedMemPerMultiprocessor = props.sharedMemPerMultiprocessor;
  ret.warpSize = static_cast<uint32_t>(props.warpSize);
  ret.multiProcessorCount = static_cast<uint32_t>(props.multiProcessorCount);
  ret.maxThreadsPerMultiProcessor = static_cast<uint32_t>(props.maxThreadsPerMultiProcessor);
  ret.maxThreadsPerBlock = static_cast<uint32_t>(props.maxThreadsPerBlock);
  ret.maxBlocksPerMultiProcessor = static_cast<uint32_t>(props.maxBlocksPerMultiProcessor);
  ret.l2CacheSize = static_cast<uint32_t>(props.l2CacheSize);
  ret.memPitch = static_cast<uint32_t>(props.memPitch);
  ret.memoryBusWidth = static_cast<uint32_t>(props.memoryBusWidth);
  ret.pciBusID = static_cast<uint32_t>(props.pciBusID);
  ret.pciDeviceID = static_cast<uint32_t>(props.pciDeviceID);
  ret.pciDomainID = static_cast<uint32_t>(props.pciDomainID);
  return nullptr;
}

void libtcFree(void* ptr) {
  free(ptr);
}

void* libtcRealloc(void* ptr, uint64_t size) {
  return realloc(ptr, size);
}

void libtcMemcpy(void* dst, void* src, uint64_t size) {
  memcpy(dst, src, size);
}