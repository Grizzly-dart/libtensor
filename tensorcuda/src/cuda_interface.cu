#include <cuda_runtime.h>
#include <pthread.h>
#include <tensorcuda.hpp>

const char *tcuCreateStream(tcuStream &ret, int32_t device) {
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

const char *tcuDestroyStream(tcuStream *ret) {
  if (ret == nullptr)
    return nullptr;
  if (ret->stream != nullptr) {
    auto err = cudaSetDevice(ret->device);
    if (err != cudaSuccess) {
      return cudaGetErrorString(err);
    }
    err = cudaStreamDestroy(static_cast<cudaStream_t>(ret->stream));
    if (err != cudaSuccess) {
      return cudaGetErrorString(err);
    }
  }
  free(ret);
  return nullptr;
}

void tcuFinalizeStream(tcuStream *ret) { tcuDestroyStream(ret); }

typedef struct {
  tcuStream *stream;
  void (*callback)(const char *);
} syncStreamArgs;

void syncStream(syncStreamArgs *args) {
  auto stream = args->stream;
  auto callback = args->callback;
  free(args);
  auto err = cudaSetDevice(stream->device);
  if (err != cudaSuccess) {
    callback(cudaGetErrorString(err));
    return;
  }
  err = cudaStreamSynchronize(static_cast<cudaStream_t>(stream->stream));
  if (err != cudaSuccess) {
    callback(cudaGetErrorString(err));
    return;
  }
  callback(nullptr);
  pthread_exit(nullptr);
}

const char *tcuSyncStream(tcuStream *stream, void (*callback)(const char *)) {
  auto args = (syncStreamArgs *)(malloc(sizeof(syncStreamArgs)));
  args->stream = stream;
  args->callback = callback;

  pthread_attr_t attr;
  int rc = pthread_attr_init(&attr);
  if (rc == -1) {
    return "cudaStreamSync: error in pthread_attr_init";
  }
  rc = pthread_attr_setdetachstate(&attr, 1);
  if (rc == -1) {
    return "cudaStreamSync: error in pthread_attr_setdetachstate";
  }

  pthread_t thread;
  pthread_create(&thread, nullptr, (void *(*)(void *))syncStream, args);
  return nullptr;
}

const char *tcuAlloc(tcuStream &stream, void **mem, uint64_t size) {
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

const char *tcuFree(tcuStream &stream, void *ptr) {
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

const char *tcuMemcpy(tcuStream &stream, void *dst, void *src, uint64_t size) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  cudaStream_t s = stream.stream;
  if (s == nullptr) {
    err = cudaStreamCreate(&s);
    if (err != cudaSuccess) {
      return cudaGetErrorString(err);
    }
  }
  err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream.stream);
  if (err != cudaSuccess) {
    cudaStreamDestroy(s);
    return cudaGetErrorString(err);
  }
  if (stream.stream == nullptr) {
    err = cudaStreamSynchronize(stream.stream);
    if (err != cudaSuccess) {
      cudaStreamDestroy(s);
      return cudaGetErrorString(err);
    }
    err = cudaStreamDestroy(s);
    if (err != cudaSuccess) {
      return cudaGetErrorString(err);
    }
  }
  return nullptr;
}

const char *tcuGetMemInfo(tcuMemInfo &memInfo, int32_t device) {
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

const char *tcuGetDeviceProps(tcuDeviceProps &ret, int32_t device) {
  cudaDeviceProp props{};
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
  ret.maxThreadsPerMultiProcessor =
      static_cast<uint32_t>(props.maxThreadsPerMultiProcessor);
  ret.maxThreadsPerBlock = static_cast<uint32_t>(props.maxThreadsPerBlock);
  ret.maxBlocksPerMultiProcessor =
      static_cast<uint32_t>(props.maxBlocksPerMultiProcessor);
  ret.l2CacheSize = static_cast<uint32_t>(props.l2CacheSize);
  ret.memPitch = static_cast<uint32_t>(props.memPitch);
  ret.memoryBusWidth = static_cast<uint32_t>(props.memoryBusWidth);
  ret.pciBusID = static_cast<uint32_t>(props.pciBusID);
  ret.pciDeviceID = static_cast<uint32_t>(props.pciDeviceID);
  ret.pciDomainID = static_cast<uint32_t>(props.pciDomainID);
  return nullptr;
}

const char *setupElementwiseKernelStrided(
    tcuStream &stream, uint64_t n, cudaLaunchConfig_t &config
) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  cudaDeviceProp props{};
  err = cudaGetDeviceProperties(&props, stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  uint32_t numThreads = props.multiProcessorCount * 128;
  if (numThreads > n) {
    numThreads = n;
  }

  config.stream = stream.stream;
  if (numThreads < props.maxThreadsPerBlock) {
    config.blockDim.x = numThreads;
    config.gridDim.x = 1;
  } else {
    config.blockDim.x = props.maxThreadsPerBlock;
    config.gridDim.x =
        (numThreads + props.maxThreadsPerBlock - 1) / props.maxThreadsPerBlock;
  }

  return nullptr;
}

const char *setupElementwiseKernel(
    tcuStream &stream, uint64_t n, cudaLaunchConfig_t &config
) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  cudaDeviceProp props{};
  err = cudaGetDeviceProperties(&props, stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }

  config.stream = stream.stream;
  config.blockDim = {(uint)props.maxThreadsPerBlock, 1, 1};
  if (n > props.maxThreadsPerBlock) {
    config.gridDim.x =
        (n + props.maxThreadsPerBlock - 1) / props.maxThreadsPerBlock;
  } else {
    config.blockDim.x = n;
  }

  return nullptr;
}