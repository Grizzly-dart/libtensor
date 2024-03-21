#include <string>
#include <cstdint>

#include <cuda_runtime.h>

#include <libgpuc_cuda.hpp>

// TODO implement stride and split
/// Adds two tensors
template<typename O, typename I1, typename I2>
__global__ void add2(O* out, const I1* in1, const I2* in2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for(uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in1[i] + in2[i];
  }
}

const char* libtcCudaAdd2(libtcCudaStream& stream, void* out, const void* in1, const void* in2, uint64_t n, dtype oType, dtype i1Type, dtype i2Type) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  cudaDeviceProp props;
  err = cudaGetDeviceProperties(&props, stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  uint32_t numThreads = props.multiProcessorCount * 128;
  if(numThreads > n) {
    numThreads = n;
  }
  
  cudaLaunchConfig_t config = {
    .stream = stream.stream,
  };
  if(numThreads < props.maxThreadsPerBlock) {
    config.blockDim.x = numThreads;
    config.gridDim.x = 1;
  } else {
    config.blockDim.x = props.maxThreadsPerBlock;
    config.gridDim.x = (n + props.maxThreadsPerBlock - 1) / props.maxThreadsPerBlock;
  }

  if(oType == i1Type && oType == i2Type) {
    if(oType == dtype::f64) {
      err = cudaLaunchKernelEx(&config, add2<double, double, double>, out, in1, in2, n);
    } else if(oType == dtype::f32) {
      err = cudaLaunchKernelEx(&config, add2<float, float, float>, out, in1, in2, n);
    } else if(oType == dtype::u8) {
      err = cudaLaunchKernelEx(&config, add2<uint8_t, uint8_t, uint8_t>, out, in1, in2, n);
    } else if(oType == dtype::u16) {
      err = cudaLaunchKernelEx(&config, add2<uint16_t, uint16_t, uint16_t>, out, in1, in2, n);
    } else if(oType == dtype::u32) {
      err = cudaLaunchKernelEx(&config, add2<uint32_t, uint32_t, uint32_t>, out, in1, in2, n);
    } else if(oType == dtype::u64) {
      err = cudaLaunchKernelEx(&config, add2<uint64_t, uint64_t, uint64_t>, out, in1, in2, n);
    } else if(oType == dtype::i8) {
      err = cudaLaunchKernelEx(&config, add2<int8_t, int8_t, int8_t>, out, in1, in2, n);
    } else if(oType == dtype::i16) {
      err = cudaLaunchKernelEx(&config, add2<int16_t, int16_t, int16_t>, out, in1, in2, n);
    } else if(oType == dtype::i32) {
      err = cudaLaunchKernelEx(&config, add2<int32_t, int32_t, int32_t>, out, in1, in2, n);
    } else if(oType == dtype::i64) {
      err = cudaLaunchKernelEx(&config, add2<int64_t, int64_t, int64_t>, out, in1, in2, n);
    } else {
      return "Unsupported data type";
    }
    if (err != cudaSuccess) {
      return cudaGetErrorString(err);
    }
    return nullptr;
  }
  if(oType == dtype::f64 && i1Type == dtype::f64 && i2Type == dtype::f64) {
    err = cudaLaunchKernelEx(&config, add2<double, double, double>, out, in1, in2, n);
  } else if(oType == dtype::f32 && i1Type == dtype::f32 && i2Type == dtype::f32) {
    err = cudaLaunchKernelEx(&config, add2<float, float, float>, out, in1, in2, n);
  } else if(oType == dtype::f32 && i1Type == dtype::f32 && i2Type == dtype::f64) {
    err = cudaLaunchKernelEx(&config, add2<float, float, double>, out, in1, in2, n);
  } else if(oType == dtype::f64 && i1Type == dtype::f64 && i2Type == dtype::f32) {
    err = cudaLaunchKernelEx(&config, add2<double, double, float>, out, in1, in2, n);
  } else if(oType == dtype::f32 && i1Type == dtype::f64 && i2Type == dtype::f32) {
    err = cudaLaunchKernelEx(&config, add2<float, double, float>, out, in1, in2, n);
  } else if(oType == dtype::f64 && i1Type == dtype::f32 && i2Type == dtype::f64) {
    err = cudaLaunchKernelEx(&config, add2<double, float, double>, out, in1, in2, n);
  } else {
    return "Unsupported data type";
  }
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

template<typename T>
__global__ void subtract2(T* out, const T* in1, const T* in2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for(uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in1[i] - in2[i];
  }
}

const char* libtcCudaSubtract2(libtcCudaStream& stream, double* out, const double* in1, const double* in2, uint64_t n) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  cudaDeviceProp props;
  err = cudaGetDeviceProperties(&props, stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  uint32_t numThreads = props.multiProcessorCount * 128;
  if(numThreads > n) {
    numThreads = n;
  }
  
  cudaLaunchConfig_t config = {
    .stream = stream.stream,
  };
  if(numThreads < props.maxThreadsPerBlock) {
    config.blockDim.x = numThreads;
    config.gridDim.x = 1;
  } else {
    config.blockDim.x = props.maxThreadsPerBlock;
    config.gridDim.x = (n + props.maxThreadsPerBlock - 1) / props.maxThreadsPerBlock;
  }

  err = cudaLaunchKernelEx(&config, subtract2<double>, out, in1, in2, n);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

template<typename T>
__global__ void multiply2(T* out, const T* in1, const T* in2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for(uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in1[i] * in2[i];
  }
}

const char* libtcCudaMultiply2(libtcCudaStream& stream, double* out, const double* in1, const double* in2, uint64_t n) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  cudaDeviceProp props;
  err = cudaGetDeviceProperties(&props, stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  uint32_t numThreads = props.multiProcessorCount * 128;
  if(numThreads > n) {
    numThreads = n;
  }
  
  cudaLaunchConfig_t config = {
    .stream = stream.stream,
  };
  if(numThreads < props.maxThreadsPerBlock) {
    config.blockDim.x = numThreads;
    config.gridDim.x = 1;
  } else {
    config.blockDim.x = props.maxThreadsPerBlock;
    config.gridDim.x = (n + props.maxThreadsPerBlock - 1) / props.maxThreadsPerBlock;
  }

  err = cudaLaunchKernelEx(&config, multiply2<double>, out, in1, in2, n);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

template<typename T>
__global__ void divide2(T* out, const T* in1, const T* in2, uint64_t n) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for(uint64_t i = thId; i < n; i += numThreads) {
    out[i] = in1[i] / in2[i];
  }
}

const char* libtcCudaDivide2(libtcCudaStream& stream, double* out, const double* in1, const double* in2, uint64_t n) {
  auto err = cudaSetDevice(stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  cudaDeviceProp props;
  err = cudaGetDeviceProperties(&props, stream.device);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  uint32_t numThreads = props.multiProcessorCount * 128;
  if(numThreads > n) {
    numThreads = n;
  }
  
  cudaLaunchConfig_t config = {
    .stream = stream.stream,
  };
  if(numThreads < props.maxThreadsPerBlock) {
    config.blockDim.x = numThreads;
    config.gridDim.x = 1;
  } else {
    config.blockDim.x = props.maxThreadsPerBlock;
    config.gridDim.x = (n + props.maxThreadsPerBlock - 1) / props.maxThreadsPerBlock;
  }

  err = cudaLaunchKernelEx(&config, divide2<double>, out, in1, in2, n);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}