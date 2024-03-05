#include <cuda_runtime.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
#include <string>
#include <memory.h>

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

Tensor makeTensor1D(uint64_t n) {
  Tensor tensor = Tensor{ndim : 1};
  tensor.dim[0] = n;
  cudaMalloc(&tensor.mem, n * sizeof(double));
  return tensor;
}

Tensor makeTensor2D(uint64_t m, uint64_t n) {
  Tensor tensor = Tensor{ndim : 2};
  tensor.dim[0] = m;
  tensor.dim[1] = n;
  cudaMalloc(&tensor.mem, m * n * sizeof(double));
  return tensor;
}

Tensor makeTensor(uint64_t* dims, uint8_t ndim) {
  if (ndim > 10) {
    throw std::string("Tensors are capped at 10 dimensions");
  } else if (ndim < 1) {
    throw std::string("Tensors must have at least 1 dimension");
  }
  Tensor tensor = Tensor{ndim : ndim};
  uint64_t size = 1;
  for (int i = 0; i < ndim; i++) {
    tensor.dim[i] = dims[i];
    size *= dims[i];
  }
  cudaMalloc(&tensor.mem, size * sizeof(double));
  return tensor;
}

void releaseTensor(Tensor t) {
  cudaFree(t.mem);
}

uint64_t getTensorNel(Tensor t) {
  uint64_t size = 1;
  for (int i = 0; i < t.ndim; i++) {
    size *= t.dim[i];
  }
  return size;
}

// TODO implement start offset
void readTensor(Tensor t, double* out, uint64_t size) {
  if (size > getTensorNel(t)) {
    throw std::string("Size mismatch");
  }
  auto err = cudaMemcpy(out, t.mem, size * sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
}

// TODO implement start offset
void writeTensor(Tensor t, double* inp, uint64_t size) {
  if (size > getTensorNel(t)) {
    throw std::string("Size mismatch");
  }
  auto err = cudaMemcpy(t.mem, inp, size * sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    throw std::string(cudaGetErrorString(err));
  }
}

uint64_t getTensorM(Tensor t) {
  if (t.ndim < 2) {
    throw std::string("Tensor must have at least 2 dimensions");
  }
  return t.dim[t.ndim - 2];
}

uint64_t getTensorN(Tensor t) {
  if (t.ndim < 2) {
    throw std::string("Tensor must have at least 2 dimensions");
  }
  return t.dim[t.ndim - 1];
}

uint64_t getTensorC(Tensor t) {
  if (t.ndim < 2) {
    throw std::string("Tensor must have at least 2 dimensions");
  } else if (t.ndim == 2) {
    return 1;
  }
  return t.dim[t.ndim - 3];
}

uint64_t getTensorB(Tensor t) {
  if (t.ndim < 2) {
    throw std::string("Tensor must have at least 2 dimensions");
  } else if (t.ndim < 3) {
    return 1;
  }
  uint64_t ret = 1;
  for (int i = 0; i < t.ndim - 3; i++) {
    ret *= t.dim[i];
  }
  return ret;
}

uint64_t getTensorCountMat(Tensor t) {
  if (t.ndim < 2) {
    throw std::string("Tensor must have at least 2 dimensions");
  }
  if (t.ndim == 2) return 1;
  uint64_t count = 1;
  for (int i = 0; i < t.ndim - 2; i++) {
    count *= t.dim[i];
  }
  return count;
}

Tensor reshapeTensor(Tensor t, uint64_t* dims, uint8_t ndim) {
  if (ndim > 10) {
    throw std::string("Tensors are capped at 10 dimensions");
  } else if (ndim < 1) {
    throw std::string("Tensors must have at least 1 dimension");
  }
  Tensor tensor = Tensor{ndim : ndim};
  if (getTensorNel(tensor) != getTensorNel(t)) {
    throw std::string("Size mismatch");
  }
  tensor.mem = t.mem;
  return tensor;
}