#include <cstdint>
#include <memory>
#include <string>

#include <cuda_runtime.h>

#include <tensorcuda.hpp>

class IntInterpreter {
public:
  virtual ~IntInterpreter() = default;

  __device__ __host__ virtual int64_t load(uint64_t index) = 0;

  __device__ __host__ virtual void store(uint64_t index, int64_t value) = 0;
};

template <typename T> class IntInterpreterImpl : public IntInterpreter {
public:
  T *ptr;
  __device__ __host__ explicit IntInterpreterImpl(T *ptr) : ptr(ptr) {}

  ~IntInterpreterImpl() override = default;

  __device__ __host__ int64_t load(uint64_t index) override {
    return ptr[index];
  }

  __device__ __host__ void store(uint64_t index, int64_t value) override {
    ptr[index] = value;
  }
};

__device__ __host__ IntInterpreter *getIntInterpreter(dtype type, void *ptr) {
  switch (type) {
  case i8:
    return new IntInterpreterImpl<int8_t>((int8_t *)ptr);
  case i16:
    return new IntInterpreterImpl<int16_t>((int16_t *)ptr);
  case i32:
    return new IntInterpreterImpl<int32_t>((int32_t *)ptr);
  case i64:
    return new IntInterpreterImpl<int64_t>((int64_t *)ptr);
  case u8:
    return new IntInterpreterImpl<uint8_t>((uint8_t *)ptr);
  case u16:
    return new IntInterpreterImpl<uint16_t>((uint16_t *)ptr);
  case u32:
    return new IntInterpreterImpl<uint32_t>((uint32_t *)ptr);
  case u64:
    return new IntInterpreterImpl<uint64_t>((uint64_t *)ptr);
  default:
    return nullptr;
  }
}

/// Adds two tensors
template <typename O, typename I1, typename I2>
__global__ void plusV2(
    O *out, I1 *inp1, I2 *inp2, dtype outType, dtype inp1Type, dtype inp2Type,
    uint64_t n, uint8_t flipScalar
) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  auto i1 = getIntInterpreter(inp1Type, inp1);
  auto i2 = getIntInterpreter(inp2Type, inp2);
  auto o = getIntInterpreter(outType, out);

  for (uint64_t i = thId; i < n; i += numThreads) {
    o->store(i1->load(i) + i2->load(i));
  }
}

/// Adds two tensors
__global__ void plusV3(
    IntInterpreter *out, IntInterpreter *inp1, IntInterpreter *inp2, uint64_t n,
    uint8_t flipScalar
) {
  uint32_t numThreads = blockDim.x * gridDim.x;
  uint32_t thId = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint64_t i = thId; i < n; i += numThreads) {
    out->store(i, inp1->load(i) + inp2->load(i));
  }
}

void tcplusV3(void *out, void *inp1, void *inp2) {
  auto o = getIntInterpreter(i64, out);
  auto i1 = getIntInterpreter(i64, inp1);
  auto i2 = getIntInterpreter(i64, inp2);
  plusV3<<<4, 5>>>(o, i1, i2, 10, 0);
}