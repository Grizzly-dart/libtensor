#ifndef LIBGPUC_CUDA_HPP
#define LIBGPUC_CUDA_HPP

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_THREADS_PER_BLOCK 1024U

typedef struct {
  uint64_t totalGlobalMem;
  uint64_t totalConstMem;
  uint64_t sharedMemPerBlock;
  uint64_t reservedSharedMemPerBlock;
  uint64_t sharedMemPerMultiprocessor;
  uint32_t warpSize;
  uint32_t multiProcessorCount;
  uint32_t maxThreadsPerMultiProcessor;
  uint32_t maxThreadsPerBlock;
  uint32_t maxBlocksPerMultiProcessor;
  uint32_t l2CacheSize;
  uint64_t memPitch;
  uint32_t memoryBusWidth;
  uint32_t pciBusID;
  uint32_t pciDeviceID;
  uint32_t pciDomainID;
} libtcDeviceProps;

typedef struct {
  uint32_t r;
  uint32_t c;
} Size2;

extern void* libtcCudaAlloc(uint64_t size, int32_t device);
extern void libtcCudaFree(void* ptr, int32_t device);
extern void libtcCudaMemcpy(void* dst, void* src, uint64_t size, uint8_t dir, int32_t device);
extern libtcDeviceProps libtcCudaGetDeviceProps(int32_t device);

void* libtcRealloc(void* ptr, uint64_t size);
void libtcMemcpy(void* dst, void* src, uint64_t size);

extern void libtcCudaAddCkern(double* out, const double* in1, const double* in2, uint32_t n);
extern void libtcCudaSum2DCkern(double* out, double* in, Size2 inSize);

typedef struct {
  uint32_t x;
  uint32_t y;
} Dim2;

typedef struct Tensor_t {
  double* mem = nullptr;
  uint8_t ndim = 0;
  uint64_t dim[10] = {0,0,0,0,0,0,0,0,0,0};
} Tensor;

typedef enum {
  CONSTANT,
  CIRCULAR,
  REFLECT,
  REPLICATION
} PaddingMode;

extern Tensor makeTensor1D(uint64_t n);
extern Tensor makeTensor2D(uint64_t m, uint64_t n);
extern Tensor makeTensor(uint64_t* dims, uint8_t ndim);
extern void releaseTensor(Tensor t);
extern uint64_t getTensorNel(Tensor t);
extern void writeTensor(Tensor t, double* inp, uint64_t size);
extern void readTensor(Tensor t, double* out, uint64_t size);

/// @brief Calculates the number of rows in the tensor
/// @param t Tensor
/// @return Returns the number of rows in the tensor
extern uint64_t getTensorM(Tensor t);

/// @brief Calculates the number of columns in the tensor
/// @param t Tensor
/// @return Returns the number of columns in the tensor
extern uint64_t getTensorN(Tensor t);

/// @brief Calculates the number of channels in the tensor
/// @param t Tensor
/// @return Returns the number of channels in the tensor
extern uint64_t getTensorC(Tensor t);

/// @brief Calculates the number of batches in the tensor
/// @param t Tensor
/// @return Returns the number of batches in the tensor
extern uint64_t getTensorB(Tensor t);

/// @brief Calculates the number of matrices in the tensor
/// @param t Tensor
/// @return Returns the number of matrices in the tensor
extern uint64_t getTensorCountMat(Tensor t);

extern void add2D(Tensor out, Tensor in1, Tensor in2);

/// @brief Computes row-wise sum of a 2D tensor
/// @param out Sum of each row
/// @param in Input 2D tensor
extern void sum2DTensor(Tensor out, Tensor in);
/// @brief Computes row-wise mean of a 2D tensor
/// @param out Mean of each row
/// @param in Input 2D tensor
extern void mean2DTensor(Tensor out, Tensor in);
/// @brief Computes row-wise variance of a 2D tensor
/// @param out Variance of each row
/// @param in Input 2D tensor
extern void variance2DTensor(Tensor out, Tensor in);

extern void matmul(Tensor out, Tensor in1, Tensor in2);

#ifdef __cplusplus
}
#endif

#endif // LIBGPUC_CUDA_HPP