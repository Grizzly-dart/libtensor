#ifndef LIBGPUC_CUDA_HPP
#define LIBGPUC_CUDA_HPP

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

extern void libtcFree(void* ptr);
extern void* libtcRealloc(void* ptr, uint64_t size);
extern void libtcMemcpy(void* dst, void* src, uint64_t size);

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
  uint64_t free;
  uint64_t total;
} libtcCudaMemInfo;

typedef struct {
  uint64_t r;
  uint64_t c;
} Size2;

typedef struct {
  cudaStream_t stream;
  int32_t device;
} libtcCudaStream;

extern const char* libtcCudaGetDeviceProps(libtcDeviceProps& ret, int32_t device);
extern const char* libtcCudaGetMemInfo(libtcCudaMemInfo& memInfo, int32_t device);

extern const char* libtcCudaCreateStream(libtcCudaStream& ret, int32_t device);
extern const char* libtcCudaDestroyStream(libtcCudaStream& stream);

extern const char* libtcCudaAlloc(libtcCudaStream& stream, void** mem, uint64_t size);
extern const char* libtcCudaFree(libtcCudaStream& stream, void* ptr);
extern const char* libtcCudaMemcpy(libtcCudaStream& stream, void* dst, void* src, uint64_t size);

extern const char* libtcCudaAddCkern(libtcCudaStream& stream, double* out, const double* in1, const double* in2, uint32_t n);
extern const char* libtcCudaSum2DCkern(libtcCudaStream& stream, double* out, double* in, Size2 inSize);

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

extern const char* libtcCudaAdd2Ckern(libtcCudaStream& stream, double* out, const double* in1, const double* in2, uint64_t n);
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

extern const char* libtcCudaMaxPool2DF64(libtcCudaStream& stream, double* out, double* inp, 
    Dim2 kernS, Dim2 outS, Dim2 inS, uint32_t matrices, Dim2 padding, 
    PaddingMode PaddingMode, double pad, Dim2 stride, Dim2 dilation);

#ifdef __cplusplus
}
#endif

#endif // LIBGPUC_CUDA_HPP