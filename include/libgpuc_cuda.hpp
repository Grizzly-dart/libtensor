#ifndef LIBGPUC_CUDA_HPP
#define LIBGPUC_CUDA_HPP

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_THREADS_PER_BLOCK 1024U

typedef struct {
  uint32_t x;
  uint32_t y;
} Dim2;

typedef struct Tensor_t {
  double* mem = nullptr;
  uint64_t dim[10] = {0,0,0,0,0,0,0,0,0,0};
  uint8_t ndim = 0;
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

/// @brief Calculates the number of matrices in the tensor
/// @param t Tensor
/// @return Returns the number of matrices in the tensor
extern uint64_t getTensorCountMat(Tensor t);

extern void ewiseF64Add2(Tensor out, Tensor in1, Tensor in2);

/// @brief Computes row-wise sum of a 2D tensor
/// @param out Sum of each row
/// @param in Input 2D tensor
extern void sum2DTensor(Tensor out, Tensor in);
/// @brief Computes row-wise mean of a 2D tensor
/// @param out Mean of each row
/// @param in Input 2D tensor
extern void mean2DTensor(Tensor out, Tensor in);

extern void matmulF64(Tensor out, Tensor in1, Tensor in2);

#ifdef __cplusplus
}
#endif

#endif // LIBGPUC_CUDA_HPP