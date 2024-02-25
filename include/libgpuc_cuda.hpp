#ifndef LIBGPUC_CUDA_HPP
#define LIBGPUC_CUDA_HPP

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_THREADS_PER_BLOCK 1024

typedef struct Tensor_t {
  double* mem = nullptr;
  uint64_t dim[10] = {0,0,0,0,0,0,0,0,0,0};
  uint8_t ndim = 0;
} Tensor;

extern Tensor makeTensor1D(uint64_t n);
extern Tensor makeTensor2D(uint64_t m, uint64_t n);
extern Tensor makeTensor(uint64_t* dims, uint8_t ndim);
extern void releaseTensor(Tensor t);
extern uint64_t getTensorNel(Tensor t);
extern void writeTensor(Tensor t, double* inp, uint64_t size);
extern void readTensor(Tensor t, double* out, uint64_t size);
extern uint64_t getTensorM(Tensor t);
extern uint64_t getTensorN(Tensor t);
extern uint64_t getTensorBatchCount(Tensor t);

extern void ewiseF64Add2(Tensor out, Tensor in1, Tensor in2);
extern void matmulF64(Tensor out, Tensor in1, Tensor in2);

#ifdef __cplusplus
}
#endif

#endif // LIBGPUC_CUDA_HPP