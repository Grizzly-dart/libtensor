#include <cstdint>

#define MAX_THREADS_PER_BLOCK 1024

typedef struct Tensor_t Tensor;

typedef struct Tensor_t {
  double* mem = nullptr;
  uint64_t x;
  bool released;

	static Tensor make1D(uint64_t n);

	void write(double* inp, uint64_t size);
  void read(double* out, uint64_t size);
  void release();
} Tensor;

extern void elementwiseAdd2(double* out, double* in1, double* in2, uint32_t size);
