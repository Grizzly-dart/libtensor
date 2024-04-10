#ifndef TENSORCUDA_HPP
#define TENSORCUDA_HPP

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_THREADS_PER_BLOCK 1024U

struct DType {
  uint8_t index;
  uint8_t bytes;
  uint8_t subIndex;
};

const DType i8 = {0, 1, 0};
const DType i16 = {1, 2, 1};
const DType i32 = {2, 4, 2};
const DType i64 = {3, 8, 3};
const DType u8 = {4, 1, 4};
const DType u16 = {5, 2, 5};
const DType u32 = {6, 4, 6};
const DType u64 = {7, 8, 7};
const DType bf16 = {8, 2, 0};
const DType f16 = {9, 2, 1};
const DType f32 = {10, 4, 2};
const DType f64 = {11, 4, 3};

const DType dtypes[] = {i8,  i16, i32,  i64, u8,  u16,
                        u32, u64, bf16, f16, f32, f64};

struct Dim2 {
  uint32_t r;
  uint32_t c;

  [[nodiscard]] uint64_t nel() const { return r * c; }
};

struct Dim3 {
  uint32_t ch;
  uint32_t r;
  uint32_t c;

  /*__device__ __host__*/ Dim2 toDim2() { return {r, c}; };

  [[nodiscard]] uint64_t nel() const { return ch * r * c; }
};

typedef enum PadMode : uint8_t {
  CONSTANT,
  CIRCULAR,
  REFLECT,
  REPLICATION
} PadMode;

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
} tcuDeviceProps;

typedef struct {
  uint64_t free;
  uint64_t total;
} tcuMemInfo;

typedef struct {
  cudaStream_t stream;
  int32_t device;
} tcuStream;

typedef struct {
  cudaStream_t *stream;
  int32_t device;
  uint32_t nStreams;
} tcuStreams;

__device__ __host__ inline Dim2 toDim2(const Dim3 &dim) {
  return {dim.r, dim.c};
}

const char *tcuGetDeviceProps(tcuDeviceProps &ret, int32_t device);
const char *tcuGetMemInfo(tcuMemInfo &memInfo, int32_t device);

const char *tcuCreateStream(tcuStream &ret, int32_t device);
const char *tcuDestroyStream(tcuStream *stream);
/// Destroys stream and frees memory. Ignores errors and
/// conforms to void (*)(void*) signature.
extern void tcuFinalizeStream(tcuStream *ret);
const char *tcuSyncStream(tcuStream *stream, void (*callback)(const char *));

const char *tcuAlloc(tcuStream &stream, void **mem, uint64_t size);
const char *tcuFree(tcuStream &stream, void *ptr);
const char *tcuMemcpy(tcuStream &stream, void *dst, void *src, uint64_t size);

/*
extern const char *tcuCast(
    tcuStream &stream, void *out, void *inp, uint64_t n, dtype outType,
    dtype inpType
);

extern const char *tcuNeg(
    tcuStream &stream, void *out, void *inp, uint64_t n, dtype outType,
    dtype inType
);

extern const char *tcuAbs(
    tcuStream &stream, void *out, void *inp, uint64_t n, dtype outType,
    dtype inType
);

extern const char *tcuSqr(
    tcuStream &stream, void *out, void *inp, uint64_t n, dtype outType,
    dtype inType
);

extern const char *tcuSqrt(
    tcuStream &stream, void *out, void *inp, uint64_t n, dtype outType,
    dtype inType
);

extern const char *tcuLog(
    tcuStream &stream, void *out, void *inp, uint64_t n, dtype outType,
    dtype inType
);

extern const char *tcuExp(
    tcuStream &stream, void *out, void *inp, uint64_t n, dtype outType,
    dtype inType
);

extern const char *tcuSin(
    tcuStream &stream, void *out, const void *inp, uint64_t n, dtype outType,
    dtype inType
);

extern const char *tcuCos(
    tcuStream &stream, void *out, const void *inp, uint64_t n, dtype outType,
    dtype inType
);

extern const char *tcuTan(
    tcuStream &stream, void *out, const void *inp, uint64_t n, dtype outType,
    dtype inType
);

extern const char *tcuSinh(
    tcuStream &stream, void *out, const void *inp, uint64_t n, dtype outType,
    dtype inType
);

extern const char *tcuCosh(
    tcuStream &stream, void *out, const void *inp, uint64_t n, dtype outType,
    dtype inType
);

extern const char *tcuTanh(
    tcuStream &stream, void *out, const void *inp, uint64_t n, dtype outType,
    dtype inType
);

extern const char *tcuMinThreshold(
    tcuStream &stream, const void *out, const void *inp, void *threshold,
    void *value, uint64_t n, dtype dtype
);

extern const char *tcuPlus(
    tcuStream &stream, void *out, void *inp1, void *inp2, void *scalar,
    uint64_t n, uint8_t flipScalar, dtype outType, dtype inp1Type,
    dtype inp2Type
);

extern const char *tcuMinus(
    tcuStream &stream, void *out, void *inp1, void *inp2, void *scalar,
    uint64_t n, uint8_t flipScalar, dtype outType, dtype inp1Type,
    dtype inp2Type
);

extern const char *tcuMul(
    tcuStream &stream, void *out, void *inp1, void *inp2, void *scalar,
    uint64_t n, uint8_t flipScalar, dtype outType, dtype inp1Type,
    dtype inp2Type
);

extern const char *tcuDiv(
    tcuStream &stream, void *out, void *inp1, void *inp2, void *scalar,
    uint64_t n, uint8_t flipScalar, dtype outType, dtype inp1Type,
    dtype inp2Type
);

extern const char *tcuMean(
    tcuStream &stream, double *out, void *inp, uint64_t nel, dtype inpType
);

extern const char *tcuVariance(
    tcuStream &stream, double *out, void *inp, uint64_t nel,
    uint64_t correction, dtype inpType
);

const char *tcuSum2d(
    tcuStream &stream, void *out, void *inp, Dim2 inpS, dtype outType,
    dtype inpType
);

const char *tcuMean2d(
    tcuStream &stream, void *out, void *inp, Dim2 inpS, dtype outType,
    dtype inpType
);

const char *tcuVariance2d(
    tcuStream &stream, void *out, void *inp, Dim2 inpS, uint64_t correction,
    uint8_t calcStd, dtype outType, dtype inpType
);

const char *tcuNormalize2d(
    tcuStream &stream, void *out, void *inp, Dim2 inpS, double epsilon,
    dtype outType, dtype inpType
);

const char *tcuTranspose2d(
    tcuStream &stream, double *out, double *inp, Dim3 size
);

char const *tcuMatMul(
    tcuStream &stream, double *out, double *inp1, double *inp2, uint32_t m,
    uint32_t n, uint32_t k, uint32_t batches
);
char const *tcuMatMulT(
    tcuStream &stream, double *out, double *inp1, double *inp2T, uint32_t m,
    uint32_t n, uint32_t k, uint32_t batches
);

char const *tcuMatMulCadd(
    tcuStream &stream, double *out, double *inp1, double *inp2, double *add,
    uint32_t m, uint32_t n, uint32_t k, uint32_t batches
);

extern char const *tcuMatMulTCadd(
    tcuStream &stream, double *out, double *inp1, double *inp2T, double *add,
    uint32_t m, uint32_t n, uint32_t k, uint32_t batches
);

extern const char *tcuMaxPool2d(
    tcuStream &stream, double *out, double *inp, Dim2 kernS, Dim2 outS,
    Dim2 inpS, uint32_t matrices, Dim2 padding, Dim2 stride, Dim2 dilation
);

extern const char *tcuConv2d(
    tcuStream &stream, double *out, double *inp, double *kernel,
    uint32_t batches, Dim3 outS, Dim3 inpS, Dim2 kernS, uint32_t groups,
    Dim2 padding, PadMode padMode, double pad, Dim2 stride, Dim2 dilation
);

extern const char *tcuPickRows(
    tcuStream &stream, void *out, const void *inp, const void *indices,
    Dim2 size, dtype type, dtype itype
);

extern const char *tcuELU(
    tcuStream &stream, const void *out, const void *inp, uint64_t n,
    double alpha, dtype dtype
);

extern const char *tcuSigmoid(
    tcuStream &stream, void *out, void *inp, uint64_t n, dtype dtype
);

extern const char *tcuSiLU(
    tcuStream &stream, const void *out, void *inp, uint64_t n, dtype dtype
);

extern const char *tcuSoftplus(
    tcuStream &stream, const void *out, void *inp, uint64_t n, int32_t beta,
    int32_t threshold, dtype dtype
);

extern const char *tcuSoftsign(
    tcuStream &stream, const void *out, void *inp, uint64_t n, dtype dtype
);

extern const char *tcuMish(
    tcuStream &stream, const void *out, void *inp, uint64_t n, dtype dtype
);
 */

#ifdef __cplusplus
} // extern "C"
#endif

const char *setupElementwiseKernelStrided(
    tcuStream &stream, uint64_t n, cudaLaunchConfig_t &config
);

const char *setupElementwiseKernel(
    tcuStream &stream, uint64_t n, cudaLaunchConfig_t &config
);

#endif // TENSORCUDA_HPP