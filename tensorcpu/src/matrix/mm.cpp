#include <cblas.h>
#include <cstdint>
#include <type_traits>

#include "tensorcpu.hpp"
#include "matrix.hpp"

template <typename T>
const char *mm(
    T *out, const T *inp1, const T *inp2, Dim2 size, uint32_t k,
    uint32_t batchSize, uint8_t bT
) {
  if constexpr (std::is_same<T, double>::value) {
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, bT ? CblasTrans : CblasNoTrans, size.r,
        size.c, k, 1.0f, inp1, k, inp2, size.c, 0.0f, out, size.c
    );
  } else if constexpr (std::is_same<T, float>::value) {
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, bT ? CblasTrans : CblasNoTrans, size.r,
        size.c, k, 1.0f, inp1, k, inp2, size.c, 0.0f, out, size.c
    );
  } else {
    if(!bT) {
      return mm_same_slow<T>(out, inp1, inp2, size, k, batchSize);
    } else {
      return mmBt_same_slow<T>(out, inp1, inp2, size, k, batchSize);
    }
  }
  return nullptr;
}