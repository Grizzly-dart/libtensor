#include <cblas.h>
#include <cstdint>
#include <type_traits>

#include "matrix.hpp"
#include "tensorcpu.hpp"

#if 0
template <typename T>
const char *mm(
    T *out, const T *inp1, const T *inp2, Dim2 size, uint32_t k,
    uint32_t batchSize, uint8_t bT, uint8_t outTID, uint8_t i1TID, uint8_t i2TID
) {
  if (outTID == i1TID && outTID == i2TID) {
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
      if (!bT) {
        return mm_same_slow<T>(out, inp1, inp2, size, k, batchSize);
      } else {
        return mmBt_same_slow<T>(out, inp1, inp2, size, k, batchSize);
      }
    }
  } else {
    if (!bT) {
      return mm_casted_slow<T>(
          out, inp1, inp2, size, k, batchSize, outTID, i1TID, i2TID
      );
    } else {
      return mmBt_casted_slow<T>(
          out, inp1, inp2, size, k, batchSize, outTID, i1TID, i2TID
      );
    }
  }
  return nullptr;
}
#endif