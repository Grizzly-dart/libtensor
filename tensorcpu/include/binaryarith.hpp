//
// Created by Teja Gudapati on 2024-05-06.
//

#ifndef TENSORCPU_BINARYARITH_HPP
#define TENSORCPU_BINARYARITH_HPP

#include "tensorcpu.hpp"
#include <cstdint>

template <typename I>
extern void binaryarith_1thread(
    I *out, I *inp1, I *inp2, BinaryOp op, uint64_t nel, uint8_t flip
);

template <typename I>
extern void binaryarith_parallel(
    I *out, I *inp1, I *inp2, BinaryOp op, uint64_t nel, uint8_t flip
);

template <typename I>
extern void binaryarith_casted_1thread(
    void *out, void *inp1, void *inp2, BinaryOp op, uint64_t nel, uint8_t flip,
    uint8_t outTID, uint8_t i1TID, uint8_t i2TID
);

template <typename I>
extern void binaryarith_casted_parallel(
    void *out, void *inp1, void *inp2, BinaryOp op, uint64_t nel, uint8_t flip,
    uint8_t outTID, uint8_t i1TID, uint8_t i2TID
);

#endif // TENSORCPU_BINARYARITH_HPP
