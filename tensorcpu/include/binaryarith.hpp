//
// Created by Teja Gudapati on 2024-05-06.
//

#ifndef TENSORCPU_BINARYARITH_HPP
#define TENSORCPU_BINARYARITH_HPP

#include "tensorcpu.hpp"
#include <cstdint>

template <typename I>
extern void binaryarith_parallel(
    I *out, I *inp1, I *inp2, BinaryOp op, uint64_t nel, uint8_t flip,
    Dim2 i2broadcaster
);

#endif // TENSORCPU_BINARYARITH_HPP
