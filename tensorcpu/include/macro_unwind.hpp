//
// Created by tejag on 2024-04-28.
//

#ifndef TENSORCPU_MACRO_UNWIND_HPP
#define TENSORCPU_MACRO_UNWIND_HPP

#define UNWIND2_SAME(A, OP) OP(A, A)

#define UNWIND2_SAME_ALL_TYPES(OP)                                             \
  UNWIND2_SAME(int8_t, OP)                                                     \
  UNWIND2_SAME(int16_t, OP)                                                    \
  UNWIND2_SAME(int32_t, OP)                                                    \
  UNWIND2_SAME(int64_t, OP)                                                    \
  UNWIND2_SAME(uint8_t, OP)                                                    \
  UNWIND2_SAME(uint16_t, OP)                                                   \
  UNWIND2_SAME(uint32_t, OP)                                                   \
  UNWIND2_SAME(uint64_t, OP)                                                   \
  UNWIND2_SAME(float, OP)                                                      \
  UNWIND2_SAME(double, OP)

#define UNWIND2_ALL_2ND(T, OP)                                                 \
  OP(T, int8_t)                                                                \
  OP(T, int16_t)                                                               \
  OP(T, int32_t)                                                               \
  OP(T, int64_t)                                                               \
  OP(T, uint8_t)                                                               \
  OP(T, uint16_t)                                                              \
  OP(T, uint32_t)                                                              \
  OP(T, uint64_t)                                                              \
  OP(T, float)                                                                 \
  OP(T, double)

#define UNWIND2_ALL_TYPES(OP)                                                  \
  UNWIND2_ALL_2ND(int8_t, OP)                                                  \
  UNWIND2_ALL_2ND(int16_t, OP)                                                 \
  UNWIND2_ALL_2ND(int32_t, OP)                                                 \
  UNWIND2_ALL_2ND(int64_t, OP)                                                 \
  UNWIND2_ALL_2ND(uint8_t, OP)                                                 \
  UNWIND2_ALL_2ND(uint16_t, OP)                                                \
  UNWIND2_ALL_2ND(uint32_t, OP)                                                \
  UNWIND2_ALL_2ND(uint64_t, OP)                                                \
  UNWIND2_ALL_2ND(float, OP)                                                   \
  UNWIND2_ALL_2ND(double, OP)

#endif // TENSORCPU_MACRO_UNWIND_HPP
