//
// Created by tejag on 2024-04-28.
//

#ifndef TENSORCPU_MACRO_UNWIND_HPP
#define TENSORCPU_MACRO_UNWIND_HPP

#define UNWIND1_ALL_TYPES(OP)                                                  \
  OP(int8_t)                                                                   \
  OP(int16_t)                                                                  \
  OP(int32_t)                                                                  \
  OP(int64_t)                                                                  \
  OP(uint8_t)                                                                  \
  OP(uint16_t)                                                                 \
  OP(uint32_t)                                                                 \
  OP(uint64_t)                                                                 \
  OP(float)                                                                    \
  OP(double)

#define UNWIND1_SIGNED(OP)                                                     \
  OP(int8_t)                                                                   \
  OP(int16_t)                                                                  \
  OP(int32_t)                                                                  \
  OP(int64_t)                                                                  \
  OP(float)                                                                    \
  OP(double)

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

#define UNWIND2_ALL_2ND(OP, T)                                                 \
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
  UNWIND2_ALL_2ND(OP, int8_t)                                                  \
  UNWIND2_ALL_2ND(OP, int16_t)                                                 \
  UNWIND2_ALL_2ND(OP, int32_t)                                                 \
  UNWIND2_ALL_2ND(OP, int64_t)                                                 \
  UNWIND2_ALL_2ND(OP, uint8_t)                                                 \
  UNWIND2_ALL_2ND(OP, uint16_t)                                                \
  UNWIND2_ALL_2ND(OP, uint32_t)                                                \
  UNWIND2_ALL_2ND(OP, uint64_t)                                                \
  UNWIND2_ALL_2ND(OP, float)                                                   \
  UNWIND2_ALL_2ND(OP, double)

#define UNWIND2_SIGNED(OP)                                                     \
  OP(int8_t, int8_t)                                                           \
  OP(int8_t, uint8_t)                                                          \
  OP(int16_t, int16_t)                                                         \
  OP(int16_t, uint16_t)                                                        \
  OP(int32_t, int32_t)                                                         \
  OP(int32_t, uint32_t)                                                        \
  OP(int64_t, int64_t)                                                         \
  OP(int64_t, uint64_t)                                                        \
  OP(float, float)                                                             \
  OP(double, double)

#define UNWIND2_UP(OP)                                                         \
  OP(int8_t, int8_t)                                                           \
  OP(int16_t, int8_t)                                                          \
  OP(int32_t, int8_t)                                                          \
  OP(int64_t, int8_t)                                                          \
  OP(int16_t, int16_t)                                                         \
  OP(int32_t, int16_t)                                                         \
  OP(int64_t, int16_t)                                                         \
  OP(int32_t, int32_t)                                                         \
  OP(int64_t, int32_t)                                                         \
  OP(int64_t, int64_t)                                                         \
  OP(uint8_t, uint8_t)                                                         \
  OP(uint16_t, uint8_t)                                                        \
  OP(uint32_t, uint8_t)                                                        \
  OP(uint64_t, uint8_t)                                                        \
  OP(uint16_t, uint16_t)                                                       \
  OP(uint32_t, uint16_t)                                                       \
  OP(uint64_t, uint16_t)                                                       \
  OP(uint32_t, uint32_t)                                                       \
  OP(uint64_t, uint32_t)                                                       \
  OP(uint64_t, uint64_t)                                                       \
  OP(float, float)                                                             \
  OP(double, float)                                                            \
  OP(double, double)

#endif // TENSORCPU_MACRO_UNWIND_HPP
