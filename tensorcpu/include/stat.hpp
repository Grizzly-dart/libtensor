//
// Created by tejag on 2024-04-30.
//

#ifndef TENSORCPU_STAT_HPP
#define TENSORCPU_STAT_HPP

#include <cstdint>

template <typename O, typename I>
extern void sum_1thread(O *out, I *inp, uint64_t nel);

template <typename O, typename I>
extern void sum_parallel(O *out, I *inp, uint64_t nel);

template <typename O, typename I>
extern void mean_1thread(O *out, I *inp, uint64_t nel);

template <typename O, typename I>
extern void mean_parallel(O *out, I *inp, uint64_t nel);

template <typename O, typename I>
extern void variance_1thread(O *out, I *inp, uint64_t nel, uint64_t correction);

template <typename O, typename I>
extern void variance_parallel(
    O *out, I *inp, uint64_t nel, uint64_t correction
);

template <typename O, typename I>
extern void sum2d_1thread(O *out, I *inp, uint64_t rows, uint64_t cols);

template <typename O, typename I>
extern void sum2d_parallel(O *out, I *inp, uint64_t rows, uint64_t cols);

template <typename O, typename I>
extern void mean2d_1thread(O *out, I *inp, uint64_t rows, uint64_t cols);

template <typename O, typename I>
extern void mean2d_parallel(O *out, I *inp, uint64_t rows, uint64_t cols);

template <typename O, typename I>
extern void variance2d_1thread(
    O *out, I *inp, uint64_t rows, uint64_t cols, uint64_t correction
);

template <typename O, typename I>
extern void variance2d_parallel(
    O *out, I *inp, uint64_t rows, uint64_t cols, uint64_t correction
);

#endif // TENSORCPU_STAT_HPP
