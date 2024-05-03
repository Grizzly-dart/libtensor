//
// Created by tejag on 2024-04-30.
//

#ifndef TENSORCPU_STAT_HPP
#define TENSORCPU_STAT_HPP

#include <cstdint>

template <typename O, typename I>
extern void sum_1thread(O *out, const I *inp, uint64_t nel);

template <typename O, typename I>
extern void sum_parsimd(O *out, I *inp, uint64_t nel);

#endif // TENSORCPU_STAT_HPP
