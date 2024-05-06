//
// Created by Teja Gudapati on 2024-05-06.
//

#ifndef TENSORCPU_NN_ACTIVATION_HPP
#define TENSORCPU_NN_ACTIVATION_HPP

#include <cstdint>

template <typename T> extern void sigmoid_parallel(T *out, const T *inp, uint64_t nel);

#endif // TENSORCPU_NN_ACTIVATION_HPP
