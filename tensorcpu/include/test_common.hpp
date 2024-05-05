//
// Created by tejag on 2024-05-05.
//

#ifndef TENSORCPU_TEST_COMMON_HPP
#define TENSORCPU_TEST_COMMON_HPP

#include <cstdint>

template <typename T> void fillRand(T *inp, uint64_t size) {
  for (uint64_t i = 0; i < size; i++) {
    if constexpr (std::is_floating_point<T>::value) {
      inp[i] = drand48();
    } else if constexpr (std::is_signed<T>::value) {
        inp[i] = T((drand48() - 0.5) * 2 * std::numeric_limits<T>::max());
    } else {
      inp[i] = drand48() * std::numeric_limits<T>::max();
    }
  }
}

template <typename T> void fillSeq(T *inp, uint64_t size) {
  for (uint64_t i = 0; i < size; i++) {
    inp[i] = T(i);
  }
}

#endif // TENSORCPU_TEST_COMMON_HPP
