//
// Created by tejag on 2024-04-26.
//

#include <cxxabi.h>
#include <experimental/simd>
#include <iostream>
#include <stdfloat>
#include <experimental/simd>
#include "typed_array.hpp"
#include <numeric>

namespace stdx = std::experimental;

int main() {
  const Caster<double>& caster = Caster<double>::lookup(f64);
  std::vector<int8_t> ints(stdx::simd_size<uint8_t>::value);
  std::cout << "ints.size(): " << ints.size() << std::endl;

  std::cout << stdx::simd_size<int64_t>::value << std::endl;

  // stdx::native_simd<int8_t> a;
  // caster.simdStorer(nullptr, 0, stdx::native_simd<double>());

  return 0;
}