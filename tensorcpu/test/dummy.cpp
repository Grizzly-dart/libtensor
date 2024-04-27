//
// Created by tejag on 2024-04-26.
//

#include <cstdint>
#include <cxxabi.h>
#include <experimental/simd>
#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <vector>

template <typename T> void printType() {
  std::cout << abi::__cxa_demangle(typeid(T).name(), NULL, NULL, NULL);
}

namespace stdx = std::experimental;

int main() {
  using toFloat = stdx::rebind_simd_t<float, stdx::fixed_size_simd<uint8_t, 4>>;
  std::vector<uint8_t> v = {1, 2, 3, 4};
  stdx::fixed_size_simd<uint8_t, 4> a(v.data(), stdx::vector_aligned);
  auto fs = stdx::simd_cast<toFloat>(a) + 0.5f;
  std::vector<float> out(4);
  fs.copy_to(out.data(), stdx::vector_aligned);
  for (auto i : out) {
    std::cout << i << std::endl;
  }

  return 0;
}