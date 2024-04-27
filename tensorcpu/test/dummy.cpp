//
// Created by tejag on 2024-04-26.
//

#include <cxxabi.h>
#include <experimental/simd>
#include <iostream>
#include <stdfloat>

template <typename T> void printType() {
  std::cout << abi::__cxa_demangle(typeid(T).name(), NULL, NULL, NULL);
}

template <typename T> void add(T a, T b) {
  T c = a + b;
  printType<T>();
  std::cout << " => a: " << a << " b: " << b << " c: " << c << std::endl;
}

namespace stdx = std::experimental;

int main() {
  add<std::float16_t>(1.0, 2.0);
  add<std::bfloat16_t>(1.0, 2.0);

  return 0;
}