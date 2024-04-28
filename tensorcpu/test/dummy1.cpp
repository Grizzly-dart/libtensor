//
// Created by tejag on 2024-04-28.
//
#include <stdio.h>
#include <iostream>
#include <experimental/simd>

namespace stdx = std::experimental;

int main() {
  std::cout << stdx::simd_size<int8_t>::value << std::endl;
  std::cout << stdx::native_simd<int8_t>::size() << std::endl;
}