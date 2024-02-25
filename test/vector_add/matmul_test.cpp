#include <libgpuc_cuda.hpp>
#include <iostream>
#include <cassert>

void test_matmulF64(double expected[2][2], double in1[2][2], double in2[2][2], uint64_t m, uint64_t n, uint64_t k) {
  auto t1 = makeTensor2D(m, n);
  writeTensor(t1, (double*)in1, m * n);
  auto t2 = makeTensor2D(n, k);
  writeTensor(t2, (double*)in2, n * k);
  auto t3 = makeTensor2D(m, k);
  matmulF64(t3, t1, t2);

  double result[2][2] = {};
  readTensor(t3, (double*)result, m * k);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      assert(result[i][j] == expected[i][j]);
    }
  }
}

int main() {
  double expected[2][2]{{19.0, 22.0}, {43.0, 50.0}};
  double in1[2][2]{{1.0, 2.0}, {3.0, 4.0}};
  double in2[2][2]{{5.0, 6.0}, {7.0, 8.0}};
  test_matmulF64(expected, in1, in2, 2, 2, 2);
  
  std::cout << "All tests passed!" << std::endl;
  return 0;
}

