#include <cassert>
#include <iostream>
#include <tensorcuda.hpp>

double sum(double* in, uint64_t n) {
  double sum = 0;
  for (int i = 0; i < n; i++) {
    sum += in[i];
  }
  return sum;
}

void test_sum2D(double* in, uint64_t m, uint64_t n) {
  auto t1 = makeTensor2D(m, n);
  writeTensor(t1, (double*)in, m * n);
  auto t2 = makeTensor1D(m);
  sum2DTensor(t2, t1);

  double* result = new double[m];
  readTensor(t2, (double*)result, m);

  for (int i = 0; i < m; i++) {
    double s = sum(in + i * n, n);
    double diff = std::abs(s - result[i]);
    if (diff > 1e-6) {
      std::cout << "Test failed at index " << i << " expected: " << s << " found: " << result[i] << std::endl;
      assert(false);
    }
  }
}

void testForMN(uint64_t m, uint64_t n) {
  double* in = new double[m * n];
  srand48(0);
  for (int i = 0; i < m * n; i++) {
    in[i] = drand48();
  }
  test_sum2D(in, m, n);
  std::cout << "Test passed!" << std::endl;
}

int main() {
  testForMN(10, 512 * 512 * 10);
  testForMN(10, 1000);
  testForMN(1, 31);
  testForMN(10000, 31);

  std::cout << "All tests passed!" << std::endl;
  return 0;
}