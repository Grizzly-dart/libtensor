#include <cassert>
#include <iostream>
#include <libgpuc_cuda.hpp>

double sum(double* in, uint64_t n) {
  double sum = 0;
  for (int i = 0; i < n; i++) {
    sum += in[i];
  }
  return sum;
}

void test_rowWiseSum(double* in, uint64_t m, uint64_t n) {
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

int main() {
  uint64_t m = 10;
  uint64_t n = 512 * 512 * 10;
  double* in = new double[m * n];
  srand48(0);
  for (int i = 0; i < m * n; i++) {
    in[i] = drand48();
  }
  test_rowWiseSum(in, m, n);

  std::cout << "All tests passed!" << std::endl;
  return 0;
}