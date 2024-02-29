#include <cassert>
#include <iostream>

#include <libgpuc_cuda.hpp>

double var(double* in, uint64_t n) {
  double mean = 0;
  double m2 = 0;
  for (int i = 0; i < n; i++) {
    double delta = in[i] - mean;
    mean += delta / (i + 1);
    m2 += delta * (in[i] - mean);
    // printf("i: %d, m2: %f\n", i, m2);
  }
  return m2 / n;
}

void test_mean2D(double* in, uint64_t m, uint64_t n) {
  auto t1 = makeTensor2D(m, n);
  writeTensor(t1, (double*)in, m * n);
  auto t2 = makeTensor1D(m);
  variance2DTensor(t2, t1);

  double* result = new double[m];
  readTensor(t2, (double*)result, m);

  for (int i = 0; i < m; i++) {
    double s = var(in + i * n, n);
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
  test_mean2D(in, m, n);
  std::cout << "Test passed!" << std::endl;
}

int main() {
  // testForMN(10, 512 * 512 * 10);
  testForMN(1, 31);

  std::cout << "All tests passed!" << std::endl;
  return 0;
}