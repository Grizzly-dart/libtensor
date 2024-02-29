#include <cassert>
#include <iostream>
#include <libgpuc_cuda.hpp>

double matmulVal(double* in1, double* in2, uint64_t m, uint64_t n, uint64_t k, uint64_t i, uint64_t j) {
  double sum = 0;
  for (int l = 0; l < n; l++) {
    sum += in1[i * n + l] * in2[l * k + j];
  }
  return sum;
}

void test_matmul(double* in1, double* in2, uint64_t m, uint64_t n, uint64_t k) {
  auto t1 = makeTensor2D(m, n);
  writeTensor(t1, (double*)in1, m * n);
  auto t2 = makeTensor2D(n, k);
  writeTensor(t2, (double*)in2, n * k);
  auto t3 = makeTensor2D(m, k);
  matmul(t3, t1, t2);

  double* result = new double[m * k];
  readTensor(t3, (double*)result, m * k);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      assert(result[i * k + j] == matmulVal(in1, in2, m, n, k, i, j));
    }
  }
  delete[] result;

  std::cout << "Test passed!" << std::endl;
}

void makeMatrix(double** m, uint64_t rows, uint64_t cols) {
  auto tmp = new double[rows * cols];
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      tmp[i * cols + j] = drand48();
    }
  }
  *m = tmp;
}

int main() {
  {
    double in1[2][2]{{1.0, 2.0}, {3.0, 4.0}};
    double in2[2][2]{{5.0, 6.0}, {7.0, 8.0}};
    test_matmulF64((double*)in1, (double*)in2, 2, 2, 2);
  }

  {
    double* in1;
    double* in2;
    makeMatrix(&in1, 512, 128);
    makeMatrix(&in2, 128, 512);
    test_matmulF64((double*)in1, (double*)in2, 2, 2, 2);
    delete[] in1;
    delete[] in2;
  }

  std::cout << "All tests passed!" << std::endl;
  return 0;
}
