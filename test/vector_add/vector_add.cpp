#include <stdio.h>
#include <libgpuc_cuda.hpp>

const auto size = 5;

int main() {
    double t1Arr[] = {1, 2, 3, 4, 5};
    double t2Arr[] = {1, 2, 3, 4, 5};
    auto t1 = Tensor::make1D(size);
    t1.write(t1Arr, size);
    auto t2 = Tensor::make1D(size);
    t2.write(t2Arr, size);

    auto t3 = Tensor::make1D(size);
    elementwiseAdd2(t3.mem, t1.mem, t2.mem, size);

    double result[size] = {};
    t3.read(result, size);

    printf("Result: ");
    for (int i = 0; i < size; i++) {
        printf("%f ", result[i]);
    }
    printf("\n");

    return 0;
}
