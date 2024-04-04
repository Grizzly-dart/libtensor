#include <stdio.h>
#include <tensorcuda.hpp>

const auto size = 5;

int main() {
    double t1Arr[] = {1, 2, 3, 4, 5};
    double t2Arr[] = {1, 2, 3, 4, 5};
    auto t1 = makeTensor1D(size);
    writeTensor(t1, t1Arr, size);
    auto t2 = makeTensor1D(size);
    writeTensor(t2, t2Arr, size);

    auto t3 = makeTensor1D(size);
    add2D(t3, t1, t2);

    double result[size] = {};
    readTensor(t3, result, size);

    printf("Result: ");
    for (int i = 0; i < size; i++) {
        printf("%f ", result[i]);
    }
    printf("\n");

    return 0;
}
