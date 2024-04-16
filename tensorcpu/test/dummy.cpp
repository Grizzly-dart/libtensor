#include <cstdint>

int main() {
  volatile double a = 10;
  volatile double b = 20;
  volatile double c = a + b;
}