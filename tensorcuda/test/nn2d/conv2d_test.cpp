#include <cstdlib>
#include <cstdio>
#include <random>

int main() {
  std::mt19937 r(0);
  for(int i = 0; i < 10; i++) {
    printf("%f ", double(r())/0xFFFFFFFF);
  }
}