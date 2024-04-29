//
// Created by Teja Gudapati on 2024-04-29.
//

#include "reducer.hpp"

void parallelFold2d(
    uint64_t rows, const std::function<void(uint64_t, uint64_t)> &kernel
) {
  uint64_t concurrency = std::thread::hardware_concurrency();
  uint64_t rowsPerThread;
  if (concurrency > rows) {
    concurrency = rows;
    rowsPerThread = 1;
  } else {
    rowsPerThread = (rows + concurrency - 1) / concurrency;
  }
  std::vector<std::future<void>> futures(concurrency);

  for (uint64_t threadNum = 0; threadNum < concurrency; threadNum++) {
    uint64_t start = threadNum * rowsPerThread;
    uint64_t last = (threadNum + 1) * rowsPerThread;
    if (last > rows) {
      last = rows;
    }

    futures[threadNum] =
        std::async(std::launch::async, [start, last, kernel]() {
          kernel(start, last);
        });
  }

  for (uint64_t i = 0; i < concurrency; i++) {
    futures[i].wait();
  }
}