//
// Created by Teja Gudapati on 2024-04-29.
//

#include <iostream>

#include "reducer.hpp"

void parallelFold2d(
    uint64_t rows, const std::function<void(uint64_t, uint64_t)> &kernel
) {
  uint64_t numThreads = std::thread::hardware_concurrency();
  uint64_t rowsPerThread = 1;
  uint64_t remainder = 0;
  if (numThreads > rows) {
    numThreads = rows;
  } else {
    rowsPerThread = rows / numThreads;
    remainder = rows % numThreads;
  }
  std::vector<std::future<void>> futures(numThreads);

  uint64_t start = 0;
  for (uint64_t threadNum = 0; threadNum < numThreads; threadNum++) {
    uint64_t last = start + rowsPerThread;
    if (threadNum < remainder) {
      last++;
    }

    futures[threadNum] =
        std::async(std::launch::async, [start, last, kernel]() {
          kernel(start, last);
        });
    start = last;
  }

  for (uint64_t i = 0; i < numThreads; i++) {
    futures[i].wait();
  }
}