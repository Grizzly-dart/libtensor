//
// Created by Teja Gudapati on 2024-04-29.
//

#include <iostream>

#include "reducer.hpp"

void parallelSimdFold(
    uint64_t nel, uint64_t laneSize,
    const std::function<void(uint16_t, uint64_t, uint64_t)> &kernel,
    uint16_t &numThreads
) {
  uint64_t totalLanes = nel / laneSize;
  numThreads = pool.getConcurrency();
  uint64_t lanesPerThread;
  uint64_t remaining = 0;
  if (numThreads > totalLanes) {
    numThreads = totalLanes;
    lanesPerThread = 1;
  } else {
    lanesPerThread = totalLanes / numThreads;
    remaining = totalLanes % numThreads;
  }

  std::chrono::steady_clock::time_point timeStart, timeEnd;
  pool.runTask([lanesPerThread, remaining, kernel, laneSize, &timeStart,
                &timeEnd](uint64_t threadId) {
    uint64_t start = threadId * lanesPerThread;
    uint64_t last;
    if (threadId < remaining) {
      start += threadId;
      last = start + lanesPerThread + 1;
    } else {
      start += remaining;
      last = start + lanesPerThread;
    }

    kernel(threadId, start * laneSize, last * laneSize);
  });
}

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