//
// Created by Teja Gudapati on 2024-04-29.
//

#include <iostream>

#include "reducer.hpp"

void parallelSimdTransform(
    uint64_t nel, uint64_t laneSize,
    const std::function<void(uint64_t, uint64_t)> &kernel
) {
  uint64_t totalLanes = nel / laneSize;

  if (totalLanes == 0) {
    return;
  }

  uint64_t numThreads = pool.getConcurrency();
  uint64_t lanesPerThread = 1;
  uint64_t remaining = 0;
  if (numThreads > totalLanes) {
    numThreads = totalLanes;
  } else {
    lanesPerThread = totalLanes / numThreads;
    remaining = totalLanes % numThreads;
  }

  pool.runTask([lanesPerThread, remaining, kernel,
                laneSize](uint64_t threadId) {
    uint64_t start = threadId * lanesPerThread;
    uint64_t last;
    if (threadId < remaining) {
      start += threadId;
      last = start + lanesPerThread + 1;
    } else {
      start += remaining;
      last = start + lanesPerThread;
    }

    kernel(start * laneSize, last * laneSize);
  });
}

void parallelSimdFold(
    uint64_t nel, uint64_t laneSize,
    const std::function<void(uint16_t, uint64_t, uint64_t)> &kernel,
    uint16_t &numThreads
) {
  uint64_t totalLanes = nel / laneSize;

  if (totalLanes == 0) {
    numThreads = 0;
    return;
  }

  numThreads = pool.getConcurrency();
  uint64_t lanesPerThread = 1;
  uint64_t remaining = 0;
  if (numThreads > totalLanes) {
    numThreads = totalLanes;
  } else {
    lanesPerThread = totalLanes / numThreads;
    remaining = totalLanes % numThreads;
  }

  pool.runTask([lanesPerThread, remaining, kernel,
                laneSize](uint64_t threadId) {
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
    uint64_t rows,
    const std::function<
        void(uint16_t threadId, uint64_t startRow, uint64_t endRow)> &kernel
) {
  uint16_t numThreads = std::thread::hardware_concurrency();
  uint64_t rowsPerThread = 1;
  uint64_t remainder = 0;
  if (numThreads > rows) {
    numThreads = rows;
  } else {
    rowsPerThread = rows / numThreads;
    remainder = rows % numThreads;
  }

  if (numThreads == 0) {
    return;
  }
  pool.runTask([rowsPerThread, remainder, kernel](uint64_t threadId) {
    uint64_t startRow = threadId * rowsPerThread;
    uint64_t endRow;
    if (threadId < remainder) {
      startRow += threadId;
      endRow = startRow + rowsPerThread + 1;
    } else {
      startRow += remainder;
      endRow = startRow + rowsPerThread;
    }
    kernel(threadId, startRow, endRow);
  });
}