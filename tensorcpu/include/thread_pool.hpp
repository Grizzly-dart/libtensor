//
// Created by tejag on 2024-05-01.
//

#ifndef TENSORCPU_THREAD_POOL_HPP
#define TENSORCPU_THREAD_POOL_HPP

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iostream>
#include <latch>
#include <pthread.h>
#include <queue>
#include <syncstream>
#include <thread>
#include <vector>

int setHighestThreadPriority(pthread_t thId);

class ThreadPool;

struct WorkerArgs {
  ThreadPool *pool;
  uint16_t threadNum;
};

void *workerFunc(WorkerArgs *arg);

/*
 * 1) Launches the first task in local thread
 * 1) Pins threads to one cores
 * 1) Uses ConditionalVariable to wakeup threads
 * 1) Uses Latch to wait for all threads to finish
 *
 * TODO:
 * 1) Busy loop before waiting for condition variable
 * 2) Only use physical cores
 */
class ThreadPool {
  uint16_t concurrency;
  std::vector<std::mutex> mutex;
  std::vector<std::condition_variable> notifier;
  std::vector<pthread_t> threads;
  int16_t jobId = -1;
  std::latch * latch = nullptr;
  std::function<void(uint64_t)> work = nullptr;
  std::atomic<bool> killed = false;
  uint16_t siblingCount = 0;

  std::chrono::steady_clock::time_point begin;
  std::vector<std::chrono::steady_clock::time_point> threadBegins;
  std::vector<std::chrono::steady_clock::time_point> threadEnds;
  int threadSchedPolicy = SCHED_OTHER;

public:
  ThreadPool()
      : concurrency(std::thread::hardware_concurrency()), mutex(concurrency),
        notifier(concurrency), threads(concurrency - 1),
        threadBegins(concurrency - 1), threadEnds(concurrency - 1) {
    int err = setHighestThreadPriority(pthread_self());
    if (err) {
      throw std::runtime_error("Failed to set thread priority");
    }

    siblingCount = std::ceil(std::sqrt(concurrency - 1));
    for (uint16_t threadNum = 0; threadNum < concurrency - 1; threadNum++) {
      auto *args = new WorkerArgs{this, threadNum};
      pthread_create(
          &threads[threadNum], nullptr,
          reinterpret_cast<void *(*)(void *)>(workerFunc), args
      );
      pthread_detach(threads[threadNum]);
      err = setHighestThreadPriority(threads[threadNum]);
      if (err != 0) {
        throw std::runtime_error("Failed to set thread priority");
      }
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(threadNum, &cpuset);
      err = pthread_setaffinity_np(
          threads[threadNum], sizeof(cpu_set_t), &cpuset
      );
      if (err != 0) {
        throw std::runtime_error("Failed to set thread affinity");
      }
    }
  }

  void runTask(const std::function<void(uint64_t)> &task) {
    work = task;
    auto l = std::latch(concurrency - 1);
    latch = &l;
    begin = std::chrono::steady_clock::now();
    jobId++;
    for (uint16_t threadNum = 0; threadNum < siblingCount; threadNum++) {
      notifier[threadNum].notify_one();
    }

    task(0);

    latch->wait();
    latch = nullptr;
    work = nullptr;
  }

  void printInfo() {
    for (uint16_t threadNum = 0; threadNum < concurrency - 1; threadNum++) {
      std::cout << "Thread " << threadNum << " started after "
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       threadBegins[threadNum] - begin
                   )
                       .count()
                << "us took "
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       threadEnds[threadNum] - threadBegins[threadNum]
                   )
                       .count()
                << "us"

                << std::endl;
    }
  }

  void kill() {
    killed = true;
    for (uint16_t threadNum = 0; threadNum < concurrency - 1; threadNum++) {
      notifier[threadNum].notify_one();
    }
  }

  friend void *workerFunc(WorkerArgs *arg) {
    ThreadPool *pool = arg->pool;
    uint16_t threadNum = arg->threadNum;
    delete arg;

    int16_t jobNum = 0;
    while (!pool->killed) {
      std::unique_lock<std::mutex> lock(pool->mutex[threadNum]);
      pool->notifier[threadNum].wait(
          lock,
          [&]() { return jobNum == pool->jobId || pool->killed; }
      );
      if (pool->killed)
        continue;
      if (pool->work == nullptr || jobNum != pool->jobId)
        continue;
      pool->threadBegins[threadNum] = std::chrono::steady_clock::now();
      if (threadNum < pool->siblingCount) {
        for (uint16_t i = 1; i < pool->siblingCount; i++) {
          pool->notifier[threadNum + i * pool->siblingCount].notify_one();
        }
      }
      jobNum++;

      try {
        pool->work(threadNum + 1);
      } catch (std::exception &e) {
      }

      pool->latch->count_down();
      pool->threadEnds[threadNum] = std::chrono::steady_clock::now();
    }
    return nullptr;
  }

  uint16_t getConcurrency() const { return concurrency; }
};

extern ThreadPool pool;

#endif // TENSORCPU_THREAD_POOL_HPP
