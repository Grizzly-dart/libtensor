//
// Created by tejag on 2024-05-01.
//

#ifndef TENSORCPU_THREAD_POOL_HPP
#define TENSORCPU_THREAD_POOL_HPP

#include "debug.hpp"
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

extern int pinThreadToCore(pthread_t thread, int coreId);

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
 * 3) Local thread should do more job
 */
class ThreadPool {
  uint16_t concurrency;
  std::vector<std::mutex> mutex;
  std::vector<std::condition_variable> notifier;
  std::vector<pthread_t> threads;
  std::atomic<int16_t> jobId = -1;
  std::atomic<uint16_t> finished = 0;
  std::function<void(uint64_t)> work = nullptr;
  std::atomic<bool> killed = false;
  uint16_t siblingCount = 0;
  std::vector<std::atomic<bool>> pending;

  std::chrono::steady_clock::time_point begin, end1, end2;
  // std::vector<std::chrono::steady_clock::time_point> threadBegins,
  // threadEnds;
  int threadSchedPolicy = SCHED_OTHER;

public:
  ThreadPool()
      : concurrency(std::thread::hardware_concurrency()), mutex(concurrency),
        notifier(concurrency), threads(concurrency - 1),
        // threadBegins(concurrency - 1), threadEnds(concurrency - 1),
        pending(std::vector<std::atomic<bool>>(concurrency - 1)) {
    int err = setHighestThreadPriority(pthread_self());
    if (err) {
      throw std::runtime_error("Failed to set thread priority");
    }
    siblingCount = std::ceil(std::sqrt(concurrency - 1));
    if (kDebug && kDebugLevel >= kDebugLevelVerbose) {
      std::cout << "Sibling count: " << siblingCount << std::endl;
    }
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
      /*err = pinThreadToCore(threads[threadNum], threadNum + 1);
      if (err != 0) {
        throw std::runtime_error("Failed to set thread affinity");
      }*/
    }
  }

  void runTask(const std::function<void(uint64_t)> &task) {
    work = task;
    finished = 0;
    begin = std::chrono::steady_clock::now();
    for (uint16_t threadNum = 0; threadNum < concurrency - 1; threadNum++) {
      pending[threadNum] = true;
    }
    jobId++;
    for (uint16_t threadNum = 0; threadNum < concurrency - 1; threadNum++) {
      notifier[threadNum].notify_one();
    }

    task(0);

    for (uint16_t threadNum = 0; threadNum < concurrency - 1; threadNum++) {
      bool expected = true;
      if (pending[threadNum].compare_exchange_strong(
              expected, false, std::memory_order_seq_cst
          )) {
        task(threadNum + 1);
        finished++;
      }
    }

    end1 = std::chrono::steady_clock::now();
    while (finished < concurrency - 1) {
    }
    work = nullptr;
    end2 = std::chrono::steady_clock::now();
  }

  void printInfo() {
    /*std::cout << "Total time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end1 - begin
                 )
                     .count()
              << "us" << std::endl;
    std::cout << "Latch time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end2 - end1
                 )
                     .count()
              << "us" << std::endl;

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
                << "us" << std::endl;
    }*/
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
      int16_t curJob;
      for (uint64_t waitCycles1 = 0; waitCycles1 < 100; waitCycles1++) {
        for (uint64_t waitCycles2 = 0; waitCycles2 < 1000; waitCycles2++) {
          curJob = pool->jobId.load(std::memory_order_relaxed);
          if (curJob != jobNum - 1 || pool->killed) {
            goto here;
          }
        }
      }

      {
        std::unique_lock<std::mutex> lock(pool->mutex[threadNum]);
        pool->notifier[threadNum].wait_for(
            lock, std::chrono::microseconds(1),
            [&]() {
              curJob = pool->jobId.load(std::memory_order_relaxed);
              return curJob != jobNum - 1 || pool->killed;
            }
        );
      }
    here:
      if (pool->killed) continue;
      curJob = pool->jobId.load(std::memory_order_relaxed);
      if (pool->work == nullptr || jobNum - 1 == curJob) continue;
      jobNum = curJob + 1;

      // pool->threadBegins[threadNum] = std::chrono::steady_clock::now();

      // wake up siblings
      for (int16_t threadToWake = pool->concurrency - 1; threadToWake >= 0;
           threadToWake--) {
        pool->notifier[threadToWake].notify_one();
      }
      uint16_t did = 0;
      for (uint16_t borrowedTask = 0; borrowedTask < pool->concurrency - 1;
           borrowedTask++) {
        bool expected = true;
        if (pool->pending[borrowedTask].compare_exchange_strong(
                expected, false, std::memory_order_seq_cst
            )) {
          did++;
          try {
            pool->work(borrowedTask + 1);
          } catch (std::exception &e) {
          }

          pool->finished++;
        }
      }
      // pool->threadEnds[threadNum] = std::chrono::steady_clock::now();
    }
    return nullptr;
  }

  [[nodiscard]] uint16_t getConcurrency() const { return concurrency; }
};

extern ThreadPool pool;

#endif // TENSORCPU_THREAD_POOL_HPP
