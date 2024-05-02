//
// Created by tejag on 2024-05-01.
//

#ifndef TENSORCPU_THREAD_POOL_HPP
#define TENSORCPU_THREAD_POOL_HPP

#include <chrono>
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

class ThreadPool {
  std::vector<std::mutex> mutex;
  std::vector<std::condition_variable> notifier;
  std::vector<pthread_t> threads;
  std::atomic<int16_t> jobId = -1;
  std::atomic<std::latch *> latch = nullptr;
  std::function<void(uint64_t)> work = nullptr;
  std::atomic<bool> killed = false;
  std::vector<std::chrono::steady_clock::time_point> threadBegins;
  std::vector<std::chrono::steady_clock::time_point> threadEnds;

public:
  ThreadPool()
      : mutex(std::thread::hardware_concurrency() - 1),
        notifier(std::thread::hardware_concurrency() - 1),
        threads(std::thread::hardware_concurrency() - 1),
        threadBegins(std::thread::hardware_concurrency() - 1),
        threadEnds(std::thread::hardware_concurrency() - 1) {
    for (uint16_t threadNum = 0;
         threadNum < std::thread::hardware_concurrency() - 1; threadNum++) {
      WorkerArgs *args = new WorkerArgs{this, threadNum};
      pthread_create(
          &threads[threadNum], nullptr,
          reinterpret_cast<void *(*)(void *)>(workerFunc), args
      );
      pthread_detach(threads[threadNum]);
      int err = setHighestThreadPriority(threads[threadNum]);
      if (err != 0) {
        throw std::runtime_error("Failed to set thread priority");
      }
      threads.push_back(threads[threadNum]);
    }
  }

  void runTask(const std::function<void(uint64_t)> &task) {
    work = task;
    latch = new std::latch(std::thread::hardware_concurrency() - 1);
    auto begin = std::chrono::steady_clock::now();
    jobId++;
    for (uint16_t threadNum = 0;
         threadNum < std::thread::hardware_concurrency() - 1; threadNum++) {
      notifier[threadNum].notify_one();
    }

    task(std::thread::hardware_concurrency() - 1);

    (*latch).wait();
    latch = nullptr;
    work = nullptr;
    auto end = std::chrono::steady_clock::now();
    std::cout << "Time taken: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end - begin
                 )
                     .count()
              << "us" << std::endl;
    for (uint16_t threadNum = 0;
         threadNum < std::thread::hardware_concurrency() - 1; threadNum++) {
      std::cout << "Thread " << threadNum << " started after "
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       threadBegins[threadNum] - begin
                   )
                       .count()
                << "us" << std::endl;
      std::cout << "Thread " << threadNum << " took "
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       threadEnds[threadNum] - threadBegins[threadNum]
                   )
                       .count()
                << "us" << std::endl;
    }
  }

  void kill() {
    killed = true;
    for (uint16_t threadNum = 0;
         threadNum < std::thread::hardware_concurrency() - 1; threadNum++) {
      notifier[threadNum].notify_one();
    }
  }

  friend void *workerFunc(WorkerArgs *arg) {
    ThreadPool *pool = arg->pool;
    uint16_t threadNum = arg->threadNum;
    delete arg;

    int16_t jobId = 0;
    while (!pool->killed) {
      std::unique_lock<std::mutex> lock(pool->mutex[threadNum]);
      pool->notifier[threadNum].wait(lock, [&]() {
        return jobId == pool->jobId || pool->killed;
      });
      if (pool->killed) continue;
      if (pool->work == nullptr) continue;
      pool->threadBegins[threadNum] = std::chrono::steady_clock::now();
      jobId++;

      try {
        pool->work(threadNum);
      } catch (std::exception &e) {
      }

      (*pool->latch).count_down();
      pool->threadEnds[threadNum] = std::chrono::steady_clock::now();
    }
    return nullptr;
  }
};

extern ThreadPool pool;

#endif // TENSORCPU_THREAD_POOL_HPP
