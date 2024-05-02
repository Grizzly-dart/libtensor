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

class ThreadPool;

void threadFunc(ThreadPool& pool);

/*
class ThreadPool {
  pthread_t *threads;
  std::mutex mutex;
  std::condition_variable notifier;

public:
  ThreadPool() {
    threads = new pthread_t[std::thread::hardware_concurrency()];
    for (uint16_t i = 0; i < std::thread::hardware_concurrency(); i++) {
      pthread_create(&threads[i], nullptr, reinterpret_cast<void *(*)(void *)>(threadFunc), this);
    }
  }

  void runTasks(std::function<void()> tasks) {
    // TODO
  }

  friend void threadFunc(ThreadPool& pool) {
    std::unique_lock<std::mutex> lock(pool.mutex);
    while (true) {
      pool.notifier.wait(lock);
      // TODO
    }
  }
};
 */

class Thread {
public:
  std::thread thread;
  std::mutex mutex;
  std::condition_variable notifier;
  std::function<void(void)> task = nullptr;
  bool killed = false;
  std::chrono::steady_clock::time_point begin;
  uint16_t threadNum = 0;

  Thread() {
    thread = std::thread([this]() {
      std::unique_lock<std::mutex> lock(mutex);
      while (true) {
        notifier.wait(lock);
        if (killed) {
          return;
        }
        if (task == nullptr) {
          continue;
        }
        try {
          task();
        } catch (std::exception &e) {
        }
      }
    });
    thread.detach();
  };

  void queueTask(
      std::chrono::steady_clock::time_point beg,
      const std::function<void(void)>& task
  ) {
    begin = beg; // std::chrono::steady_clock::now();
    // mutex.lock();
    this->task = task;
    // mutex.unlock();
    notifier.notify_one();
  }

  void kill() {
    killed = true;
    begin = std::chrono::steady_clock::now();
    notifier.notify_one();
  }
};

class ThreadPool {
  std::vector<Thread> threads;

public:
  ThreadPool() : threads(std::thread::hardware_concurrency()) {
    for (uint64_t i = 0; i < threads.size(); i++) {
      threads[i].threadNum = i;
    }
  }

  void runTask(uint16_t threadNum, const std::function<void()>& task) {
    if(threadNum >= threads.size()) {
      throw std::runtime_error("Invalid exceeds maximum concurrency");
    }

    threads[threadNum].queueTask(std::chrono::steady_clock::now(), task);
  }

  void kill() {
    for (auto &thread : threads) {
      thread.kill();
    }
  }
};

extern ThreadPool pool;

#endif // TENSORCPU_THREAD_POOL_HPP
