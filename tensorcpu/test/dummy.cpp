//
// Created by tejag on 2024-04-26.
//

#include "thread_pool.hpp"
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <latch>
#include <queue>
#include <syncstream>
#include <thread>
#include <vector>

constexpr int iterations = 1000;
std::chrono::steady_clock::time_point schedBegins[iterations];
std::chrono::steady_clock::time_point schedEnds[iterations];
std::chrono::steady_clock::time_point threadBegins[32][iterations];
std::atomic<int16_t> jobId = -1;
std::atomic<uint16_t> total = 0;

std::mutex *mutex;
std::condition_variable *notifier;
std::mutex *mutex1;
std::condition_variable *notifier1;

std::latch *latch = nullptr;

[[noreturn]] void thfunc(void *args) {
  int threadNum = *(int *)args;
  int16_t prevJobId = -1;
  while (true) {
    std::unique_lock<std::mutex> lock(mutex[threadNum]);
    notifier[threadNum].wait(lock, [&]() { return jobId != prevJobId; });
    prevJobId = jobId;
    /*std::osyncstream(std::cout)
        << "Job " << prevJobId << " thread " << threadNum << std::endl;*/
    threadBegins[threadNum][jobId] = std::chrono::steady_clock::now();
    total++;
    latch->count_down();
    /*std::cout << "Job " << prevJobId << " thread " << threadNum << " done"
              << std::endl;*/
  }
}

int main() {
  int err = setHighestThreadPriority(pthread_self());
  if (err) {
    std::cerr << "Failed to set highest thread priority" << std::endl;
    exit(1);
  }

  mutex = new std::mutex[std::thread::hardware_concurrency()];
  notifier = new std::condition_variable[std::thread::hardware_concurrency()];
  mutex1 = new std::mutex[std::thread::hardware_concurrency()];
  notifier1 = new std::condition_variable[std::thread::hardware_concurrency()];

  pthread_t th[std::thread::hardware_concurrency()];
  int is[std::thread::hardware_concurrency()];

  for (int i = 0; i < std::thread::hardware_concurrency() - 1; i++) {
    is[i] = i;
    pthread_create(
        &th[i], nullptr, reinterpret_cast<void *(*)(void *)>(thfunc), &is[i]
    );
    err = setHighestThreadPriority(th[i]);
    if (err) {
      std::cerr << "Failed to set highest child thread priority" << std::endl;
      exit(1);
    }
  }
  std::this_thread::sleep_for(std::chrono::seconds(1));
  for (int iteration = 0; iteration < iterations; iteration++) {
    std::cout << "--------------------";
    total = 0;
    latch = new std::latch(std::thread::hardware_concurrency() - 1);
    jobId = iteration;
    schedBegins[iteration] = std::chrono::steady_clock::now();
    for (int threadNum = 0; threadNum < std::thread::hardware_concurrency() - 1;
         threadNum++) {
      notifier[threadNum].notify_one();
    }
    latch->wait();
    schedEnds[iteration] = std::chrono::steady_clock::now();
    std::cout << "Total: " << total << std::endl;
  }
  uint64_t average = 0;
  for (int iteration = 0; iteration < iterations; iteration++) {
    std::cout << "--------------------" << std::endl;
    std::cout << "Scheduling took "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     schedEnds[iteration] - schedBegins[iteration]
                 )
                     .count()
              << "us" << std::endl;
    for (int jobNum = 0; jobNum < std::thread::hardware_concurrency() - 1;
         jobNum++) {
      average += std::chrono::duration_cast<std::chrono::microseconds>(
                     threadBegins[jobNum][iteration] - schedBegins[iteration]
      )
                     .count();
      std::cout << "Thread " << jobNum << " took "
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       threadBegins[jobNum][iteration] - schedBegins[iteration]
                   )
                       .count()
                << "us" << std::endl;
    }
  }
  std::cout << "Average: "
            << average /
                   (iterations * (std::thread::hardware_concurrency() - 1))
            << "us" << std::endl;
}