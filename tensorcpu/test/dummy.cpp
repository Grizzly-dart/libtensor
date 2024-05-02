//
// Created by tejag on 2024-04-26.
//

#include <cstdint>
#include <functional>
#include <iostream>
#include <queue>
#include <thread>
#include <vector>

#include "thread_pool.hpp"

std::chrono::steady_clock::time_point begin;

std::mutex *mutex;
std::condition_variable *notifier;

[[noreturn]] void thfunc(void *args) {
  int threadNum = *(int *)args;
  while (true) {
    mutex[threadNum].lock();
    auto end = std::chrono::steady_clock::now();
    std::osyncstream(std::cout) << "Time taken: " << threadNum << " "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end - begin
                 )
                     .count()
              << std::endl;
  }
}

int main() {
  /*
  auto th = std::thread([begin]() {
    auto end = std::chrono::steady_clock::now();
    std::cout << "Time taken: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end - begin
                 )
                     .count()
              << "us" << std::endl;
  });*/
  mutex = new std::mutex[std::thread::hardware_concurrency()];
  notifier = new std::condition_variable[std::thread::hardware_concurrency()];

  pthread_t th[std::thread::hardware_concurrency()];
  int is[std::thread::hardware_concurrency()];

  for (int i = 0; i < std::thread::hardware_concurrency(); i++) {
    is[i] = i;
    pthread_create(
        &th[i], nullptr, reinterpret_cast<void *(*)(void *)>(thfunc), &is[i]
    );
  }
  for(int j = 0; j < 3; j++) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "--------------------";
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < std::thread::hardware_concurrency(); i++) {
      mutex[i].unlock();
    }
  }
  auto end = std::chrono::steady_clock::now();
  std::osyncstream(std::cout)
      << "Time taken: "
      << std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
             .count()
      << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(10));
}