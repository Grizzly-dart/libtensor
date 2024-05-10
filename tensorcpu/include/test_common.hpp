//
// Created by tejag on 2024-05-05.
//

#ifndef TENSORCPU_TEST_COMMON_HPP
#define TENSORCPU_TEST_COMMON_HPP

#include <cstdint>
#include <thread>

template <typename T> void fillRand(T *inp, uint64_t size) {
  for (uint64_t i = 0; i < size; i++) {
    if constexpr (std::is_floating_point<T>::value) {
      inp[i] = drand48();
    } else if constexpr (std::is_signed<T>::value) {
      inp[i] = T((drand48() - 0.5) * 2 * std::numeric_limits<T>::max());
    } else {
      inp[i] = drand48() * std::numeric_limits<T>::max();
    }
  }
}

template <typename T> void fillSeq(T *inp, uint64_t size) {
  for (uint64_t i = 0; i < size; i++) {
    inp[i] = T(i);
  }
}

void make1dTestSizes(std::vector<uint64_t> &sizes, uint64_t laneSize) {
  sizes.resize(0);
  for (uint64_t i = 1; i < laneSize * 3; i++) {
    sizes.push_back(i);
  }
  uint64_t concurrency = std::thread::hardware_concurrency();
  for (uint64_t i = 1; i < concurrency * 2; i++) {
    sizes.push_back(i * laneSize - 1);
    sizes.push_back(i * laneSize);
    sizes.push_back(i * laneSize + 1);
  }

  {
    uint64_t tmp[] = {
        laneSize * 10000 - 1, (laneSize - 1) * 100000 - 1,
        (laneSize - 1) * 1000000 - 1
    };
    sizes.insert(sizes.end(), tmp, tmp + sizeof(tmp) / sizeof(tmp[0]));
  }
  {
    uint64_t tmp[] = {laneSize * 10000, laneSize * 100000, laneSize * 1000000};
    sizes.insert(sizes.end(), tmp, tmp + sizeof(tmp) / sizeof(tmp[0]));
  }
  {
    uint64_t tmp[] = {
        laneSize * 10000 + 1, laneSize * 100000 + 1, laneSize * 1000000 + 1
    };
    sizes.insert(sizes.end(), tmp, tmp + sizeof(tmp) / sizeof(tmp[0]));
  }
}

void make1dBenchSizes(std::vector<uint64_t> &sizes, uint64_t laneSize) {
  uint64_t con = std::thread::hardware_concurrency();
  sizes.resize(0);
  for(uint64_t i = 1; i < laneSize * 3; i++) {
    sizes.push_back(i);
  }
  for (uint64_t size : {con * laneSize * 2, con * laneSize * 5, con * laneSize * 10}) {
    sizes.push_back(size);
  }
  {
    uint64_t tmp[] = {
        laneSize * 10000 - 1, (laneSize - 1) * 100000 - 1,
        (laneSize - 1) * 1000000 - 1
    };
    sizes.insert(sizes.end(), tmp, tmp + sizeof(tmp) / sizeof(tmp[0]));
  }
  {
    uint64_t tmp[] = {laneSize * 10000, laneSize * 100000, laneSize * 1000000};
    sizes.insert(sizes.end(), tmp, tmp + sizeof(tmp) / sizeof(tmp[0]));
  }
  {
    uint64_t tmp[] = {
        laneSize * 10000 + 1, laneSize * 100000 + 1, laneSize * 1000000 + 1
    };
    sizes.insert(sizes.end(), tmp, tmp + sizeof(tmp) / sizeof(tmp[0]));
  }
}

void make2dTestSizes(std::vector<Dim2> &sizes, uint64_t laneSize) {
  uint64_t concurrency = std::thread::hardware_concurrency();

  sizes.resize(0);
  for (uint64_t col = 1; col < laneSize * 3; col++) {
    for (uint64_t row = 1; row < concurrency * 3; row++) {
      sizes.push_back(Dim2(row, col));
    }
  }
  for (uint64_t col :
       {laneSize * 1000 - 1, laneSize * 1000, laneSize * 1000 + 1}) {
    for (uint64_t row :
         {concurrency * 100 - 1, concurrency * 100, concurrency * 100 + 1}) {
      sizes.push_back(Dim2(row, col));
    }
  }
}

#endif // TENSORCPU_TEST_COMMON_HPP
