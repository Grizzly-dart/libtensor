//
// Created by Teja Gudapati on 2024-05-03.
//

#ifndef TENSORCPU_DEBUG_HPP
#define TENSORCPU_DEBUG_HPP

#include <cstdint>

constexpr uint8_t kDebugLevelWarn = 1;
constexpr uint8_t kDebugLevelInfo = 2;
constexpr uint8_t kDebugLevelDebug = 3;
constexpr uint8_t kDebugLevelVerbose = 4;

extern bool kDebug;

extern uint8_t kDebugLevel;

#endif // TENSORCPU_DEBUG_HPP
