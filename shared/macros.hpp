#pragma once
#include <chrono>
#include <cstdint>
#include "nlohmann/json.hpp"
#include <thread>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Timestamp = Clock::time_point;
using ns = std::chrono::nanoseconds;
using us = std::chrono::microseconds;
using ms = std::chrono::milliseconds;

#define castV(ptr) (reinterpret_cast<void *>(ptr))
#define castI(ptr) (reinterpret_cast<std::uintptr_t>(ptr))
#define castIs(ptr) (static_cast<std::uintptr_t>(ptr))
#define castu64(ptr) (reinterpret_cast<uint64_t>(ptr))
#define casti64(ptr) (static_cast<int64_t>(ptr))

#define us(num) us(num)
#define sleepns(num) std::this_thread::sleep_for(ns(num))
#define sleepus(num) std::this_thread::sleep_for(us(num))
#define sleepms(num) std::this_thread::sleep_for(ms(num))
#define LOCK_FLAGS 1
#define STOP_POS 0
#define CID_MASK 0xFFFFFFFF00000000
#define CAS_SWAP_MASK 0xFFFFFFFF00000000
#define CAS_CMP_MASK 0x00000000FFFFFFFF
#define FAA_MASK 0x00000000FFFFFFFE
