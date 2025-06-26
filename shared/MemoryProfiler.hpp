#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

class memProfiler {
 public:
  memProfiler() { is_stop.store(false, std::memory_order_release); }
  void startMeasure() {
    profiler = std::thread(&memProfiler::runProfiler, this);
  }
  double getAvgMemUtil() {
    is_stop.store(true, std::memory_order_release);
    if (profiler.joinable()) profiler.join();
    return mem_avg_util;
  }
  ~memProfiler() {
    if (profiler.joinable()) profiler.join();
  }
 private:
  std::thread profiler;
  double mem_avg_util = 0.0;
  std::atomic<bool> is_stop;
  void runProfiler() {
    cpu_set_t cpuSet;
    CPU_ZERO(&cpuSet);
    CPU_SET(15, &cpuSet);
    CPU_SET(31, &cpuSet);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuSet) != 0)
      std::cerr << "Failed to set CPU affinity in main" << std::endl;
    double mem_util = 0.0;
    long int count = 0;
    while (!is_stop.load(std::memory_order_acquire)) {
      mem_util += getMemUtil();
      count++;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    // avg mem utilization
    mem_avg_util = mem_util / count;
  }
  double getMemUtil() {
    // RAM
    long long total_mem = read_MemInfo("MemTotal:");
    long long free_mem = read_MemInfo("MemFree:");
    double usedMemoryPercentage = 100.0 * (total_mem - free_mem) / total_mem;
    return usedMemoryPercentage;
  }
  long long read_MemInfo(const std::string& parameter) {
    std::ifstream memInfoFile("/proc/meminfo");
    std::string line;
    while (std::getline(memInfoFile, line)) {
      if (line.find(parameter) != std::string::npos) {
        std::istringstream iss(line);
        std::string paramName;
        long long paramValue;
        iss >> paramName >> paramValue;
        return paramValue;
      }
    }
    return -1;  // Parameter not found
  }
};