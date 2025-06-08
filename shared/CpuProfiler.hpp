#pragma once
#include <sys/syscall.h>  // For syscall
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "macros.hpp"
//#include "../shared/include/shared.hpp"

class CPUProfiler {
 public:
  double getCoreUtil(int core_id) {
    end.cpu_time = read_cpu_stat(core_id);

    unsigned long long total_time_1 =
        std::accumulate(start.cpu_time.begin(), start.cpu_time.end(), 0ULL);
    unsigned long long total_time_2 =
        std::accumulate(end.cpu_time.begin(), end.cpu_time.end(), 0ULL);

    unsigned long long total_time_delta = total_time_2 - total_time_1;

    unsigned long long total_idle_time_1 =
        start.cpu_time[3] + start.cpu_time[4];
    unsigned long long total_idle_time_2 = end.cpu_time[3] + end.cpu_time[4];
    unsigned long long total_idle_delta = total_idle_time_2 - total_idle_time_1;

    unsigned long long total_used_delta = total_time_delta - total_idle_delta;

    return (static_cast<double>(total_used_delta) / total_time_delta) * 100.0;
  }
  double getTotalCpuUtil() {
    end_total.cpu_time = read_cpu_stat();
    // Calculate the total CPU time spent
    unsigned long long total_time_1 = std::accumulate(
        start_total.cpu_time.begin(), start_total.cpu_time.end(), 0ULL);
    unsigned long long total_time_2 = std::accumulate(
        end_total.cpu_time.begin(), end_total.cpu_time.end(), 0ULL);
    // get cpu_delta total time
    unsigned long long total_time_delta = total_time_2 - total_time_1;
    // Calculate the total time spent in all idle modes (including iowait)
    unsigned long long total_idle_time_1 =
        start_total.cpu_time[3] + start_total.cpu_time[4];  // idle + iowait
    unsigned long long total_idle_time_2 =
        end_total.cpu_time[3] + end_total.cpu_time[4];  // idle + iowait
    // get delta idle
    unsigned long long total_idle_delta = total_idle_time_2 - total_idle_time_1;
    // get cpu_used delta
    unsigned long long total_used_delta = total_time_delta - total_idle_delta;
    // Calculate CPU utilization percentage
    double cpu_utilization =
        (static_cast<double>(total_used_delta) / total_time_delta) * 100.0;

    return cpu_utilization;
  }
  void startTotalMeasure() { start_total.cpu_time = read_cpu_stat(); }
  void startCoreMeasure(int core_id) {
    start.cpu_time = read_cpu_stat(core_id);
  }

 private:
  struct data_thrd {
    Timestamp ts;
    uint64_t cpu_time = 0;
  };
  struct data_total {
    std::vector<unsigned long long> cpu_time;
  };
  data_total start;
  data_total end;
  data_total start_total;
  data_total end_total;

  inline uint64_t getThreadCpuTime() {
    std::ifstream statFile("/proc/self/task/" + tidStr() + "/stat");
    if (!statFile.is_open()) {
      std::cerr << "Failed to open thread stat file: " << strerror(errno)
                << std::endl;
      throw std::runtime_error("Failed to open thread stat file");
    }
    std::string statLine;
    std::getline(statFile, statLine);
    statFile.close();

    // Parse the 14th and 15th fields for user and system CPU time
    std::istringstream iss(statLine);
    std::string field;
    for (int i = 1; i <= 13; ++i) {
      iss >> field;
    }
    uint64_t utime, stime;
    iss >> utime >> stime;

    return utime + stime;
  }
  inline std::vector<unsigned long long> read_cpu_stat() {
    std::ifstream stat_file("/proc/stat");
    std::string line;

    // Read the first line of /proc/stat which contains total CPU statistics
    std::getline(stat_file, line);
    std::istringstream iss(line);

    // Skip the "cpu" string
    std::string cpu_label;
    iss >> cpu_label;

    // Read the CPU statistics
    std::vector<unsigned long long> cpu_stats;
    unsigned long long value;
    while (iss >> value) {
      cpu_stats.push_back(value);
    }

    return cpu_stats;
  }
  inline std::vector<unsigned long long> read_cpu_stat(int core_id) {
    std::ifstream stat_file("/proc/stat");
    std::string line;
    std::string target1 = "cpu" + std::to_string(core_id);
    std::string target2 = "cpu" + std::to_string(core_id + 16);

    std::vector<unsigned long long> stats1, stats2;

    while (std::getline(stat_file, line)) {
      std::istringstream iss(line);
      std::string label;
      iss >> label;

      if (label == target1 || label == target2) {
        std::vector<unsigned long long> current_stats;
        unsigned long long value;
        while (iss >> value) {
          current_stats.push_back(value);
        }

        if (label == target1)
          stats1 = std::move(current_stats);
        else if (label == target2)
          stats2 = std::move(current_stats);
      }

      if (!stats1.empty() && !stats2.empty()) break;
    }

    if (stats1.empty() || stats2.empty()) {
      throw std::runtime_error("Could not read stats for both hyperthreads");
    }

    std::vector<unsigned long long> combined;
    size_t size = std::min(stats1.size(), stats2.size());
    for (size_t i = 0; i < size; ++i) {
      combined.push_back(stats1[i] + stats2[i]);
    }

    return combined;
  }
  inline std::string tidStr() {
    // std::ostringstream oss;
    // oss << std::this_thread::get_id();
    pid_t tid = syscall(SYS_gettid);
    std::cout << tid << "\n";
    return std::to_string(tid);
  }
};