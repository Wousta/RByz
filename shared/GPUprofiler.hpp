#pragma once
#include </usr/local/cuda-12.9/targets/x86_64-linux/include/cuda_runtime.h>
#include </usr/local/cuda-12.9/targets/x86_64-linux/include/nvml.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

class GPUProfiler {
public:
  GPUProfiler() : is_stop(false), device_id(0), use_nvml(true) {
    // Initialize NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
      std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result)
                << ". Falling back to nvidia-smi method." << std::endl;
      nvml_initialized = false;
      use_nvml = false;
    } else {
      nvml_initialized = true;
      // Get device handle for GPU 0
      result = nvmlDeviceGetHandleByIndex(0, &device);
      if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get device handle: " << nvmlErrorString(result)
                  << ". Falling back to nvidia-smi method." << std::endl;
        nvml_initialized = false;
        use_nvml = false;
      }
    }
  }

  ~GPUProfiler() {
    if (profiler.joinable()) {
      profiler.join();
    }
    if (nvml_initialized) {
      nvmlShutdown();
    }
  }

  void startMeasure() {
    if (!nvml_initialized && !use_nvml) {
      std::cerr << "GPU profiling methods not available" << std::endl;
      return;
    }
    is_stop.store(false, std::memory_order_release);
    profiler = std::thread(&GPUProfiler::runProfiler, this);
  }

  double getAvgGpuUtil() {
    is_stop.store(true, std::memory_order_release);
    if (profiler.joinable()) {
      profiler.join();
    }
    return avg_gpu_util;
  }

  double getAvgMemUtil() { return avg_mem_util; }

  double getCurrentGpuUtil() {
    if (nvml_initialized) {
      return getCurrentGpuUtilNVML();
    } else {
      return getCurrentGpuUtilSMI();
    }
  }

  double getCurrentMemUtil() {
    if (nvml_initialized) {
      return getCurrentMemUtilNVML();
    } else {
      return getCurrentMemUtilSMI();
    }
  }

  size_t getTotalMemory() {
    if (!nvml_initialized)
      return getTotalMemorySMI();

    nvmlMemory_t memory;
    nvmlReturn_t result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (result != NVML_SUCCESS) {
      std::cerr << "Failed to get GPU memory info: " << nvmlErrorString(result)
                << std::endl;
      return getTotalMemorySMI();
    }
    return memory.total;
  }

  size_t getUsedMemory() {
    if (!nvml_initialized)
      return getUsedMemorySMI();

    nvmlMemory_t memory;
    nvmlReturn_t result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (result != NVML_SUCCESS) {
      std::cerr << "Failed to get GPU memory info: " << nvmlErrorString(result)
                << std::endl;
      return getUsedMemorySMI();
    }
    return memory.used;
  }

  int getTemperature() {
    if (!nvml_initialized)
      return getTemperatureSMI();

    unsigned int temp;
    nvmlReturn_t result =
        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
    if (result != NVML_SUCCESS) {
      std::cerr << "Failed to get GPU temperature: " << nvmlErrorString(result)
                << std::endl;
      return getTemperatureSMI();
    }
    return static_cast<int>(temp);
  }

  unsigned int getPowerUsage() {
    if (!nvml_initialized)
      return getPowerUsageSMI();

    unsigned int power;
    nvmlReturn_t result = nvmlDeviceGetPowerUsage(device, &power);
    if (result != NVML_SUCCESS) {
      std::cerr << "Failed to get GPU power usage: " << nvmlErrorString(result)
                << std::endl;
      return getPowerUsageSMI();
    }
    return power; // Returns power in milliwatts
  }

private:
  std::thread profiler;
  std::atomic<bool> is_stop;
  int device_id;
  nvmlDevice_t device;
  bool nvml_initialized;
  bool use_nvml;

  double avg_gpu_util = 0.0;
  double avg_mem_util = 0.0;

  void runProfiler() {
    double total_gpu_util = 0.0;
    double total_mem_util = 0.0;
    long int count = 0;

    while (!is_stop.load(std::memory_order_acquire)) {
      double gpu_util = getCurrentGpuUtil();
      double mem_util = getCurrentMemUtil();

      if (gpu_util >= 0) {
        total_gpu_util += gpu_util;
      }
      if (mem_util >= 0) {
        total_mem_util += mem_util;
      }

      count++;
      std::this_thread::sleep_for(
          std::chrono::milliseconds(100)); // Sample every 100ms
    }

    if (count > 0) {
      avg_gpu_util = total_gpu_util / count;
      avg_mem_util = total_mem_util / count;
    }
  }

  // NVML-based methods
  double getCurrentGpuUtilNVML() {
    nvmlUtilization_t utilization;
    nvmlReturn_t result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (result != NVML_SUCCESS) {
      std::cerr << "Failed to get GPU utilization: " << nvmlErrorString(result)
                << std::endl;
      return -1.0;
    }
    return static_cast<double>(utilization.gpu);
  }

  double getCurrentMemUtilNVML() {
    nvmlMemory_t memory;
    nvmlReturn_t result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (result != NVML_SUCCESS) {
      std::cerr << "Failed to get GPU memory info: " << nvmlErrorString(result)
                << std::endl;
      return -1.0;
    }
    return (static_cast<double>(memory.used) / memory.total) * 100.0;
  }

  // nvidia-smi fallback methods
  double getCurrentGpuUtilSMI() {
    std::string command = "nvidia-smi --query-gpu=utilization.gpu "
                          "--format=csv,noheader,nounits -i 0";
    return executeAndParseDouble(command);
  }

  double getCurrentMemUtilSMI() {
    std::string command = "nvidia-smi --query-gpu=utilization.memory "
                          "--format=csv,noheader,nounits -i 0";
    return executeAndParseDouble(command);
  }

  size_t getTotalMemorySMI() {
    std::string command = "nvidia-smi --query-gpu=memory.total "
                          "--format=csv,noheader,nounits -i 0";
    double result = executeAndParseDouble(command);
    return static_cast<size_t>(result * 1024 * 1024); // Convert MB to bytes
  }

  size_t getUsedMemorySMI() {
    std::string command =
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0";
    double result = executeAndParseDouble(command);
    return static_cast<size_t>(result * 1024 * 1024); // Convert MB to bytes
  }

  int getTemperatureSMI() {
    std::string command = "nvidia-smi --query-gpu=temperature.gpu "
                          "--format=csv,noheader,nounits -i 0";
    return static_cast<int>(executeAndParseDouble(command));
  }

  unsigned int getPowerUsageSMI() {
    std::string command =
        "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0";
    double result = executeAndParseDouble(command);
    return static_cast<unsigned int>(result * 1000); // Convert W to mW
  }

  double executeAndParseDouble(const std::string &command) {
    std::string result = executeCommand(command);
    if (result.empty())
      return -1.0;

    // Simple parsing - if it fails, return -1.0
    char *endptr;
    double value = std::strtod(result.c_str(), &endptr);
    if (endptr == result.c_str() || *endptr != '\0') {
      std::cerr << "Failed to parse GPU metric from: " << result << std::endl;
      return -1.0;
    }
    return value;
  }

  std::string executeCommand(const std::string &command) {
    char buffer[128];
    std::string result = "";
    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe) {
      std::cerr << "Failed to execute command: " << command << std::endl;
      return "";
    }

    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }

    pclose(pipe);

    // Remove trailing newline
    if (!result.empty() && result.back() == '\n') {
      result.pop_back();
    }

    return result;
  }
};