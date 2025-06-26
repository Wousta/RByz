#include "logger.hpp"
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include "CpuProfiler.hpp"
#include "MemoryProfiler.hpp"

// Global flag to indicate when to stop
std::atomic<bool> stop_requested(false);

// Signal handler for SIGINT (Ctrl+C)
void signalHandler(int signal) {
    if (signal == SIGINT) {
        stop_requested = true;
    }
}

int main(int argc, char* argv[]) {
    CPUProfiler cpuProfiler; 
    memProfiler memoryProfiler;

    Logger::instance().log("Profiler starting execution\n");
    std::signal(SIGINT, signalHandler);

    cpuProfiler.startTotalMeasure();
    memoryProfiler.startMeasure();
    
    while (!stop_requested) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Log final CPU usage when stopping
    double cpuUtil = cpuProfiler.getTotalCpuUtil();
    double memUtil = memoryProfiler.getAvgMemUtil();
    std::string message = "CPU Utilization: "  + std::to_string(cpuUtil) + "%\n";
    message += "Memory Utilization: " + std::to_string(memUtil) + "%\n";
    Logger::instance().log(message);
    
    Logger::instance().log("Profiler execution completed\n");
    return 0;
}