#include "logger.hpp"
#include <csignal>
#include <atomic>
#include <string>
#include <thread>
#include <chrono>
#include <unistd.h>
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
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <name> <only_flt>" << std::endl;
        return 1;
    }

    std::string name = argv[1];
    int only_flt = std::stoi(argv[2]);
    CPUProfiler cpuProfiler; 
    memProfiler memoryProfiler;

    std::signal(SIGINT, signalHandler);

    cpuProfiler.startTotalMeasure();
    memoryProfiler.startMeasure();
    
    while (!stop_requested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::string header;
    if (only_flt) {
        header = "F_";
    } else {
        header = "R_";
    }

    // Log final CPU and memory utilization
    double cpuUtil = cpuProfiler.getTotalCpuUtil();
    double memUtil = memoryProfiler.getAvgMemUtil();
    std::string message = "CPU " + std::to_string(cpuUtil) + "\n";
    message += "MEM " + std::to_string(memUtil) + "\n";
    std::string filename = header + name + "_cpu_mem_util.log";
    Logger::instance().logCustom("", filename, message);
    Logger::instance().logCustom("", filename, "$ END OF EXECUTION $\n");
    
    return 0;
}