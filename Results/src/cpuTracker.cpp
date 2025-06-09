#include "logger.hpp"
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>

// Global flag to indicate when to stop
std::atomic<bool> stop_requested(false);

// Signal handler for SIGINT (Ctrl+C)
void signalHandler(int signal) {
    if (signal == SIGINT) {
        stop_requested = true;
    }
}

int main(int argc, char* argv[]) {
    Logger::instance().log("CPU Tracker starting execution\n");

    // Register signal handler for SIGINT
    std::signal(SIGINT, signalHandler);

    Logger::instance().startCpuProfilingTotal();
    
    while (!stop_requested) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Log final CPU usage when stopping
    Logger::instance().logCpuState("Final CPU Usage");
    
    Logger::instance().log("CPU Tracker execution completed\n");
    return 0;
}