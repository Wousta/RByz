#pragma once

#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <ctime>
#include <unistd.h>
#include <sched.h>

#include "CpuProfiler.hpp"

class Logger {
public:
    // Get the singleton instance of Logger.
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    // Log a message with timestamp. Thread-safe.
    void log(const std::string &message) {
        std::lock_guard<std::mutex> lock(mtx_);
        logFile_ << currentDateTime() << " - " << message;
        logFile_.flush();
    }

    // CPU profiling methods
    void startCpuProfiling() {
        std::lock_guard<std::mutex> lock(mtx_);
        cpuProfiler_.startTotalMeasure();
        cpuProfiler_.startCoreMeasure(sched_getcpu());
    }

    void logCpuState(const std::string& context = "") {
        std::lock_guard<std::mutex> lock(mtx_);
        std::cout << "Logging CPU state" << std::endl;
        double cpuUtil = cpuProfiler_.getTotalCpuUtil();
        std::string message = "CPU Utilization";
        if (!context.empty()) {
            message += " (" + context + ")";
        }
        message += ": " + std::to_string(cpuUtil) + "%\n";
        logFile_ << currentDateTime() << " - " << message;
        logFile_.flush();
    }

    void logCoreCpuState(const std::string& context = "") {
        std::lock_guard<std::mutex> lock(mtx_);

        int core_id = sched_getcpu();
        double coreUtil = cpuProfiler_.getCoreUtil(core_id);
        std::string message = "Core " + std::to_string(core_id) + " Utilization";
        if (!context.empty()) {
            message += " (" + context + ")";
        }
        message += ": " + std::to_string(coreUtil) + "%\n";
        logFile_ << currentDateTime() << " - " << message;
        logFile_.flush();
    }

    // Delete copying to enforce singleton behavior.
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

private:
    Logger() {
        // Create the logs/ directory if it does not exist.
        system("mkdir -p logs");

        // Use the process ID to create a unique log file name.
        int pid = getpid();
        std::string filename = "logs/execution_" + std::to_string(pid) + ".log";

        logFile_.open(filename, std::ios::app);
        if (!logFile_.is_open())
            std::cerr << "Failed to open log file: " << filename << std::endl;
    }

    ~Logger() {
        if (logFile_.is_open())
            logFile_.close();
    }

    // Helper function to get current date and time as a string.
    std::string currentDateTime() {
        std::time_t now = std::time(nullptr);
        char buffer[100];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        return std::string(buffer);
    }

    std::ofstream logFile_;
    std::mutex mtx_;
    CPUProfiler cpuProfiler_;  // Add CPU profiler instance
};