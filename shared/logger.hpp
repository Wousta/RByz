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
        logFile_ << message;
        logFile_.flush();
    }

    // CPU profiling methods
    void startCpuProfilingTotal() {
        std::lock_guard<std::mutex> lock(mtx_);
        cpuProfiler_.startTotalMeasure();
    }

    void startCpuProfilingCore() {
        std::lock_guard<std::mutex> lock(mtx_);
        int core_id = sched_getcpu();
        cpuProfiler_.startCoreMeasure(core_id);
    }

    void logCpuState(const std::string& context = "") {
        std::lock_guard<std::mutex> lock(mtx_);
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

    void openRByzAccLog() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!rbyzAccLogFile_.is_open()) {
            std::string accLogsDir = "/home/bustaman/rbyz/Results/accLogs";
            int pid = getpid();
            std::string rbyzAccFilename = accLogsDir + "/R_acc_" + std::to_string(pid) + ".log";
            rbyzAccLogFile_.open(rbyzAccFilename, std::ios::app);
            if (!rbyzAccLogFile_.is_open()) {
                std::cerr << "Failed to open RByz accuracy log file: " << rbyzAccFilename << std::endl;
            }
        }
    }

    void openFLAccLog() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!flAccLogFile_.is_open()) {
            std::string accLogsDir = "/home/bustaman/rbyz/Results/accLogs";
            int pid = getpid();
            std::string flAccFilename = accLogsDir + "/F_acc_" + std::to_string(pid) + ".log";
            flAccLogFile_.open(flAccFilename, std::ios::app);
            if (!flAccLogFile_.is_open()) {
                std::cerr << "Failed to open FL accuracy log file: " << flAccFilename << std::endl;
            }
        }
    }

    void logRByzAcc(const std::string &message) {
        std::lock_guard<std::mutex> lock(mtx_);
        rbyzAccLogFile_ << message;
        rbyzAccLogFile_.flush();
    }

    void logFLAcc(const std::string &message) {
        std::lock_guard<std::mutex> lock(mtx_);
        flAccLogFile_ << message;
        flAccLogFile_.flush();
    }

    // Delete copying to enforce singleton behavior.
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

private:
    Logger() {
        // Create the logs/ directory if it does not exist.
        std::string accLogsDir = "/home/bustaman/rbyz/Results/accLogs";
        int ret = system("mkdir -p logs");
        ret = system(("mkdir -p " + accLogsDir).c_str());

        // Use the process ID to create a unique log file name.
        int pid = getpid();
        std::string filename = "logs/execution_" + std::to_string(pid) + ".log";
        std::string rbyzAccFilename = accLogsDir + "/R_acc_" + std::to_string(pid) + ".log";
        std::string flAccFilename = accLogsDir + "/F_acc_" + std::to_string(pid) + ".log";

        logFile_.open(filename, std::ios::app);
    }

    ~Logger() {
        if (logFile_.is_open())
            logFile_.close();
        if (rbyzAccLogFile_.is_open())
            rbyzAccLogFile_.close();
        if (flAccLogFile_.is_open())
            flAccLogFile_.close();
    }

    // Helper function to get current date and time as a string.
    std::string currentDateTime() {
        std::time_t now = std::time(nullptr);
        char buffer[100];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        return std::string(buffer);
    }

    std::ofstream logFile_;
    std::ofstream rbyzAccLogFile_;
    std::ofstream flAccLogFile_;
    std::mutex mtx_;
    CPUProfiler cpuProfiler_;  // Add CPU profiler instance
};