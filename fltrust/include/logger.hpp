#pragma once

#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <ctime>
#include <unistd.h> // for getpid()

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

    // Delete copying to enforce singleton behavior.
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

private:
    Logger() {
        // Create the logs/ directory if it does not exist.
        // (This is a quick hack; in production code consider using a cross-platform API.)
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
};