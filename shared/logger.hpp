#pragma once

#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <ctime>
#include <unistd.h>
#include <sched.h>
#include <unordered_map>


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
        if (!logFile_.is_open()) {
            std::string filename = "logs/execution_" + std::to_string(pid_) + ".log";
            logFile_.open(filename, std::ios::app);
        }
        logFile_ << message;
        logFile_.flush();
    }

    void openRByzAccLog() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!rbyzAccLogFile_.is_open()) {
            std::string accLogsDir = "/home/bustaman/rbyz/Results/accLogs";
            std::string rbyzAccFilename = accLogsDir + "/R_acc_" + std::to_string(pid_) + ".log";
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
            std::string flAccFilename = accLogsDir + "/F_acc_" + std::to_string(pid_) + ".log";
            flAccLogFile_.open(flAccFilename, std::ios::app);
            if (!flAccLogFile_.is_open()) {
                std::cerr << "Failed to open FL accuracy log file: " << flAccFilename << std::endl;
            }
        }
    }

    void logAcc(int only_flt, const std::string &message) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (only_flt) {
            flAccLogFile_ << message;
            flAccLogFile_.flush();
        } else {
            rbyzAccLogFile_ << message;
            rbyzAccLogFile_.flush();
        }
    }

    void logFLAcc(const std::string &message) {
        std::lock_guard<std::mutex> lock(mtx_);
        flAccLogFile_ << message;
        flAccLogFile_.flush();
    }

    /**
     * Log a custom message to a specific file in a directory.
     * 
     * @param dir Subdirectory within Results/logs/ (e.g., "training", "metrics")
     * @param filename Log file name (e.g., "accuracy.log")
     * @param message Message to write to the file
     * 
     * Creates: /home/bustaman/rbyz/Results/logs/{dir}/{filename}
     * Files are opened in append mode. Thread-safe.
     * 
     * Example: logger.logCustom("training", "loss.log", "Epoch 1: Loss=0.5\n");
     */
    void logCustom(const std::string &dir,  std::string &filename, const std::string &message) {
        std::lock_guard<std::mutex> lock(mtx_);
        std::string dirPath = resultsDir_ + dir;
        std::string fullPath = dirPath + "/" + filename;

        if (accLogFiles_.find(fullPath) == accLogFiles_.end()) {
            int ret = system(("mkdir -p " + dirPath).c_str());
            if (ret != 0) {
                std::cerr << "Failed to create directory: " << dirPath << std::endl;
            }
            accLogFiles_[fullPath] = std::ofstream(fullPath, std::ios::app);
        }

        std::ofstream &file = accLogFiles_[fullPath];
        file << message;
        file.flush();
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
        pid_ = getpid();
        std::string rbyzAccFilename = accLogsDir + "/R_acc_" + std::to_string(pid_) + ".log";
        std::string flAccFilename = accLogsDir + "/F_acc_" + std::to_string(pid_) + ".log";

    }

    ~Logger() {
        if (logFile_.is_open())
            logFile_.close();
        if (rbyzAccLogFile_.is_open())
            rbyzAccLogFile_.close();
        if (flAccLogFile_.is_open())
            flAccLogFile_.close();

        for (auto &pair : accLogFiles_) {
            if (pair.second.is_open()) {
                pair.second.close();
            }
        }
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
    std::unordered_map<std::string, std::ofstream> accLogFiles_;
    std::string resultsDir_ = "/home/bustaman/rbyz/Results/logs/";
    int pid_;
};