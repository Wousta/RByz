#include "FlippableMNIST.hpp"
#include "logger.hpp"
#include <algorithm>
#include <iomanip>
#include <sstream>

FlippableMNIST::ExampleType FlippableMNIST::get(size_t index) {
    auto example = mnist_dataset.get(index);
    
    // Check if this index has a flipped label
    std::unordered_map<size_t, int64_t>::iterator it = flipped_labels.find(index);
    if (it != flipped_labels.end()) {
        // Create a new example with the flipped label
        return {example.data, torch::tensor(it->second, torch::kInt64)};
    }
    
    return example;
}

torch::optional<size_t> FlippableMNIST::size() const {
    return mnist_dataset.size();
}

void FlippableMNIST::flipLabelsRandom(float flip_ratio, std::mt19937& rng) {
    if (flip_ratio <= 0.0f || flip_ratio > 1.0f) {
        std::ostringstream oss;
        oss << "Invalid flip_ratio: " << flip_ratio << ". Must be in (0, 1]";
        Logger::instance().log(oss.str() + "\n");
        return;
    }
    
    std::uniform_real_distribution<float> flip_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> label_dist(0, 9);
    
    size_t dataset_size = mnist_dataset.size().value();
    size_t flipped_count = 0;
    
    // Clear existing flipped labels to avoid conflicts
    flipped_labels.clear();
    
    for (size_t i = 0; i < dataset_size; ++i) {
        if (flip_dist(rng) < flip_ratio) {
            auto original_example = mnist_dataset.get(i);
            int original_label = original_example.target.item<int64_t>();
            int new_label;
            
            // Generate a different random label
            do {
                new_label = label_dist(rng);
            } while (new_label == original_label);
            
            flipped_labels[i] = new_label;
            flipped_count++;
        }
    }
    
    std::ostringstream oss;
    oss << "Random label flipping completed: " << flipped_count << "/" << dataset_size 
        << " labels flipped (" << (100.0f * flipped_count / dataset_size) << "%)";
    Logger::instance().log(oss.str() + "\n");
}

void FlippableMNIST::flipLabelsTargeted(int source_label, int target_label, float flip_ratio, std::mt19937& rng) {
    if (flip_ratio <= 0.0f || flip_ratio > 1.0f) {
        std::ostringstream oss;
        oss << "Invalid flip_ratio: " << flip_ratio << ". Must be in (0, 1]";
        Logger::instance().log(oss.str() + "\n");
        return;
    }
    
    if (source_label < 0 || source_label > 9 || target_label < 0 || target_label > 9) {
        Logger::instance().log("Invalid labels. MNIST labels must be in range [0, 9]\n");
        return;
    }
    
    if (source_label == target_label) {
        Logger::instance().log("Source and target labels are the same. No flipping needed.\n");
        return;
    }
    
    std::uniform_real_distribution<float> flip_dist(0.0f, 1.0f);
    
    size_t dataset_size = mnist_dataset.size().value();
    size_t source_count = 0;
    size_t flipped_count = 0;
    
    // Don't clear existing flipped labels - allow multiple targeted attacks
    for (size_t i = 0; i < dataset_size; ++i) {
        // Get the current effective label (considering any previous flips)
        auto example = get(i);
        int current_label = example.target.item<int64_t>();
        
        if (current_label == source_label) {
            source_count++;
            if (flip_dist(rng) < flip_ratio) {
                flipped_labels[i] = target_label;
                flipped_count++;
            }
        }
    }
    
    std::ostringstream oss;
    oss << "Targeted label flipping completed: " << flipped_count << "/" << source_count 
        << " instances of label " << source_label << " flipped to " << target_label << " (";
    if (source_count > 0) {
        oss << (100.0f * flipped_count / source_count);
    } else {
        oss << "0.0";
    }
    oss << "%)";
    Logger::instance().log(oss.str() + "\n");
}

void FlippableMNIST::clearFlippedLabels() {
    flipped_labels.clear();
    Logger::instance().log("Cleared all flipped labels\n");
}

size_t FlippableMNIST::getFlippedCount() const {
    return flipped_labels.size();
}

std::vector<int64_t> FlippableMNIST::getLabelDistribution() const {
    std::vector<int64_t> distribution(10, 0);
    
    size_t dataset_size = mnist_dataset.size().value();
    for (size_t i = 0; i < dataset_size; ++i) {
        auto example = const_cast<FlippableMNIST*>(this)->get(i);
        int label = example.target.item<int64_t>();
        if (label >= 0 && label <= 9) {
            distribution[label]++;
        }
    }
    
    return distribution;
}

void FlippableMNIST::printLabelDistribution() const {
    std::vector<int64_t> distribution = getLabelDistribution();
    
    std::ostringstream oss;
    oss << "Label distribution:\n";
    for (int i = 0; i < 10; ++i) {
        oss << "Label " << i << ": " << distribution[i] << " samples\n";
    }
    
    Logger::instance().log(oss.str());
}