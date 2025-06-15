#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include <random>
#include <iostream>

class FlippableMNIST : public torch::data::datasets::Dataset<FlippableMNIST> {
private:
    torch::data::datasets::MNIST mnist_dataset;
    std::unordered_map<size_t, int64_t> flipped_labels;
    
public:
    // Use the correct Example type
    using ExampleType = torch::data::Example<torch::Tensor, torch::Tensor>;
    
    explicit FlippableMNIST(const std::string& root, 
                           torch::data::datasets::MNIST::Mode mode = torch::data::datasets::MNIST::Mode::kTrain)
        : mnist_dataset(root, mode) {}
    
    // Required Dataset interface methods
    ExampleType get(size_t index) override;
    torch::optional<size_t> size() const override;
    
    // Label flipping methods
    void flipLabelsRandom(float flip_ratio, std::mt19937& rng);
    void flipLabelsTargeted(int source_label, int target_label, float flip_ratio, std::mt19937& rng);
    void clearFlippedLabels();
    size_t getFlippedCount() const;
    
    // Utility methods
    std::vector<int64_t> getLabelDistribution() const;
    void printLabelDistribution() const;
};