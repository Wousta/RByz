#pragma once

#include <torch/torch.h>
#include <cstddef>
#include <vector>

#include "global/logger.hpp"


// Memory layout:
// For each image i:
// - Pixels: registered_images[i * data_size] to registered_images[i * data_size + 783]
// - Original index: reinterpret_cast<uint32_t*>(&registered_images[i * data_size + 784])
class RegisteredMNIST : public torch::data::Dataset<RegisteredMNIST> {
    private:
    float* images;            // Pointer to image data in registered memory
    int64_t* labels;          // Pointer to label data in registered memory
    size_t num_samples;       // Number of samples in the dataset
    std::unordered_map<size_t, size_t> index_map; // Map original indices to registered indices
    size_t data_size;        // Size of each flattened image (784 for MNIST)
    torch::TensorOptions options; // Options for creating tensors
    bool owns_memory;         // Whether this class owns the memory (for cleanup)

    public:
    // takes pointers to registered memory
    RegisteredMNIST(float* images, int64_t* labels, size_t num_samples, std::unordered_map<size_t, size_t> index_map,
                          size_t data_size, bool owns_memory = false)
        : images(images), 
          labels(labels), 
          num_samples(num_samples),
          index_map(index_map),
          data_size(data_size),
          options(torch::TensorOptions().dtype(torch::kFloat32)),
          owns_memory(owns_memory) {}

    // clean up if we own the memory
    ~RegisteredMNIST() {
        if (owns_memory) {
            free(images);
            free(labels);
        }
    }

    torch::data::Example<> get(size_t original_index) override {
        size_t index = index_map.at(original_index);

        // Get pointer to the beginning of this sample's data
        float* img_ptr = images + (index * data_size);
        uint32_t* index_ptr = reinterpret_cast<uint32_t*>(&img_ptr[784]);

        // Create a tensor that references this memory (no copy)
        // The from_blob function creates a tensor that points to existing memory
        torch::Tensor image = torch::from_blob(
            img_ptr,
            {1, 28, 28}, // MNIST image dimensions [channels, height, width]
            options
        ).clone(); // Clone to make a copy that's safe to return

        int64_t label = labels[index];
        if (label < 0 || label >= 10) {
            throw std::runtime_error("Invalid label in RegisteredMNIST::get(): " + std::to_string(label));
        }
        torch::Tensor target = torch::tensor(label, torch::kInt64);

        return {image, target};
    }

    std::optional<size_t> size() const override {
        return num_samples;
    }

};