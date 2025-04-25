#pragma once

#include <torch/torch.h>
#include <cstddef>
#include <vector>

class RegisteredMNIST : public torch::data::Dataset<RegisteredMNIST> {
    private:
    float* images;            // Pointer to image data in registered memory
    int64_t* labels;          // Pointer to label data in registered memory
    size_t num_samples;       // Number of samples in the dataset
    size_t image_size;        // Size of each flattened image (784 for MNIST)
    torch::TensorOptions options; // Options for creating tensors
    bool owns_memory;         // Whether this class owns the memory (for cleanup)

    public:
    // takes pointers to registered memory
    RegisteredMNIST(float* images, int64_t* labels, size_t num_samples, 
                          size_t image_size = 784, bool owns_memory = false)
        : images(images), 
          labels(labels), 
          num_samples(num_samples), 
          image_size(image_size),
          options(torch::TensorOptions().dtype(torch::kFloat32)),
          owns_memory(owns_memory) {}

    // clean up if we own the memory
    ~RegisteredMNIST() {
        if (owns_memory) {
            free(images);
            free(labels);
        }
    }

    torch::data::Example<> get(size_t index) override {
        // Check bounds
        if (index >= num_samples) {
            throw std::runtime_error("Index out of bounds in RegisteredMemoryMNIST");
        }

        // Get pointer to the beginning of this sample's data
        float* img_ptr = images + (index * image_size);

        // Create a tensor that references this memory (no copy)
        // The from_blob function creates a tensor that points to existing memory
        torch::Tensor image = torch::from_blob(
            img_ptr,
            {1, 28, 28}, // MNIST image dimensions [channels, height, width]
            options
        ).clone(); // Clone to make a copy that's safe to return

        int64_t label = labels[index];
        torch::Tensor target = torch::tensor(label, torch::kInt64);

        return {image, target};
    }

    std::optional<size_t> size() const override {
        return num_samples;
    }

};