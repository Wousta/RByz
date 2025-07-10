#pragma once

#include <cstdint>
#include <torch/torch.h>
#include <cstddef>
#include <vector>

#include "logger.hpp"
#include "structs.hpp"


/**
 * @brief RegisteredMNIST class for handling registered MNIST dataset.
 * 
 * Memory layout per sample:
 * [1 uint32_t (original index)][1 int64_t (label)][784 floats (pixels)]
 */
class RegisteredMNIST : public torch::data::Dataset<RegisteredMNIST> {
    private:
    RegTrainData* data_info;                   // Registered MNIST dataset
    std::unordered_map<size_t, size_t> index_map;   // Map original indices to registered indices          
    torch::TensorOptions options;                   // Options for creating tensors
    size_t num_samples = 0;                         // Number of samples in the dataset
    size_t sample_size = 0;
    size_t index_size = 0;
    size_t label_size = 0;
    size_t image_size = 0;

    public:
    // takes pointers to registered memory
    RegisteredMNIST(RegTrainData& data_info, std::unordered_map<size_t, size_t> index_map)
        : data_info(&data_info),  // Match the member name
        index_map(index_map),
        //options(torch::TensorOptions().dtype(torch::kUInt8)) { // UINT8CHANGE
        options(torch::TensorOptions().dtype(torch::kFloat32)) {

        num_samples = data_info.num_samples;
        index_size = data_info.index_size;
        label_size = data_info.label_size;
        image_size = data_info.image_size;
        sample_size = data_info.get_sample_size();
    }

    // clean up if we own the memory
    ~RegisteredMNIST() = default;

    torch::data::Example<> get(size_t original_index) override {
        size_t index = index_map.at(original_index);

        if (index >= num_samples) {
            throw std::out_of_range("Index out of range in RegisteredMNIST::get()");
        }

        void* sample = static_cast<char*>(data_info->reg_data) + (index * sample_size);
        //uint8_t* img_ptr = reinterpret_cast<uint8_t*>(reinterpret_cast<uint8_t*>(sample) + index_size + label_size);
        float* img_ptr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sample) + index_size + label_size);

        torch::Tensor image = torch::from_blob(
            img_ptr,
            {1, 28, 28}, // MNIST image dimensions [channels, height, width]
            options
        ).clone(); // Clone to make a copy that's safe to return

        //auto data_normalized = (image.to(torch::kFloat32) - 0.1307f) / 0.3081f; // UINT8CHANGE

        int64_t label = *reinterpret_cast<int64_t*>(reinterpret_cast<uint8_t*>(sample) + index_size);
        if (label < 0 || label >= 10) {
            throw std::runtime_error("Invalid label in RegisteredMNIST::get(): " + std::to_string(label));
        }
        torch::Tensor target = torch::tensor(label, torch::kInt64);

        //return {data_normalized, target}; // UINT8CHANGE
        return {image, target};
    }

    std::optional<size_t> size() const override {
        return num_samples;
    }

};