#pragma once

#include "registeredMnistTrain.hpp"
#include "logger.hpp"
#include "global/globalConstants.hpp"
#include "entities.hpp"

#include <ctime>
#include <random>
#include <algorithm>
#include <numeric>

/**
 * @brief Class to split the registered MNIST dataset into n_clients.
 * During RByz, the server will insert data from its registered dataset into the clients' datasets.
 * The server will split the registered dataset into n_clients Validation Datasets (VD), 
 * and each client will receive data from a different Validation Dataset in each round.
 */
class RegMnistSplitter {
    private:
    const int n_clients;
    const int sgl_size;
    uint32_t chunk_size;
    int vd_test_size;
    RegisteredMnistTrain& mnist;
    std::vector<ClientDataRbyz>& clnt_data_vec;
    std::vector<std::vector<size_t>> vd_indexes;              // Vector of indices for the start of each VD
    std::vector<size_t> clnt_chunks;                          // Vector of offsets for the clients till n-1
    std::vector<size_t> last_clnt_chunks;                     // Offsets for the last client (has remainder of the dataset split)

    // Vector of integers corresponding to the indices in vd_indexes and lbl_offsets that each client will use, 
    // indices cannot be repeated in consecutive rounds per client 
    std::vector<int> prev_indexes_arrangement;  
    std::mt19937 rng;

    public:
    RegMnistSplitter(int sgl_size, RegisteredMnistTrain& mnist, std::vector<ClientDataRbyz>& clnt_data_vec)
        : n_clients(clnt_data_vec.size()), sgl_size(sgl_size), mnist(mnist), clnt_data_vec(clnt_data_vec),  
          vd_indexes(n_clients), prev_indexes_arrangement(n_clients), 
          rng((static_cast<unsigned int>(std::time(nullptr)))) {

        for (int i = 0; i < n_clients; i++) {
            prev_indexes_arrangement[i] = i;
        }

        // Split the server registered data into n_clients VDs
        size_t vd_size = mnist.getNumSamples() / n_clients;

        Logger::instance().log("Splitting registered dataset into " + std::to_string(n_clients) + " VDs of size " + std::to_string(vd_size) + "\n");

        for (int i = 0; i < n_clients; i++) {
            size_t start_idx = i * vd_size;
            size_t end_idx;

            if (i == n_clients - 1) {
                end_idx = mnist.getNumSamples();
            } else {
                end_idx = (i + 1) * vd_size;
            }

            std::vector<size_t> indices(end_idx - start_idx);
            std::iota(indices.begin(), indices.end(), start_idx);
            vd_indexes[i] = indices;

            Logger::instance().log("Client " + std::to_string(i) + " image offset: " + std::to_string(vd_indexes[i][0]) + " size: " + std::to_string(vd_indexes[i].size()) + "\n");
        }

        // For each VD, only a part of the data will be used
        vd_test_size = vd_indexes[0].size();
        for (const auto& indices : vd_indexes) {
            if (indices.size() < vd_test_size) {
                vd_test_size = indices.size();
            }
        }

        if (vd_test_size < sgl_size) {
            throw std::runtime_error("VD split size cannot be less than " + std::to_string(sgl_size) + 
                                     " VD split size: " + std::to_string(vd_test_size));
        }

        size_t num_samples_clnt = clnt_data_vec[0].dataset_size / mnist.getSampleSize();
        size_t num_samples_last_clnt = clnt_data_vec.back().dataset_size / mnist.getSampleSize();
        double chunks_proportion = vd_test_size / static_cast<double>(num_samples_clnt);
        double chunks_proportion_last = vd_test_size / static_cast<double>(num_samples_last_clnt);

        // chunks will be the size of two SGLs, (Scatter Gather Lists)
        chunk_size = mnist.getSampleSize() * sgl_size;

        size_t total_offsets = clnt_data_vec[0].dataset_size / chunk_size;
        size_t total_offsets_last = clnt_data_vec.back().dataset_size / chunk_size;
        size_t num_offsets = total_offsets * std::min(0.25, chunks_proportion);
        size_t num_offsets_last = total_offsets_last * std::min(0.25, chunks_proportion_last);

        if (num_offsets <= 2 || num_offsets_last <= 2) {
            throw std::runtime_error("Not enough offsets for the clients, please increase the dataset size or decrease the number of clients");
        }

        Logger::instance().log("Total offsets: " + std::to_string(total_offsets) + 
                               " Total offsets last client: " + std::to_string(total_offsets_last) + "\n");

        clnt_chunks.reserve(num_offsets);
        last_clnt_chunks.reserve(num_offsets_last);

        // Evenly distribute offsets
        // (total_offsets - 1) / (num_offsets - 1) ensures the last offset lands exactly at the end of the available range.
        double step = static_cast<double>(total_offsets - 1) / (num_offsets - 1);
        for (size_t i = 0; i < num_offsets; ++i) {
            size_t offset_index = static_cast<size_t>(i * step);
            clnt_chunks.push_back(offset_index * chunk_size);
        }

        double step_last = static_cast<double>(total_offsets_last - 1) / (num_offsets_last - 1);
        for (size_t i = 0; i < num_offsets_last; ++i) {
            size_t offset_index = static_cast<size_t>(i * step_last);
            last_clnt_chunks.push_back(offset_index * chunk_size);
        }

        Logger::instance().log("Client offsets initialized with " + std::to_string(clnt_chunks.size()) + 
                               " offsets for first " + std::to_string(n_clients - 1) + " clients and " +
                               std::to_string(last_clnt_chunks.size()) + " offsets for last client\n");
    }

    int getSamplesPerChunk() const {
        return sgl_size;
    }

    uint32_t getChunkSize() const {
        return chunk_size;
    }

    /**
     * Generates a derangement of the previous offset arrangement.
     * A derangement is a permutation where no element appears in its original position.
     * Time Complexity: O(n) where n is the number of clients
     * 
     * @return A vector containing the new arrangement of indices for the offset vectors 
     *         where no client receives the same partition index as in the previous round
     */
    std::vector<int> generateDerangement() {
        if (n_clients < 2) {
            return prev_indexes_arrangement; // No derangement possible
        }

        std::vector<int> new_arrangement(n_clients);
        
        do {
            std::shuffle(prev_indexes_arrangement.begin(), prev_indexes_arrangement.end(), rng);
            for (int i = 0; i < n_clients; i++) {
                new_arrangement[i] = prev_indexes_arrangement[i];
            }
            
            // Simple swap to fix any remaining fixed points
            for (int i = 0; i < n_clients; i++) {
                if (new_arrangement[i] == i) {
                    // If we find a fixed point, swap it with any other element
                    for (int j = 0; j < n_clients; j++) {
                        if (i != j && new_arrangement[j] != j) {
                            std::swap(new_arrangement[i], new_arrangement[j]);
                            break;
                        }
                    }
                }
            }
        } while (!isDerangement(new_arrangement));
        
        prev_indexes_arrangement = new_arrangement;
        return new_arrangement;
    }

    bool isDerangement(const std::vector<int>& arrangement) {
        for (int i = 0; i < n_clients; i++) {
            if (arrangement[i] == i) {
                return false;
            }
        }
        return true;
    }

    std::vector<size_t> getServerIndices(int clnt_idx, std::vector<int> derangement) {
        std::vector<size_t>& all_indices = vd_indexes[derangement[clnt_idx]];

        if (all_indices.size() < vd_test_size) {
            throw std::runtime_error("getServerIndices: Not enough indices in the VD for client " + std::to_string(clnt_idx));
        }

        std::shuffle(all_indices.begin(), all_indices.end(), rng);
        std::vector<size_t> indexes(all_indices.begin(), all_indices.begin() + vd_test_size);

        return indexes;
    }

    std::vector<size_t> getClientChunks(int clnt_idx) {
        if (clnt_idx < 0 || clnt_idx >= n_clients) {
            throw std::out_of_range("getClientChunks: Client index out of range");
        }

        if (clnt_idx == n_clients - 1) {
            return last_clnt_chunks; // Last client has its own offsets
        } else {
            return clnt_chunks; // All other clients share the same offsets
        }
    }

};