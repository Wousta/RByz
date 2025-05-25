#pragma once

#include "registeredMnistTrain.hpp"
#include "global/logger.hpp"
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
    int n_clients;
    int vd_split_size;
    RegisteredMnistTrain& mnist;
    std::vector<ClientDataRbyz>& clnt_data_vec;
    std::vector<std::vector<size_t>> vd_indexes;              // Vector of indices for the start of each VD

    // Vector of integers corresponding to the indices in vd_indexes and lbl_offsets that each client will use, 
    // indices cannot be repeated in consecutive rounds per client 
    std::vector<int> prev_indexes_arrangement;  
    std::mt19937 rng;

    public:
    RegMnistSplitter(int n_clients, RegisteredMnistTrain& mnist, std::vector<ClientDataRbyz>& clnt_data_vec)
        : n_clients(n_clients), mnist(mnist), clnt_data_vec(clnt_data_vec), 
          vd_indexes(n_clients), prev_indexes_arrangement(n_clients),
          rng((static_cast<unsigned int>(std::time(nullptr)))) {

        for (int i = 0; i < n_clients; i++) {
            prev_indexes_arrangement[i] = i;
        }

        // Split the server registered data into n_clients VDs
        size_t vd_size = mnist.getNumSamples() / n_clients;

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

            Logger::instance().log("Client " + std::to_string(i) + " image offset: " + std::to_string(vd_indexes[i][0]) + "\n");
        }

        // For each VD, only a part of the data will be used
        vd_split_size = vd_size / VD_SPLIT;
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

        // Create a copy of the previous arrangement
        std::vector<int> new_arrangement(n_clients);
        
        // Generate a true derangement using a more reliable algorithm
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

    // Helper function to check if an arrangement is a derangement
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

        std::shuffle(all_indices.begin(), all_indices.end(), rng);
        std::vector<size_t> indexes(all_indices.begin(), 
                               all_indices.begin() + std::min(vd_split_size, 
                                                                  static_cast<int>(all_indices.size())));

        return indexes;
    }

    std::vector<size_t> getClientOffsets(ClientDataRbyz &clnt_data, std::vector<int> derangement) {
        int clnt_idx = clnt_data.index;
        size_t dataset_size = clnt_data.dataset_size;
        Logger::instance().log("Client " + std::to_string(clnt_idx) + " dataset size: " + std::to_string(dataset_size) + "\n");
        size_t sample_size = mnist.getSampleSize();

        size_t num_possible_positions = dataset_size / sample_size;
        std::vector<size_t> all_possible_offsets(num_possible_positions);

        // Fill with multiples of sample_size: 0, sample_size, sample_size*2, etc.
        for (size_t i = 0; i < all_possible_offsets.size(); ++i) {
            all_possible_offsets[i] = i * sample_size;
        }

        // Choose random offsets for the client
        std::shuffle(all_possible_offsets.begin(), all_possible_offsets.end(), rng);
        size_t num_offsets = std::min(static_cast<size_t>(vd_split_size), num_possible_positions);
        std::vector<size_t> offsets(all_possible_offsets.begin(), all_possible_offsets.begin() + num_offsets);

        return offsets;
    }

};