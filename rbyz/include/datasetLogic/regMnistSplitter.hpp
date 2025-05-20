#pragma once

#include "registeredMnistTrain.hpp"
#include "global/logger.hpp"
#include "global/globalConstants.hpp"
#include "entities.hpp"

#include <ctime>
#include <random>
#include <algorithm>

/**
 * @brief Class to split the registered MNIST dataset into n_clients.
 * During RByz, the server will insert data from its registered dataset into the clients' datasets.
 * The server will split the registered dataset into n_clients Validation Datasets, 
 * and each client will receive data from a different Validation Dataset in each round.
 */
class RegMnistSplitter {
    private:
    int n_clients;
    int vd_split_size;
    RegisteredMnistTrain& mnist;
    std::vector<ClientDataRbyz>& clnt_data_vec;
    std::vector<size_t> img_offsets;
    std::vector<size_t> lbl_offsets;

    // Vector of integers corresponding to the indices in img_offsets and lbl_offsets that each client will use, 
    // indices cannot be repeated in consecutive rounds per client 
    std::vector<int> prev_offsets_arrangement;  
    std::mt19937 rng;

    public:
    RegMnistSplitter(int n_clients, RegisteredMnistTrain& mnist, std::vector<ClientDataRbyz>& clnt_data_vec)
        : n_clients(n_clients), mnist(mnist), clnt_data_vec(clnt_data_vec), 
          img_offsets(n_clients), lbl_offsets(n_clients), prev_offsets_arrangement(n_clients),
          rng((static_cast<unsigned int>(std::time(nullptr)))) {

        for (int i = 0; i < n_clients; i++) {
            prev_offsets_arrangement[i] = i;
        }

        // Split the server registered data into n_clients with offsets (indices)
        size_t srvr_samples_count = mnist.getRegisteredSamplesCount();

        size_t vd_size = srvr_samples_count / n_clients;
        size_t data_size = mnist.getDataSize();
        size_t labels_size = mnist.getLabelSize();
        
        for (int i = 0; i < n_clients; i++) {
            img_offsets[i] = i * vd_size * data_size;
            lbl_offsets[i] = i * vd_size * labels_size;
            Logger::instance().log("Client " + std::to_string(i) + " image offset: " + std::to_string(img_offsets[i]) + "\n");
        }

        vd_split_size = vd_size / VD_SPLIT;
    }

    std::vector<size_t> getImageOffsets(std::vector<int> derangement) {
        std::vector<size_t> offsets;
        for (int i : derangement) {
            offsets.push_back(img_offsets[i]);
        }
        return offsets;
    }

    std::vector<size_t> getLabelOffsets(std::vector<int> derangement) {
        std::vector<size_t> offsets;
        for (int i : derangement) {
            offsets.push_back(lbl_offsets[i]);
        }
        return offsets;
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
            return prev_offsets_arrangement; // No derangement possible
        }

        std::vector<int> new_arrangement = prev_offsets_arrangement;
        std::shuffle(new_arrangement.begin(), new_arrangement.end(), rng);
        
        // Check for fixed points and fix them
        for (size_t i = 0; i < new_arrangement.size(); ++i) {
            if (new_arrangement[i] == prev_offsets_arrangement[i]) {
                // Found an element that didn't move
                // Find another position to swap with (guaranteed to exist if n ≥ 2)
                size_t j = (i + 1) % new_arrangement.size();
                
                // Avoid creating another fixed point
                while (new_arrangement[j] == prev_offsets_arrangement[j] || 
                       prev_offsets_arrangement[j] == prev_offsets_arrangement[i] ||
                       new_arrangement[i] == prev_offsets_arrangement[j]) {
                    j = (j + 1) % new_arrangement.size();
                }
                
                // Swap to fix the fixed point
                std::swap(new_arrangement[i], new_arrangement[j]);
            }
        }

        prev_offsets_arrangement = new_arrangement;
        
        return new_arrangement;
    }

    std::vector<size_t> getImageOffsetsSrvr(ClientDataRbyz clnt_data, std::vector<int> derangement) {
        int clnt_idx = clnt_data.clnt_index;
        size_t start_offset = img_offsets[derangement[clnt_idx]];
        size_t end_offset;
        if (clnt_idx == n_clients - 1) {
            end_offset = mnist.getRegisteredImagesMemSize() - mnist.getDataSize();
        } else {
            end_offset = img_offsets[derangement[clnt_idx + 1]] - mnist.getDataSize();
        }

        std::vector<size_t> offsets(vd_split_size);
        // TODO return offsets
        return offsets;
    }

};