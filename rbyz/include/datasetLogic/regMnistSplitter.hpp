#pragma once

#include "datasetLogic/iRegDatasetMngr.hpp"
#include "logger.hpp"
#include "entities.hpp"

#include <ctime>
#include <random>
#include <algorithm>
#include <vector>

/**
 * @brief Class to split the registered MNIST dataset into n_clients.
 * During RByz, the server will insert data from its registered dataset into the clients' datasets.
 * The server will split the registered dataset into n_clients Validation Datasets (VD), 
 * and each client will receive data from a different Validation Dataset in each round.
 */
class RegMnistSplitter {
    private:
    const int n_clients;
    const int samples_per_chunk;
    const float clnt_vd_proportion;                 // Proportion of the client dataset that will be overwritten with server data
    int samples_per_vd_split = 0;
    uint32_t chunk_sz_bytes;
    IRegDatasetMngr& mngr;
    std::vector<ClientDataRbyz>& clnt_data_vec;
    std::vector<std::vector<size_t>> vd_indexes;    // Vector of indices for the start of each VD
    std::vector<size_t> clnt_chunks;                // Vector of offsets for the clients till n-1

    // Vector of integers corresponding to the indices in vd_indexes and lbl_offsets that each client will use, 
    // indices cannot be repeated in consecutive rounds per client 
    std::vector<int> prev_indexes_arrangement;  
    std::mt19937 rng;

    /**
     * @brief Initializes client chunk offsets for test data injection.
     * 
     * Calculates and distributes chunk offsets evenly across the client dataset based
     * on the clnt_vd_proportion passed. These offsets determine where server data will be injected
     * into client datasets during RByz operations. The offsets are distributed uniformly
     * to ensure even coverage across the client's data space.
     *
     * @param overwrite_poisoned If true, the server will overwrite poisoned samples in the client dataset.
     * @throws std::runtime_error if clnt_vd_proportion exceeds 0.25 (25% limit)
     * @return The number of samples in total to be inserted to each client.
     */
    void initializeClientChunkOffsets(int overwrite_poisoned) {
        if (clnt_vd_proportion > 0.25) {
            throw std::runtime_error("clnt_vd_proportion must be <= 0.25, max 25%' overwrite of the client dataset");
        }

        chunk_sz_bytes = mngr.data_info.get_sample_size() * samples_per_chunk;
        int total_offsets = clnt_data_vec[0].dataset_size / chunk_sz_bytes;
        int num_offsets = total_offsets * clnt_vd_proportion;     
        clnt_chunks.reserve(num_offsets);

        Logger::instance().log("Total offsets: " + std::to_string(total_offsets) + 
                            ", datset size: " + std::to_string(clnt_data_vec[0].dataset_size) +
                            ", num_offsets: " + std::to_string(num_offsets) + 
                            ", chunk_sz_bytes: " + std::to_string(chunk_sz_bytes) + 
                            ", samples per chunk: " + std::to_string(samples_per_chunk) + "\n");

        if (overwrite_poisoned) {
            // (total_offsets - 1) / (num_offsets - 1) ensures the last offset lands exactly at the end of the available range.
            double step = static_cast<double>(total_offsets - 1) / (num_offsets - 1);
            for (size_t i = 0; i < num_offsets; i++) {
                size_t offset_index = static_cast<size_t>(i * step);
                clnt_chunks.push_back(offset_index * chunk_sz_bytes);
            }
        } else{
            // all the client chunks to overwrite are put together at the top
            for (size_t i = 0; i < num_offsets; i++) {
                clnt_chunks.push_back(i * chunk_sz_bytes);
            }
        }

        Logger::instance().log("Client offsets initialized with " + std::to_string(clnt_chunks.size()) + 
                            " offsets for " + std::to_string(n_clients) + " clients\n");

        // Samples per VD split is the number of test samples the server can insert into each client at most
        samples_per_vd_split = num_offsets * samples_per_chunk;
    }

    /**
     * @brief Initializes validation dataset (VD) partitions.
     * 
     * Divides the server's registered dataset into n_clients equal sections and creates
     * sampling indices for each partition. Each VD contains indices that start at regular
     * intervals (samples_per_chunk) within its assigned section. If the samples per VD split
     * exceeds the number of samples in the section, it wraps around to the start of the section
     * This may cause to write a lot of repeated samples to the clients if the clnt_vd_proportion is high
     * But this can be mitigated by refreshing the server VD dataset often.
     * 
     * The last client receives any remaining samples to handle cases where the dataset
     * size is not perfectly divisible by n_clients.
     */
    void initializeValidationDatasetPartitions() {
        // Split the server registered data into n_clients VD sections
        size_t vd_size = mngr.data_info.num_samples / n_clients;

        if (vd_size < samples_per_chunk) {
            throw std::runtime_error("Not enough samples in the server VD with the given clnt_vd_proportion. "
                                    "Increase the registered dataset size or decrease the clnt_vd_proportion.");
        }

        if (samples_per_vd_split == 0) {
            throw std::runtime_error("samples_per_vd_split cannot be zero, check the clnt_vd_proportion and samples_per_chunk values.");
        }

        Logger::instance().log("Splitting registered dataset into " + std::to_string(n_clients) + 
                            " VDs of size " + std::to_string(samples_per_vd_split) + "\n");

        // Initialize server VD indexes, it contains the dataset sample indices for each client split
        for (int i = 0; i < n_clients; i++) {
            size_t start_idx = i * vd_size;
            size_t end_idx;

            if (i == n_clients - 1) {
                end_idx = mngr.data_info.num_samples;
            } else {
                end_idx = (i + 1) * vd_size;
            }

            std::vector<size_t> indices;
            indices.reserve(samples_per_vd_split);

            int indices_put = 0;
            size_t idx = start_idx;
            while (indices_put < samples_per_vd_split) {
                // If we reach the end of the current VD split, wrap around to the start
                if (idx + samples_per_chunk > end_idx) {
                    idx = start_idx;
                }

                indices.push_back(idx);
                idx += samples_per_chunk;
                indices_put++;
            }
            
            vd_indexes[i] = indices;
            Logger::instance().log("Client " + std::to_string(i) + " image index: " + 
                                std::to_string(vd_indexes[i][0]) + " size: " + 
                                std::to_string(vd_indexes[i].size()) + "\n");
        }
    }

    public:
    // I dislike how C++ constructors work, java is so much better in this regard, look at this mess
    RegMnistSplitter(TrainInputParams& t_params, IRegDatasetMngr& mngr, std::vector<ClientDataRbyz>& clnt_data_vec)
        : n_clients(clnt_data_vec.size()), samples_per_chunk(t_params.chunk_size), clnt_vd_proportion(t_params.clnt_vd_proportion), 
          mngr(mngr), clnt_data_vec(clnt_data_vec), vd_indexes(n_clients), prev_indexes_arrangement(n_clients), 
          rng((static_cast<unsigned int>(std::time(nullptr)))) {

        // Used to select the VD splits for each client
        for (int i = 0; i < n_clients; i++) {
            prev_indexes_arrangement[i] = i;
        }

        initializeClientChunkOffsets(t_params.overwrite_poisoned);
        initializeValidationDatasetPartitions();
    }

    int getSamplesPerChunk() const {
        return samples_per_chunk;
    }

    uint32_t getChunkSize() const {
        return chunk_sz_bytes;
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

    /**
     * @brief Gets the server indices for a given client index based on the derangement.
     * This function retrieves the indices of the server's validation dataset (VD) that will be sent to a specific client.
     * @param clnt_idx The index of the client for which to get the server indices.
     * @param derangement The derangement vector that determines the current arrangement of VD samples.
     * @return A vector containing the selected server indices for the client.
     */
    std::vector<size_t> getServerIndices(int clnt_idx, std::vector<int> derangement) {
        return vd_indexes[derangement[clnt_idx]];
    }

    /**
     * @brief Gets the client chunks for a given client index. I proportion < 1.0, it returns a random subset of the chunks.
     * @param clnt_idx The index of the client for which to get the chunks.
     * @param proportion The proportion of chunks to return (default is 1.0, meaning all chunks).
     * @return A vector containing the selected chunks for the client.
     */
    std::vector<size_t> getClientChunks(int clnt_idx, float proportion = 1.0) {
        if (clnt_idx < 0 || clnt_idx >= n_clients) {
            throw std::out_of_range("[getClientChunks] Client index out of range");
        }

        if (proportion < 0.0 || proportion > 1.0) {
            throw std::invalid_argument("[getClientChunks] Proportion must be between 0 and 1");
        }

        if (proportion == 1.0) {
            return clnt_chunks;
        }

        std::shuffle(clnt_chunks.begin(), clnt_chunks.end(), rng);
        size_t num_chunks = std::max(static_cast<size_t>(clnt_chunks.size() * proportion), 1UL);
        std::vector<size_t> clnt_chunks_subset(clnt_chunks.begin(), clnt_chunks.begin() + num_chunks);

        return clnt_chunks_subset; 
    }

};