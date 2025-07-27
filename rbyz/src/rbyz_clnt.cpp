#include "rbyz_clnt.hpp"
#include "tensorOps.hpp"
#include "attacks.hpp"
#include "global/globalConstants.hpp"


void RByzClnt::runRByzClient(std::vector<torch::Tensor> &w, RegMemClnt &regMem) {
  Logger::instance().log("\n\n==============  STARTING RBYZ  ==============\n");
  Logger::instance().log("Client: Starting RByz with accuracy\n");
  // std::string log_file = "stepTimes_" + std::to_string(regMem.id) + ".log";
  // Set manager epochs to 1, the epochs will be controled by RByz
  mngr.kNumberOfEpochs = 1;
  regMem.clnt_ready_flag.store(0);

  while (regMem.round <= global_rounds) {
    if (regMem.round == 2) {
      Logger::instance().log("POST: first mnist samples\n");
      for (int i = 0; i < 320; i++) {
        if (i % 32 == 0)
          Logger::instance().log(
              "Sample " + std::to_string(i) +
              ": label = " + std::to_string(*mngr.getLabel(i)) +
              " | og_idx = " + std::to_string(*mngr.getOriginalIndex(i)) +
              "\n");
      }
    }

    // Reset
    regMem.local_step.store(0);

    Logger::instance().log("\n//////////////// Round " +
                           std::to_string(regMem.round) +
                           " started ////////////////\n");
    // Wait for the server to be ready
    do {
      rdma_ops.exec_rdma_read(sizeof(int), SRVR_READY_RB_IDX);
      std::this_thread::sleep_for(millis(1));
    } while (regMem.srvr_ready_rb < regMem.round &&
             regMem.srvr_ready_rb != SRVR_FINISHED);

    if (regMem.srvr_ready_rb == SRVR_FINISHED) {
      Logger::instance().log("Server finished early, exiting...\n");
      return;
    }

    if (regMem.round < regMem.srvr_ready_rb) {
      regMem.round.store(regMem.srvr_ready_rb);
    }

    rdma_ops.exec_rdma_read(regMem.reg_sz_data, SRVR_W_IDX);
    tops::writeToTensorVec(w, regMem.srvr_w, regMem.reg_sz_data);
    std::vector<torch::Tensor> w_pre_train = mngr.updateModelParameters(w);

    Logger::instance().log(
        "Steps to run: " + std::to_string(regMem.CAS.load()) + "\n");

    while (regMem.local_step.load() < regMem.CAS.load()) {
      // auto start = std::chrono::high_resolution_clock::now();
      int step = regMem.local_step.load();
      Logger::instance().log(" ...... Client: Running step " +
                             std::to_string(step) + " of RByz in round " +
                             std::to_string(regMem.round.load()) + "\n");
      mngr.runTraining();

      regMem.local_step.store(step + 1);

      // auto end = std::chrono::high_resolution_clock::now();
      // std::string time =
      // std::to_string(std::chrono::duration_cast<millis>(end
      // - start).count()); Logger::instance().logCustom("./stepTimes",
      // log_file, time + "\n");
    }

    // Write the updates to the registered memory and write to the server
    std::vector<torch::Tensor> g = mngr.calculateUpdate(w_pre_train);
    tops::memcpyTensorVec(regMem.clnt_w, g, regMem.reg_sz_data);
    rdma_ops.exec_rdma_write(regMem.reg_sz_data, CLNT_W_IDX);

    // If overwriting poisoned labels is enabled, byz client has to renew
    // poisoned labels (50% chance)
    if (byz_clnt && t_params.overwrite_poisoned && coinFlip()) {
      Logger::instance().log("Client: Overwriting poisoned labels\n");
      data_poison_attack(t_params.use_mnist, t_params, mngr);
    }

    regMem.clnt_ready_flag.store(regMem.round);
    rdma_ops.exec_rdma_write(sizeof(int), CLNT_READY_IDX);
    regMem.round.store(regMem.round + 1);

    Logger::instance().log("\n//////////////// Client: Round " +
                           std::to_string(regMem.round - 1) +
                           " completed ////////////////\n");
  }

  // Notify the server that the client is done
  regMem.round.store(CLNT_FINISHED_RBYZ);
  rdma_ops.exec_rdma_write(sizeof(int), CLNT_ROUND_IDX);
  Logger::instance().log("Finish round " + std::to_string(regMem.round.load()) +
                         " and notify server\n");
  Logger::instance().log("Client: Finished RByz\n");
}