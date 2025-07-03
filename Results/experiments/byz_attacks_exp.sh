#!/bin/bash

# Byzantine Attacks Experiment Script
# Follows the methodology from the paper: https://arxiv.org/pdf/2007.08432

trap 'echo "Script interrupted. Exiting..."; exit 1' INT TERM
ORIGINAL_DIR=$(pwd)

# Common parameters
clients=50
epochs=5
glob_learning_rate=0.05
local_learn_rate=0.05
byz_clients=2
chunk_size=2
label_flip_type=2
flip_ratio=1.0
only_flt=0  # Run FLtrust only

rm -rf ../logs/*
rm -rf ../accLogs/*
cd ../../rbyz

run() {
    local name=$1
    local i_start=$2    # Controls the VD proportion to use

    echo "Starting experiment $name from iteration $i_start"

    for ((i=$i_start; i<=10; i++)); do
        echo "Running experiment iteration $i"
        vd_prop=$(awk "BEGIN {printf \"%.1f\", 0.1 * $i}")
        echo "VD proportion: $vd_prop"
        
        ./run_all.sh $use_mnist $clients $epochs $batch_size $glob_learning_rate $local_learn_rate $byz_clients \
            $clnt_subset_size $srvr_subset_size $glob_iters_fl $local_steps_rbyz $glob_iters_rbyz \
            $chunk_size $label_flip_type $flip_ratio $only_flt $vd_prop

        cd ../Results/accLogs
        percent=$(awk "BEGIN {printf \"%.1f\", $vd_prop * 100}")
        mv R_acc*.log "R_vdprop_acc_${percent}%.log"
        cd ../../rbyz
    done

    cd ../Results/logs
    mkdir -p "$name"
    mv *.log "$name/"
    mv ../accLogs/*.log "$name/"
    cd ../../rbyz
}

cd "$ORIGINAL_DIR"