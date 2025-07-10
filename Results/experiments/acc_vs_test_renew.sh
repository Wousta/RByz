#!/bin/bash

# Accuracy vs Test Size Experiment
trap 'echo "Script interrupted. Exiting..."; exit 1' INT TERM
ORIGINAL_DIR=$(pwd)
EXPERIMENT="acc_vs_test_renew"
IP_ADDRESS=$(ip addr show | grep -A2 "ibp.*UP" | grep "inet " | head -1 | awk '{print $2}' | cut -d'/' -f1)
REMOTE_HOSTS=("lpdquatro2")

echo "Running experiment $EXPERIMENT on Server IP: $IP_ADDRESS"

# Common parameters
clients=10
byz_clients=2
glob_learning_rate=0.08
local_learn_rate=0.01
epochs=5                    # Local rounds of FLtrust
local_steps_rbyz=5          # Local rounds of RByz
glob_iters_fl=3
glob_iters_rbyz=47
chunk_size=2                # Slab size for RByz VDsampling
label_flip_type=1           # 1: Random label flip
flip_ratio=0.5              # 50% of the data will be flipped
overwrite_poisoned=1        # Can overwrite poisoned data
only_flt=0                  # Run RByz
vd_prop=0.2                 # Use the best value obtained from acc_vs_test_size.sh
vd_prop_write=0.1           # Proportion of total chunks writable on client to write each time the test is renewed

rm -rf ../logs/$EXPERIMENT/*
cd ../../rbyz

run() {
    local name=$1
    echo "=========================================================="
    echo "---- Starting experiment $name ----"
    echo "=========================================================="
    echo "     VD prop: $vd_prop"

    for ((i=1; i<=5; i++)); do
        echo "______________________________________________________"
        echo "---- Running experiment $name iteration $i ----"
        ./run_all.sh $EXPERIMENT $IP_ADDRESS "${REMOTE_HOSTS[*]}" $use_mnist $clients $epochs $batch_size $glob_learning_rate \
            $local_learn_rate $byz_clients $clnt_subset_size $srvr_subset_size $glob_iters_fl $local_steps_rbyz $glob_iters_rbyz \
            $chunk_size $label_flip_type $flip_ratio $only_flt $vd_prop $vd_prop_write $test_renewal_freq $overwrite_poisoned

        # cd ../Results/accLogs
        # percent=$(awk "BEGIN {printf \"%.1f\", $vd_prop * 100}")
        # mv R_acc*.log "R_vdprop_acc_${percent}%.log"
        # cd ../../rbyz
    done

    cd ../Results/logs/$EXPERIMENT
    mkdir -p "$name"
    mv *.log "$name/"
    cd ../../../rbyz
}

#######################################
########## MNIST Experiments ##########
use_mnist="true"
batch_size=32
clnt_subset_size=5900
srvr_subset_size=1000

test_renewal_freq=1
run "mnist_test_ren_fq_1"

test_renewal_freq=3
run "mnist_test_ren_fq_3"

test_renewal_freq=5
run "mnist_test_ren_fq_5"

test_renewal_freq=10
run "mnist_test_ren_fq_10"

test_renewal_freq=15
run "mnist_test_ren_fq_15"

#######################################
########## CIFAR Experiments ##########
use_mnist="false"
batch_size=64
clnt_subset_size=4900
srvr_subset_size=1000
glob_learning_rate=0.8

test_renewal_freq=1
run "cifar_test_ren_fq_1"

test_renewal_freq=3
run "cifar_test_ren_fq_3"

test_renewal_freq=5
run "cifar_test_ren_fq_5"

test_renewal_freq=10
run "cifar_test_ren_fq_10"

test_renewal_freq=15
run "cifar_test_ren_fq_15"

cd "$ORIGINAL_DIR"


