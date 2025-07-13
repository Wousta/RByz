#!/bin/bash
cleanup() {
    echo "Script interrupted. Cleaning up..."
    kill $current_pid 2>/dev/null
    echo "Cleanup complete. Exiting..."
    exit 1
}
trap cleanup INT TERM QUIT EXIT

# Accuracy vs Test renewal frequency Experiment
ORIGINAL_DIR=$(pwd)
EXPERIMENT="acc_vs_test_renew"
IP_ADDRESS=$(ip addr show | grep -A2 "ibp.*UP" | grep "inet " | head -1 | awk '{print $2}' | cut -d'/' -f1)
REMOTE_HOSTS=("dcldelta3")
PORT="2300"

echo "Running experiment $EXPERIMENT on Server IP: $IP_ADDRESS"

# Launch redis (disowned so it is not affected)
echo "Starting Redis server on $IP_ADDRESS:$PORT"
redis-server --bind "$IP_ADDRESS" --port "$PORT" >/dev/null &
disown
sleep 1
redis-cli -h "$IP_ADDRESS" -p "$PORT" SET srvr "0" >/dev/null
redis-cli -h "$IP_ADDRESS" -p "$PORT" SET clnt "0" >/dev/null
redis-cli -h "$IP_ADDRESS" -p "$PORT" SET nid "0" >/dev/null

# Common parameters
clients=10
byz_clients=3
local_learn_rate=0.01
epochs=1                    # Local rounds of FLtrust
local_steps_rbyz=5          # Local rounds of RByz
glob_iters_fl=1
glob_iters_rbyz=49
chunk_size=2                # Slab size for RByz VDsampling
label_flip_type=1           # 1: Random label flip
flip_ratio=0.75             # 50% of the data will be flipped
overwrite_poisoned=0        # Can overwrite poisoned data
only_flt=0                  # Run RByz
vd_prop_write=1.0           # Proportion of total chunks writable on client to write each time the test is renewed

rm -rf ../logs/$EXPERIMENT/*
cd ../../rbyz

run() {
    local name=$1
    echo "=========================================================="
    echo "---- Starting experiment $name ----"
    echo "=========================================================="
    echo "     Renewal freq: $test_renewal_freq"

    for ((i=1; i<=1; i++)); do
        echo "______________________________________________________"
        echo "---- Running experiment $name iteration $i ----"
        ./run_all.sh "${REMOTE_HOSTS[*]}" $EXPERIMENT $IP_ADDRESS $PORT $use_mnist $clients $epochs $batch_size $glob_learning_rate \
            $local_learn_rate $byz_clients $clnt_subset_size $srvr_subset_size $glob_iters_fl $local_steps_rbyz $glob_iters_rbyz \
            $chunk_size $label_flip_type $flip_ratio $only_flt $vd_prop $vd_prop_write $test_renewal_freq $overwrite_poisoned

        current_pid=$!
        wait $current_pid
        sleep 0.1
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
clnt_subset_size=4800
srvr_subset_size=12000
glob_learning_rate=0.04
vd_prop=0.09                # Use the best value obtained from acc_vs_test_size.sh


test_renewal_freq=1
run "mnist_test_ren_fq_1"

test_renewal_freq=3
run "mnist_test_ren_fq_3"

test_renewal_freq=5
run "mnist_test_ren_fq_5"

test_renewal_freq=7
run "mnist_test_ren_fq_7"

test_renewal_freq=9
run "mnist_test_ren_fq_9"

test_renewal_freq=11
run "mnist_test_ren_fq_11"

test_renewal_freq=13
run "mnist_test_ren_fq_13"

test_renewal_freq=15
run "mnist_test_ren_fq_15"

#######################################
########## CIFAR Experiments ##########
use_mnist="false"
batch_size=64
clnt_subset_size=4000
srvr_subset_size=10000
glob_learning_rate=0.4

test_renewal_freq=1
run "cifar_test_ren_fq_1"

test_renewal_freq=3
run "cifar_test_ren_fq_3"

test_renewal_freq=5
run "cifar_test_ren_fq_5"

test_renewal_freq=7
run "cifar_test_ren_fq_7"

test_renewal_freq=9
run "cifar_test_ren_fq_9"

test_renewal_freq=11
run "cifar_test_ren_fq_11"

test_renewal_freq=13
run "cifar_test_ren_fq_13"

test_renewal_freq=15
run "cifar_test_ren_fq_15"

cd "$ORIGINAL_DIR"


