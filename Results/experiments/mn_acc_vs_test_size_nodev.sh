#!/bin/bash

cleanup() {
    echo "Script interrupted. Cleaning up..."
    kill $current_pid 2>/dev/null
    echo "Cleanup complete. Exiting..."
    exit 1
}
trap cleanup INT TERM QUIT EXIT SIGINT SIGTERM

# Accuracy vs Test Size Experiment
ORIGINAL_DIR=$(pwd)
EXPERIMENT="mn_acc_vs_test_size_nodev"
IP_ADDRESS=$(ip addr show | grep -A2 "ibp.*UP" | grep "inet " | head -1 | awk '{print $2}' | cut -d'/' -f1)
PORT=2450
REMOTE_HOSTS=("dcldelta3")

echo "Running experiment $EXPERIMENT on Server IP: $IP_ADDRESS"

# Launch redis (disowned so it is not affected)
echo "Starting Redis server on $IP_ADDRESS:$PORT"
redis-server --bind "$IP_ADDRESS" --port "$PORT" >/dev/null &
disown
sleep 1
redis-cli -h "$IP_ADDRESS" -p "$PORT" SET srvr "0" >/dev/null
redis-cli -h "$IP_ADDRESS" -p "$PORT" SET clnt "0" >/dev/null
redis-cli -h "$IP_ADDRESS" -p "$PORT" SET nid "0" >/dev/null

echo "Redis server started on $IP_ADDRESS:$PORT"

# Common parameters
clients=10
byz_clients=3
epochs=1                    # Local rounds of FLtrust
local_steps_rbyz=5          # Local rounds of RByz
glob_iters_fl=1
glob_iters_rbyz=49
chunk_size=2                # Slab size for RByz VDsampling
label_flip_type=1           # 1: Random label flip
flip_ratio=0.75             # 50% of the data will be flipped
overwrite_poisoned=0        # Do not overwrite poisoned data
only_flt=0                  # Run RByz
test_renewal_freq=1000      # Renew test samples every 5 rounds (fixed 50% of VD is renewed)
vd_prop_write=1.0           # Proportion of total chunks writable on client to write each time the test is renewed

rm -rf ../logs/$EXPERIMENT/*
cd ../../rbyz

run() {
    local name=$1
    local iters=${2:-5}

    echo "=========================================================="
    echo "---- Starting experiment $name iters: $iters ----"
    echo "=========================================================="
    echo "     VD prop: $vd_prop"

    for ((i=1; i<=$iters; i++)); do
        echo "______________________________________________________"
        echo "---- Running experiment $name iteration $i ----"
        ./run_all.sh "${REMOTE_HOSTS[*]}" $EXPERIMENT $IP_ADDRESS $PORT $use_mnist $clients $epochs $batch_size $glob_learning_rate \
            $local_learn_rate $byz_clients $clnt_subset_size $srvr_subset_size $glob_iters_fl $local_steps_rbyz $glob_iters_rbyz \
            $chunk_size $label_flip_type $flip_ratio $only_flt $vd_prop $vd_prop_write $test_renewal_freq $overwrite_poisoned &

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
glob_learning_rate=0.04
local_learn_rate=0.01
clnt_subset_size=4800
srvr_subset_size=12000

vd_prop=0.25
run "mnist_25%vd" 

vd_prop=0.23
run "mnist_23%vd"

vd_prop=0.21
run "mnist_21%vd"

vd_prop=0.19
run "mnist_19%vd"

vd_prop=0.17
run "mnist_17%vd"

vd_prop=0.15
run "mnist_15%vd"

vd_prop=0.13
run "mnist_13%vd" 

vd_prop=0.11
run "mnist_11%vd" 

vd_prop=0.09
run "mnist_9%vd" 

vd_prop=0.07
run "mnist_7%vd" 

vd_prop=0.05
run "mnist_5%vd" 

vd_prop=0.03
run "mnist_3%vd"

vd_prop=0.01
run "mnist_1%vd" 

#######################################
########## CIFAR Experiments ##########
use_mnist="false"
batch_size=64
glob_learning_rate=0.5
local_learn_rate=0.01
clnt_subset_size=4900
srvr_subset_size=1000

# vd_prop=0.25
# run "cifar_25%vd"

# vd_prop=0.23
# run "cifar_23%vd"

# vd_prop=0.2
# run "cifar_20%vd"

# vd_prop=0.17
# run "cifar_17%vd"

# vd_prop=0.14
# run "cifar_14%vd"

# vd_prop=0.11
# run "cifar_11%vd"

# vd_prop=0.08
# run "cifar_8%vd"

# vd_prop=0.05
# run "cifar_5%vd"

# vd_prop=0.02
# run "cifar_2%vd"

cd "$ORIGINAL_DIR"