#!/bin/bash
cleanup() {
    echo "Script interrupted. Cleaning up..."
    kill -TERM $current_pid 2>/dev/null
    echo "Cleanup complete. Exiting..."
    exit 1
}
trap cleanup INT TERM QUIT EXIT

# Accuracy vs Test Size Experiment
ORIGINAL_DIR=$(pwd)
EXPERIMENT="acc_vs_test_size"
IP_ADDRESS=$(ip addr show | grep -A2 "ibp.*UP" | grep "inet " | head -1 | awk '{print $2}' | cut -d'/' -f1)
REMOTE_HOSTS=("dcldelta4")
PORT="2000"

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
byz_clients=2
epochs=5                    # Local rounds of FLtrust
local_steps_rbyz=5          # Local rounds of RByz
glob_iters_fl=3
glob_iters_rbyz=47
chunk_size=2                # Slab size for RByz VDsampling
label_flip_type=1           # 1: Random label flip
flip_ratio=0.5              # 50% of the data will be flipped
overwrite_poisoned=0        # Do not overwrite poisoned data
only_flt=0                  # Run RByz
test_renewal_freq=1         # Renew test samples every 5 rounds (fixed 50% of VD is renewed)
vd_prop_write=0.1           # Proportion of total chunks writable on client to write each time the test is renewed

rm -rf ../logs/$EXPERIMENT/*
cd ../../rbyz

run() {
    local name=$1
    echo "=========================================================="
    echo "---- Starting experiment $name ----"
    echo "=========================================================="
    echo "     VD prop: $vd_prop"

    for ((i=1; i<=3; i++)); do
        echo "______________________________________________________"
        echo "---- Running experiment $name iteration $i ----"
        ./run_all.sh "${REMOTE_HOSTS[*]}" $EXPERIMENT $IP_ADDRESS $PORT $use_mnist $clients $epochs $batch_size $glob_learning_rate \
            $local_learn_rate $byz_clients $clnt_subset_size $srvr_subset_size $glob_iters_fl $local_steps_rbyz $glob_iters_rbyz \
            $chunk_size $label_flip_type $flip_ratio $only_flt $vd_prop $vd_prop_write $test_renewal_freq $overwrite_poisoned

        current_pid=$!

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
glob_learning_rate=0.09
local_learn_rate=0.01
clnt_subset_size=5900
srvr_subset_size=1000

# vd_prop=0.25
# run "mnist_25%vd"

# vd_prop=0.2
# run "mnist_20%vd"

# vd_prop=0.15
# run "mnist_15%vd"

# vd_prop=0.10
# run "mnist_10%vd"

# vd_prop=0.05
# run "mnist_5%vd"

#######################################
########## CIFAR Experiments ##########
use_mnist="false"
batch_size=64
glob_learning_rate=0.9
clnt_subset_size=4900
srvr_subset_size=1000

# vd_prop=0.25
# run "cifar_25%vd"

vd_prop=0.2
run "cifar_20%vd"

vd_prop=0.15
run "cifar_15%vd"

vd_prop=0.10
run "cifar_10%vd"

vd_prop=0.05
run "cifar_5%vd"

cd "$ORIGINAL_DIR"