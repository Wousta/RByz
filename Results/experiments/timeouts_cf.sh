#!/bin/bash

cleanup() {
    echo "Script interrupted. Cleaning up..."
    echo "Killing process $current_pid"
    kill $current_pid 2>/dev/null
    echo "Cleanup complete. Exiting..."
    exit 1
}
trap cleanup INT TERM QUIT EXIT SIGINT SIGTERM

# Comparison of timing out medium speed clients vs waiting for all Experiment
ORIGINAL_DIR=$(pwd)
EXPERIMENT="timeouts_cf"
IP_ADDRESS=$(ip addr show | grep -A2 "ibp.*UP" | grep "inet " | head -1 | awk '{print $2}' | cut -d'/' -f1)
REMOTE_HOSTS=("dcldelta4")
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
local_steps_rbyz=3          # Local rounds of RByz
glob_iters_fl=1
chunk_size=2                # Slab size for RByz VDsampling
label_flip_type=1           # 1: Random label flip
flip_ratio=0.75             # 50% of the data will be flipped
overwrite_poisoned=0        # Can overwrite poisoned data
only_flt=0                  # Run RByz
batches_fpass=0.2

rm -rf ../logs/$EXPERIMENT/*
cd ../../rbyz

run() {
    local name=$1
    echo "=========================================================="
    echo "---- Starting experiment $name ----"
    echo "=========================================================="
    echo "     Renewal freq: $test_renewal_freq"

    for ((i=1; i<=3; i++)); do
        echo "______________________________________________________"
        echo "---- Running experiment $name iteration $i ----"
        ./run_all.sh "${REMOTE_HOSTS[*]}" $EXPERIMENT $IP_ADDRESS $PORT $use_mnist $clients $epochs $batch_size $glob_learning_rate \
            $local_learn_rate $byz_clients $clnt_subset_size $srvr_subset_size $glob_iters_fl $local_steps_rbyz $glob_iters_rbyz \
            $chunk_size $label_flip_type $flip_ratio $only_flt $vd_prop $vd_prop_write $test_renewal_freq $overwrite_poisoned \
            $wait_all $batches_fpass $srvr_wait_inc &

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
########## CIFAR Experiments ##########
use_mnist="false"
batch_size=64
clnt_subset_size=4504
srvr_subset_size=4960
glob_learning_rate=0.4
glob_iters_rbyz=49
vd_prop=0.11                # Use the best value obtained from acc_vs_test_size.sh
vd_prop_write=0.2
test_renewal_freq=2       

wait_all=0

srvr_wait_inc=1
run "cifar_wait_1"

srvr_wait_inc=2
run "cifar_wait_2"

srvr_wait_inc=3
run "cifar_wait_3"

srvr_wait_inc=4
run "cifar_wait_4"

srvr_wait_inc=5
run "cifar_wait_5"

srvr_wait_inc=6
run "cifar_wait_6"

srvr_wait_inc=7
run "cifar_wait_7"

srvr_wait_inc=8
run "cifar_wait_8"

srvr_wait_inc=9
run "cifar_wait_9"

srvr_wait_inc=1
wait_all=1
run "cifar_wait_10"


cd "$ORIGINAL_DIR"