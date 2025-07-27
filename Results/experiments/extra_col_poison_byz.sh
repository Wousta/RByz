#!/bin/bash

cleanup() {
    echo "Script interrupted. Cleaning up..."
    echo "Killing process $current_pid"
    kill $current_pid 2>/dev/null
    echo "Cleanup complete. Exiting..."
    exit 1
}
trap cleanup SIGTERM SIGINT INT TERM QUIT EXIT 

ORIGINAL_DIR=$(pwd)
EXPERIMENT="extra_col_poison_byz"
IP_ADDRESS=$(ip addr show | grep -A2 "ibp.*UP" | grep "inet " | head -1 | awk '{print $2}' | cut -d'/' -f1)
#REMOTE_HOSTS=("dcldelta4" "dcldelta2")
PORT="2400"
REMOTE_HOSTS=("dcldelta4")

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
local_learn_rate=0.01
chunk_size=2                # Slab size for RByz VDsampling
overwrite_poisoned=1        # Do not overwrite poisoned data
test_renewal_freq=3000
flip_ratio=1.0
batches_fpass=0.2
wait_all=1
epochs=3
local_steps_rbyz=3          # Local rounds of RByz
label_flip_type=2  
byz_clients=4
srvr_wait_inc=0
only_flt=0

rm -rf ../logs/$EXPERIMENT/*
cd ../../rbyz

run() {
    local name=$1
    echo "=========================================================="
    echo "---- Starting experiment $name iters: $iters ----"
    echo "=========================================================="

    for ((i=1; i<=3; i++)); do
        echo "______________________________________________________"
        echo "---- Running experiment $name iteration $i ----"
        ./run_all.sh "${REMOTE_HOSTS[*]}" $EXPERIMENT $IP_ADDRESS $PORT $use_mnist $clients $epochs $batch_size $glob_learning_rate \
            $local_learn_rate $byz_clients $clnt_subset_size $srvr_subset_size $glob_iters_fl $local_steps_rbyz $glob_iters_rbyz \
            $chunk_size $label_flip_type $flip_ratio $only_flt $vd_prop $vd_prop_write $test_renewal_freq $overwrite_poisoned \
            $wait_all $batches_fpass $srvr_wait_inc $extra_col_poison_prop &

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
clnt_subset_size=4854
srvr_subset_size=1460
vd_prop=0.03
vd_prop_write=1.0         
glob_learning_rate=1.0
local_learn_rate=0.01

#---------------# RByz #--------------#
glob_iters_fl=1
glob_iters_rbyz=49


extra_col_poison_prop=0.0
run "R_cifar_0%_poison"

extra_col_poison_prop=0.2
run "R_cifar_20%_poison"

extra_col_poison_prop=0.4
run "R_cifar_40%_poison"

extra_col_poison_prop=0.6
run "R_cifar_60%_poison"

extra_col_poison_prop=0.8
run "R_cifar_80%_poison"

extra_col_poison_prop=1.0
run "R_cifar_100%_poison"

cd "$ORIGINAL_DIR"