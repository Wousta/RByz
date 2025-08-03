#!/bin/bash
cleanup() {
    echo "Script interrupted. Cleaning up..."
    echo "Killing process $current_pid"
    kill $current_pid 2>/dev/null
    echo "Cleanup complete. Exiting..."
    exit 1
}
trap cleanup INT TERM QUIT EXIT SIGINT SIGTERM

# CPU and memory Utilization Experiment
ORIGINAL_DIR=$(pwd)
EXPERIMENT="cpu_gpu_mem_util"
IP_ADDRESS=$(ip addr show | grep -A2 "ibp.*UP" | grep "inet " | head -1 | awk '{print $2}' | cut -d'/' -f1)
REMOTE_HOSTS=("dcldelta4")
PORT="2350"

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
label_flip_type=1           # 1: Random label flip
flip_ratio=0.75  
chunk_size=2                # Slab size for RByz VDsampling
overwrite_poisoned=1        # Do not overwrite poisoned data
vd_prop=0.03
vd_prop_write=1.0           # Proportion of total chunks writable on client to write each time the test is renewed
flip_ratio=1.0
batches_fpass=0.2
wait_all=1
epochs=5
local_steps_rbyz=5
test_renewal_freq=1
label_flip_type=1


rm -rf ../logs/$EXPERIMENT/*
cd ../../rbyz

run() {
    local name=$1
    echo "=========================================================="
    echo "---- Starting experiment $name iters: $iters ----"
    echo "=========================================================="

    for ((i=1; i<=1; i++)); do
        echo "______________________________________________________"
        echo "---- Running experiment $name iteration $i ----"
        ./run_all.sh "${REMOTE_HOSTS[*]}" $EXPERIMENT $IP_ADDRESS $PORT $use_mnist $clients $epochs $batch_size $glob_learning_rate \
            $local_learn_rate $byz_clients $clnt_subset_size $srvr_subset_size $glob_iters_fl $local_steps_rbyz $glob_iters_rbyz \
            $chunk_size $label_flip_type $flip_ratio $only_flt $vd_prop $vd_prop_write $test_renewal_freq $overwrite_poisoned \
            $wait_all $batches_fpass &

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
glob_learning_rate=1.0
clnt_subset_size=4854
srvr_subset_size=1460
local_learn_rate=0.01


#-------------# FLtrust #-------------#
only_flt=1    
glob_iters_fl=100
glob_iters_rbyz=0  

run "F_cifar"

#---------------# RByz #--------------#
only_flt=0 
glob_iters_fl=1
glob_iters_rbyz=99

run "R_cifar"

cd "$ORIGINAL_DIR"


