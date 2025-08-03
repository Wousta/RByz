#!/bin/bash

# Byz attacks FLtrust vs RByz Experiment. 
# Methodology based in: https://arxiv.org/pdf/2007.08432
cleanup() {
    echo "Script interrupted. Cleaning up..."
    echo "Killing process $current_pid"
    kill $current_pid 2>/dev/null
    echo "Cleanup complete. Exiting..."
    exit 1
}
trap cleanup SIGTERM SIGINT INT TERM QUIT EXIT 

ORIGINAL_DIR=$(pwd)
EXPERIMENT="attacks_cf"
IP_ADDRESS=$(ip addr show | grep -A2 "ibp.*UP" | grep "inet " | head -1 | awk '{print $2}' | cut -d'/' -f1)
#REMOTE_HOSTS=("dcldelta4" "dcldelta2")
PORT="2200"
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
test_renewal_freq=3
vd_prop=0.03
vd_prop_write=1.0           # Proportion of total chunks writable on client to write each time the test is renewed
flip_ratio=1.0
batches_fpass=0.2
wait_all=1
epochs=3
local_steps_rbyz=3          # Local rounds of RByz

rm -rf ../logs/$EXPERIMENT/*
cd ../../rbyz

byz_clients_arr=(1 2)
run() {
    local name=$1
    local rounds=${2:-1}
    echo "=========================================================="
    echo "---- Starting experiment $name ----"
    echo "=========================================================="

    for byz_clients in "${byz_clients_arr[@]}"; do
        for ((i=1; i<=$rounds; i++)); do
            echo "______________________________________________________________________"
            echo "-- Experiment $name with $byz_clients byzantine clients round $i --"
            echo "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾"

            ./run_all.sh "${REMOTE_HOSTS[*]}" $EXPERIMENT $IP_ADDRESS $PORT $use_mnist $clients $epochs $batch_size $glob_learning_rate \
                $local_learn_rate $byz_clients $clnt_subset_size $srvr_subset_size $glob_iters_fl $local_steps_rbyz $glob_iters_rbyz \
                $chunk_size $label_flip_type $flip_ratio $only_flt $vd_prop $vd_prop_write $test_renewal_freq $overwrite_poisoned \
                $wait_all $batches_fpass &

            current_pid=$!
            wait $current_pid
            sleep 0.1
        done
        
        cd ../Results/logs/$EXPERIMENT
        mkdir -p "${byz_clients}_byz"
        mv *.log "${byz_clients}_byz/"
        cd ../../../rbyz
    done

    cd ../Results/logs/$EXPERIMENT
    mkdir -p "$name"
    mv *_byz "$name/"
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
glob_iters_fl=50
glob_iters_rbyz=0               

# label_flip_type=6          
# run "F_cifar_set_6" 3

label_flip_type=2
run "F_cifar_set_2" 3

# label_flip_type=4          
# run "F_cifar_set_4" 3

# label_flip_type=5          
# run "F_cifar_set_5" 3

#---------------# RByz #--------------#
only_flt=0 
glob_iters_fl=1
glob_iters_rbyz=49

# label_flip_type=6
# run "R_cifar_set_6" 3

# label_flip_type=4         
# run "R_cifar_set_4" 3

# label_flip_type=5         
# run "R_cifar_set_5" 3

cd "$ORIGINAL_DIR"