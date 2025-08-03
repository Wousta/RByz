#!/bin/bash

# Byz attacks FLtrust vs RByz Experiment. 
# Methodology based in: https://arxiv.org/pdf/2007.08432
trap 'echo "Script interrupted. Exiting..."; exit 1' INT TERM
ORIGINAL_DIR=$(pwd)
EXPERIMENT="byz_attacks"
IP_ADDRESS=$(ip addr show | grep -A2 "ibp.*UP" | grep "inet " | head -1 | awk '{print $2}' | cut -d'/' -f1)
#REMOTE_HOSTS=("dcldelta4" "dcldelta2")
PORT="2200"
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
chunk_size=2                # Slab size for RByz VDsampling
overwrite_poisoned=1        # Do not overwrite poisoned data
test_renewal_freq=3
vd_prop=0.03
vd_prop_write=1.0           # Proportion of total chunks writable on client to write each time the test is renewed
flip_ratio=1.0
batches_fpass=0.2
wait_all=0
epochs=4
local_steps_rbyz=4          # Local rounds of RByz

rm -rf ../logs/$EXPERIMENT/*
cd ../../rbyz

byz_clients_arr=(0 1 2 3 4 5 6 7)
run() {
    local name=$1
    local rounds=$2
    echo "=========================================================="
    echo "---- Starting experiment $name ----"
    echo "=========================================================="

    for byz_clients in "${byz_clients_arr[@]}"; do
        echo "______________________________________________________________________"
        echo "---- Running experiment $name with $byz_clients byzantine clients ----"
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
    mkdir -p "$name"
    mv *_byz "$name/"
    cd ../../../rbyz
}

#######################################
########## MNIST Experiments ##########
use_mnist="true"
batch_size=32
clnt_subset_size=5825
srvr_subset_size=1750
glob_learning_rate=0.04

#-------------# FLtrust #-------------#
only_flt=1    
glob_iters_fl=150
glob_iters_rbyz=0      

label_flip_type=0           
run "F_mnist_set_0"

label_flip_type=3           
run "F_mnist_set_2"

#---------------# RByz #--------------#
only_flt=0 
glob_iters_fl=1
glob_iters_rbyz=149     

label_flip_type=0         
run "R_mnist_set_0"

label_flip_type=3         
run "R_mnist_set_3"

#######################################
########## CIFAR Experiments ##########
use_mnist="false"
batch_size=64
glob_learning_rate=0.4
clnt_subset_size=4854
srvr_subset_size=1460


#-------------# FLtrust #-------------#
only_flt=1    
glob_iters_fl=50
glob_iters_rbyz=0               

label_flip_type=0          
run "F_cifar_set_0"

label_flip_type=3          
run "F_cifar_set_3"

label_flip_type=4          
run "F_cifar_set_4"

label_flip_type=5          
run "F_cifar_set_5"

#---------------# RByz #--------------#
only_flt=0 
glob_iters_fl=1
glob_iters_rbyz=49


label_flip_type=0         
run "R_cifar_set_0"

label_flip_type=3         
run "R_cifar_set_3"

label_flip_type=4         
run "R_cifar_set_4"

label_flip_type=5         
run "R_cifar_set_5"

cd "$ORIGINAL_DIR"