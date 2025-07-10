#!/bin/bash

# Accuracy vs Test Size Experiment. 
# Methodology based in: https://arxiv.org/pdf/2007.08432
trap 'echo "Script interrupted. Exiting..."; exit 1' INT TERM
ORIGINAL_DIR=$(pwd)
EXPERIMENT="byz_attacks"
IP_ADDRESS=$(ip addr show | grep -A2 "ibp.*UP" | grep "inet " | head -1 | awk '{print $2}' | cut -d'/' -f1)
#REMOTE_HOSTS=("dcldelta4" "dcldelta2")
REMOTE_HOSTS=("dcldelta4")

echo "Running experiment $EXPERIMENT on Server IP: $IP_ADDRESS"

# Common parameters
clients=50
epochs=5                    # Local rounds of FLtrust
local_steps_rbyz=5          # Local rounds of RByz
chunk_size=1                # Slab size for RByz VDsampling
overwrite_poisoned=0        # Do not overwrite poisoned data
test_renewal_freq=1         # Renew test samples every 5 rounds (fixed 50% of VD is renewed)
vd_prop=0.25
vd_prop_write=0.1           # Proportion of total chunks writable on client to write each time the test is renewed
flip_ratio=1.0

rm -rf ../logs/$EXPERIMENT/*
cd ../../rbyz

byz_clients_arr=(0 1 2 5 10 15 20 25)
run() {
    local name=$1
    echo "=========================================================="
    echo "---- Starting experiment $name ----"
    echo "=========================================================="

    for byz_clients in "${byz_clients_arr[@]}"; do
        echo "______________________________________________________________________"
        echo "---- Running experiment $name with $byz_clients byzantine clients ----"
        
        ./run_all.sh $EXPERIMENT $IP_ADDRESS "${REMOTE_HOSTS[*]}" $use_mnist $clients $epochs $batch_size $glob_learning_rate \
            $local_learn_rate $byz_clients $clnt_subset_size $srvr_subset_size $glob_iters_fl $local_steps_rbyz $glob_iters_rbyz \
            $chunk_size $label_flip_type $flip_ratio $only_flt $vd_prop $vd_prop_write $test_renewal_freq $overwrite_poisoned
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
glob_learning_rate=0.01
local_learn_rate=0.01
clnt_subset_size=200
srvr_subset_size=1196

#-------------# FLtrust #-------------#
only_flt=1    
glob_iters_fl=75
glob_iters_rbyz=0      

label_flip_type=0           
run "F_mnist_set_0"

label_flip_type=2           
run "F_mnist_set_2"

label_flip_type=3           
run "F_mnist_set_3"

label_flip_type=4           
run "F_mnist_set_4"

#---------------# RByz #--------------#
only_flt=0 
glob_iters_fl=3
glob_iters_rbyz=72     

label_flip_type=0         
run "R_mnist_set_0"

label_flip_type=2         
run "R_mnist_set_2"

label_flip_type=3         
run "R_mnist_set_3"

label_flip_type=4         
run "R_mnist_set_4"

#######################################
########## CIFAR Experiments ##########
use_mnist="false"
batch_size=64
glob_learning_rate=1.0
clnt_subset_size=200
srvr_subset_size=996

#-------------# FLtrust #-------------#
only_flt=1    
glob_iters_fl=75
glob_iters_rbyz=0               

label_flip_type=0          
run "F_cifar_set_0"

label_flip_type=2          
run "F_cifar_set_2"

label_flip_type=3          
run "F_cifar_set_3"

label_flip_type=4          
run "F_cifar_set_4"

#---------------# RByz #--------------#
only_flt=0 
glob_iters_fl=3
glob_iters_rbyz=72 

label_flip_type=0         
run "R_cifar_set_0"

label_flip_type=2         
run "R_cifar_set_2"

label_flip_type=3         
run "R_cifar_set_3"

label_flip_type=4         
run "R_cifar_set_4"

cd "$ORIGINAL_DIR"