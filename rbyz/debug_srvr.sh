#!/bin/bash

srvr_ip=192.168.128.103
port=2000
remote_user="bustaman"
#remote_hosts=("dcldelta2" "dcldelta3" "dcldelta4")
remote_hosts=("dcldelta4")
remote_script_path="/home/bustaman/rbyz/rbyz"
results_path="/home/bustaman/rbyz/Results"

use_mnist=${1:-true}       # First argument: true/false for MNIST vs CIFAR-10
n_clients=${2:-50}          
epochs=${3:-5}             
batch_size=${4:-100}        
glob_learn_rate=${5:-0.05}  # Global learning rate for FLtrust aggregation
local_learn_rate=${6:-0.05}  
n_byz_clnts=${7:-0}         
if [ "$use_mnist" = true ]; then
  # MNIST dataset 60000 training images
  load_use_mnist_param="--load"
  # Override defaults for MNIST if not explicitly set
  if [ $# -lt 3 ]; then epochs=3; fi
  if [ $# -lt 4 ]; then batch_size=32; fi
  clnt_subset_size=${8:-1196}
  srvr_subset_size=${9:-200}
  glob_iters_fl=${10:-5}
  local_steps_rbyz=${11:-5}
  glob_iters_rbyz=${12:-5}
else
  # CIFAR-10 dataset 50000 training images
  load_use_mnist_param="" 
  glob_learn_rate=${5:-1.0}
  local_learn_rate=${6:-0.001}  
  clnt_subset_size=${8:-4900}
  srvr_subset_size=${9:-1000}
  glob_iters_fl=${10:-15}
  local_steps_rbyz=${11:-5}
  glob_iters_rbyz=${12:-5}
fi
chunk_size=${13:-1}      # slab size for RByz VDsampling

# 0: no label flip, 1: random label flip 
# 2: targeted label flip setting (1) 3: targeted label flip setting (2) 4: targeted label flip setting (3)
# references for the settings: CIFAR-10 -> https://arxiv.org/pdf/2007.08432 | MNIST -> https://arxiv.org/pdf/2407.07818v1
label_flip_type=${14:-2}

flip_ratio=${15:-1.0}
only_flt=${16:-0}  # Terminate after running FLtrust, to test FLtrust only (1) or run all (0)
vd_prop=${17:-1.0}  # Proportion of validation data for each client

# rm -rf logs/*
# rm -rf $results_path/logs/*
# rm -rf $results_path/accLogs/*

gdb -ex "start" \
    --args build/srvr --srvr_ip $srvr_ip --port $port --n_clients $n_clients $load_use_mnist_param --n_byz $n_byz_clnts \
  --epochs $epochs --batch_size $batch_size --global_learn_rate $glob_learn_rate --local_learn_rate $local_learn_rate --clnt_subset_size $clnt_subset_size \
  --srvr_subset_size $srvr_subset_size --global_iters_fl $glob_iters_fl --local_steps_rbyz $local_steps_rbyz \
  --global_iters_rbyz $glob_iters_rbyz --chunk_size $chunk_size --only_flt $only_flt --vd_prop $vd_prop 


# build/srvr --srvr_ip $srvr_ip --port $port --n_clients $n_clients $load_use_mnist_param --n_byz $n_byz_clnts \
#   --epochs $epochs --batch_size $batch_size --global_learn_rate $glob_learn_rate --clnt_subset_size $clnt_subset_size \
#   --srvr_subset_size $srvr_subset_size --global_iters_fl $glob_iters_fl --local_steps_rbyz $local_steps_rbyz \
#   --global_iters_rbyz $glob_iters_rbyz --chunk_size $chunk_size --only_flt $only_flt --vd_prop $vd_prop 