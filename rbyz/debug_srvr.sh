#!/bin/bash

srvr_ip=192.168.128.103
port=2000
n_clients=10
remote_user="bustaman"
#remote_hosts=("dcldelta2" "dcldelta3" "dcldelta4")
remote_hosts=("dcldelta4")
remote_script_path="/home/bustaman/rbyz/rbyz"
results_path="/home/bustaman/rbyz/Results"
use_mnist=false   # MNIST or CIFAR-10 dataset

# Lyra handling of boolean flag
if [ "$use_mnist" = true ]; then
  # MNIST dataset 60000 training images
  load_use_mnist_param="--load"
  n_clients=10 #$1
  epochs=5
  batch_size=32
  glob_learn_rate=0.05
  clnt_subset_size=5900
  srvr_subset_size=1000
  glob_iters_fl=5
  local_steps_rbyz=6
  glob_iters_rbyz=10
else
  # CIFAR-10 dataset 50000 training images
  load_use_mnist_param=""
  n_clients=0
  epochs=30
  batch_size=100
  glob_learn_rate=0.01
  clnt_subset_size=50000
  srvr_subset_size=50000
  glob_iters_fl=100
  local_steps_rbyz=6
  glob_iters_rbyz=5
fi
n_byz_clnts=0
chunk_size=1      # slab size for RByz VDsampling
label_flip_type=0
flip_ratio=0.25
only_flt=1  # Terminate after running FLtrust, to test FLtrust only (1) or run all (0)

rm -rf logs/*
rm -rf $results_path/logs/*
rm -rf $results_path/accLogs/*

gdb -ex "break /home/bustaman/rbyz/rbyz/src/datasetLogic/baseRegDatasetMngr.cpp:298" \
    -ex "break /home/bustaman/rbyz/rbyz/src/datasetLogic/baseRegDatasetMngr.cpp:192" \
    -ex "break /home/bustaman/rbyz/rbyz/src/datasetLogic/baseRegDatasetMngr.cpp:246" \
    -ex "start" \
    --args build/srvr --srvr_ip $srvr_ip --port $port --n_clients $n_clients $load_use_mnist_param --n_byz $n_byz_clnts \
  --epochs $epochs --batch_size $batch_size --global_learn_rate $glob_learn_rate --clnt_subset_size $clnt_subset_size \
  --srvr_subset_size $srvr_subset_size --global_iters_fl $glob_iters_fl --local_steps_rbyz $local_steps_rbyz \
  --global_iters_rbyz $glob_iters_rbyz --chunk_size $chunk_size --only_flt $only_flt 
    #-ex "break /home/bustaman/rbyz/rdma-api/src/rdma-api.cpp:22" \