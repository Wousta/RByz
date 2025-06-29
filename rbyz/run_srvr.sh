#!/bin/bash
srvr_ip=192.168.128.103 # Delta
#srvr_ip=192.168.117.103  # quatro
port=2000
debug_client_id=1

use_mnist=${1:-true}       # First argument: true/false for MNIST vs CIFAR-10
n_clients=${2:-1}          
epochs=${3:-5}             
batch_size=${4:-100}        
glob_learn_rate=${5:-0.05}  # Global learning rate for FLtrust aggregation
n_byz_clnts=${6:-0}         
if [ "$use_mnist" = true ]; then
  # MNIST dataset 60000 training images
  load_use_mnist_param="--load"
  # Override defaults for MNIST if not explicitly set
  if [ $# -lt 3 ]; then epochs=5; fi
  if [ $# -lt 4 ]; then batch_size=32; fi
  clnt_subset_size=${7:-58000}
  srvr_subset_size=${8:-2000}
  glob_iters_fl=${9:-3}
  local_steps_rbyz=${10:-5}
  glob_iters_rbyz=${11:-3}
else
  # CIFAR-10 dataset 50000 training images
  load_use_mnist_param=""
  clnt_subset_size=${7:-50000}
  srvr_subset_size=${8:-50000}
  glob_iters_fl=${9:-100}
  local_steps_rbyz=${10:-4}
  glob_iters_rbyz=${11:-2}
fi
chunk_size=${12:-2}      # slab size for RByz VDsampling
label_flip_type=${13:-0}
flip_ratio=${14:-0.25}
only_flt=${15:-1}  # Terminate after running FLtrust, to test FLtrust only (1) or run all (0)
vd_prop=${16:-1.0}  # Proportion of validation data for each client

# Array for client PIDs
CLNT_PIDS=()

cleanup() {
  echo "Terminating srvr and clnt processes..."
  kill $SRVR_PID "${CLNT_PIDS[@]}" 2>/dev/null
  #kill $SRVR_PID $CLNT_PID 2>/dev/null 
  exit 0
}

trap cleanup SIGINT SIGTERM

# Launch redis
echo "Starting Redis server on $srvr_ip:$port"
redis-server --bind "$srvr_ip" --port "$port" >/dev/null &
disown
sleep 1
redis-cli -h "$srvr_ip" -p "$port" SET srvr "0" >/dev/null
redis-cli -h "$srvr_ip" -p "$port" SET clnt "0" >/dev/null
redis-cli -h "$srvr_ip" -p "$port" SET nid "0" >/dev/null

echo "Redis server started on $srvr_ip:$port"

rm -rf logs/*

build/srvr --srvr_ip $srvr_ip --port $port --n_clients $n_clients $load_use_mnist_param --n_byz $n_byz_clnts \
  --epochs $epochs --batch_size $batch_size --global_learn_rate $glob_learn_rate --clnt_subset_size $clnt_subset_size \
  --srvr_subset_size $srvr_subset_size --global_iters_fl $glob_iters_fl --local_steps_rbyz $local_steps_rbyz \
  --global_iters_rbyz $glob_iters_rbyz --chunk_size $chunk_size --only_flt $only_flt --vd_prop $vd_prop & 
SRVR_PID=$!


for id in $(seq 1 $n_clients); do
  sleep 1
  
  if [ "$debug_client_id" -eq "$id" ]; then
    # Run this client with gdb
    echo "Running client $id with gdb for debugging..."
    gdb -ex "break /home/bustaman/rbyz/rbyz/src/entities/clnt.cpp:214" \
        -ex "start" \
        --args build/clnt --srvr_ip $srvr_ip --port $port --id $id --n_clients $n_clients $load_use_mnist_param --n_byz $n_byz_clnts \
          --epochs $epochs --batch_size $batch_size --global_learn_rate $glob_learn_rate --clnt_subset_size $clnt_subset_size \
          --srvr_subset_size $srvr_subset_size --global_iters_fl $glob_iters_fl --local_steps_rbyz $local_steps_rbyz \
          --global_iters_rbyz $glob_iters_rbyz --only_flt $only_flt --label_flip_type $label_flip_type --flip_ratio $flip_ratio
    CLNT_PIDS+=($!)
  else
    # Run this client normally in background
    build/clnt --srvr_ip $srvr_ip --port $port --id $id --n_clients $n_clients $load_use_mnist_param --n_byz $n_byz_clnts \
          --epochs $epochs --batch_size $batch_size --global_learn_rate $glob_learn_rate --clnt_subset_size $clnt_subset_size \
          --srvr_subset_size $srvr_subset_size --global_iters_fl $glob_iters_fl --local_steps_rbyz $local_steps_rbyz \
          --global_iters_rbyz $glob_iters_rbyz --only_flt $only_flt --label_flip_type $label_flip_type --flip_ratio $flip_ratio &
    CLNT_PIDS+=($!)
  fi
done

# gdb -ex "break srvr.cpp:200" \
#     -ex "break /home/bustaman/rbyz/rbyz/src/mnistTrain.cpp:463" \
#     -ex "break /home/bustaman/rbyz/rbyz/src/mnistTrain.cpp:465" \
#     -ex "break /home/bustaman/rbyz/rbyz/src/mnistTrain.cpp:477" \
#     -ex "break /home/bustaman/rbyz/rbyz/include/registeredMNIST.hpp:43" \
#     -ex "start" \
#     --args build/srvr --srvr_ip $srvr_ip --port $port --n_clients $n_clients $load_model_param --file $model_file 
# #valgrind --leak-check=full build/srvr --srvr_ip $srvr_ip --port $port --n_clients $n_clients $load_model_param --file $model_file &
# SRVR_PID=$!

wait $SRVR_PID "${CLNT_PIDS[@]}"
