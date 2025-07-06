#!/bin/bash

# Configuration
srvr_ip=192.168.128.103
port=2000
remote_user="bustaman"
#remote_hosts=("dcldelta2" "dcldelta3" "dcldelta4")
remote_hosts=("dcldelta4")
remote_script_path="/home/bustaman/rbyz/rbyz"
results_path="/home/bustaman/rbyz/Results"

use_mnist=${1:-false}       # First argument: true/false for MNIST vs CIFAR-10
n_clients=${2:-10}          
epochs=${3:-7}             
batch_size=${4:-64}        
glob_learn_rate=${5:-0.05}  # Global learning rate for FLtrust aggregation
local_learn_rate=${6:-0.05}  
n_byz_clnts=${7:-0}         
if [ "$use_mnist" = true ]; then
  # MNIST dataset 60000 training images
  load_use_mnist_param="--load"
  # Override defaults for MNIST if not explicitly set
  if [ $# -lt 3 ]; then epochs=3; fi
  if [ $# -lt 4 ]; then batch_size=32; fi
  clnt_subset_size=${8:-5900}
  srvr_subset_size=${9:-1000}
  glob_iters_fl=${10:-8}
  local_steps_rbyz=${11:-5}
  glob_iters_rbyz=${12:-5}
else
  # CIFAR-10 dataset 50000 training images
  load_use_mnist_param="" 
  glob_learn_rate=${5:-1.0}
  local_learn_rate=${6:-0.1}  
  clnt_subset_size=${8:-4900}
  srvr_subset_size=${9:-1000}
  glob_iters_fl=${10:-200}
  local_steps_rbyz=${11:-5}
  glob_iters_rbyz=${12:-20}
fi
chunk_size=${13:-1}      # slab size for RByz VDsampling

# 0: no label flip, 1: random label flip 
# 2: targeted label flip setting (1) 3: targeted label flip setting (2) 4: targeted label flip setting (3)
# references for the settings: CIFAR-10 -> https://arxiv.org/pdf/2007.08432 | MNIST -> https://arxiv.org/pdf/2407.07818v1
label_flip_type=${14:-0}

flip_ratio=${15:-1.0}
only_flt=${16:-1}  # Terminate after running FLtrust, to test FLtrust only (1) or run all (0)
vd_prop=${17:-0.02}  # Proportion of validation data for each client
overwrite_poisoned=${18:-0}  # Allow VD samples to overwrite poisoned samples (1) or not (0)


# Calculate clients per machine (even distribution)
clients_per_machine=$((n_clients / ${#remote_hosts[@]}))
remainder=$((n_clients % ${#remote_hosts[@]}))

# Cleanup function: kill local and remote processes
cleanup() {
  echo "Terminating all processes..."
  
  # Kill local server process
  echo "Killing local server process..."
  kill $SRVR_PID 2>/dev/null
  ps aux | grep profiler | grep -v grep | awk '{print $2}' | xargs kill -2 2>/dev/null || true
  
  # Kill remote client processes on all machines
  echo "Killing remote client processes..."
  for host in "${remote_hosts[@]}"; do
    echo "Stopping remote profiler on $host..."
    ssh $remote_user@$host "ps aux | grep profiler | grep -v grep | awk '{print \$2}' | xargs -r kill -2 2>/dev/null; sleep 0.5" 
    ssh $remote_user@$host "ps aux | grep clnt | grep -v grep | awk '{print \$2}' | xargs -r kill -9 2>/dev/null || true" &
  done
  
  exit 0
}

# Trap SIGINT and SIGTERM to run cleanup
trap cleanup SIGINT SIGTERM

# Launch redis (disowned so it is not affected)
echo "Starting Redis server on $srvr_ip:$port"
redis-server --bind "$srvr_ip" --port "$port" >/dev/null &
disown
sleep 1
redis-cli -h "$srvr_ip" -p "$port" SET srvr "0" >/dev/null
redis-cli -h "$srvr_ip" -p "$port" SET clnt "0" >/dev/null
redis-cli -h "$srvr_ip" -p "$port" SET nid "0" >/dev/null

echo "Redis server started on $srvr_ip:$port"

rm -rf logs/*
rm -rf $results_path/logs/*
rm -rf $results_path/accLogs/*

# Start the server process locally
echo "Starting server locally..."
taskset -c 0 build/srvr --srvr_ip $srvr_ip --port $port --n_clients $n_clients $load_use_mnist_param --n_byz $n_byz_clnts \
  --epochs $epochs --batch_size $batch_size --global_learn_rate $glob_learn_rate --local_learn_rate $local_learn_rate --clnt_subset_size $clnt_subset_size \
  --srvr_subset_size $srvr_subset_size --global_iters_fl $glob_iters_fl --local_steps_rbyz $local_steps_rbyz \
  --global_iters_rbyz $glob_iters_rbyz --chunk_size $chunk_size --label_flip_type $label_flip_type --flip_ratio $flip_ratio --only_flt $only_flt --vd_prop $vd_prop \
  --overwrite_poisoned $overwrite_poisoned & 
SRVR_PID=$!

echo "Starting clients on remote machines..."
#sleep 10
client_id=1

for i in "${!remote_hosts[@]}"; do
  host=${remote_hosts[$i]}
  # Calculate how many clients to run on this machine
  local_clients=$clients_per_machine
  if [ $i -lt $remainder ]; then
    # Distribute remainder clients one per machine until used up
    local_clients=$((local_clients + 1))
  fi
  
  # Only start clients if there are any allocated to this machine
  if [ $local_clients -gt 0 ]; then
    echo "Starting $local_clients clients on $host..."
    
    # Create a range of client IDs for this machine
    client_ids=()
    for ((j=0; j<local_clients; j++)); do
      client_ids+=($client_id)
      client_id=$((client_id + 1))
    done

    # Start profiling CPU
    ssh $remote_user@$host "cd $results_path && \
      echo \"Starting CPU profiling on $host\" && \
      build/profiler C $only_flt" &

    ssh $remote_user@$host "cd $remote_script_path && \
      core_id=0; \
      for id in ${client_ids[@]}; do \
        echo \"Starting client \$id on $host with physical core \$core_id\" && \
        taskset -c \$core_id build/clnt --srvr_ip $srvr_ip --port $port --id \$id --n_clients $n_clients $load_use_mnist_param --n_byz $n_byz_clnts \
          --epochs $epochs --batch_size $batch_size --global_learn_rate $glob_learn_rate --local_learn_rate $local_learn_rate --clnt_subset_size $clnt_subset_size \
          --srvr_subset_size $srvr_subset_size --global_iters_fl $glob_iters_fl --local_steps_rbyz $local_steps_rbyz \
          --global_iters_rbyz $glob_iters_rbyz --only_flt $only_flt --label_flip_type $label_flip_type --flip_ratio $flip_ratio --overwrite_poisoned $overwrite_poisoned & \
        core_id=\$((core_id + 1)); \
        if [ \$core_id -eq 16 ]; then core_id=0; fi; \
        sleep 0.1; \
      done" &
  fi
done

cd $results_path
build/profiler S $only_flt &
CPU_TRACKER_PID=$!

# Wait for the local server process
wait $SRVR_PID
cleanup

