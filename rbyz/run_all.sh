#!/bin/bash

# Configuration
logs_dir=${1:-""}
srvr_ip=${2:-"192.168.128.103"}
port=2000
remote_user="bustaman"
#remote_hosts=("dcldelta2" "dcldelta3" "dcldelta4")
if [ -n "$3" ]; then
  # Convert space-separated string back to array
  read -ra remote_hosts <<< "$3"
else
  echo "No remote hosts provided, exiting"
  exit 1
fi
remote_script_path="/home/bustaman/rbyz/rbyz"
results_path="/home/bustaman/rbyz/Results"

# Must change the order of parameters here, number one is logs_dir, etc
use_mnist=${4:-false}       # First argument: true/false for MNIST vs CIFAR-10
n_clients=${5:-10}          
epochs=${6:-5}             
batch_size=${7:-64}        
glob_learn_rate=${8:-0.01}  # Global learning rate for FLtrust aggregation
local_learn_rate=${9:-0.01}  
n_byz_clnts=${10:-2}         
if [ "$use_mnist" = true ]; then
  # MNIST dataset 60000 training images
  load_use_mnist_param="--load"
  # Override defaults for MNIST if not explicitly set
  if [ $# -lt 6 ]; then epochs=5; fi
  if [ $# -lt 7 ]; then batch_size=32; fi
  clnt_subset_size=${11:-5900}
  srvr_subset_size=${12:-1000}
  glob_iters_fl=${13:-3}
  local_steps_rbyz=${14:-5}
  glob_iters_rbyz=${15:-3}
else
  # CIFAR-10 dataset 50000 training images
  load_use_mnist_param="" 
  glob_learn_rate=${8:-1.0}
  local_learn_rate=${9:-0.1}  
  clnt_subset_size=${11:-4900}
  srvr_subset_size=${12:-1000}
  glob_iters_fl=${13:-3}
  local_steps_rbyz=${14:-5}
  glob_iters_rbyz=${15:-50}
fi
chunk_size=${16:-1}      # slab size for RByz VDsampling

# 0: no label flip, 1: random label flip 2: random image corruption (RNG) setting
# 3: targeted label flip setting (1) 4: targeted label flip setting (2) 5: targeted label flip setting (3)
# references for the settings: CIFAR-10 -> https://arxiv.org/pdf/2007.08432 | MNIST -> https://arxiv.org/pdf/2407.07818v1
label_flip_type=${17:-1}

flip_ratio=${18:-0.50}
only_flt=${19:-0}             # Terminate after running FLtrust, to test FLtrust only (1) or run all (0)
vd_prop=${20:-0.20}           # Proportion of validation data for each client (proportion of total chunks writable on client)
vd_prop_write=${21:-0.1}      # Proportion of total chunks writable on client to write each time the test is renewed
test_renewal_freq=${22:-5}    # Frequency of test renewal (every n rounds)
overwrite_poisoned=${23:-0}   # Allow VD samples to overwrite poisoned samples (1) or not (0)
wait_all=${24:-0}             # Wait indefinitely for all clients (1) or not (0) in RByz


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
# rm -rf $results_path/logs/*

# Start the server process locally
echo "Starting server locally..."
taskset -c 0 build/srvr --logs_dir $logs_dir --srvr_ip $srvr_ip --port $port --n_clients $n_clients $load_use_mnist_param --n_byz $n_byz_clnts \
  --epochs $epochs --batch_size $batch_size --global_learn_rate $glob_learn_rate --local_learn_rate $local_learn_rate --clnt_subset_size $clnt_subset_size \
  --srvr_subset_size $srvr_subset_size --global_iters_fl $glob_iters_fl --local_steps_rbyz $local_steps_rbyz \
  --global_iters_rbyz $glob_iters_rbyz --chunk_size $chunk_size --label_flip_type $label_flip_type --flip_ratio $flip_ratio --only_flt $only_flt --vd_prop $vd_prop \
  --vd_prop_write $vd_prop_write --test_renewal_freq $test_renewal_freq --overwrite_poisoned $overwrite_poisoned --wait_all $wait_all & 
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
      build/profiler C $only_flt $logs_dir" &

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
build/profiler S $only_flt $logs_dir &
CPU_TRACKER_PID=$!

# Wait for the local server process
wait $SRVR_PID
cleanup

