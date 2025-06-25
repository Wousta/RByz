#!/bin/bash

# Configuration
srvr_ip=192.168.128.103
port=2000
remote_user="bustaman"
#remote_hosts=("dcldelta2" "dcldelta3" "dcldelta4")
remote_hosts=("dcldelta4")
remote_script_path="/home/bustaman/rbyz/rbyz"
results_path="/home/bustaman/rbyz/Results"
use_mnist=true   # MNIST or CIFAR-10 dataset

# Lyra handling of boolean flag
if [ "$use_mnist" = true ]; then
  # MNIST dataset 60000 training images
  load_use_mnist_param="--load"
  n_clients=10
  n_byz_clnts=2
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
  n_byz_clnts=2
  epochs=5
  batch_size=128
  glob_learn_rate=0.05
  clnt_subset_size=4900
  srvr_subset_size=1000
  glob_iters_fl=100
  local_steps_rbyz=6
  glob_iters_rbyz=10
fi

# Calculate clients per machine (even distribution)
clients_per_machine=$((n_clients / ${#remote_hosts[@]}))
remainder=$((n_clients % ${#remote_hosts[@]}))

# Cleanup function: kill local and remote processes
cleanup() {
  echo "Terminating all processes..."
  
  # Kill local server process
  echo "Killing local server process..."
  kill $SRVR_PID 2>/dev/null
  
  # Kill remote client processes on all machines
  echo "Killing remote client processes..."
  for host in "${remote_hosts[@]}"; do
    ssh $remote_user@$host "ps aux | grep clnt | grep -v grep | awk '{print \$2}' | xargs kill -9 2>/dev/null || true" &
    ssh $remote_user@$host "ps aux | grep cpuTracker | grep -v grep | awk '{print \$2}' | xargs kill -2 2>/dev/null || true" &
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
  --epochs $epochs --batch_size $batch_size --global_learn_rate $glob_learn_rate --clnt_subset_size $clnt_subset_size \
  --srvr_subset_size $srvr_subset_size --global_iters_fl $glob_iters_fl --local_steps_rbyz $local_steps_rbyz \
  --global_iters_rbyz $glob_iters_rbyz & 
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
      build/cpuTracker" &

    ssh $remote_user@$host "cd $remote_script_path && \
      core_id=0; \
      for id in ${client_ids[@]}; do \
        echo \"Starting client \$id on $host with physical core \$core_id\" && \
        taskset -c \$core_id build/clnt --srvr_ip $srvr_ip --port $port --id \$id --n_clients $n_clients $load_use_mnist_param --n_byz $n_byz_clnts \
          --epochs $epochs --batch_size $batch_size --global_learn_rate $glob_learn_rate --clnt_subset_size $clnt_subset_size \
          --srvr_subset_size $srvr_subset_size --global_iters_fl $glob_iters_fl --local_steps_rbyz $local_steps_rbyz \
          --global_iters_rbyz $glob_iters_rbyz & \
        core_id=\$((core_id + 1)); \
        if [ \$core_id -eq 16 ]; then core_id=0; fi; \
        sleep 0.1; \
      done" &
  fi
done

# Wait for the local server process
wait $SRVR_PID

#sleep 999999

