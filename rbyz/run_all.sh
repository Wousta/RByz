#!/bin/bash

# Configuration
#remote_hosts=("dcldelta2" "dcldelta3" "dcldelta4")
if [ -n "$1" ]; then
  # Convert space-separated string back to array
  read -ra remote_hosts <<< "$1"
else
  echo "No remote hosts provided, exiting"
  exit 1
fi
logs_dir=${2:-""}
srvr_ip=${3:-"192.168.128.103"}
port=${4:-"2000"}
remote_user="bustaman"
remote_script_path="/home/bustaman/rbyz/rbyz"
results_path="/home/bustaman/rbyz/Results"

# Must change the order of parameters here, number one is logs_dir, etc
use_mnist=${5:-false}       # First argument: true/false for MNIST vs CIFAR-10
n_clients=${6:-10}          
epochs=${7:-5}             
batch_size=${8:-64}        
glob_learn_rate=${9:-0.01}  # Global learning rate for FLtrust aggregation
local_learn_rate=${10:-0.01}  
n_byz_clnts=${11:-2}         
if [ "$use_mnist" = true ]; then
  # MNIST dataset 60000 training images
  load_use_mnist_param="--load"
  # Override defaults for MNIST if not explicitly set
  if [ $# -lt 7 ]; then epochs=5; fi
  if [ $# -lt 8 ]; then batch_size=32; fi
  clnt_subset_size=${12:-5900}
  srvr_subset_size=${13:-1000}
  glob_iters_fl=${14:-3}
  local_steps_rbyz=${15:-5}
  glob_iters_rbyz=${16:-3}
else
  # CIFAR-10 dataset 50000 training images
  load_use_mnist_param="" 
  glob_learn_rate=${9:-1.0}
  local_learn_rate=${10:-0.1}  
  clnt_subset_size=${12:-4900}
  srvr_subset_size=${13:-1000}
  glob_iters_fl=${14:-3}
  local_steps_rbyz=${15:-5}
  glob_iters_rbyz=${16:-50}
fi
chunk_size=${17:-1}      # slab size for RByz VDsampling

# 0: no label flip, 1: random label flip 2: random image corruption (RNG) setting
# 3: targeted label flip setting (1) 4: targeted label flip setting (2) 5: targeted label flip setting (3)
# references for the settings: CIFAR-10 -> https://arxiv.org/pdf/2007.08432 | MNIST -> https://arxiv.org/pdf/2407.07818v1
# 6: targeted label flip of FLtrust paper: https://arxiv.org/abs/2012.13995)
label_flip_type=${18:-1}

flip_ratio=${19:-0.50}
only_flt=${20:-0}             # Terminate after running FLtrust, to test FLtrust only (1) or run all (0)
vd_prop=${21:-0.20}           # Proportion of validation data for each client (proportion of total chunks writable on client)
vd_prop_write=${22:-0.1}      # Proportion of total chunks writable on client to write each time the test is renewed
test_renewal_freq=${23:-5}    # Frequency of test renewal (every n rounds)
overwrite_poisoned=${24:-0}   # Allow VD samples to overwrite poisoned samples (1) or not (0)
wait_all=${25:-0}             # Wait indefinitely for all clients (1) or not (0) in RByz
batches_fpass=${26:-0.2}
srvr_wait_inc=${27:-0}        # Server wait increment for slow clients in timeouts experiment (0 to not run experiment)
extra_col_poison_prop=${28:-0.0}  # For the progressive poisoning of trusted clients column experiment

# Calculate clients per machine (even distribution)
clients_per_machine=$((n_clients / ${#remote_hosts[@]}))
remainder=$((n_clients % ${#remote_hosts[@]}))

monitor_server_errors() {
    # Monitor server process for crashes that might indicate RDMA errors
    while kill -0 $SRVR_PID 2>/dev/null; do
        sleep 0.1
    done
    
    # Check exit code when process dies
    wait $SRVR_PID 2>/dev/null
    local exit_code=$?
    
    # Common exit codes for RDMA errors or crashes
    if [ $exit_code -ne 0 ] && [ $exit_code -ne 130 ] && [ $exit_code -ne 143 ]; then
        echo "Server crashed with exit code $exit_code, likely RDMA error"
        touch "/tmp/rdma_error_detected"
    fi
}

restart_experiment() {
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        echo "Starting attempt $((retry_count + 1)) of $max_retries"
        
        # Start server without log capture
        build/srvr --logs_dir $logs_dir --srvr_ip $srvr_ip --port $port --n_clients $n_clients $load_use_mnist_param --n_byz $n_byz_clnts \
          --epochs $epochs --batch_size $batch_size --global_learn_rate $glob_learn_rate --local_learn_rate $local_learn_rate --clnt_subset_size $clnt_subset_size \
          --srvr_subset_size $srvr_subset_size --global_iters_fl $glob_iters_fl --local_steps_rbyz $local_steps_rbyz \
          --global_iters_rbyz $glob_iters_rbyz --chunk_size $chunk_size --label_flip_type $label_flip_type --flip_ratio $flip_ratio  --only_flt $only_flt --vd_prop $vd_prop \
          --vd_prop_write $vd_prop_write --test_renewal_freq $test_renewal_freq --overwrite_poisoned $overwrite_poisoned --wait_all $wait_all \
          --batches_fpass $batches_fpass --srvr_wait_inc $srvr_wait_inc --extra_col_poison_prop $extra_col_poison_prop &
        
        SRVR_PID=$!
        echo "Started server with PID $SRVR_PID"
        
        # Start clients after a brief delay
        sleep 2
        start_clients
        
        # Monitor server process for crashes in background
        monitor_server_errors &
        MONITOR_PID=$!
        
        # Wait for server to complete or fail
        wait $SRVR_PID
        server_exit_code=$?
        
        # Kill the monitor process
        kill $MONITOR_PID 2>/dev/null
        wait $MONITOR_PID 2>/dev/null
        
        # Check if RDMA error was detected
        if [ -f "/tmp/rdma_error_detected" ]; then
            echo "RDMA error detected during execution. Cleaning up and retrying..."
            rm -f "/tmp/rdma_error_detected"
            cleanup_for_restart
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                sleep 5  # Wait before retry
            fi
        elif [ $server_exit_code -eq 0 ]; then
            echo "Experiment completed successfully"
            return 0
        else
            echo "Server failed with exit code $server_exit_code"
            # For non-zero exit codes, assume it might be RDMA related and retry
            echo "Assuming RDMA-related error. Cleaning up and retrying..."
            cleanup_for_restart
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                sleep 5
            fi
        fi
    done
    
    echo "Max retries reached. Experiment failed."
    exit 1
}

start_clients() {
    echo "Starting clients on remote machines..."
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
            taskset -c \$core_id build/clnt --srvr_ip $srvr_ip --port $port --id \$id --n_clients $n_clients $load_use_mnist_param --n_byz $n_byz_clnts \
              --epochs $epochs --batch_size $batch_size --global_learn_rate $glob_learn_rate --local_learn_rate $local_learn_rate --clnt_subset_size $clnt_subset_size \
              --srvr_subset_size $srvr_subset_size --global_iters_fl $glob_iters_fl --local_steps_rbyz $local_steps_rbyz \
              --global_iters_rbyz $glob_iters_rbyz --only_flt $only_flt --label_flip_type $label_flip_type --flip_ratio $flip_ratio --overwrite_poisoned $overwrite_poisoned \
              --vd_prop $vd_prop --batches_fpass $batches_fpass --srvr_wait_inc $srvr_wait_inc & \

            echo \"Client \$id started on $host with physical core \$core_id\ (PID: \$!)\" && \
            core_id=\$((core_id + 1)); \
            if [ \$core_id -eq 16 ]; then core_id=0; fi; \
            sleep 0.2; \
          done" &
      fi
    done
}

cleanup_for_restart() {
    echo "Cleaning up for restart..."
    
    # Kill server process
    if [ ! -z "$SRVR_PID" ]; then
        kill $SRVR_PID 2>/dev/null
    fi
    
    # Kill monitor process if it exists
    if [ ! -z "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null
        wait $MONITOR_PID 2>/dev/null
    fi
    
    # Kill local profiler processes
    ps aux | grep profiler | grep -v grep | awk '{print $2}' | xargs -r kill -2 2>/dev/null
    
    # Clean up remote clients and profilers
    for host in "${remote_hosts[@]}"; do
        ssh $remote_user@$host "
            # Kill client processes
            ps aux | grep clnt | grep -v grep | awk '{print \$2}' | xargs -r kill -9 2>/dev/null
            # Kill profiler processes
            ps aux | grep profiler | grep -v grep | awk '{print \$2}' | xargs -r kill -2 2>/dev/null
        " &
    done
    wait
    
    # Clear RDMA resources and temporary files
    echo "Clearing RDMA resources..."
    rm -f "/tmp/rdma_error_detected" 2>/dev/null
    sleep 2
}

# Cleanup function: kill local and remote processes
cleanup() {
  echo "Terminating all processes..."
  
  # Kill local server process
  echo "Killing local server process..."
  echo "Profiler PID: $CPU_TRACKER_PID"
  kill $SRVR_PID 2>/dev/null
  ps aux | grep profiler | grep -v grep | awk '{print $2}' | xargs -r kill -2 2>/dev/null
  
  # Kill remote client processes on all machines
  for host in "${remote_hosts[@]}"; do
    echo "Stopping remote profiler on $host..."
    ssh $remote_user@$host "ps aux | grep profiler | grep -v grep | awk '{print \$2}' | xargs -r kill -2 2>/dev/null; sleep 0.5" 
    
    # Gather client PIDs, kill them, then remove logs
    ssh $remote_user@$host "
      cd $remote_script_path
      # Gather all client PIDs first
      client_pids=\$(ps aux | grep clnt | grep -v grep | awk '{print \$2}')
      
      # Kill all client processes
      if [ -n \"\$client_pids\" ]; then
        echo \$client_pids | xargs -r kill -9 2>/dev/null || true
        sleep 0.2
        
        # Now remove their log files
        for pid in \$client_pids; do
          if [ -f logs/execution_\${pid}.log ]; then
            rm -f logs/execution_\${pid}.log
          fi
        done
      fi
    " &
  done

  #rm -f /home/bustaman/rbyz/rbyz/logs/execution_${SRVR_PID}.log 2>/dev/null

  echo "All processes terminated."
  
  exit 0
}

# Trap SIGINT and SIGTERM to run cleanup
trap cleanup SIGINT SIGTERM

# # Launch redis (disowned so it is not affected)
# echo "Starting Redis server on $srvr_ip:$port"
# redis-server --bind "$srvr_ip" --port "$port" >/dev/null &
# disown
# sleep 1
# redis-cli -h "$srvr_ip" -p "$port" SET srvr "0" >/dev/null
# redis-cli -h "$srvr_ip" -p "$port" SET clnt "0" >/dev/null
# redis-cli -h "$srvr_ip" -p "$port" SET nid "0" >/dev/null

# echo "Redis server started on $srvr_ip:$port"

# rm -rf logs/*
# rm -rf $results_path/logs/*

# Start the server process locally
restart_experiment

# Start CPU tracker for the successful server process
cd $results_path
echo "Starting CPU tracker for local server process..."
build/profiler S $only_flt $logs_dir &
CPU_TRACKER_PID=$!

# Wait for the local server process (this should only run after successful restart)
wait $SRVR_PID
cleanup

