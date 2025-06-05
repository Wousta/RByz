#!/bin/bash

# Configuration
srvr_ip=192.168.128.101
port=2000
n_clients=10
remote_user="bustaman"
#remote_hosts=("dcldelta2" "dcldelta3" "dcldelta4")
remote_hosts=("dcldelta4")
remote_script_path="/home/bustaman/rbyz/fltrust"

# Calculate clients per machine (even distribution)
clients_per_machine=$((n_clients / ${#remote_hosts[@]}))
remainder=$((n_clients % ${#remote_hosts[@]}))

# Server runs locally, clients run remotely
run_server_remote=false
run_clients_remote=true

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

# Start the server process locally
echo "Starting server locally..."
taskset -c 0 build/srvr --srvr_ip $srvr_ip --port $port --n_clients $n_clients & 
SRVR_PID=$!

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
    
    # Start clients on this machine using SSH with keys
    # ssh $remote_user@$host "cd $remote_script_path && rm -f $remote_script_path/clients_${host}.pid && \
    #   for id in ${client_ids[@]}; do \
    #     echo \"Starting client \$id on $host\" && \
    #     build/clnt --srvr_ip $srvr_ip --port $port --id \$id --n_clients $n_clients & \
    #     echo \$! >> $remote_script_path/clients_${host}.pid; \
    #     sleep 0.5; \
    #   done" &

    ssh $remote_user@$host "cd $remote_script_path && rm -f $remote_script_path/clients_${host}.pid && \
      core_id=0; \
      for id in ${client_ids[@]}; do \
        echo \"Starting client \$id on $host with physical core \$core_id\" && \
        taskset -c \$core_id build/clnt --srvr_ip $srvr_ip --port $port --id \$id --n_clients $n_clients & \
        echo \$! >> $remote_script_path/clients_${host}.pid; \
        core_id=\$((core_id + 1)); \
        if [ \$core_id -eq 16 ]; then core_id=0; fi; \
        sleep 0.5; \
      done" &
  fi
done

# Wait for the local server process
wait $SRVR_PID

