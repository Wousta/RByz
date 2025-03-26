#!/bin/bash

# Configuration
srvr_ip=192.168.117.103
port=2000
n_clients=6
remote_user="bustaman"
remote_host="lpdquatro4"
remote_script_path="/home/bustaman/rbyz/fltrust"

# Server runs locally, clients run remotely
run_server_remote=false
run_clients_remote=true

# Array for client PIDs and SSH PIDs
CLNT_PIDS=()
SSH_PIDS=()

# Cleanup function: kill local and remote processes
cleanup() {
  echo "Terminating all processes..."
  
  # Kill local server process
  echo "Killing local server process..."
  kill $SRVR_PID 2>/dev/null
  
  # Kill remote client processes
  echo "Killing remote client processes..."
  ssh $remote_user@$remote_host "for pid in \$(cat $remote_script_path/clients.pid); do kill \$pid 2>/dev/null; done"
  
  # Kill SSH connections
  kill "${SSH_PIDS[@]}" 2>/dev/null
  
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
build/srvr --srvr_ip $srvr_ip --port $port --n_clients $n_clients &
SRVR_PID=$!

# Start the client processes remotely
echo "Starting clients on remote machine..."
ssh $remote_user@$remote_host "cd $remote_script_path && rm -f $remote_script_path/clients.pid && for id in \$(seq 1 $n_clients);\
                              do sleep 1; build/clnt --srvr_ip $srvr_ip --port $port --id \$id & echo \$! >> $remote_script_path/clients.pid;\
                              done" &
SSH_PIDS+=($!)

# Wait for the local server process
wait $SRVR_PID