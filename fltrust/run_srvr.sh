#!/bin/bash

srvr_ip=192.168.117.103
port=2000
n_clients=1

# Array for client PIDs
CLNT_PIDS=()

# Cleanup function: kill the srvr process and every client process.
cleanup() {
  echo "Terminating srvr and clnt processes..."
  kill $SRVR_PID "${CLNT_PIDS[@]}" 2>/dev/null
  #kill $SRVR_PID $CLNT_PID 2>/dev/null 
  exit 0
}

# Trap SIGINT and SIGTERM to run cleanup.
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

# Start the srvr process and capture its PID. srvr has id 0.
build/srvr --srvr_ip $srvr_ip --port $port --n_clients $n_clients &
#valgrind --leak-check=full build/srvr --srvr_ip $srvr_ip --port $port --n_clients $n_clients &
SRVR_PID=$!

# Start the client processes and store their PIDs.
#ssh bustaman@lpdquatro4 "/home/bustaman/rbyz/fltrust/run_client.sh" &
#CLNT_PID=$!
for id in $(seq 1 $n_clients); do
  build/clnt --srvr_ip $srvr_ip --port $port --id $id &
  # gdb -ex "break /home/bustaman/usr-rdma-api-main/fltrust/src/clnt.cpp:92" \
  #     -ex "break /home/bustaman/usr-rdma-api-main/rdma-api/src/rdma-api.cpp:125" \
  #     -ex "break /home/bustaman/usr-rdma-api-main/rdma-api/src/rdma-api.cpp:215" \
  #     -ex "start" \
  #     --args build/clnt --srvr_ip $srvr_ip --port $port --id $id 
  #valgrind --leak-check=full build/clnt --srvr_ip $srvr_ip --port $port --id $id &
  CLNT_PIDS+=($!)
done

# # Wait for the srvr and all client processes.
wait $SRVR_PID "${CLNT_PIDS[@]}"
#wait $SRVR_PID $CLNT_PID
