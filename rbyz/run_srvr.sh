#!/bin/bash

srvr_ip=192.168.117.103
port=2000
n_clients=1

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

build/srvr --srvr_ip $srvr_ip --port $port --n_clients $n_clients &
SRVR_PID=$!


for id in $(seq 1 $n_clients); do
  sleep 1
  build/clnt --srvr_ip $srvr_ip --port $port --id $id &
  # gdb -ex "break /home/bustaman/usr-rdma-api-main/rbyz/src/clnt.cpp:92" \
  #     -ex "start" \
  #     --args build/clnt --srvr_ip $srvr_ip --port $port --id $id 
  #valgrind --leak-check=full build/clnt --srvr_ip $srvr_ip --port $port --id $id &
  CLNT_PIDS+=($!)
done

# gdb -ex "break srvr.cpp:107" \
#     -ex "start" \
#     --args build/srvr --srvr_ip $srvr_ip --port $port --n_clients $n_clients 
# #valgrind --leak-check=full build/srvr --srvr_ip $srvr_ip --port $port --n_clients $n_clients &
# SRVR_PID=$!

wait $SRVR_PID "${CLNT_PIDS[@]}"
