#!/bin/bash
srvr_ip=192.168.128.103 # Delta
#srvr_ip=192.168.117.103  # quatro
port=2000
n_clients=2
use_mnist=false
model_file="mnist_model_params.pt"
debug_client_id=2

# Lyra handling of boolean flag
if [ "$use_mnist" = true ]; then
  load_model_param="--load"
else
  load_model_param=""
fi

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

build/srvr --srvr_ip $srvr_ip --port $port --n_clients $n_clients $load_model_param --file $model_file & 
SRVR_PID=$!


for id in $(seq 1 $n_clients); do
  sleep 1
  
  if [ "$debug_client_id" -eq "$id" ]; then
    # Run this client with gdb
    echo "Running client $id with gdb for debugging..."
    gdb -ex "break /home/bustaman/rbyz/rbyz/src/datasetLogic/baseRegDatasetMngr.cpp:474" \
        -ex "break rbyz/src/datasetLogic/regCIFAR10Mngr.cpp:20" \
        -ex "break rbyz/src/datasetLogic/regCIFAR10Mngr.cpp:72" \
        -ex "break rbyz/src/datasetLogic/regCIFAR10Mngr.cpp:30" \
        -ex "start" \
        --args build/clnt --srvr_ip $srvr_ip --port $port --id $id --n_clients $n_clients $load_model_param --file $model_file 
    CLNT_PIDS+=($!)
  else
    # Run this client normally in background
    build/clnt --srvr_ip $srvr_ip --port $port --id $id --n_clients $n_clients $load_model_param --file $model_file &
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
