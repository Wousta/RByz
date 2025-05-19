#!/bin/bash
srvr_ip=192.168.128.104 # Delta
#srvr_ip=192.168.117.103  # quatro
port=2000
n_clients=3
load_model=true
model_file="mnist_model_params.pt"

# Lyra handling of boolean flag
if [ "$load_model" = true ]; then
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
  build/clnt --srvr_ip $srvr_ip --port $port --id $id --n_clients $n_clients $load_model_param --file $model_file &
  # gdb -ex "break /home/bustaman/usr-rdma-api-main/rbyz/src/mnistTrain.cpp:409" \
  #     -ex "break /home/bustaman/usr-rdma-api-main/rbyz/src/clnt.cpp:154" \
  #     -ex "start" \
  #     --args build/clnt --srvr_ip $srvr_ip --port $port --id $id --n_clients $n_clients $load_model_param --file $model_file 
  #valgrind --leak-check=full build/clnt --srvr_ip $srvr_ip --port $port --id $id $load_model_param --file $model_file &
  CLNT_PIDS+=($!)
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
