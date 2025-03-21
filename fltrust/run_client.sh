#!/bin/bash

# Run on server Node

srvr_ip=192.168.117.103
port=2000
n_clients=1

for id in $(seq 1 $n_clients); do
  /home/bustaman/rbyz/fltrust/build/clnt --srvr_ip $srvr_ip --port $port --id $id &
  # gdb -ex "break /home/bustaman/usr-rdma-api-main/fltrust/src/clnt.cpp:92" \
  #     -ex "break /home/bustaman/usr-rdma-api-main/rdma-api/src/rdma-api.cpp:125" \
  #     -ex "break /home/bustaman/usr-rdma-api-main/rdma-api/src/rdma-api.cpp:215" \
  #     -ex "start" \
  #     --args build/clnt --srvr_ip $srvr_ip --port $port --id $id 
  #valgrind --leak-check=full build/clnt --srvr_ip $srvr_ip --port $port --id $id &
  #CLNT_PIDS+=($!)
done
