#!/bin/bash

srvr_ip=192.168.128.103
port=2000
n_clients=10
remote_user="bustaman"
#remote_hosts=("dcldelta2" "dcldelta3" "dcldelta4")
remote_hosts=("dcldelta4")
remote_script_path="/home/bustaman/rbyz/rbyz"
results_path="/home/bustaman/rbyz/Results"
load_model=false
model_file="mnist_model_params.pt"

gdb -ex "break srvr.cpp:297" \
    -ex "break /home/bustaman/rbyz/rbyz/src/rbyzAux.cpp:197" \
    -ex "break /home/bustaman/rbyz/rbyz/src/rbyzAux.cpp:274" \
    -ex "start" \
    --args build/srvr --srvr_ip $srvr_ip --port $port --n_clients $n_clients $load_model_param --file $model_file 